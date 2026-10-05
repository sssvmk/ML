import pytest
import torch
import torch.nn as nn
from smoke_over import SMOKE
from torchvision.ops.misc import FrozenBatchNorm2d

from ssd_voc.config import load_config
from ssd_voc.model import SSD, L2Norm, build_model, count_params, load_compatible
from ssd_voc.serve import export_torchscript


def tiny_cfg(*extra):
    return load_config("voc", [*SMOKE, *extra])


def test_ssd300_resnet50_has_the_papers_feature_maps_and_box_count():
    m = build_model(load_config("voc", ["model.pretrained=none"])).eval()
    assert m.feature_sizes == [38, 19, 10, 5, 3, 1] and m.channels == [512, 1024, 512, 256, 256, 256]
    assert m.num_priors == 8732 and m.boxes_per_loc == [4, 6, 6, 6, 4, 4]
    with torch.no_grad():
        loc, conf = m(torch.zeros(2, 3, 300, 300))
    assert loc.shape == (2, 8732, 4) and conf.shape == (2, 8732, 21)


def test_ssd512_variant():
    cfg = load_config("voc", ["model.pretrained=none", "model.backbone=resnet18", "model.input_size=512",
                              "model.extras=[[256,512,2,1],[128,256,2,1],[128,256,2,1],[128,256,2,1],[128,256,2,1]]",
                              "model.boxes_per_loc=[4,6,6,6,6,4,4]"])
    m = build_model(cfg)
    assert m.feature_sizes == [64, 32, 16, 8, 4, 2, 1] and m.num_priors == 24564


def test_freezing_and_frozen_batchnorm():
    m = build_model(load_config("voc", ["model.pretrained=none"]))
    assert not any(p.requires_grad for p in m.stem.parameters()) and not any(p.requires_grad for p in m.layer1.parameters())
    assert all(p.requires_grad for p in m.layer2.parameters()) and all(p.requires_grad for p in m.layer3.parameters())
    assert any(isinstance(x, FrozenBatchNorm2d) for x in m.modules()) and not any(type(x) is nn.BatchNorm2d for x in m.modules())
    assert count_params(m, True) < count_params(m)
    none_frozen = build_model(load_config("voc", ["model.pretrained=none", 'model.freeze_up_to=""', "model.freeze_bn=false"]))
    assert count_params(none_frozen, True) == count_params(none_frozen)
    layer2_frozen = build_model(load_config("voc", ["model.pretrained=none", "model.freeze_up_to=layer2"]))
    assert not any(p.requires_grad for p in layer2_frozen.layer2.parameters())
    with pytest.raises(ValueError, match="freeze_up_to"):
        build_model(load_config("voc", ["model.pretrained=none", "model.freeze_up_to=layer3"]))
    with pytest.raises(ValueError, match="backbone"):
        build_model(load_config("voc", ["model.backbone=vgg16"]))


def test_initialisation_follows_the_paper_with_documented_class_head_deviation():
    m = build_model(tiny_cfg())
    for mod in [*m.extras.modules(), *m.loc_heads.modules()]:
        if isinstance(mod, nn.Conv2d):
            fan_in, fan_out = mod.in_channels * mod.kernel_size[0] ** 2, mod.out_channels * mod.kernel_size[0] ** 2
            assert float(mod.weight.abs().max()) <= (6 / (fan_in + fan_out)) ** 0.5 + 1e-6      # Xavier uniform bound
            assert float(mod.bias.abs().sum()) == 0
    for h in m.conf_heads:
        fan_in = h.in_channels * 9
        assert float(h.weight.std()) == pytest.approx(0.1 / fan_in ** 0.5, rel=0.1) and float(h.bias.abs().sum()) == 0
    assert isinstance(m.l2norm, L2Norm) and torch.all(m.l2norm.weight == 20.0)                  # paper: scale 20


def test_weight_decay_only_on_weights():
    m = build_model(tiny_cfg())
    groups = m.param_groups(5e-4)
    assert groups[0]["weight_decay"] == 5e-4 and groups[1]["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in groups[0]["params"]) and all(p.ndim <= 1 for p in groups[1]["params"])
    assert any(p is m.l2norm.weight for p in groups[1]["params"])


def test_dropout_is_off_by_default_and_optional():
    assert not any(isinstance(x, nn.Dropout2d) for x in build_model(tiny_cfg()).modules())
    assert any(isinstance(x, nn.Dropout2d) for x in build_model(tiny_cfg("model.dropout=0.2")).modules())


def test_pretrained_backbone_weights_are_loaded_and_frozen_bn_gets_their_statistics(monkeypatch):
    import torchvision
    from torchvision.models._api import WeightsEnum
    src = torchvision.models.resnet18(weights=None)
    for p in src.parameters():
        nn.init.normal_(p, std=0.05)
    src.bn1.running_mean.fill_(3.0)
    monkeypatch.setattr(WeightsEnum, "get_state_dict", lambda self, *a, **k: src.state_dict())
    cfg = tiny_cfg("model.freeze_bn=true", "model.pretrained=imagenet")
    m = build_model(cfg)
    assert torch.equal(m.layer2[0].conv1.weight, src.layer2[0].conv1.weight)
    assert torch.equal(m.stem[0].weight, src.conv1.weight) and float(m.stem[1].running_mean[0]) == 3.0
    assert not torch.equal(build_model(cfg, pretrained=False).layer2[0].conv1.weight, src.layer2[0].conv1.weight)


def test_load_compatible_skips_class_heads_when_the_number_of_classes_changes():
    a = build_model(tiny_cfg("data.synthetic.classes=3"))
    b = build_model(tiny_cfg("data.synthetic.classes=5"))
    rep = load_compatible(b, a.state_dict(), ("conf_heads.",))
    assert rep["skipped"] and all(k.startswith("conf_heads.") for k in rep["skipped"])
    assert torch.equal(b.loc_heads[0].weight, a.loc_heads[0].weight) and torch.equal(b.layer2[0].conv1.weight, a.layer2[0].conv1.weight)
    same = load_compatible(a, a.state_dict(), ())
    assert not same["skipped"] and not same["missing"]


def test_torchscript_export_matches_eager_for_any_batch_size(tmp_path):
    m = build_model(tiny_cfg()).eval()
    export_torchscript(m, tmp_path / "m.pt")
    ts = torch.jit.load(str(tmp_path / "m.pt"))
    x = torch.randn(3, 3, 128, 128)
    with torch.no_grad():
        (l1, c1), (l2, c2) = m(x), ts(x)
    assert torch.allclose(l1, l2, atol=1e-5) and torch.allclose(c1, c2, atol=1e-5)
    assert isinstance(m, SSD)
