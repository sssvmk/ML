import numpy as np
import pytest
import torch
import torch.nn as nn
from smoke_over import SMOKE
from torchvision.ops.misc import FrozenBatchNorm2d

import detr_voc.ops as ops
from detr_voc.config import STAGES, load_config, parse_overrides, registered_name
from detr_voc.model import DETR, PositionEmbeddingSine, build_model, count_params, load_compatible


def tiny_cfg(*extra):
    return load_config("voc", [*SMOKE, *extra])


# ------------------------------------------------------------------ config
def test_defaults_follow_the_paper():
    c = load_config("voc")
    assert (c["model"]["hidden_dim"], c["model"]["nheads"], c["model"]["enc_layers"], c["model"]["dec_layers"]) == (256, 8, 6, 6)
    assert c["model"]["num_queries"] == 100 and c["model"]["dropout"] == 0.1 and c["model"]["aux_loss"] and c["model"]["freeze_bn"]
    assert (c["loss"]["cost_class"], c["loss"]["cost_bbox"], c["loss"]["cost_giou"]) == (1.0, 5.0, 2.0)
    assert (c["loss"]["w_ce"], c["loss"]["w_bbox"], c["loss"]["w_giou"], c["loss"]["eos_coef"]) == (1.0, 5.0, 2.0, 0.1)
    s = c["schedule"]
    assert s["lr_backbone"] == 1e-5 and [(p["lr"], p["epochs"]) for p in s["phases"]] == [(1e-4, 200), (1e-5, 100)]
    assert s["weight_decay"] == 1e-4 and s["grad_clip"] == 0.1
    a = c["aug"]
    assert a["scales"][0] == 480 and a["scales"][-1] == 800 and a["max_size"] == 1333 and a["crop_resize"] == [400, 500, 600]
    assert (a["crop_min"], a["crop_max"], a["crop_prob"]) == (384, 600, 0.5)


def test_coco_stage_uses_the_500_epoch_schedule_and_overrides_work():
    c = load_config("coco")
    assert c["data"]["num_foreground"] == 80 and c["data"]["test_set"] is None
    assert [(p["lr"], p["epochs"]) for p in c["schedule"]["phases"]] == [(1e-4, 400), (1e-5, 100)]
    assert set(STAGES) == {"voc", "coco"}
    assert load_config("voc", ["schedule.eval_every=7"])["schedule"]["eval_every"] == 7
    assert parse_overrides(["a.b=1", "c=[1,2]"]) == {"a.b": 1, "c": [1, 2]}
    for bad, msg in ((["broken"], "a.b=value"), (["model.hidden_dim=100"], "divisible"), (["data.dataset=imagenet"], "unknown dataset")):
        with pytest.raises(ValueError, match=msg):
            load_config("voc", bad)
    with pytest.raises(ValueError, match="unknown stage"):
        load_config("nope")
    assert registered_name(load_config("voc")) == "detr-resnet50-voc2012"


# ------------------------------------------------------------------ hungarian + boxes
def test_numpy_hungarian_matches_scipy_on_random_rectangular_matrices():
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(0)
    for shape in [(5, 5), (100, 7), (7, 100), (100, 1), (1, 100), (60, 60), (100, 40)]:
        for _ in range(8):
            c = rng.standard_normal(shape)
            r1, c1 = scipy.linear_sum_assignment(c)
            r2, c2 = ops.hungarian_numpy(c)
            assert len(r2) == min(shape) and len(set(c2.tolist())) == len(c2) and len(set(r2.tolist())) == len(r2)
            assert abs(c[r1, c1].sum() - c[r2, c2].sum()) < 1e-9
    r, c = ops.hungarian_numpy(np.zeros((4, 0)))
    assert len(r) == 0 and len(c) == 0


def test_a_broken_scipy_install_falls_back_to_the_numpy_solver(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("scipy"):
            raise AttributeError("module 'numpy' has no attribute 'long'")      # the error your cluster raised
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    monkeypatch.setattr(ops, "_SCIPY", {})
    r, c = ops.solve_assignment(np.array([[4.0, 1.0], [2.0, 3.0]]))
    assert r.tolist() == [0, 1] and c.tolist() == [1, 0] and ops._SCIPY["fn"] is ops.hungarian_numpy


def test_giou_and_box_conversions():
    a, b = torch.tensor([[0.0, 0, 2, 2]]), torch.tensor([[1.0, 1, 3, 3], [0, 0, 2, 2]])
    g = ops.giou_matrix(a, b)[0]
    assert float(g[1]) == pytest.approx(1.0) and float(g[0]) == pytest.approx(1 / 7 - 2 / 9, abs=1e-6)   # IoU 1/7, enclosing 9, union 7
    x = torch.tensor([[0.3, 0.4, 0.2, 0.1]])
    assert torch.allclose(ops.box_xyxy_to_cxcywh(ops.box_cxcywh_to_xyxy(x)), x, atol=1e-6)


# ------------------------------------------------------------------ model
def test_detr_r50_has_the_papers_parameter_count():
    m = build_model(load_config("voc", ["model.pretrained=none"]))
    assert 41.0e6 < count_params(m) < 41.8e6 and 23.3e6 < count_params(m.backbone) < 23.7e6       # paper: 41.3M, ResNet-50 23.5M
    assert 41.0e6 < count_params(m, True) < 41.5e6                                                  # stem + layer1 frozen
    assert m.class_embed.out_features == 21 and m.query_embed.num_embeddings == 100


def test_forward_shapes_aux_outputs_and_box_range():
    m = build_model(tiny_cfg()).eval()
    x, mask = torch.randn(2, 3, 96, 128), torch.zeros(2, 96, 128, dtype=torch.bool)
    mask[1, 64:, :] = True
    with torch.no_grad():
        out = m(x, mask)
    assert out["pred_logits"].shape == (2, 10, 4) and out["pred_boxes"].shape == (2, 10, 4)       # 3 classes + no-object
    assert len(out["aux_outputs"]) == 1 and 0 <= float(out["pred_boxes"].min()) and float(out["pred_boxes"].max()) <= 1
    six = build_model(tiny_cfg("model.dec_layers=6")).eval()
    with torch.no_grad():
        assert len(six(x, mask)["aux_outputs"]) == 5                                                  # one per decoder layer except the last
    assert "aux_outputs" not in build_model(tiny_cfg("model.aux_loss=false")).eval()(x, mask)


def test_sine_position_encoding_ignores_the_padding():
    pe = PositionEmbeddingSine(32)
    small = pe(torch.zeros(1, 10, 12, dtype=torch.bool))
    padded_mask = torch.ones(1, 16, 20, dtype=torch.bool)
    padded_mask[:, :10, :12] = False
    big = pe(padded_mask)
    assert small.shape == (1, 64, 10, 12) and torch.allclose(big[:, :, :10, :12], small, atol=1e-5)


def test_backbone_freezing_and_frozen_batchnorm():
    m = build_model(load_config("voc", ["model.pretrained=none"]))
    b = m.backbone
    assert not any(p.requires_grad for p in b.stem.parameters()) and not any(p.requires_grad for p in b.layer1.parameters())
    assert all(p.requires_grad for p in b.layer2.parameters()) and all(p.requires_grad for p in b.layer4.parameters())
    assert any(isinstance(x, FrozenBatchNorm2d) for x in b.modules()) and not any(type(x) is nn.BatchNorm2d for x in b.modules())
    assert any(type(x) is nn.BatchNorm2d for x in build_model(tiny_cfg()).backbone.modules())          # freeze_bn=false in SMOKE
    with pytest.raises(ValueError, match="DC5"):
        build_model(tiny_cfg("model.dilation=true"))
    plain = build_model(tiny_cfg("model.backbone=resnet50", "model.enc_layers=1", "model.dec_layers=1")).eval()
    dc5 = build_model(tiny_cfg("model.backbone=resnet50", "model.enc_layers=1", "model.dec_layers=1", "model.dilation=true")).eval()
    x, mask = torch.randn(1, 3, 64, 64), torch.zeros(1, 64, 64, dtype=torch.bool)
    with torch.no_grad():
        assert plain.backbone(x, mask)[0].shape[-1] == 2 and dc5.backbone(x, mask)[0].shape[-1] == 4      # stride 32 vs 16
        assert dc5(x, mask)["pred_logits"].shape == (1, 10, 4)
    with pytest.raises(ValueError, match="backbone"):
        build_model(load_config("voc", ["model.backbone=vgg16"]))


def test_transformer_uses_xavier_init_and_dropout_layers():
    m = build_model(tiny_cfg())
    for name, p in m.transformer.named_parameters():
        if p.dim() > 1:
            fan_out, fan_in = p.shape[0], p.shape[1]
            assert float(p.abs().max()) <= (6 / (fan_in + fan_out)) ** 0.5 + 1e-6, name
    assert sum(isinstance(x, nn.Dropout) for x in m.transformer.modules()) == 3 * 1 + 4 * 2          # 1 enc layer, 2 dec layers
    assert all(att.dropout == 0.1 for att in m.modules() if isinstance(att, nn.MultiheadAttention))


def test_param_groups_give_the_backbone_a_tenth_of_the_learning_rate():
    m = build_model(tiny_cfg())
    g = m.param_groups(1e-4, 1e-5, 1e-4)
    assert [x["lr"] for x in g] == [1e-4, 1e-5] and g[1]["lr_mult"] == pytest.approx(0.1) and all(x["weight_decay"] == 1e-4 for x in g)
    assert all(n.startswith("backbone") for n, p in m.named_parameters() if any(p is q for q in g[1]["params"]))
    assert sum(len(x["params"]) for x in g) == len([p for p in m.parameters() if p.requires_grad])


def test_pretrained_weights_are_loaded_into_the_backbone(monkeypatch):
    import torchvision
    from torchvision.models._api import WeightsEnum
    src = torchvision.models.resnet18(weights=None)
    for p in src.parameters():
        nn.init.normal_(p, std=0.05)
    src.bn1.running_mean.fill_(3.0)
    monkeypatch.setattr(WeightsEnum, "get_state_dict", lambda self, *a, **k: src.state_dict())
    m = build_model(tiny_cfg("model.freeze_bn=true", "model.pretrained=imagenet"))
    assert torch.equal(m.backbone.layer2[0].conv1.weight, src.layer2[0].conv1.weight) and float(m.backbone.stem[1].running_mean[0]) == 3.0
    assert not torch.equal(build_model(tiny_cfg(), pretrained=False).backbone.layer2[0].conv1.weight, src.layer2[0].conv1.weight)


def test_load_compatible_skips_the_class_head_when_the_classes_change():
    a, b = build_model(tiny_cfg("data.synthetic.classes=3")), build_model(tiny_cfg("data.synthetic.classes=5"))
    rep = load_compatible(b, a.state_dict())
    assert rep["skipped"] and all(k.startswith("class_embed.") for k in rep["skipped"])
    assert torch.equal(b.bbox_embed.layers[0].weight, a.bbox_embed.layers[0].weight) and isinstance(b, DETR)
