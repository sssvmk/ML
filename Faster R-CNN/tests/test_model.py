import pytest
import torch
import torch.nn as nn

from src.ema import ModelEMA
from src.model import ARCHS, build_model, count_params, set_loss_mode
from src.train import EarlyStopper, LRController, make_optimizer

MCFG = {"arch": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": "none", "min_size": [96], "max_size": 128,
        "box_batch_size_per_image": 32}


def test_predictor_shape_and_small_init():
    m = build_model(MCFG, 20)
    p = m.roi_heads.box_predictor
    assert p.cls_score.out_features == 21 and p.bbox_pred.out_features == 84  # 20 classes + background
    assert abs(float(p.cls_score.weight.std()) - 0.01) < 0.003 and float(p.cls_score.bias.abs().sum()) == 0
    assert float(p.bbox_pred.weight.std()) < 0.003
    assert m.transform.min_size == (96,) and m.transform.max_size == 128


def test_box_head_dropout_is_optional():
    assert not any(isinstance(x, nn.Dropout) for x in build_model(MCFG, 3).roi_heads.box_head.modules())
    m = build_model({**MCFG, "box_head_dropout": 0.3}, 3)
    drops = [x for x in m.roi_heads.box_head.modules() if isinstance(x, nn.Dropout)]
    assert len(drops) == 1 and drops[0].p == 0.3


def test_loss_mode_returns_losses_without_touching_norm_or_dropout():
    m = build_model({**MCFG, "box_head_dropout": 0.3}, 3)
    set_loss_mode(m)
    assert m.training and all(not x.training for x in m.modules() if isinstance(x, (nn.BatchNorm2d, nn.Dropout)))
    bn = next(x for x in m.modules() if isinstance(x, nn.BatchNorm2d))
    before = bn.running_mean.clone()
    img = [torch.rand(3, 96, 96)]
    tg = [{"boxes": torch.tensor([[10., 10., 60., 60.]]), "labels": torch.tensor([1])}]
    with torch.no_grad():
        losses = m(img, tg)
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    assert torch.equal(bn.running_mean, before)


def test_invalid_choices_rejected_and_registry_complete():
    with pytest.raises(ValueError):
        build_model({**MCFG, "arch": "fasterrcnn_nonexistent"}, 3)
    with pytest.raises(ValueError):
        build_model({**MCFG, "pretrained": "magic"}, 3)
    assert {"fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_resnet50_fpn"} <= set(ARCHS)


def test_optimizer_groups_decay_only_weights_and_scale_backbone_lr():
    m = build_model(MCFG, 3)
    t = {"optimizer": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-3, "backbone_lr_mult": 0.1}
    opt = make_optimizer(m, t)
    assert sum(len(g["params"]) for g in opt.param_groups) == len([p for p in m.parameters() if p.requires_grad])
    for g in opt.param_groups:
        assert g["weight_decay"] in (0.0, 1e-3)
    lrs = sorted({round(g["lr"], 6) for g in opt.param_groups})
    assert lrs == [0.001, 0.01]
    assert make_optimizer(m, {**t, "optimizer": "adamw", "lr": 1e-4}).__class__.__name__ == "AdamW"
    assert count_params(m, True) <= count_params(m)


def _opt():
    return torch.optim.SGD([{"params": [nn.Parameter(torch.zeros(1))], "lr": 0.1}], lr=0.1)


def test_lr_controller_warmup_then_cosine():
    t = {"max_epochs": 10, "warmup_iters": 20, "warmup_factor": 0.001, "schedule": "cosine", "lr_min_ratio": 0.01}
    lrc = LRController(_opt(), 10, t)
    f = [lrc.factor(i) for i in range(100)]
    assert f[0] < 0.01 and all(a <= b for a, b in zip(f[:20], f[1:21], strict=False)) and abs(f[20] - 1.0) < 1e-6
    assert all(a >= b for a, b in zip(f[20:99], f[21:100], strict=False)) and abs(lrc.factor(100) - 0.01) < 1e-6
    lrc.step()
    assert lrc.opt.param_groups[0]["lr"] == pytest.approx(0.1 * f[0])


def test_lr_controller_multistep_and_plateau():
    ms = LRController(_opt(), 10, {"max_epochs": 10, "warmup_iters": 0, "schedule": "multistep",
                                   "milestones": [0.5, 0.8], "gamma": 0.1})
    assert ms.factor(10) == 1.0 and ms.factor(50) == pytest.approx(0.1) and ms.factor(80) == pytest.approx(0.01)
    pl = LRController(_opt(), 10, {"max_epochs": 10, "warmup_iters": 0, "schedule": "plateau", "plateau_patience": 2,
                                   "plateau_gamma": 0.5})
    pl.it = 50
    for v in (0.5, 0.5, 0.5):  # first sets best, then two epochs without improvement
        pl.on_epoch_end(v)
    assert pl.plateau_scale == 0.5 and pl.factor(60) == 0.5


def test_early_stopper_behaviour():
    es = EarlyStopper(patience=3, min_delta=0.01)
    res = [es.step(i, v) for i, v in enumerate([0.10, 0.20, 0.205, 0.20, 0.19])]
    assert [r[0] for r in res] == [True, True, False, False, False]  # +0.005 < min_delta is not an improvement
    assert [r[1] for r in res] == [False, False, False, False, True] and es.best_epoch == 1
    assert not any(EarlyStopper(2, enabled=False).step(i, 0.1)[1] for i in range(10))
    late = EarlyStopper(patience=1, min_epochs=5)
    assert [late.step(i, 0.5)[1] for i in range(6)] == [False, False, False, False, True, True]  # not before 5 epochs


def test_ema_tracks_model_and_copies_buffers():
    m = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = ModelEMA(m, decay=0.9, tau=1.0)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(1.0)
    start = float(ema.module[0].weight.abs().mean())
    for _ in range(60):
        ema.update(m)
    end = float(ema.module[0].weight.mean())
    assert abs(end - 1.0) < 0.05 and start != end and not any(p.requires_grad for p in ema.module.parameters())
    assert ema.module[1].num_batches_tracked == m[1].num_batches_tracked
    ema2 = ModelEMA(m)
    ema2.load_state_dict(ema.state_dict())
    assert ema2.updates == ema.updates


def _patch_weights(monkeypatch):
    """No network: serve locally built random 'pretrained' state dicts for the COCO and ImageNet weight enums."""
    from torchvision.models import detection as det
    from torchvision.models import mobilenet_v3_large
    from torchvision.models._api import WeightsEnum

    coco_sd = det.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None).state_dict()
    imnet_sd = mobilenet_v3_large(weights=None).state_dict()
    monkeypatch.setattr(WeightsEnum, "get_state_dict",
                        lambda self, *a, **k: coco_sd if type(self).__name__.startswith("FasterRCNN") else imnet_sd)
    return coco_sd, imnet_sd


def test_coco_pretrained_path_loads_backbone_swaps_predictor_and_freezes(monkeypatch):
    from torchvision.ops.misc import FrozenBatchNorm2d
    coco_sd, _ = _patch_weights(monkeypatch)
    m = build_model({**MCFG, "pretrained": "coco", "trainable_backbone_layers": 2}, 20)
    assert m.roi_heads.box_predictor.cls_score.out_features == 21           # 91 COCO classes replaced by 20 + bg
    assert torch.equal(m.backbone.fpn.layer_blocks[0][0].weight, coco_sd["backbone.fpn.layer_blocks.0.0.weight"])
    assert any(isinstance(x, FrozenBatchNorm2d) for x in m.modules()) and not any(
        type(x) is nn.BatchNorm2d for x in m.backbone.modules())            # frozen BN with pretrained weights
    assert count_params(m, True) < count_params(m)                          # early backbone layers frozen
    assert float(m.roi_heads.box_predictor.cls_score.weight.detach().std()) < 0.02  # re-initialised head


def test_imagenet_backbone_path(monkeypatch):
    _, imnet_sd = _patch_weights(monkeypatch)
    m = build_model({**MCFG, "pretrained": "imagenet", "trainable_backbone_layers": 3}, 20)
    assert m.roi_heads.box_predictor.cls_score.out_features == 21
    assert count_params(m, True) < count_params(m)
