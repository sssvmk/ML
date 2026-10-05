import math

import numpy as np
import pytest
import torch

from detr_voc.infer import aspect_size, denormalize, postprocess, to_normalised_tensor
from detr_voc.loss import HungarianMatcher, SetCriterion

W = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}


def crit(k=3, eos=0.1):
    return SetCriterion(k, HungarianMatcher(), W, eos)


def tg(boxes, labels):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), "labels": torch.tensor(labels, dtype=torch.long)}


TARGETS = [tg([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.3, 0.2]], [0, 2]), tg([], [])]


def perfect_outputs():
    logits = torch.full((2, 5, 4), -20.0)
    logits[:, :, 3] = 20.0                                         # everything "no object" ...
    boxes = torch.rand(2, 5, 4).clamp(0.1, 0.9)
    logits[0, 1] = torch.tensor([20.0, -20, -20, -20]); boxes[0, 1] = TARGETS[0]["boxes"][0]   # ... except the two true objects  # noqa: E702
    logits[0, 3] = torch.tensor([-20.0, -20, 20, -20]); boxes[0, 3] = TARGETS[0]["boxes"][1]   # noqa: E702
    return {"pred_logits": logits, "pred_boxes": boxes}


def test_uniform_logits_cost_exactly_ln_of_classes_plus_one():
    out = crit()({"pred_logits": torch.zeros(2, 5, 4), "pred_boxes": torch.rand(2, 5, 4).clamp(0.1, 0.9)}, TARGETS)
    assert float(out["loss_ce"]) == pytest.approx(math.log(4), abs=1e-5)         # weighted mean of identical losses


def test_perfect_predictions_give_zero_loss():
    out = crit()(perfect_outputs(), TARGETS)
    assert float(out["loss_ce"]) < 1e-6 and float(out["loss_bbox"]) < 1e-6 and float(out["loss_giou"]) < 1e-6 and float(out["total"]) < 1e-5


def test_matching_picks_the_obvious_pairs_and_is_invariant_to_query_order():
    out = perfect_outputs()
    idx = HungarianMatcher()(out, TARGETS)
    assert sorted(zip(idx[0][0].tolist(), idx[0][1].tolist(), strict=True)) == [(1, 0), (3, 1)] and len(idx[1][0]) == 0
    perm = torch.tensor([4, 3, 2, 1, 0])
    shuffled = {"pred_logits": out["pred_logits"][:, perm], "pred_boxes": out["pred_boxes"][:, perm]}
    assert float(crit()(shuffled, TARGETS)["total"]) == pytest.approx(float(crit()(out, TARGETS)["total"]), abs=1e-6)


def test_each_target_is_matched_exactly_once_even_with_more_targets_than_queries():
    out = {"pred_logits": torch.randn(1, 3, 4), "pred_boxes": torch.rand(1, 3, 4).clamp(0.1, 0.9)}
    rows, cols = HungarianMatcher()(out, [tg([[0.2, 0.2, 0.1, 0.1]] * 5, [0] * 5)])[0]
    assert len(rows) == len(cols) == 3 and len(set(rows.tolist())) == 3 and len(set(cols.tolist())) == 3


def test_auxiliary_losses_are_added_per_decoder_layer_with_the_same_weights():
    out = perfect_outputs()
    out["aux_outputs"] = [{"pred_logits": torch.zeros(2, 5, 4), "pred_boxes": torch.rand(2, 5, 4).clamp(0.1, 0.9)} for _ in range(2)]
    res = crit()(out, TARGETS)
    assert {"loss_ce_0", "loss_bbox_1", "loss_giou_1"} <= set(res)
    expected = sum(W[k] * float(res[f"{k}_{n}"]) for n in range(2) for k in W) + sum(W[k] * float(res[k]) for k in W)
    assert float(res["total"]) == pytest.approx(expected, rel=1e-4)
    assert float(res["total"]) > float(crit()({k: v for k, v in out.items() if k != "aux_outputs"}, TARGETS)["total"])


def test_no_object_class_is_downweighted():
    out = {"pred_logits": torch.zeros(1, 4, 4), "pred_boxes": torch.full((1, 4, 4), 0.5)}
    out["pred_logits"][0, :, 3] = 3.0                                              # confidently "no object" everywhere
    tgt = [tg([[0.5, 0.5, 0.5, 0.5]], [0])]
    heavy, light = float(crit(eos=1.0)(out, tgt)["loss_ce"]), float(crit(eos=0.1)(out, tgt)["loss_ce"])
    assert light > heavy                                       # the one missed object dominates when "no object" counts less


def test_empty_batch_is_finite_and_gradients_flow():
    out = {"pred_logits": torch.randn(2, 5, 4, requires_grad=True), "pred_boxes": torch.rand(2, 5, 4, requires_grad=True)}
    res = crit()(out, [tg([], []), tg([], [])])
    assert torch.isfinite(res["total"]) and float(res["loss_bbox"]) == 0.0
    res["total"].backward()
    assert out["pred_logits"].grad.abs().sum() > 0
    out2 = {"pred_logits": torch.randn(2, 5, 4, requires_grad=True), "pred_boxes": torch.rand(2, 5, 4).clamp(0.1, 0.9).requires_grad_()}
    crit()(out2, TARGETS)["total"].backward()
    assert out2["pred_boxes"].grad.abs().sum() > 0


def test_postprocess_uses_the_best_real_class_scales_boxes_and_returns_one_detection_per_query():
    logits = torch.tensor([[[0.0, 5.0, 0.0, 6.0],         # "no object" wins, class 1 is the best real class (paper override)
                            [4.0, 0.0, 0.0, -9.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.25, 0.5, 0.5, 1.0]]])
    out = postprocess({"pred_logits": logits, "pred_boxes": boxes}, torch.tensor([[200, 100]]))[0]
    assert out["labels"].tolist() == [2, 1] and out["boxes"].shape == (2, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([80.0, 30.0, 120.0, 70.0]))               # xyxy in original pixels
    assert float(out["scores"][1]) == pytest.approx(54.598 / 56.598, abs=1e-3) and float(out["scores"][0]) < 0.5


def test_official_resize_rule_and_normalisation_roundtrip():
    assert aspect_size(640, 480, 800, 1333) == (800, 1066) and aspect_size(480, 640, 800, 1333) == (1066, 800)
    assert aspect_size(2000, 500, 800, 1333) == (333, 1332)                      # the longer side is capped at max_size
    assert aspect_size(100, 100, 100, None) == (100, 100)
    from PIL import Image
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8))
    x = to_normalised_tensor(img)
    assert x.shape == (3, 30, 40) and torch.allclose(denormalize(x) * 255, torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float(), atol=1e-3)
