import math

import pytest
import torch

from ssd_voc.boxes import (
    MultiBoxLoss,
    box_iou_batched,
    cxcywh_to_xyxy,
    decode,
    encode,
    generate_priors,
    match_priors,
    pad_targets,
    postprocess,
    prior_scales,
    xyxy_to_cxcywh,
)

SIZES300, PER300 = [38, 19, 10, 5, 3, 1], [4, 6, 6, 6, 4, 4]
SIZES512, PER512 = [64, 32, 16, 8, 4, 2, 1], [4, 6, 6, 6, 6, 4, 4]


def test_default_box_counts_match_the_paper():
    p300 = generate_priors(SIZES300, PER300)
    p512 = generate_priors(SIZES512, PER512)
    assert p300.shape == (8732, 4) and p512.shape == (24564, 4)           # paper: 8732 (SSD300), 24564 (SSD512)
    assert float(p300.min()) >= 0 and float(p300.max()) <= 1


def test_scales_follow_paper_equation():
    s = prior_scales(6, 0.1, 0.2, 0.9)
    assert s[0] == 0.1 and s[1] == pytest.approx(0.2) and s[5] == pytest.approx(0.9)
    assert [round(b - a, 6) for a, b in zip(s[1:6], s[2:7], strict=False)] == [0.175] * 5   # regular spacing
    p = generate_priors([2], [4], 0.3, 0.2, 0.9, clip=False)
    assert p.shape == (2 * 2 * 4, 4) and torch.allclose(p[0], torch.tensor([0.25, 0.25, 0.3, 0.3]))
    ar2 = p[2]                                                              # aspect ratio 2: wider than tall
    assert ar2[2] > ar2[3] and ar2[2] * ar2[3] == pytest.approx(0.09, rel=1e-5)
    assert generate_priors([1], [6]).shape == (6, 4)
    with pytest.raises(ValueError):
        generate_priors([1], [5])


def test_box_conversions_iou_and_codec_roundtrip():
    b = torch.tensor([[0.1, 0.2, 0.5, 0.8]])
    assert torch.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)), b, atol=1e-6)
    a = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    pr = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.0, 1.5, 1.0], [2.0, 2.0, 3.0, 3.0]])
    assert torch.allclose(box_iou_batched(a, pr)[0, 0], torch.tensor([1.0, 1 / 3, 0.0]), atol=1e-6)
    priors = generate_priors([4], [4])
    gt = torch.tensor([[0.2, 0.2, 0.7, 0.9]]).expand(len(priors), 4)
    assert torch.allclose(decode(encode(gt, priors, (0.1, 0.2)), priors, (0.1, 0.2)), gt, atol=1e-4)


def _t(boxes, labels):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), "labels": torch.tensor(labels)}


def test_matching_rules():
    priors = generate_priors([8, 4], [4, 4])
    pxy = cxcywh_to_xyxy(priors)
    gb, gl, valid = pad_targets([_t([[0.3, 0.3, 0.7, 0.7]], [5]), _t([[0.05, 0.05, 0.1, 0.1]], [2]), _t([], [])], "cpu")
    labels, boxes = match_priors(gb, gl, valid, pxy, 0.5)
    assert labels.shape == (3, len(priors)) and (labels[0] == 5).sum() >= 1 and set(labels[0].tolist()) <= {0, 5}
    # a tiny object overlaps no default box by 0.5, but its best-matching box is still forced to be positive
    assert (labels[1] == 2).sum() >= 1
    assert (labels[2] > 0).sum() == 0                                        # image without objects: all background
    pos = labels[0] > 0
    assert torch.allclose(boxes[0][pos], gb[0, 0].expand(int(pos.sum()), 4))
    empty = match_priors(*pad_targets([_t([], [])], "cpu"), pxy, 0.5)
    assert empty[0].sum() == 0


def test_loss_uniform_logits_equal_four_ln_c_by_construction():
    """Uniform logits cost ln(C) per sampled box; positives + 3x hard negatives per positive => conf loss = 4 ln C."""
    priors = generate_priors([10, 5], [6, 6])
    loss = MultiBoxLoss(priors, 0.5, 3)
    tg = [_t([[0.2, 0.2, 0.6, 0.6]], [1]), _t([[0.5, 0.1, 0.9, 0.5], [0.1, 0.5, 0.4, 0.9]], [2, 3])]
    out = loss(torch.zeros(2, len(priors), 4), torch.zeros(2, len(priors), 21), tg)
    assert float(out["conf"]) == pytest.approx(4 * math.log(21), rel=1e-4)
    assert float(out["num_pos"]) >= 3


def test_perfect_predictions_give_zero_loss_and_gradients_flow():
    priors = generate_priors([10, 5], [6, 6])
    loss = MultiBoxLoss(priors, 0.5, 3)
    tg = [_t([[0.2, 0.2, 0.6, 0.6]], [4])]
    gb, gl, valid = pad_targets(tg, "cpu")
    labels, matched = match_priors(gb, gl, valid, loss.priors_xyxy, 0.5)
    loc = encode(matched, priors, (0.1, 0.2))
    conf = torch.nn.functional.one_hot(labels, 6).float() * 60.0
    out = loss(loc, conf, tg)
    assert float(out["total"]) < 1e-3
    loc_p = torch.randn(1, len(priors), 4, requires_grad=True)
    conf_p = torch.randn(1, len(priors), 6, requires_grad=True)
    loss(loc_p, conf_p, tg)["total"].backward()
    assert loc_p.grad.abs().sum() > 0 and conf_p.grad.abs().sum() > 0


def test_no_objects_gives_a_finite_zero_loss():
    priors = generate_priors([6], [4])
    out = MultiBoxLoss(priors)(torch.randn(2, len(priors), 4, requires_grad=True), torch.randn(2, len(priors), 5, requires_grad=True),
                               [_t([], []), _t([], [])])
    assert float(out["total"]) == 0.0 and torch.isfinite(out["total"])


def test_postprocess_threshold_nms_and_limits():
    priors = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.51, 0.5, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]])
    loc = torch.zeros(1, 4, 4)
    conf = torch.full((1, 4, 3), -10.0)
    conf[0, :, 0] = 5.0                                    # background everywhere ...
    conf[0, 0] = torch.tensor([-5.0, 6.0, -5.0])           # two overlapping class-1 boxes -> NMS keeps the better one
    conf[0, 1] = torch.tensor([-5.0, 4.0, -5.0])
    conf[0, 2] = torch.tensor([-5.0, -5.0, 6.0])           # a class-2 box elsewhere
    out = postprocess(loc, conf, priors, (0.1, 0.2), score_thresh=0.01, nms_iou=0.45, max_det=200)[0]
    assert sorted(out["labels"].tolist()) == [1, 2] and out["scores"].max() > 0.99
    assert out["boxes"].min() >= 0 and out["boxes"].max() <= 1
    assert len(postprocess(loc, conf, priors, (0.1, 0.2), 0.01, 0.99, 200)[0]["labels"]) == 3   # lenient NMS keeps the overlap
    assert len(postprocess(loc, conf, priors, (0.1, 0.2), 0.01, 0.45, max_det=1)[0]["labels"]) == 1
    assert len(postprocess(loc, conf, priors, (0.1, 0.2), score_thresh=0.99999)[0]["labels"]) == 0
