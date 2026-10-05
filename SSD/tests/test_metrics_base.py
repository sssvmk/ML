import numpy as np

from ssd_voc.metrics import DetectionEvaluator, ap_11_point, ap_all_points, box_iou_np, error_breakdown

GT = {"boxes": [[0, 0, 10, 10]], "labels": [1], "difficult": [False]}


def _ev(preds, gts, names=("a", "b")):
    ev = DetectionEvaluator(list(names))
    ev.update(preds, gts)
    return ev.compute()


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10.0]])
    b = np.array([[0, 0, 10, 10.0], [5, 0, 15, 10.0], [20, 20, 30, 30.0]])
    np.testing.assert_allclose(box_iou_np(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-9)
    assert box_iou_np(np.zeros((0, 4)), b).shape == (0, 3)


def test_pr_curve_aps_match_hand_computation():
    rec, prec = np.array([.5, .5, 1.]), np.array([1, .5, 2 / 3])
    assert abs(ap_all_points(rec, prec) - (0.5 + 0.5 * 2 / 3)) < 1e-9
    assert abs(ap_11_point(rec, prec) - (6 + 5 * 2 / 3) / 11) < 1e-9


def test_perfect_detection_scores_one_everywhere():
    r = _ev([{"boxes": [[0, 0, 10, 10]], "scores": [0.9], "labels": [1]}], [GT])
    assert r["map50"] == r["map75"] == r["map"] == r["mar100"] == 1.0
    assert abs(r["map50_voc07"] - 1.0) < 1e-9


def test_wrong_class_and_missing_detection_score_zero():
    wrong = _ev([{"boxes": [[0, 0, 10, 10]], "scores": [0.9], "labels": [2]}], [GT])
    assert wrong["map50"] == 0.0
    none = _ev([{"boxes": np.zeros((0, 4)), "scores": [], "labels": []}], [GT])
    assert none["map50"] == 0.0 and none["mar100"] == 0.0


def test_iou_threshold_behaviour():
    # shifted by 3px: IoU = 70/130 = 0.538 -> true positive at 0.5 only
    r = _ev([{"boxes": [[3, 0, 13, 10]], "scores": [0.9], "labels": [1]}], [GT])
    assert r["map50"] == 1.0 and r["map75"] == 0.0 and abs(r["map"] - 0.1) < 1e-9


def test_duplicate_detection_is_false_positive():
    p = {"boxes": [[0, 0, 10, 10], [0, 0, 10, 10]], "scores": [0.9, 0.8], "labels": [1, 1]}
    r = _ev([p], [GT])
    assert r["map50"] == 1.0  # the duplicate comes after full recall: precision envelope keeps AP at 1
    # but a duplicate ranked ABOVE a true positive of another object hurts
    gt2 = {"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "labels": [1, 1], "difficult": [False, False]}
    p2 = {"boxes": [[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], "scores": [0.9, 0.8, 0.7], "labels": [1, 1, 1]}
    r2 = _ev([p2], [gt2])
    assert 0.5 < r2["map50"] < 1.0  # precision at full recall is 2/3, not 1


def test_difficult_ground_truth_is_ignored():
    gt = {"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "labels": [1, 1], "difficult": [False, True]}
    # detecting the difficult object is neither TP nor FP; missing it does not hurt recall
    p = {"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "scores": [0.9, 0.8], "labels": [1, 1]}
    assert _ev([p], [gt])["map50"] == 1.0
    p_only_easy = {"boxes": [[0, 0, 10, 10]], "scores": [0.9], "labels": [1]}
    assert _ev([p_only_easy], [gt])["map50"] == 1.0


def test_classes_without_ground_truth_are_excluded_from_the_mean():
    r = _ev([{"boxes": [[0, 0, 10, 10]], "scores": [0.9], "labels": [1]}], [GT], names=("a", "b", "c"))
    assert r["map50"] == 1.0 and r["ap50_per_class"]["b"] is None


def test_error_breakdown_categories():
    gt = {"boxes": [[0, 0, 10, 10], [100, 100, 110, 110]], "labels": [1, 2], "difficult": [False, False]}
    dets = {"boxes": [[0, 0, 10, 10],        # correct
                      [0, 0, 10, 10],        # duplicate
                      [20, 0, 24, 10],       # background (no overlap)
                      [100, 100, 110, 110],  # confusion: class 1 box on the class-2 object
                      [0, 0, 10, 40]],       # localization: IoU 0.25 with the class-1 object
            "scores": [0.9, 0.8, 0.7, 0.6, 0.55], "labels": [1, 1, 1, 1, 1]}
    e = error_breakdown([dets], [gt], score_thr=0.5)
    assert (e["correct"], e["duplicate"], e["background"], e["confusion"], e["localization"]) == (1, 1, 1, 1, 1)
    assert e["missed"] == 1 and e["n_gt"] == 2  # the class-2 object was never found
    assert error_breakdown([dets], [gt], score_thr=0.95)["n_det"] == 0
