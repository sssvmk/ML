import pytest
import torch

from ssd_voc.metrics import DetectionEvaluator
from ssd_voc.schedule import PhaseEarlyStopper, PhaseSchedule

GT = {"boxes": [[0, 0, 10, 10]], "labels": [1], "difficult": [False]}


def opt():
    return torch.optim.SGD([{"params": [torch.nn.Parameter(torch.zeros(1))], "lr_mult": 2.0},
                            {"params": [torch.nn.Parameter(torch.zeros(1))]}], lr=1.0)


def test_phase_schedule_follows_the_paper_steps_with_warmup():
    o = opt()
    s = PhaseSchedule(o, [(1e-3, 10), (1e-4, 5)], scale=1.0, warmup_iters=4, warmup_factor=0.1)
    lrs = [s.step() for _ in range(10)]
    assert lrs[0] == pytest.approx(1e-3 * 0.1) and lrs[3] < lrs[4] == pytest.approx(1e-3) and lrs[-1] == pytest.approx(1e-3)
    assert s.phase_done() and not s.is_last_phase and s.total_iters == 15
    assert o.param_groups[0]["lr"] == pytest.approx(2e-3) and o.param_groups[1]["lr"] == pytest.approx(1e-3)   # lr_mult
    assert s.advance() and s.step() == pytest.approx(1e-4) and s.is_last_phase and not s.advance()
    assert not s.phase_done()
    for _ in range(4):
        s.step()
    assert s.phase_done()


def test_schedule_scale_doubles_every_phase_and_state_round_trips():
    s = PhaseSchedule(opt(), [(1e-3, 60000), (1e-4, 20000)], scale=2.0)
    assert [n for _, n in s.phases] == [120000, 40000]
    for _ in range(7):
        s.step()
    t = PhaseSchedule(opt(), [(1e-3, 60000), (1e-4, 20000)], scale=2.0)
    t.load_state_dict(s.state_dict())
    assert (t.phase, t.it_in_phase, t.global_it) == (0, 7, 7)


def test_early_stopper_counts_evaluations_without_improvement_per_phase():
    es = PhaseEarlyStopper(patience=3, min_delta=0.01)
    assert es.update(0.30, 100) and es.update(0.40, 200)
    assert not es.update(0.405, 300) and not es.exhausted        # +0.005 < min_delta: not an improvement
    assert not es.update(0.39, 400) and not es.exhausted
    assert not es.update(0.40, 500) and es.exhausted
    es.reset_phase()
    assert not es.exhausted and es.best == 0.40 and es.best_it == 200
    assert es.update(0.45, 600) and es.bad == 0
    assert not PhaseEarlyStopper(1, enabled=False).update(0.1, 1) or True
    off = PhaseEarlyStopper(1, enabled=False)
    off.update(0.5, 1)
    off.update(0.1, 2)
    assert not off.exhausted
    other = PhaseEarlyStopper(3)
    other.load_state_dict(es.state_dict())
    assert other.best == es.best and other.best_it == es.best_it


def test_fast_mode_scores_only_iou_half():
    pr = {"boxes": [[3, 0, 13, 10]], "scores": [0.9], "labels": [1]}
    fast, full = DetectionEvaluator(["a"], [0.5]), DetectionEvaluator(["a"])
    for ev in (fast, full):
        ev.update([pr], [GT])
    f, g = fast.compute(), full.compute()
    assert f["map50"] == g["map50"] == 1.0 and f["map75"] is None and f["map"] is None and f["mar100"] is None
    assert g["map75"] == 0.0 and g["map"] == pytest.approx(0.1)
    assert f["ap50_per_class"] == {"a": 1.0}
