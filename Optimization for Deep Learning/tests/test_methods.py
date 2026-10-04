"""Contract tests that every one of the 40 methods must satisfy."""
import numpy as np
import optuna
import pytest
import torch
import torch.nn.functional as F

from optpipe.bundle import Predictor, save_bundle
from optpipe.core import Ctx
from optpipe.data import to_float
from optpipe.methods import ORDER, REGISTRY, get_methods
from optpipe.tuning import default_trial_params, suggest_hp


def test_registry_has_all_methods_with_metadata():
    assert len(ORDER) == 40 and len(REGISTRY) == 40
    groups = {cls.group for cls in ORDER}
    assert groups == {"reference", "update rules", "adaptive learning rate", "second-order", "initialization",
                      "gradient clipping", "meta-algorithms"}
    for cls in ORDER:
        assert cls.title and cls.section and cls.group and cls.name


def test_method_counts_per_group_match_the_requested_list():
    count = lambda g: sum(1 for c in ORDER if c.group == g)
    assert count("update rules") == 5 and count("adaptive learning rate") == 4 and count("second-order") == 4
    assert count("initialization") == 14 and count("gradient clipping") == 3 and count("meta-algorithms") == 9


@pytest.mark.parametrize("cls", ORDER, ids=lambda c: c.name)
def test_default_trial_is_valid_for_the_search_space(cls, cfg):
    m = cls()
    params = default_trial_params(m, cfg)
    assert {"l1", "l2"} <= set(params)                                           # L1 + L2 are tuned for every method
    assert ("lr" in params) == m.tune_lr
    hp = suggest_hp(m, optuna.trial.FixedTrial({**m.defaults()}), cfg)
    assert set(params) <= set(hp)


@pytest.mark.parametrize("cls", ORDER, ids=lambda c: c.name)
def test_every_method_trains_and_its_bundle_reloads(cls, cfg, data, tmp_path):
    m = cls()
    ctx = Ctx(cfg, torch.device("cpu"), 0, epochs=m.budget(cfg, "final") if m.mode == "fullbatch" else 3)
    res = m.fit(data, m.defaults(), ctx)
    assert len(res.history) >= 1 and all(np.isfinite(h["val_loss"]) for h in res.history)
    assert {"train_loss", "val_loss", "val_acc", "val_auc", "passes"} <= set(res.history[-1])
    assert "init_loss_over_lnK" in res.extras
    save_bundle(tmp_path / "b", res.model, {"method": m.name})
    x = to_float(data.x_val[:5]).numpy()
    ref = F.softmax(res.model.eval()(torch.from_numpy(x)), -1).detach().numpy()
    assert np.allclose(Predictor(tmp_path / "b").predict_proba(x), ref, atol=1e-5)


def test_l1_and_l2_penalties_are_part_of_every_objective(ctx, data):
    from optpipe.core import penalty, set_l1_smoothing
    set_l1_smoothing(1e-3)
    m = get_methods(["baseline"])[0]
    model = m.build(m.defaults(), ctx)
    batch = {"x": to_float(data.x_train[:32]), "y": data.y_train[:32]}
    plain = float(m.loss(model, batch, {"l1": 0.0, "l2": 0.0}))
    both = float(m.loss(model, batch, {"l1": 1e-3, "l2": 1e-2}))
    ws = [p for p in model.parameters() if p.ndim == 2]
    l2 = 0.5 * 1e-2 * sum(float((w ** 2).sum()) for w in ws)
    l1 = 1e-3 * sum(float(torch.sqrt(w ** 2 + 1e-6).sum()) for w in ws)
    assert both - plain == pytest.approx(l1 + l2, rel=1e-4)
    biases = [p for p in model.parameters() if p.ndim == 1]
    assert biases and float(penalty(model, 1.0, 1.0)) == pytest.approx(
        0.5 * sum(float((w ** 2).sum()) for w in ws) + sum(float(torch.sqrt(w ** 2 + 1e-6).sum()) for w in ws), rel=1e-4)


def test_smooth_l1_approximates_abs_for_large_weights():
    from optpipe.core import set_l1_smoothing
    set_l1_smoothing(1e-3)
    w = torch.tensor([2.0, -3.0])
    assert float(torch.sqrt(w ** 2 + 1e-6).sum()) == pytest.approx(5.0, abs=1e-5)


def test_update_rule_group_uses_constant_learning_rate(ctx):
    for name in ("sgd", "momentum", "nesterov", "adagrad", "rmsprop", "rmsprop_nesterov", "adam"):
        m = REGISTRY[name]()
        assert m.lr_factor(7.5, m.defaults(), ctx) == 1.0
    assert REGISTRY["baseline"]().lr_factor(3, {}, ctx) == pytest.approx(ctx.lr_decay ** 3)


def test_full_batch_methods_are_marked_and_have_no_lr(cfg):
    for name in ("batch_gd", "newton", "conjugate_gradients", "bfgs", "lbfgs"):
        assert REGISTRY[name].mode == "fullbatch"
    for name in ("newton", "conjugate_gradients", "bfgs", "lbfgs"):
        assert REGISTRY[name].tune_lr is False and "lr" not in REGISTRY[name]().defaults()
    assert REGISTRY["batch_gd"]().budget(cfg, "final") == cfg["full_batch"]["iterations"]


def test_bfgs_is_reduced_regime_and_excluded_from_champion_selection(cfg, ctx):
    m = REGISTRY["bfgs"]()
    model = m.build(m.defaults(), ctx)
    assert model.hidden[0].out_features == 6 and not m.champion_eligible and "reduced" in m.regime


def test_training_is_reproducible_on_cpu(cfg, data, ctx):
    for name in ("baseline", "adam", "init_random_walk", "curriculum"):
        m = REGISTRY[name]()
        a = m.fit(data, m.defaults(), ctx)
        b = m.fit(data, m.defaults(), ctx)
        assert [h["val_loss"] for h in a.history] == [h["val_loss"] for h in b.history], name


def test_unknown_method_name_is_rejected():
    with pytest.raises(KeyError):
        get_methods(["nope"])
    assert [m.name for m in get_methods(["adam"])] == ["baseline", "adam"]
