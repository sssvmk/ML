"""One property test per method: each checks that the regularizer does what the book says."""
import copy

import numpy as np
import optuna
import pytest
import torch
import torch.nn.functional as F

from regpipe.evaluate import fgsm_accuracy, evaluate
from regpipe.methods import ORDER, REGISTRY, get_methods
from regpipe.methods.base import soft_cross_entropy, weights_only
from regpipe.methods.constrained_norm import ConstrainedNorm, project_max_norm
from regpipe.methods.l1 import L1
from regpipe.methods.l2_weight_decay import L2WeightDecay
from regpipe.model import MLP, Ensemble, TiedDepthMLP, n_params
from regpipe.transforms import affine_images, affine_tangents, random_affine

X = torch.rand(16, 784)
Y = torch.randint(0, 10, (16,))


def batch():
    return {"x": X, "y": Y}


# ---------------------------------------------------------------------------- registry
def test_all_methods_registered_with_metadata():
    assert len(ORDER) == 19 and len(REGISTRY) == 19
    for cls in ORDER:
        assert cls.title and cls.section


@pytest.mark.parametrize("cls", ORDER, ids=lambda c: c.name)
def test_search_space_contains_defaults(cls):
    m = cls()
    trial = optuna.trial.FixedTrial({**{"lr": 0.05}, **m.defaults()})
    sampled = m.space(trial)
    assert set(sampled) <= set(m.defaults())            # every tuned name has a default (first trial)


# ---------------------------------------------------------------------------- 7.1 / 7.2
def test_l2_penalty_matches_formula_and_skips_biases():
    model = MLP([8])
    pen = L2WeightDecay().penalty(model, batch(), {"alpha": 0.01})
    expected = 0.5 * 0.01 * sum(float((w ** 2).sum()) for w in [model.hidden[0].weight, model.out.weight])
    assert float(pen) == pytest.approx(expected, rel=1e-5)


def test_l1_penalty_matches_formula():
    model = MLP([8])
    pen = L1().penalty(model, batch(), {"alpha": 0.01})
    expected = 0.01 * sum(float(w.abs().sum()) for w in [model.hidden[0].weight, model.out.weight])
    assert float(pen) == pytest.approx(expected, rel=1e-5)


def test_l1_gradient_is_alpha_sign_w():
    model = MLP([8])
    for p in model.parameters():
        p.grad = None
    L1().penalty(model, batch(), {"alpha": 0.5}).backward()
    w = model.hidden[0].weight
    assert torch.allclose(w.grad, 0.5 * torch.sign(w), atol=1e-6)


def test_max_norm_projection_bounds_every_unit_and_leaves_small_rows():
    model = MLP([16])
    with torch.no_grad():
        model.hidden[0].weight.mul_(10)
        small = model.out.weight.clone() * 1e-3
        model.out.weight.copy_(small)
    project_max_norm(model, 2.0)
    assert (model.hidden[0].weight.norm(dim=1) <= 2.0 + 1e-5).all()
    assert torch.allclose(model.out.weight, small)       # already inside the constraint: unchanged


# ---------------------------------------------------------------------------- 7.4 / 7.5
def test_affine_identity_and_shift():
    img = torch.zeros(1, 784)
    img[0, 14 * 28 + 14] = 1.0
    assert torch.allclose(affine_images(img), img, atol=1e-5)
    shifted = affine_images(img, tx=2.0)
    row, col = divmod(int(shifted.argmax()), 28)
    assert (row, col) == (14, 16)                        # +2 px in x


def test_augmentation_keeps_range_and_is_identity_at_zero_strength():
    out = random_affine(X, 3.0, 20.0, 0.2)
    assert out.shape == X.shape and out.min() >= 0 and out.max() <= 1 + 1e-6
    assert torch.allclose(random_affine(X, 0, 0, 0), X, atol=1e-5)


def test_input_noise_only_in_transform():
    from regpipe.methods.input_noise import InputNoise
    m = InputNoise()
    noisy = m.transform(batch(), None, {"sigma": 0.3})["x"]
    assert not torch.allclose(noisy, X)
    assert (noisy - X).std() == pytest.approx(0.3, rel=0.2)


def test_weight_noise_perturbs_inside_and_restores_after():
    from regpipe.methods.weight_noise import perturbed_weights
    model = MLP([8])
    before = [w.clone() for w in weights_only(model)]
    with perturbed_weights(model, 0.1):
        assert not torch.allclose(weights_only(model)[0], before[0])
    for w, b in zip(weights_only(model), before):
        assert torch.equal(w, b)                         # exact restore


def test_label_smoothing_matches_definition():
    logits = torch.randn(6, 10)
    y = torch.randint(0, 10, (6,))
    assert float(soft_cross_entropy(logits, y, 0.0)) == pytest.approx(float(F.cross_entropy(logits, y)))
    eps = 0.2
    target = torch.full((6, 10), eps / 9)
    target[torch.arange(6), y] = 1 - eps
    assert float(target.sum(1).mean()) == pytest.approx(1.0)
    manual = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
    assert float(soft_cross_entropy(logits, y, eps)) == pytest.approx(float(manual), rel=1e-5)


# ---------------------------------------------------------------------------- 7.6 / 7.7
def test_semi_supervised_split_is_disjoint_and_unlabeled_has_no_labels(data, ctx):
    from regpipe.methods.semi_supervised import SemiSupervised
    m = SemiSupervised()
    ts = m.prepare(data, {}, ctx)
    assert len(ts.y) == 200 and len(ts.extras["x_unl"]) == len(data.y_train) - 200
    assert "y_unl" not in ts.extras
    b = next(m.batches(ts, {}, ctx, torch.Generator().manual_seed(0)))
    assert set(b) == {"x", "y", "x_unl"}


def test_semi_supervised_lambda_zero_ignores_reconstruction(data, ctx):
    from regpipe.methods.semi_supervised import SemiSupervised
    m = SemiSupervised()
    model = m.build({}, ctx)
    b = {"x": X, "y": Y, "x_unl": torch.rand(16, 784)}
    assert float(m.data_loss(model, b, {"lam": 0.0})) == pytest.approx(float(F.cross_entropy(model(X), Y)), rel=1e-5)
    assert float(m.data_loss(model, b, {"lam": 5.0})) > float(m.data_loss(model, b, {"lam": 0.0}))


def test_multitask_aux_labels_and_loss(ctx):
    from regpipe.methods.multitask import Multitask, aux_labels
    y = torch.arange(10)
    a = aux_labels(y)
    assert a["parity"].tolist() == [0, 1] * 5 and a["high"].tolist() == [0] * 5 + [1] * 5
    m = Multitask()
    model = m.build({}, ctx)
    assert float(m.data_loss(model, batch(), {"lam": 1.0})) > float(m.data_loss(model, batch(), {"lam": 0.0}))


# ---------------------------------------------------------------------------- 7.8 / 7.9 / 7.10
def test_early_stopping_rule_and_best_weights(data, ctx):
    from regpipe.methods.early_stopping import EarlyStopping
    m = EarlyStopping()
    hist = [{"val_loss": v} for v in [1.0, 0.8, 0.7, 0.71, 0.72, 0.73]]
    assert m.should_stop(hist, {"patience": 3}) is True
    assert m.should_stop(hist[:5], {"patience": 3}) is False
    assert m.selection == "best"
    res = m.fit(data, {"lr": 0.05, "patience": 1}, ctx)
    assert res.extras["best_epoch"] >= 1


def test_parameter_sharing_hard_has_fewer_parameters_and_shares_the_tensor():
    hard, soft = TiedDepthMLP(32, 4, "hard"), TiedDepthMLP(32, 4, "soft")
    assert n_params(hard) < n_params(soft)
    assert len(hard.blocks) == 1 and len(soft.blocks) == 4
    assert float(soft.tying_penalty()) == pytest.approx(0.0, abs=1e-10)
    with torch.no_grad():
        soft.blocks[1].weight.add_(1.0)
    assert float(soft.tying_penalty()) > 0
    hard.zero_grad()
    hard(X).sum().backward()
    assert hard.blocks[0].weight.grad.abs().sum() > 0       # gradients from all 4 uses accumulate in one tensor


def test_sparse_loss_adds_activation_penalty(ctx):
    from regpipe.methods.sparse_representations import SparseRepresentations
    m = SparseRepresentations()
    model = m.build({}, ctx)
    ce = float(F.cross_entropy(model(X), Y))
    assert float(m.loss(model, batch(), {"lam": 0.5})) > ce
    assert float(m.loss(model, batch(), {"lam": 0.0})) == pytest.approx(ce, rel=1e-5)


# ---------------------------------------------------------------------------- 7.11 / 7.12 / 7.13
def test_bagging_members_differ_and_ensemble_is_a_distribution(data, ctx):
    from regpipe.methods.bagging import Bagging
    m = Bagging()
    res = m.fit(data, {"lr": 0.05}, ctx)
    assert isinstance(res.model, Ensemble) and len(res.model.members) == 2
    w0, w1 = (mm.hidden[0].weight for mm in res.model.members)
    assert not torch.allclose(w0, w1)
    probs = F.softmax(res.model(X), -1)
    assert torch.allclose(probs.sum(1), torch.ones(16), atol=1e-5)
    ex = m.extras(res, data, ctx, {})
    assert ex["ensemble_val_acc"] >= ex["member_val_acc_min"] - 1e-9


def test_dropout_is_stochastic_in_train_and_deterministic_in_eval(ctx):
    from regpipe.methods.dropout import Dropout
    model = Dropout().build({"p_in": 0.2, "p_hidden": 0.5}, ctx)
    model.train()
    assert not torch.allclose(model(X), model(X))
    model.eval()
    assert torch.allclose(model(X), model(X))


def test_adversarial_eps_zero_equals_clean_loss_and_attack_is_bounded(ctx):
    from regpipe.methods.adversarial import AdversarialTraining
    m = AdversarialTraining()
    model = m.build({}, ctx)
    clean = float(F.cross_entropy(model(X), Y))
    assert float(m.data_loss(model, batch(), {"eps": 0.0, "mix": 0.5})) == pytest.approx(clean, rel=1e-5)
    assert float(m.data_loss(model, batch(), {"eps": 0.2, "mix": 0.0})) >= clean - 1e-6   # attack cannot lower the loss much
    x = X.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(F.cross_entropy(model(x), Y), x)
    adv = (x + 0.1 * g.sign()).clamp(0, 1)
    assert (adv - X).abs().max() <= 0.1 + 1e-6


def test_fgsm_accuracy_at_eps_zero_is_plain_accuracy(data, ctx):
    model = MLP([16])
    cpu = torch.device("cpu")
    assert fgsm_accuracy(model, data.x_val, data.y_val, cpu, 0.0) == pytest.approx(
        evaluate(model, data.x_val, data.y_val, cpu)["accuracy"])


# ---------------------------------------------------------------------------- 7.14
def test_tangents_are_unit_norm_and_penalty_has_gradient(data, ctx):
    from regpipe.methods.tangent import Tangent, manifold_tangents
    v = affine_tangents(X)
    assert v.shape == (16, 5, 784) and torch.allclose(v.norm(dim=-1), torch.ones(16, 5), atol=1e-4)
    m = Tangent()
    ts = m.prepare(data, {}, ctx)
    model = m.build({}, ctx)
    b = next(m.batches(ts, {}, ctx, torch.Generator().manual_seed(0)))
    for source in ("affine", "manifold"):
        model.zero_grad()
        pen = m.penalty(model, b, {"lam": 1.0, "source": source})
        assert torch.isfinite(pen) and float(pen) >= 0
        pen.backward()
        assert model.out.weight.grad.abs().sum() > 0
    t = manifold_tangents(b["x"][:4], b["y"][:4], m._pool, m._pool_y)
    assert t.shape == (4, 5, 784) and torch.allclose(t.norm(dim=-1), torch.ones(4, 5), atol=1e-3)


def test_tangent_penalty_vanishes_for_an_invariant_function(ctx):
    from regpipe.methods.tangent import Tangent
    m = Tangent()
    model = MLP([8])
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()                                     # constant function: derivative along ANY direction is 0
    assert float(m.penalty(model, batch(), {"lam": 1.0, "source": "affine"})) == pytest.approx(0.0, abs=1e-10)


def test_tangent_distance_scores_are_distributions(data):
    from regpipe.data import stratified_subset, to_float
    from regpipe.methods.tangent import tangent_distance_scores
    idx = stratified_subset(data.y_train, 100, 0)
    p_tan, p_euc = tangent_distance_scores(to_float(data.x_val[:30]), to_float(data.x_train[idx]), data.y_train[idx])
    assert torch.allclose(p_tan.sum(1), torch.ones(30), atol=1e-5) and torch.allclose(p_euc.sum(1), torch.ones(30), atol=1e-5)


# ---------------------------------------------------------------------------- 7.3
def test_under_constrained_regime_is_singular_and_pinv_is_ridge_limit(data, ctx):
    from regpipe.methods.under_constrained import UnderConstrained
    m = UnderConstrained()
    hp = m.defaults()
    res = m.fit(data, hp, ctx)
    ex = m.extras(res, data, ctx, hp)
    assert ex["n_train_examples"] == 200 and ex["rank_XtX"] <= 200 < ex["n_inputs"] and ex["XtX_is_singular"]
    assert ex["pinv_vs_ridge_alpha0_max_abs_diff"] < 1e-3


# ---------------------------------------------------------------------------- generic contract
@pytest.mark.parametrize("name", [c.name for c in ORDER])
def test_every_method_trains_and_is_serialisable(name, data, ctx, tmp_path):
    from regpipe.bundle import Predictor, save_bundle
    m = get_methods([name])[-1]
    hp = m.defaults()
    res = m.fit(data, hp, ctx)
    assert len(res.history) >= 1 and np.isfinite(res.history[-1]["val_loss"])
    save_bundle(tmp_path / "b", res.model, {"method": name})
    pred = Predictor(tmp_path / "b")
    x = (data.x_val[:5].float() / 255).numpy()
    ref = F.softmax(res.model.eval()(torch.from_numpy(x)), -1).detach().numpy()
    assert np.allclose(pred.predict_proba(x), ref, atol=1e-5)


def test_training_is_reproducible_on_cpu(data, ctx):
    from regpipe.methods.baseline import Baseline
    a = Baseline().fit(data, {"lr": 0.05}, ctx)
    b = Baseline().fit(data, {"lr": 0.05}, ctx)
    assert [h["val_loss"] for h in a.history] == [h["val_loss"] for h in b.history]
