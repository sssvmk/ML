"""Each initialisation strategy (section 8.4) does what its definition says."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from optpipe.data import to_float
from optpipe.methods.bias_gate import BiasGate
from optpipe.methods.bias_output_marginal import BiasOutputMarginal
from optpipe.methods.bias_relu_positive import BiasReluPositive
from optpipe.methods.bias_zero import BiasZero
from optpipe.methods.greedy_pretraining import GreedyPretraining
from optpipe.methods.init_fixed_scale import InitFixedScale
from optpipe.methods.init_glorot import InitGlorot
from optpipe.methods.init_orthogonal import InitOrthogonal
from optpipe.methods.init_random import InitRandom
from optpipe.methods.init_random_walk import InitRandomWalk, random_walk_init
from optpipe.methods.init_scale_search import InitScaleSearch, lsuv_init
from optpipe.methods.init_sparse import InitSparse, sparse_init_
from optpipe.methods.init_supervised_pretrain import InitSupervisedPretrain, task_labels
from optpipe.methods.init_unsupervised_pretrain import InitUnsupervisedPretrain
from optpipe.methods.variance_precision import VariancePrecision


def make(cls, ctx, data, **over):
    m = cls()
    hp = {**m.defaults(), **over}
    model = m.build(hp, ctx)
    info = m.initialize(model, hp, ctx, data)
    return m, hp, model, info


def test_random_normal_has_requested_std_and_zero_bias(ctx, data):
    _, _, model, _ = make(InitRandom, ctx, data, dist="normal", std=0.1)
    w = model.hidden[0].weight
    assert w.std().item() == pytest.approx(0.1, rel=0.03)
    assert all(float(l.bias.abs().sum()) == 0 for l in model.weight_layers())


def test_random_uniform_has_requested_std_and_bounded_support(ctx, data):
    _, _, model, _ = make(InitRandom, ctx, data, dist="uniform", std=0.1)
    w = model.hidden[0].weight
    assert w.std().item() == pytest.approx(0.1, rel=0.03)
    assert float(w.abs().max()) <= math.sqrt(3) * 0.1 + 1e-6


def test_random_init_breaks_symmetry(ctx, data):
    _, _, model, _ = make(InitRandom, ctx, data)
    w = model.hidden[0].weight
    assert torch.unique(w, dim=0).shape[0] == w.shape[0]                    # no two units start identical


def test_fixed_scale_bound_is_one_over_sqrt_m(ctx, data):
    _, _, model, _ = make(InitFixedScale, ctx, data, gain=1.0)
    for lin in model.weight_layers():
        assert float(lin.weight.abs().max()) <= 1 / math.sqrt(lin.in_features) + 1e-7


def test_glorot_bound(ctx, data):
    _, _, model, _ = make(InitGlorot, ctx, data, gain=1.0)
    for lin in model.weight_layers():
        assert float(lin.weight.abs().max()) <= math.sqrt(6 / (lin.in_features + lin.out_features)) + 1e-7


def test_orthogonal_rows_are_orthogonal_with_norm_g(ctx, data):
    g = 1.7
    _, _, model, _ = make(InitOrthogonal, ctx, data, gain=g)
    for lin in model.weight_layers():                                       # all layers have rows < columns here
        W = lin.weight
        assert torch.allclose(W @ W.T, g * g * torch.eye(W.shape[0]), atol=1e-4)


def test_sparse_init_has_exactly_k_nonzeros_per_unit(ctx, data):
    _, _, model, _ = make(InitSparse, ctx, data, k=7, std=1.0)
    for lin in model.weight_layers():
        nnz = (lin.weight != 0).sum(1)
        assert (nnz == min(7, lin.in_features)).all()


def test_sparse_init_helper_clamps_k_to_fan_in():
    w = torch.empty(4, 3)
    sparse_init_(w, 10, 1.0)
    assert ((w != 0).sum(1) == 3).all()


def test_random_walk_calibrates_backward_norm_ratio(ctx, data):
    model = ctx_model = None
    from optpipe.model import MLP
    model = MLP([32, 16])
    x = to_float(data.x_train[:256])
    for target in (0.7, 1.0, 1.4):
        measured = random_walk_init(model, x, ratio=target, seed=1)
        assert all(m == pytest.approx(target, rel=1e-3) for m in measured)
    assert all(float(l.bias.abs().sum()) == 0 for l in model.weight_layers())


def test_random_walk_preserves_relu_masks_when_rescaling(ctx, data):
    from optpipe.model import MLP
    model = MLP([32, 16])
    x = to_float(data.x_train[:128])
    random_walk_init(model, x, ratio=1.0, seed=3)
    h = model.norm(x)
    signs_after = []
    for lin in model.hidden:
        z = lin(h)
        signs_after.append((z > 0))
        h = F.relu(z)
    assert all(s.float().mean() > 0.2 for s in signs_after)                   # units are alive, not all dead / all on


def test_lsuv_gives_unit_preactivation_variance(ctx, data):
    from optpipe.model import MLP
    model = MLP([32, 16])
    x = to_float(data.x_train[:256])
    var = lsuv_init(model, x, seed=0)
    assert all(abs(v - 1.0) < 0.05 for v in var)
    assert float(model.hidden[0](model.norm(x)).var()) == pytest.approx(1.0, abs=0.05)


def test_scale_search_mode_multiplies_he_std(ctx, data):
    _, _, a, _ = make(InitScaleSearch, ctx, data, mode="search", scale_0=1.0)
    _, _, b, _ = make(InitScaleSearch, ctx, data, mode="search", scale_0=3.0)
    assert (b.hidden[0].weight.std() / a.hidden[0].weight.std()).item() == pytest.approx(3.0, rel=0.1)


def test_bias_zero(ctx, data):
    _, _, model, _ = make(BiasZero, ctx, data)
    assert all(float(l.bias.abs().sum()) == 0 for l in model.weight_layers())


def test_output_bias_reproduces_class_frequencies(ctx):
    y = torch.cat([torch.zeros(700), torch.ones(200), torch.full((100,), 2.0)]).long()
    fake = SimpleNamespace(y_train=y)
    m = BiasOutputMarginal()
    hp = {**m.defaults(), "strength": 1.0}
    model = m.build(hp, ctx)
    info = m.initialize(model, hp, ctx, fake)
    p = F.softmax(model.out.bias.detach(), dim=0)
    assert p[0].item() == pytest.approx(0.7, abs=1e-3) and p[1].item() == pytest.approx(0.2, abs=1e-3)
    assert p[2].item() == pytest.approx(0.1, abs=1e-3)
    assert info["class_prior_entropy"] > 0


def test_output_bias_strength_zero_leaves_bias_zero(ctx, data):
    _, _, model, _ = make(BiasOutputMarginal, ctx, data, strength=0.0)
    assert float(model.out.bias.abs().sum()) == 0


def test_relu_bias_value(ctx, data):
    _, _, model, _ = make(BiasReluPositive, ctx, data, bias=0.25)
    assert all(torch.all(l.bias == 0.25) for l in model.hidden) and float(model.out.bias.abs().sum()) == 0


def test_gate_bias_value_and_gating_effect(ctx, data):
    _, _, model, _ = make(BiasGate, ctx, data, gate_bias=3.0)
    assert model.gated and all(torch.all(g.bias == 3.0) for g in model.gates)
    _, _, closed, _ = make(BiasGate, ctx, data, gate_bias=-6.0)
    x = to_float(data.x_train[:64])
    open_act = model.trunk(x)[1][0].abs().mean()
    closed_act = closed.trunk(x)[1][0].abs().mean()
    assert open_act > 5 * closed_act                                           # a closed gate silences the unit


def test_precision_init_one_vs_marginal(ctx, data):
    _, _, a, ia = make(VariancePrecision, ctx, data, precision_init="one")
    _, _, b, ib = make(VariancePrecision, ctx, data, precision_init="marginal")
    assert ia["beta_init"] == pytest.approx(1.0)
    assert ib["beta_init"] == pytest.approx(1.0 / ib["marginal_variance"], rel=1e-5)
    assert ib["marginal_variance"] == pytest.approx(0.09, abs=0.01)           # 10 balanced classes: c(1-c) = 0.09


def test_gaussian_loss_formula(ctx, data):
    m = VariancePrecision()
    model = m.build(m.defaults(), ctx)
    f = torch.tensor([[1.0] + [0.0] * 9, [0.0] * 10])
    model.mean_output = lambda x: f
    with torch.no_grad():
        model.log_beta.fill_(math.log(4.0))
    y = torch.tensor([0, 3])
    got = float(m.data_loss(model, {"x": None, "y": y}, {}))
    sse = [0.0, 1.0 + 1.0]                                                     # sample 1 exact; sample 2: (0-1)^2 + (0-0)^2 ... = 1
    sse = [0.0, 1.0]
    expected = sum(0.5 * 4.0 * s for s in sse) / 2 - 0.5 * 10 * math.log(4.0)
    assert got == pytest.approx(expected, rel=1e-5)


def test_precision_minimiser_is_k_over_sse():
    K, sse = 10, 0.8
    betas = torch.linspace(0.5, 40, 4000)
    nll = 0.5 * betas * sse - 0.5 * K * torch.log(betas)
    assert float(betas[nll.argmin()]) == pytest.approx(K / sse, rel=0.02)


@pytest.mark.parametrize("E", [2, 5, 20])
@pytest.mark.parametrize("cls", [InitUnsupervisedPretrain, InitSupervisedPretrain])
def test_pretraining_keeps_total_epoch_budget(cls, E, cfg, data):
    from optpipe.core import Ctx
    ctx = Ctx(cfg, torch.device("cpu"), 0, epochs=E)
    m = cls()
    hp = {**m.defaults(), "pre_frac": 0.4}
    model = m.build(hp, ctx)
    info = m.initialize(model, hp, ctx, data)
    assert info["pretrain_epochs"] + m.epochs(hp, ctx) == E
    assert m.epochs(hp, ctx) >= 1


def test_unsupervised_pretraining_changes_the_hidden_weights(ctx, data):
    m = InitUnsupervisedPretrain()
    hp = m.defaults()
    a, b = m.build(hp, ctx), m.build(hp, ctx)
    torch.manual_seed(0)
    m.initialize(b, hp, ctx, data)
    assert not torch.allclose(a.hidden[0].weight, b.hidden[0].weight)


def test_task_labels(data):
    y = torch.arange(10)
    lab, k = task_labels("related", None, y, 0)
    assert k == 4 and lab.tolist() == [0, 2, 0, 2, 0, 3, 1, 3, 1, 3]
    l1, k1 = task_labels("unrelated", data.x_train[:200], data.y_train[:200], 0)
    l2, _ = task_labels("unrelated", data.x_train[:200], data.y_train[:200], 0)
    assert k1 == 10 and torch.equal(l1, l2) and len(torch.unique(l1)) > 3


def test_greedy_pretraining_budget_and_stage_layers(cfg, data):
    from optpipe.core import Ctx
    cfg3 = {**cfg, "model": {**cfg["model"], "hidden": [32, 16, 8]}}
    for E in (3, 6, 20):
        ctx = Ctx(cfg3, torch.device("cpu"), 0, epochs=E)
        m = GreedyPretraining()
        hp = {**m.defaults(), "pre_frac": 0.5}
        model = m.build(hp, ctx)
        before = model.hidden[0].weight.clone()
        info = m.initialize(model, hp, ctx, data)
        assert info["stages"] == 2
        assert info["pretrain_epochs"] + m.epochs(hp, ctx) == E
        assert not torch.allclose(before, model.hidden[0].weight)             # layer 1 came from a pretrained stage
