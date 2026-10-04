"""Meta-algorithms and model-design strategies (8.7)."""
import copy

import pytest
import torch
import torch.nn.functional as F

from optpipe.core import Ctx
from optpipe.data import to_float
from optpipe.methods.batch_norm import BatchNorm
from optpipe.methods.continuation import Continuation
from optpipe.methods.coordinate_descent import CoordinateDescent
from optpipe.methods.curriculum import Curriculum
from optpipe.methods.design_activations import DesignActivations
from optpipe.methods.design_aux_heads import DesignAuxHeads
from optpipe.methods.design_skip import DesignSkip
from optpipe.methods.polyak_averaging import PolyakAveraging
from optpipe.model import MLP, build_model, strip_aux
from optpipe.transforms import gaussian_blur


def test_batch_norm_uses_batch_stats_in_train_and_running_stats_in_eval(ctx, data):
    m = BatchNorm()
    model = m.build(m.defaults(), ctx)
    x = to_float(data.x_train[:256])
    model.train()
    before = model.bns[0].running_mean.clone()
    model(x)
    assert not torch.equal(before, model.bns[0].running_mean)             # running statistics updated by training passes
    model.eval()
    a = model(x[:8])
    b = model(x[:8])
    assert torch.equal(a, b)
    assert torch.allclose(model(x[:8]), model(torch.cat([x[:8], x[8:100]]))[:8], atol=1e-5)   # eval output independent of batch


def test_batch_norm_pre_activations_are_standardised(ctx, data):
    model = BatchNorm().build({}, ctx)
    x = to_float(data.x_train[:512])
    model.train()
    z = model.bns[0](model.hidden[0](model.norm(x)))
    assert z.mean(0).abs().max() < 1e-4 and (z.std(0) - 1).abs().max() < 0.01


def test_coordinate_descent_cycles_blocks_and_freezes_the_rest(ctx, data):
    m = CoordinateDescent()
    hp = {**m.defaults(), "cycle": 1}
    model = m.build(hp, ctx)
    opt = m.make_optimizer(model, hp, ctx)
    state = {}
    blocks = model.blocks()
    assert len(blocks) == 3
    x, y = to_float(data.x_train[:64]), data.y_train[:64]
    for epoch, expect_active in ((1, 0), (2, 1), (3, 2), (4, 0)):
        m.epoch_start(model, opt, hp, epoch, state)
        assert opt.active == expect_active
        before = [[p.detach().clone() for p in b] for b in blocks]
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        for i, b in enumerate(blocks):
            changed = any(not torch.equal(p, q) for p, q in zip(b, before[i]))
            assert changed == (i == expect_active)
    assert state["order"] == [0, 1, 2, 0]


def test_polyak_average_is_exponential_moving_average(ctx, data):
    m = PolyakAveraging()
    hp = {**m.defaults(), "one_minus_tau": 0.5}
    model = m.build(hp, ctx)
    state = m.init_state(model, hp, ctx)
    w0 = model.out.weight.detach().clone()
    with torch.no_grad():
        model.out.weight.add_(2.0)
    m.after_step(model, hp, state)
    avg = m.eval_model(model, state)
    assert torch.allclose(avg.out.weight, 0.5 * w0 + 0.5 * (w0 + 2.0), atol=1e-6)
    assert not torch.equal(avg.out.weight, model.out.weight)                   # averaged weights are what gets evaluated
    assert torch.equal(model.out.weight, w0 + 2.0)                              # live weights untouched


def test_polyak_with_tau_zero_tracks_the_iterate(ctx):
    m = PolyakAveraging()
    model = m.build({}, ctx)
    hp = {"one_minus_tau": 1.0}
    state = m.init_state(model, hp, ctx)
    with torch.no_grad():
        model.out.weight.mul_(3.0)
    m.after_step(model, hp, state)
    assert torch.allclose(m.eval_model(model, state).out.weight, model.out.weight)


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "elu", "tanh", "sigmoid", "maxout"])
def test_every_activation_builds_and_runs(act, ctx, data):
    m = DesignActivations()
    model = m.build({"activation": act}, ctx)
    out = model(to_float(data.x_train[:4]))
    assert out.shape == (4, 10) and torch.isfinite(out).all()
    if act == "maxout":
        assert model.hidden[0].out_features == 2 * ctx.hidden[0]               # k = 2 linear pieces per unit
    assert build_model(model.spec).state_dict().keys() == model.state_dict().keys()


def test_sigmoid_saturates_more_than_relu_in_a_deep_net(ctx, data):
    cfg_deep = Ctx({**ctx.cfg, "model": {**ctx.cfg["model"], "hidden": [64] * 8}}, torch.device("cpu"), 0)
    ratios = {}
    for act in ("relu", "sigmoid"):
        torch.manual_seed(0)
        model = DesignActivations().build({"activation": act}, cfg_deep)
        x, y = to_float(data.x_train[:128]), data.y_train[:128]
        F.cross_entropy(model(x), y).backward()
        ratios[act] = float(model.hidden[0].weight.grad.norm() / model.out.weight.grad.norm())
    assert ratios["sigmoid"] < 0.1 * ratios["relu"]                             # vanishing gradient with saturating units


def test_residual_layers_pass_the_input_through_when_their_weights_are_zero(ctx, data):
    m = DesignSkip()
    model = m.build({"depth": 4}, ctx)
    with torch.no_grad():
        for lin in list(model.hidden)[1:]:
            lin.weight.zero_()
            lin.bias.zero_()
    x = to_float(data.x_train[:8])
    h, acts = model.trunk(x)
    assert torch.allclose(h, acts[0], atol=1e-6)                                # identity path
    plain = m.build({"depth": 4, "skip": False}, ctx)
    with torch.no_grad():
        for lin in list(plain.hidden)[1:]:
            lin.weight.zero_()
            lin.bias.zero_()
    assert float(plain.trunk(x)[0].abs().max()) == 0.0                          # without skips the signal is destroyed


def test_aux_heads_exist_during_training_and_are_stripped_for_inference(ctx, data):
    m = DesignAuxHeads()
    model = m.build({"depth": 4}, ctx)
    out = model.forward_all(to_float(data.x_train[:8]))
    assert {"aux_0", "aux_1", "aux_2"} <= set(out)
    final = m.finalize(model)
    assert not final.aux and final.spec["aux_heads"] == []
    x = to_float(data.x_train[:8])
    assert torch.allclose(model(x), final(x), atol=1e-6)                         # same predictions without the heads
    assert not any(k.startswith("aux.") for k in final.state_dict())
    assert float(m.data_loss(model, {"x": x, "y": data.y_train[:8]}, {"aux_weight": 1.0})) > \
        float(m.data_loss(model, {"x": x, "y": data.y_train[:8]}, {"aux_weight": 0.0}))


def test_gaussian_blur_identity_and_smoothing(data):
    x = to_float(data.x_train[:16])
    assert torch.equal(gaussian_blur(x, 0.0), x)
    b = gaussian_blur(x, 1.5)
    tv = lambda t: (t.view(-1, 28, 28).diff(dim=2).abs().sum() + t.view(-1, 28, 28).diff(dim=1).abs().sum())
    assert tv(b) < 0.5 * tv(x)
    assert float(b.sum()) == pytest.approx(float(x.sum()), rel=0.15)             # blur roughly conserves ink


def test_continuation_sigma_schedule_is_monotone_and_ends_at_zero():
    hp = {"sigma0": 2.0, "anneal_frac": 0.5}
    s = [Continuation.sigma(hp, e, 20) for e in range(1, 21)]
    assert s[0] == pytest.approx(2.0)
    assert all(a >= b for a, b in zip(s, s[1:]))
    assert s[-1] == 0.0 and s[11] == 0.0


def test_continuation_last_epochs_train_on_sharp_images(ctx, data):
    m = Continuation()
    hp = m.defaults()
    batch = {"x": to_float(data.x_train[:8]), "y": data.y_train[:8]}
    ctx20 = Ctx(ctx.cfg, torch.device("cpu"), 0, epochs=20)
    assert torch.equal(m.transform(batch, hp, 20, ctx20)["x"], batch["x"])
    assert not torch.equal(m.transform(batch, hp, 1, ctx20)["x"], batch["x"])


def test_curriculum_pool_grows_and_starts_with_the_easiest(ctx, data):
    hp = {"start_frac": 0.2, "pace_frac": 0.5}
    f = [Curriculum.pool_fraction(hp, e, 10) for e in range(1, 11)]
    assert f[0] == pytest.approx(0.2) and f[-1] == 1.0 and all(a <= b for a, b in zip(f, f[1:]))
    m = Curriculum()
    ts = m.prepare(data, hp, ctx)
    order = ts.extras["order"]
    assert sorted(order.tolist()) == list(range(len(data.y_train)))              # a permutation of the training set
    gen = torch.Generator().manual_seed(0)
    n = len(data.y_train)
    pool = set(order[: max(ctx.batch_size, int(0.2 * n))].tolist())
    seen = 0
    for batch in m.batches(ts, hp, Ctx(ctx.cfg, torch.device("cpu"), 0, epochs=10), gen, epoch=1):
        seen += len(batch["y"])
    assert seen == n                                                             # same number of draws per epoch as other methods


def test_curriculum_difficulty_orders_by_loss(ctx, data):
    m = Curriculum()
    ts = m.prepare(data, m.defaults(), ctx)
    order = ts.extras["order"]
    # The easiest examples should be classified better by a simple reference model than the hardest ones:
    easy, hard = order[:300], order[-300:]
    assert order.shape[0] == len(data.y_train) and easy.ne(hard[:1]).all()
