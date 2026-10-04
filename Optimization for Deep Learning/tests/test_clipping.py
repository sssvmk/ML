"""Gradient clipping variants (8.2.4)."""
import pytest
import torch
import torch.nn.functional as F

from optpipe.data import to_float
from optpipe.methods.clip_backprop import ClipBackprop
from optpipe.methods.clip_norm import ClipNorm
from optpipe.methods.clip_value import ClipValue
from optpipe.model import MLP


def grads(model, data):
    model.zero_grad()
    F.cross_entropy(model(to_float(data.x_train[:128])), data.y_train[:128]).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def test_value_clipping_bounds_every_component_and_keeps_small_ones(ctx, data):
    model = MLP([32, 16])
    g0 = grads(model, data)
    v = float(g0.abs().quantile(0.9))
    state = {}
    ClipValue().clip(model, {"v": v}, state)
    g1 = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert float(g1.abs().max()) <= v + 1e-9
    small = g0.abs() <= v
    assert torch.equal(g1[small], g0[small])
    assert ClipValue().final_stats(model, state)["clipped_element_fraction"] == pytest.approx(0.1, abs=0.02)


def test_norm_clipping_bounds_norm_and_preserves_direction(ctx, data):
    model = MLP([32, 16])
    g0 = grads(model, data)
    v = float(g0.norm()) / 4
    state = {}
    ClipNorm().clip(model, {"v": v}, state)
    g1 = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert float(g1.norm()) == pytest.approx(v, rel=1e-4)
    assert float(F.cosine_similarity(g0, g1, dim=0)) == pytest.approx(1.0, abs=1e-6)     # same direction
    assert ClipNorm().final_stats(model, state)["clipped_step_fraction"] == 1.0


def test_norm_clipping_is_inactive_below_the_threshold(ctx, data):
    model = MLP([32, 16])
    g0 = grads(model, data)
    ClipNorm().clip(model, {"v": float(g0.norm()) * 10}, {})
    assert torch.equal(torch.cat([p.grad.flatten() for p in model.parameters()]), g0)


def test_backprop_hook_clamps_and_counts():
    m = MLP([8])
    m.act_grad_clip = 0.5
    out = m._clip_hook(torch.tensor([-2.0, -0.1, 0.3, 4.0]))
    assert out.tolist() == pytest.approx([-0.5, -0.1, 0.3, 0.5])
    assert m.clip_hits == [2, 4]


def test_backprop_clipping_limits_error_signals_reaching_lower_layers(ctx, data):
    x, y = to_float(data.x_train[:128]), data.y_train[:128]

    def first_layer_grad(clip):
        torch.manual_seed(0)
        model = MLP([32, 16])
        model.act_grad_clip = clip
        model.train()
        model.zero_grad()
        F.cross_entropy(model(x), y).backward()
        return float(model.hidden[0].weight.grad.norm()), model

    free, _ = first_layer_grad(None)
    clipped, model = first_layer_grad(1e-5)
    assert clipped < 0.1 * free
    assert model.clip_hits[0] > 0


def test_backprop_clipping_inactive_in_eval_mode(ctx, data):
    model = MLP([32, 16])
    model.act_grad_clip = 1e-9
    model.eval()
    x = to_float(data.x_train[:16]).requires_grad_(True)
    model(x).sum().backward()
    assert model.clip_hits == [0, 0]


def test_clip_backprop_method_sets_threshold_and_finalize_removes_it(ctx):
    m = ClipBackprop()
    model = m.build({"v": 0.01}, ctx)
    assert model.act_grad_clip == 0.01
    assert m.finalize(model).act_grad_clip is None
