"""First-order update rules against reference implementations (torch.optim or the book's formulas written out by hand)."""
import numpy as np
import pytest
import torch

from optpipe.methods.lr_decay import LRDecay
from optpipe.optim import SGD, AdaGrad, Adam, BlockCoordinateSGD, RMSProp

torch.manual_seed(0)
A = torch.randn(8, 8)
A = A @ A.T / 8 + torch.eye(8)           # well-conditioned: all tests stay in the stable regime
B = torch.randn(8)


def grad_at(w):
    return A @ w - B


def run_pair(ours, ref, steps=40):
    w1 = torch.zeros(8, requires_grad=True)
    w2 = torch.zeros(8, requires_grad=True)
    o1, o2 = ours([w1]), ref([w2])
    for _ in range(steps):
        for w, o in ((w1, o1), (w2, o2)):
            if hasattr(o, "prepare"):
                o.prepare()
            o.zero_grad()
            (0.5 * w @ A @ w - B @ w).backward()
            o.step()
    return float((w1 - w2).detach().abs().max())


def test_sgd_matches_torch():
    assert run_pair(lambda p: SGD(p, 0.05), lambda p: torch.optim.SGD(p, 0.05)) < 1e-6


def test_momentum_matches_torch():
    assert run_pair(lambda p: SGD(p, 0.05, 0.9), lambda p: torch.optim.SGD(p, 0.05, momentum=0.9)) < 1e-5


def test_nesterov_follows_the_book_formula():
    """Book Algorithm 8.3: gradient at theta + a v;  v <- a v - e g;  theta <- theta + v."""
    lr, a = 0.02, 0.9
    w = torch.zeros(8, requires_grad=True)
    opt = SGD([w], lr, a, nesterov=True)
    th, v = torch.zeros(8), torch.zeros(8)
    for _ in range(30):
        g = grad_at(th + a * v)
        v = a * v - lr * g
        th = th + v
        opt.prepare()
        opt.zero_grad()
        (0.5 * w @ A @ w - B @ w).backward()
        opt.step()
    assert float((w.detach() - th).abs().max()) < 1e-5


def test_nesterov_differs_from_plain_momentum():
    w1, w2 = torch.zeros(8, requires_grad=True), torch.zeros(8, requires_grad=True)
    o1, o2 = SGD([w1], 0.05, 0.9, True), SGD([w2], 0.05, 0.9, False)
    for _ in range(10):
        for w, o in ((w1, o1), (w2, o2)):
            o.prepare()
            o.zero_grad()
            (0.5 * w @ A @ w - B @ w).backward()
            o.step()
    assert not torch.allclose(w1, w2)


def test_adam_matches_torch():
    assert run_pair(lambda p: Adam(p, 0.01), lambda p: torch.optim.Adam(p, 0.01)) < 1e-5


def test_adagrad_matches_torch():
    assert run_pair(lambda p: AdaGrad(p, 0.1, delta=1e-7), lambda p: torch.optim.Adagrad(p, 0.1, eps=1e-7)) < 1e-6


def test_rmsprop_follows_algorithm_8_5():
    lr, rho, delta = 0.01, 0.9, 1e-6
    w = torch.zeros(8, requires_grad=True)
    opt = RMSProp([w], lr, rho, delta)
    th, r = torch.zeros(8), torch.zeros(8)
    for _ in range(30):
        g = grad_at(th)
        r = rho * r + (1 - rho) * g * g
        th = th - lr / torch.sqrt(delta + r) * g
        opt.zero_grad()
        (0.5 * w @ A @ w - B @ w).backward()
        opt.step()
    assert float((w.detach() - th).abs().max()) < 1e-5


def test_rmsprop_nesterov_follows_algorithm_8_6():
    lr, rho, a, delta = 0.005, 0.9, 0.9, 1e-6
    w = torch.zeros(8, requires_grad=True)
    opt = RMSProp([w], lr, rho, delta, momentum=a)
    th, v, r = torch.zeros(8), torch.zeros(8), torch.zeros(8)
    for _ in range(30):
        g = grad_at(th + a * v)
        r = rho * r + (1 - rho) * g * g
        v = a * v - lr / torch.sqrt(r + delta) * g
        th = th + v
        opt.prepare()
        opt.zero_grad()
        (0.5 * w @ A @ w - B @ w).backward()
        opt.step()
    assert float((w.detach() - th).abs().max()) < 1e-5


@pytest.mark.parametrize("make", [lambda p: SGD(p, 0.05, 0.9), lambda p: Adam(p, 0.05), lambda p: AdaGrad(p, 0.3),
                                  lambda p: RMSProp(p, 0.02)], ids=["momentum", "adam", "adagrad", "rmsprop"])
def test_every_optimizer_minimises_a_quadratic(make):
    w = torch.zeros(8, requires_grad=True)
    opt = make([w])
    for _ in range(400):
        opt.zero_grad()
        (0.5 * w @ A @ w - B @ w).backward()
        opt.step()
    best = torch.linalg.solve(A, B)
    assert float((0.5 * w @ A @ w - B @ w) - (0.5 * best @ A @ best - B @ best)) < 1e-2


def test_block_coordinate_updates_only_the_active_block():
    a, b = torch.randn(3, requires_grad=True), torch.randn(3, requires_grad=True)
    opt = BlockCoordinateSGD([[a], [b]], lr=0.1, momentum=0.9)
    a0, b0 = a.detach().clone(), b.detach().clone()
    opt.set_active(0)
    opt.zero_grad()
    (a.sum() + b.sum()).backward()
    opt.step()
    assert not torch.equal(a, a0) and torch.equal(b, b0)
    opt.set_active(1)
    a1 = a.detach().clone()
    opt.zero_grad()
    (a.sum() + b.sum()).backward()
    opt.step()
    assert torch.equal(a, a1) and not torch.equal(b, b0)


class _Ctx:
    epochs = 10


def test_linear_decay_matches_equation_8_14():
    m = LRDecay()
    hp = {"schedule": "linear", "tau_frac": 0.5, "final_frac": 0.1}
    tau = 5.0
    for k in (0.0, 1.0, 2.5, 5.0, 7.0, 10.0):
        a = min(k / tau, 1.0)
        assert m.lr_factor(k, hp, _Ctx) == pytest.approx((1 - a) * 1.0 + a * 0.1)
    assert m.lr_factor(8, hp, _Ctx) == pytest.approx(0.1)            # constant after tau


def test_other_schedules():
    m = LRDecay()
    assert m.lr_factor(3, {"schedule": "exponential", "gamma": 0.5}, _Ctx) == pytest.approx(0.125)
    assert m.lr_factor(2, {"schedule": "inverse", "d": 1.0}, _Ctx) == pytest.approx(1 / 3)
    assert m.lr_factor(7, {"schedule": "step", "step_epochs": 3}, _Ctx) == pytest.approx(0.25)
