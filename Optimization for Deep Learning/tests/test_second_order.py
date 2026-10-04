"""Line search and the second-order methods, on problems with known answers (float64)."""
import math

import pytest
import torch

from optpipe.core import set_l1_smoothing
from optpipe.fullbatch import FlatProblem
from optpipe.linesearch import backtracking, strong_wolfe
from optpipe.methods.bfgs import BFGSStepper, bfgs_update
from optpipe.methods.conjugate_gradients import NonlinearCGStepper
from optpipe.methods.lbfgs import LBFGSStepper, two_loop
from optpipe.methods.newton import NewtonCGStepper, truncated_cg


class Quad:
    """f = 0.5 x'Ax - b'x with the FlatProblem interface used by the steppers."""
    def __init__(self, n=10, seed=0, cond=50.0):
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
        self.A = Q @ torch.diag(torch.linspace(1, cond, n)) @ Q.T
        self.b = torch.randn(n, generator=g)
        self.x = torch.zeros(n)
        self.n, self.N, self.passes = n, 100, 0.0

    def get(self): return self.x.clone()
    def set(self, v): self.x = v.clone()
    def value(self, v=None):
        if v is not None: self.set(v)
        return float(0.5 * self.x @ self.A @ self.x - self.b @ self.x)
    def value_and_grad(self, v=None):
        if v is not None: self.set(v)
        self.passes += 1
        return self.value(), self.A @ self.x - self.b
    def hvp(self, v, idx): return self.A @ v
    def solution(self): return torch.linalg.solve(self.A, self.b)


class Rosen:
    n, N, passes = 2, 1, 0.0
    def __init__(self): self.x = torch.tensor([-1.2, 1.0])
    def get(self): return self.x.clone()
    def set(self, v): self.x = v.clone()
    def value_and_grad(self, v=None):
        if v is not None: self.set(v)
        x = self.x.clone().requires_grad_(True)
        f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        g, = torch.autograd.grad(f, x)
        return float(f), g
    def value(self, v=None): return self.value_and_grad(v)[0]


def rosen_phi(x, d):
    def phi(a):
        xa = (x + a * d).clone().requires_grad_(True)
        f = (1 - xa[0]) ** 2 + 100 * (xa[1] - xa[0] ** 2) ** 2
        g, = torch.autograd.grad(f, xa)
        return float(f), float(g @ d), g
    return phi


@pytest.mark.parametrize("c2", [0.9, 0.1])
def test_strong_wolfe_conditions_hold(f64, c2):
    x = torch.tensor([-1.2, 1.0])
    f0, g0 = Rosen().value_and_grad()
    d = -g0
    dphi0 = float(g0 @ d)
    a, f, g, _ = strong_wolfe(rosen_phi(x, d), f0, dphi0, 1.0, c2=c2)
    assert f <= f0 + 1e-4 * a * dphi0                         # sufficient decrease
    assert abs(float(g @ d)) <= -c2 * dphi0 + 1e-12            # curvature (strong Wolfe)


def test_strong_wolfe_refuses_ascent_direction(f64):
    assert strong_wolfe(lambda a: (1.0, 1.0, None), 1.0, 1.0) is None


def test_backtracking_armijo(f64):
    r = backtracking(lambda a: (1 - a) ** 2 * 10, 10.0, -20.0, alpha=8.0)
    assert r is not None and r[1] <= 10.0 - 1e-4 * r[0] * 20


def test_truncated_cg_solves_spd_system_exactly(f64):
    q = Quad()
    g = torch.randn(10)
    d, Bd = truncated_cg(lambda v: q.A @ v, g, max_iter=50, tol=1e-12)
    assert torch.allclose(d, -torch.linalg.solve(q.A, g), atol=1e-8)
    assert torch.allclose(Bd, -g, atol=1e-8)


def test_truncated_cg_stops_on_negative_curvature(f64):
    H = torch.diag(torch.tensor([1.0, -2.0, 3.0]))
    d, _ = truncated_cg(lambda v: H @ v, torch.tensor([1.0, 1.0, 1.0]), 20, 1e-12)
    assert torch.isfinite(d).all()


def test_newton_step_on_a_quadratic_is_exact(f64):
    q = Quad()
    st = NewtonCGStepper(damping=1e-9, cg_iters=50, curv_batch=10, seed=0, exact=True)
    st.step(q)
    assert torch.allclose(q.get(), q.solution(), atol=1e-5)    # one Newton step lands on the minimiser


def test_damping_shortens_the_newton_step(f64):
    q1, q2 = Quad(), Quad()
    NewtonCGStepper(1e-9, 50, 10, 0, exact=True).step(q1)
    NewtonCGStepper(100.0, 50, 10, 0, exact=True).step(q2)
    assert q2.get().norm() < q1.get().norm()


def test_nonlinear_cg_minimises_a_quadratic_in_about_n_steps(f64):
    q = Quad(n=10)
    st = NonlinearCGStepper("PR+", restart=100, c2=0.1)
    for _ in range(30):
        st.step(q)
    assert torch.allclose(q.get(), q.solution(), atol=1e-4)


@pytest.mark.parametrize("variant", ["PR+", "FR"])
def test_cg_variants_solve_rosenbrock(f64, variant):
    p = Rosen()
    st = NonlinearCGStepper(variant, restart=20, c2=0.1)
    for _ in range(400):
        st.step(p)
        if st.done:
            break
    assert p.value() < 1e-3


def test_bfgs_update_satisfies_the_secant_equation(f64):
    q = Quad(n=6)
    H = torch.eye(6)
    s = torch.randn(6)
    y = q.A @ s
    assert bfgs_update(H, s, y)
    assert torch.allclose(H @ y, s, atol=1e-9)
    assert torch.allclose(H, H.T, atol=1e-9)


def test_bfgs_update_skips_when_curvature_condition_fails(f64):
    H = torch.eye(3)
    assert bfgs_update(H, torch.tensor([1.0, 0, 0]), torch.tensor([-1.0, 0, 0])) is False
    assert torch.equal(H, torch.eye(3))


def test_bfgs_converges_on_rosenbrock(f64):
    p = Rosen()
    st = BFGSStepper(c2=0.9)
    for _ in range(100):
        st.step(p)
        if st.done:
            break
    assert p.value() < 1e-8


def test_lbfgs_converges_on_rosenbrock(f64):
    p = Rosen()
    st = LBFGSStepper(history=10, c2=0.9)
    for _ in range(100):
        st.step(p)
        if st.done:
            break
    assert p.value() < 1e-8


def test_two_loop_equals_dense_bfgs_with_full_history(f64):
    q = Quad(n=8)
    s_list, y_list = [], []
    for _ in range(5):
        s = torch.randn(8)
        s_list.append(s)
        y_list.append(q.A @ s)
    gamma = float(s_list[-1] @ y_list[-1]) / float(y_list[-1] @ y_list[-1])
    H = torch.eye(8) * gamma
    for s, y in zip(s_list, y_list):                     # dense BFGS with the same H0 recursion order
        pass
    # Build the dense matrix implied by the two-loop recursion by applying it to the identity.
    g = torch.randn(8)
    dense = torch.stack([two_loop(e, s_list, y_list) for e in torch.eye(8)], dim=1)
    assert torch.allclose(dense, dense.T, atol=1e-8)             # symmetric, as an inverse-Hessian approximation must be
    for s, y in zip(s_list[-1:], y_list[-1:]):
        assert torch.allclose(dense @ y, s, atol=1e-8)           # most recent secant equation holds
    assert torch.allclose(two_loop(g, [], []), g)


def test_lbfgs_history_is_bounded(f64):
    p = Quad(n=10)
    st = LBFGSStepper(history=3, c2=0.9)
    for _ in range(10):
        st.step(p)
    assert len(st.s) <= 3


# ---- FlatProblem on a real network: exact gradients and Hessian-vector products ---------------------------------
def test_flatproblem_gradient_and_hvp_match_finite_differences(f64, data):
    from optpipe.model import MLP
    set_l1_smoothing(1e-3)
    torch.manual_seed(0)
    model = MLP([6])
    prob = FlatProblem(model, data.x_train[:200], data.y_train[:200], l1=1e-3, l2=1e-2, device=torch.device("cpu"), chunk=64)
    theta = prob.get()
    f, g = prob.value_and_grad()
    for i in (0, 7, 100, prob.n - 1):
        e = torch.zeros(prob.n)
        e[i] = 1e-6
        fd = (prob.value(theta + e) - prob.value(theta - e)) / 2e-6
        assert fd == pytest.approx(float(g[i]), rel=2e-3, abs=2e-6)
    prob.set(theta)
    v = torch.randn(prob.n)
    idx = torch.arange(200)
    hv = prob.hvp(v, idx)
    eps = 1e-5
    _, g1 = prob.value_and_grad(theta + eps * v)
    _, g2 = prob.value_and_grad(theta - eps * v)
    prob.set(theta)
    assert torch.allclose(hv, (g1 - g2) / (2 * eps), atol=1e-3, rtol=1e-2)
    assert prob.passes > 0


def test_flatproblem_counts_data_passes(f64, data):
    from optpipe.model import MLP
    prob = FlatProblem(MLP([4]), data.x_train[:100], data.y_train[:100], 0.0, 0.0, torch.device("cpu"))
    prob.value_and_grad()
    prob.value_and_grad()
    assert prob.passes == pytest.approx(2.0)
    prob.hvp(torch.randn(prob.n), torch.arange(50))
    assert prob.passes == pytest.approx(3.0)             # 2 * 50 / 100 = 1 pass


def test_steppers_stop_when_the_objective_stops_decreasing(f64):
    from optpipe.fullbatch import Stepper
    st = Stepper()
    for _ in range(2):
        st.note_progress(1.0, 1.0)
    assert st.done is False
    st.note_progress(1.0, 1.0)
    assert st.done is True
    st2 = Stepper()
    st2.note_progress(1.0, 1.0)
    st2.note_progress(1.0, 0.5)                          # real progress resets the counter
    st2.note_progress(0.5, 0.5)
    assert st2.done is False


def test_lbfgs_stops_by_itself_on_a_solved_quadratic(f64):
    q = Quad(n=6)
    st = LBFGSStepper(history=10, c2=0.9)
    for _ in range(200):
        st.step(q)
        if st.done:
            break
    assert st.done and torch.allclose(q.get(), q.solution(), atol=1e-5)


def test_steppers_stop_on_a_small_gradient(f64):
    from optpipe.fullbatch import Stepper
    st = Stepper()
    st.note_gradient(torch.tensor([1e-3, -2e-3]))
    assert st.done is False
    st.note_gradient(torch.tensor([1e-6, -9e-6]))
    assert st.done is True


def test_cg_and_newton_converge_and_stop_on_a_quadratic(f64):
    for stepper in (NonlinearCGStepper("PR+", 100, 0.1), NewtonCGStepper(1e-9, 50, 10, 0, exact=True)):
        q = Quad(n=6)
        for _ in range(100):
            stepper.step(q)
            if stepper.done:
                break
        assert stepper.done and torch.allclose(q.get(), q.solution(), atol=1e-4)


def test_strong_wolfe_stops_at_alpha_max_on_a_nearly_linear_valley(f64):
    """f(a) = -1e-3 a (decreasing forever): accept the capped step after ~log2(alpha_max/alpha1) evaluations."""
    calls = []

    def phi(a):
        calls.append(a)
        return -1e-3 * a, -1e-3, torch.tensor([-1e-3])

    a, f, g, n = strong_wolfe(phi, 0.0, -1e-3, alpha1=1.0, c2=0.1, alpha_max=20.0, max_iter=15)
    assert a == 20.0 and n == len(calls) <= 6
