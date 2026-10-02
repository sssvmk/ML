"""Verify the kernel machinery against brute-force / scikit-learn references."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common, kern
import nw_classification as nwc, kernel_density_classifier as kdc, naive_bayes as nb, gaussian_mixture_classifier as gmc, rbf_network as rbf
from scipy.optimize import approx_fprime
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
# ---- 1-D local polynomial vs brute force at the target point ----
n = 300; x = np.sort(rng.rand(n)); y = np.sin(5 * x) + 0.3 * rng.randn(n); span = 0.15
g = np.array([0.0, 0.2, 0.55, 0.9, 1.0])
def brute(x0, deg):
    k = int(np.ceil(span * n)); d = np.abs(x - x0); h = np.sort(d)[k - 1] * (1 + 1e-12) + 1e-12
    w = kern.tricube(d / h); B = np.vander(x - x0, deg + 1, increasing=True)
    return np.linalg.solve((B * w[:, None]).T @ B, (B * w[:, None]).T @ y)[0]
for deg in (0, 1, 2):
    r = kern.local_poly_1d(x, y, g, span, deg)
    check(f"local polynomial degree {deg} == brute-force weighted least squares", np.abs(r["fit"] - np.array([brute(t, deg) for t in g])).max() < 1e-9)
# ---- smoother matrix: S y == fit and trace(S) == df at the data points ----
r = kern.local_poly_1d(x, y, x, span, 1)
k = int(np.ceil(span * n)); S = np.zeros((n, n))
for j, x0 in enumerate(x):
    d = np.abs(x - x0); h = np.sort(d)[k - 1] * (1 + 1e-12) + 1e-12; w = kern.tricube(d / h); B = np.vander((x - x0) / h, 2, increasing=True)
    S[j] = w * (B @ np.linalg.inv((B * w[:, None]).T @ B)[:, 0])
check("equivalent-kernel smoother matrix: S y == fitted values", np.abs(S @ y - r["fit"]).max() < 1e-9)
dg = kern.diag_from_grid(x, y, x, r)
check("df = trace(S) from self-leverage", abs(dg["df"] - np.trace(S)) < 1e-8, f"{dg['df']:.4f} vs {np.trace(S):.4f}")
check("pointwise variance factor == ||l(x)||^2", abs(r["varfac"][50] - (S[50] @ S[50])) < 1e-9)
# ---- local binomial / batched local logistic vs scikit-learn ----
g0 = 0.5; k = int(np.ceil(0.3 * n)); d = np.abs(x - g0); h = np.sort(d)[k - 1] * (1 + 1e-12) + 1e-12; w = kern.tricube(d / h); yb = (rng.rand(n) < 1 / (1 + np.exp(-3 * (x - 0.5)))).astype(float)
res = kern.local_glm_1d(x, yb, np.array([g0]), 0.3, 1, "binomial", ridge=1e-10)
m = w > 0
sk = LogisticRegression(C=1e9, tol=1e-12, max_iter=1000).fit(((x - g0) / h)[m][:, None], yb[m].astype(int), sample_weight=w[m])
check("1-D local logistic == weighted scikit-learn logistic", abs(res["fit"][0] - sk.intercept_[0]) < 1e-4, f"{res['fit'][0]:.5f} vs {sk.intercept_[0]:.5f}")
Xn = rng.randn(400, 3); ynn = (rng.rand(400) < kern.sigmoid(Xn[:, 0] - 0.5 * Xn[:, 1] - 1)).astype(float); wn = kern.tricube(np.linalg.norm(Xn, axis=1) / 2.5)
eta, se = kern.local_logistic_batch((Xn[None] - 0.0), ynn[None], wn[None], ridge=1e-10, iters=25)
mm = wn > 0
sk = LogisticRegression(C=1e9, tol=1e-12, max_iter=1000).fit(Xn[mm], ynn[mm].astype(int), sample_weight=wn[mm])
check("batched local logistic (R^3) == weighted scikit-learn", abs(eta[0] - sk.intercept_[0]) < 1e-3, f"{eta[0]:.4f} vs {sk.intercept_[0]:.4f}")
# ---- N-W classification vs brute force kernel-weighted proportion ----
Xr = rng.randn(500, 4); yr = (rng.rand(500) < 0.3).astype(float); Xq = rng.randn(5, 4); sp = 0.1
P = nwc.predict_path(Xr, yr, [{"span": sp}], Xq, common.RunConfig(), {})[0]
kk = int(np.ceil(sp * 500)); bf = []
for q in Xq:
    d = np.linalg.norm(Xr - q, axis=1); h = np.sort(d)[kk - 1] * (1 + 1e-12) + 1e-12; w_ = kern.tricube(d / h); bf.append((w_ * yr).sum() / w_.sum())
check("N-W classifier == kernel-weighted class proportion", np.abs(P - np.array(bf)).max() < 1e-9)
# ---- kernel density vs scikit-learn ----
Xc = rng.randn(800, 3); Xq = rng.randn(40, 3)
ld = kdc.log_density(Xc, Xq, [0.4])[0]
ref = KernelDensity(kernel="gaussian", bandwidth=0.4).fit(Xc).score_samples(Xq)
check("Gaussian KDE (exact, chunked) == scikit-learn KernelDensity", np.abs(ld - ref).max() < 2e-3, f"{np.abs(ld - ref).max():.1e}")
# ---- naive Bayes marginal KDE on a grid vs direct sum ----
z = rng.randn(4000, 2); c = 1.0; tab = nb._logdens(z, c); hb = c * z[:, 0].std() * 4000 ** -0.2
cent = nb.LO + (np.array([2400 // 2, 2400 // 2 + 300, 2400 // 2 - 400]) + 0.5) * nb.BW
direct = [np.log(np.mean(np.exp(-0.5 * ((q - z[:, 0]) / hb) ** 2) / (hb * np.sqrt(2 * np.pi)))) for q in cent]
check("naive Bayes binned 1-D KDE == direct kernel sum", np.abs(tab[0][[1200, 1500, 800]] - direct).max() < 0.02, f"{np.abs(tab[0][[1200, 1500, 800]] - direct).max():.1e}")
# ---- Gaussian mixture M=1 == Gaussian naive Bayes with class variances ----
Xg = np.r_[rng.randn(300, 5) + 0.5, rng.randn(1200, 5) * 1.3]; yg = np.r_[np.ones(300), np.zeros(1200)]
Xqg = rng.randn(30, 5)
P = gmc.predict_path(Xg, yg, [{"M": 1}], Xqg, common.RunConfig(), {})[0]
def lg(X, Q):
    mu, v = X.mean(0), X.var(0); return -0.5 * (np.log(2 * np.pi * v) + (Q - mu) ** 2 / v).sum(1)
lo = lg(Xg[yg == 1], Xqg) - lg(Xg[yg == 0], Xqg) + np.log(300 / 1200)
check("mixture classifier (M=1, diag) == Gaussian naive Bayes", np.abs(np.log(P / (1 - P)) - lo).max() < 1e-2, f"{np.abs(np.log(P / (1 - P)) - lo).max():.1e}")
# ---- varying-coefficient local regression ----
nz = 4000; z = np.sort(rng.rand(nz)); x1 = rng.randn(nz); yv = np.sin(3 * z) + (1 + 2 * z) * x1 + 0.2 * rng.randn(nz)
gp = kern.make_grid_points(z, 100); rv = kern.local_vc_1d(z, x1[:, None], yv, gp, 0.25, 1, 1)
fit_beta = rv["coef"][:, 2]
check("varying-coefficient fit recovers b(z) = 1 + 2z", np.sqrt(np.mean((fit_beta[10:-10] - (1 + 2 * gp[10:-10])) ** 2)) < 0.1, f"rmse={np.sqrt(np.mean((fit_beta[10:-10] - (1 + 2 * gp[10:-10])) ** 2)):.3f}")
r0 = kern.local_vc_1d(z, np.zeros((nz, 0)), yv, gp, 0.25, 0, 0); lp = kern.local_poly_1d(z, yv, gp, 0.25, 1)
check("varying-coefficient with M=0 == local linear regression", np.abs(r0["coef"][:, 0] - lp["fit"]).max() < 1e-8)
# ---- RBF analytic gradient ----
Xb = rng.randn(60, 3); yb = rng.randn(60); M, p = 4, 3
th = np.concatenate([[0.1], rng.randn(M) * 0.3, rng.randn(M * p), np.log(np.full(M, 1.5))])
num = approx_fprime(th, lambda t: rbf._loss_grad(t, Xb, yb, M, p)[0], 1e-7); ana = rbf._loss_grad(th, Xb, yb, M, p)[1]
check("RBF analytic gradient == numerical gradient", np.abs(num - ana).max() / max(np.abs(ana).max(), 1) < 1e-5, f"{np.abs(num - ana).max():.1e}")
Pn = rbf._phi(Xb, kern.kmeans(Xb, 5), 1.2, True)
check("renormalised RBF rows sum to 1 (no holes)", np.allclose(Pn.sum(1), 1.0))
# ---- tests / rules ----
e = rng.randn(3000) ** 2
check("Diebold-Mariano: clearly different losses -> tiny p", common.dm_test(e, e + 0.5)[2] < 1e-6)
fm = np.array([[1.0, 0.5, 0.4], [3.0, 2.5, 2.41], [2.0, 1.5, 1.38], [5.0, 4.5, 4.39]])
check("paired one-SE vs plain one-SE", common.one_se_paired(fm, np.array([1., 2., 3.]))[0] >= 1 and common.one_se(fm.mean(0), fm.std(0, ddof=1) / 2, np.array([1., 2., 3.]))[0] == 0)
check("Holm", np.allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
