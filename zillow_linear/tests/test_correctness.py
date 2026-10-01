"""Verify every method's fit_path against independent references (sklearn / brute force / KKT) on a strong-signal problem."""
import itertools, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common, solvers
import ols, ridge, lasso, elastic_net, lar, pcr, pls, best_subset, backward_stepwise, forward_stagewise, dantzig, grouped_lasso
from sklearn.linear_model import Ridge, Lasso, ElasticNet, lars_path
from sklearn.cross_decomposition import PLSRegression

rng = np.random.RandomState(1)
n, p = 3000, 12
Z = rng.randn(n, p); Z[:, 3] = 0.8 * Z[:, 2] + 0.6 * Z[:, 3]            # some correlation
X = (Z - Z.mean(0)) / Z.std(0)
beta_true = np.array([1.0, -0.8, 0.5, 0, 0.3, 0, 0, 0.2, 0, 0, 0, 0.1])
y = 2.0 + X @ beta_true + rng.randn(n)
d = common.build_stats(X, {"y": y})["y"]
class Bun: pass
Bun.tr_fit = d
bun = Bun()
cfg = common.RunConfig(subset_pool=8, dantzig_pool=12, fs_steps=3000)
ok = True
def check(name, cond, info=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

# OLS
P = ols.fit_path(d, None, cfg); b_ref = np.linalg.lstsq(np.c_[np.ones(n), X], y, rcond=None)[0]
check("ols", np.allclose(P.B[:, 0], b_ref[1:], atol=1e-8) and abs(P.b0[0] - b_ref[0]) < 1e-8)

# ridge (RSS + lam*|b|^2)
lams = n * np.array([1.0, 0.1, 1e-3]); P = ridge.fit_path(d, lams, cfg)
err = max(np.abs(P.B[:, i] - Ridge(alpha=l).fit(X, y).coef_).max() for i, l in enumerate(lams))
check("ridge vs sklearn Ridge", err < 1e-8, f"maxerr={err:.1e}")

# lasso (RSS + lam*|b|_1  == sklearn alpha = lam/(2n))
lm = lasso.make_grid(bun, None, cfg); P = lasso.fit_path(d, np.geomspace(lm[0], lm[-1], 30), cfg)
err = max(np.abs(P.B[:, i] - Lasso(alpha=l / (2 * n), tol=1e-12, max_iter=100000).fit(X, y).coef_).max() for i, l in list(enumerate(np.geomspace(lm[0], lm[-1], 30)))[::7])
check("lasso vs sklearn Lasso", err < 1e-5, f"maxerr={err:.1e}")

# elastic net (ESLII parametrisation: alpha weights L2)
for a in (0.25, 0.75):
    lg = solvers.lam_max_abs(d, a) * np.geomspace(1, 1e-3, 20)
    B, b0 = solvers.enet_path_abs(d, lg, a)
    errs = []
    for i in (3, 10, 19):
        l = lg[i]; sk = ElasticNet(alpha=l * (1 + a) / (2 * n), l1_ratio=(1 - a) / (1 + a), tol=1e-12, max_iter=100000).fit(X, y)
        errs.append(np.abs(B[:, i] - sk.coef_).max())
    check(f"elastic net alpha={a} vs sklearn", max(errs) < 1e-5, f"maxerr={max(errs):.1e}")

# LAR knots vs sklearn lars_path
Gc, cc, yyc, xb, yb = d.centered()
knots = lar.lar_knots(Gc, cc)
_, _, coefs = lars_path(X, y - y.mean(), method="lar")
m = min(knots.shape[1], coefs.shape[1])
err = np.abs(knots[:, :m] - coefs[:, :m]).max()
check("LAR knots vs sklearn lars_path", err < 1e-6 and knots.shape[1] == coefs.shape[1], f"maxerr={err:.1e} knots={knots.shape[1]}/{coefs.shape[1]}")

# PCR
P = pcr.fit_path(d, None, cfg); U, s, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
errs = []
for M in (1, 3, 7):
    T = (X - X.mean(0)) @ Vt[:M].T; g = np.linalg.lstsq(T, y - y.mean(), rcond=None)[0]; errs.append(np.abs(P.B[:, M] - Vt[:M].T @ g).max())
check("PCR vs explicit PCA regression", max(errs) < 1e-8, f"maxerr={max(errs):.1e}")

# PLS (PLS1 == SIMPLS)
P = pls.fit_path(d, 8, cfg); errs = []
for M in (1, 2, 5):
    sk = PLSRegression(n_components=M, scale=False).fit(X, y); errs.append(np.abs(P.B[:, M] - sk.coef_.ravel()).max())
check("PLS vs sklearn PLSRegression", max(errs) < 1e-6, f"maxerr={max(errs):.1e}")

# best subset vs brute force (pool = all 8 of first 8 columns is not guaranteed; compare on the pool it picks)
P = best_subset.fit_path(d, {"pool": 8}, cfg); pool = common.forward_pool(Gc, cc, 8)
errs = []
for k in (1, 2, 3, 5):
    best = min(itertools.combinations(pool, k), key=lambda S: np.linalg.lstsq(np.c_[np.ones(n), X[:, S]], y, rcond=None)[1][0])
    bb = np.zeros(p); bb[list(best)] = np.linalg.lstsq(np.c_[np.ones(n), X[:, list(best)]], y, rcond=None)[0][1:]
    errs.append(np.abs(P.B[:, k] - bb).max())
check("best subset (Gray-code sweep) vs brute force", max(errs) < 1e-6, f"maxerr={max(errs):.1e}")
# swap heuristic should match exact on this easy problem
Gs, cs = Gc[np.ix_(pool, pool)], cc[pool]
me, ms = best_subset._exact_masks(Gs, cs, yyc, n), best_subset._swap_masks(Gs, cs, yyc, n)
check("best subset swap-heuristic == exact (easy case)", np.array_equal(me, ms))

# backward stepwise: manual greedy elimination by smallest |t|
P = backward_stepwise.fit_path(d, None, cfg); act = list(range(p)); okb = True
for size in range(p, 1, -1):
    A = np.c_[np.ones(n), X[:, act]]; coef = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ coef; s2 = res @ res / (n - A.shape[1]); cov = s2 * np.linalg.inv(A.T @ A)
    z = coef[1:] / np.sqrt(np.diag(cov)[1:]); drop = act[int(np.argmin(np.abs(z)))]
    act.remove(drop)
    bb = np.zeros(p); bb[act] = np.linalg.lstsq(np.c_[np.ones(n), X[:, act]], y, rcond=None)[0][1:]
    okb &= np.abs(P.B[:, size - 1] - bb).max() < 1e-4
check("backward stepwise vs manual Z-score elimination", okb)

# forward stagewise approaches OLS path direction: at large T coefficients close to OLS
g = forward_stagewise.make_grid(bun, None, cfg); P = forward_stagewise.fit_path(d, {**g, "T": 40000, "steps": np.array([0, 40000])}, cfg)
check("forward stagewise -> OLS for many steps", np.abs(P.B[:, -1] - b_ref[1:]).max() < 0.03, f"maxdiff={np.abs(P.B[:, -1] - b_ref[1:]).max():.3f}")

# Dantzig: feasibility + sparsity monotone
g = dantzig.make_grid(bun, None, cfg); P = dantzig.fit_path(d, g, cfg)
cn = cc / n; feas = True
for l, t in enumerate(g["t"]):
    r = np.abs(cn - Gc / n @ P.B[:, l]).max(); feas &= r <= t * np.abs(cn).max() * (1 + 1e-5) + 1e-9
check("dantzig constraint ||X'(y-Xb)||inf <= s satisfied along path", feas, f"l1 first/last {np.abs(P.B[:,0]).sum():.3f}/{np.abs(P.B[:,-1]).sum():.3f}")
check("dantzig t->0 approaches OLS", np.abs(P.B[:, -1] - b_ref[1:]).max() < 0.02)

# grouped lasso: KKT conditions
class _P: group_ids = np.array([0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8])
class _B: tr_fit = d
g = grouped_lasso.make_grid(_B, _P, cfg); g["max_iter"] = 20000; g["lams"] = g["lams"][[5, 20, 40, 59]]
P = grouped_lasso.fit_path(d, g, cfg); okg = True; worst = 0
for l, lam in enumerate(g["lams"]):
    b = P.B[:, l]; grad = (cc - Gc @ b) / n; lg = lam / (2 * n)
    for gi in range(g["ng"]):
        idx = np.where(g["inv"] == gi)[0]; gn, bn = np.linalg.norm(grad[idx]), np.linalg.norm(b[idx])
        if bn > 1e-9: viol = np.linalg.norm(grad[idx] - lg * g["sqrtp"][gi] * b[idx] / bn)
        else: viol = max(0.0, gn - lg * g["sqrtp"][gi])
        worst = max(worst, viol / max(lg, 1e-12))
check("grouped lasso KKT conditions", worst < 1e-3, f"worst relative violation={worst:.1e}")
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
