"""Verify the Chapter 9 machinery against direct solutions / planted structure."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common, selection, gam, prim, mars, hme, cart

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
cfg = common.RunConfig()
# ---- GAM: backfitting converges to the joint penalised least-squares solution ----
n = 1500; X = rng.rand(n, 2); y = np.sin(4 * X[:, 0]) + (X[:, 1] - 0.5) ** 2 * 4 + 0.2 * rng.randn(n)
terms = [gam._Term(X[:, j], np.float64) for j in range(2)]
lam = [gam._lams(terms, [5.0])[j][5.0] for j in range(2)]
m = gam._fit_gam(terms, lam, y, "regression", cfg)
Bc = [t.B - t.B.mean(0) for t in terms]
Dm = np.hstack(Bc); Pen = np.zeros((Dm.shape[1], Dm.shape[1])); o = 0
for t, l in zip(terms, lam):
    k_ = t.B.shape[1]; Pen[o:o + k_, o:o + k_] = l * t.Om; o += k_
theta = np.linalg.lstsq(np.vstack([Dm.T @ Dm + Pen]), Dm.T @ (y - y.mean()), rcond=None)[0]
direct = Dm @ theta
check("backfitting == joint penalised least squares (Gauss-Seidel solution)", np.abs((y.mean() + m.f.sum(0)) - (y.mean() + direct)).max() < 2e-3, f"{np.abs(m.f.sum(0) - direct).max():.1e}; cycles={len(m.hist)}")
check("backfitting change decreases monotonically to the tolerance", m.hist[-1] < 1e-5 and all(m.hist[i + 1] <= m.hist[i] * 1.0001 for i in range(len(m.hist) - 1)))
check("GAM total df ~ 1 + sum(df_j - 1) with df_j = 5", abs(m.df_total - (1 + 2 * 4)) < 0.6, f"{m.df_total:.2f}")
yb = (rng.rand(n) < common.sigmoid(2 * np.sin(4 * X[:, 0]) - 0.5)).astype(float)
mb = gam._fit_gam(terms, lam, yb, "classification", cfg)
check("local scoring (additive logistic) fits better than the constant", mb.rss < -2 * np.sum(yb * np.log(yb.mean()) + (1 - yb) * np.log(1 - yb.mean())))
# ---- PRIM: first peel == brute force; recovers a planted box ----
Xp = rng.rand(3000, 4); yp = ((Xp[:, 0] > 0.2) & (Xp[:, 0] < 0.5) & (Xp[:, 1] > 0.6)).astype(float) * 0.9 + 0.1 * rng.rand(3000)
traj = prim._peel(Xp, yp, 0.1, 0.05)
lo, hi = traj[1]
best = (-np.inf, None)
for j in range(4):
    ql, qh = np.quantile(Xp[:, j], [0.1, 0.9])
    for side, keep, q in (("lo", Xp[:, j] > ql, ql), ("hi", Xp[:, j] < qh, qh)):
        if yp[keep].mean() > best[0]: best = (yp[keep].mean(), (j, side, q))
j, side, q = best[1]
check("PRIM first peel == brute-force best peel", (np.isfinite(lo[j]) if side == "lo" else np.isfinite(hi[j])) and abs((lo[j] if side == "lo" else hi[j]) - q) < 1e-12)
boxes, rest, base = prim._fit(Xp, yp, 0.07, 0.03, 1, 0)
b = boxes[0]
check("PRIM recovers the planted box (mean ~0.95, bounds near 0.2/0.5/0.6)", b["mean"] > 0.85 and abs(b["lo"][1] - 0.6) < 0.1 and abs(b["lo"][0] - 0.2) < 0.1 and abs(b["hi"][0] - 0.5) < 0.1, f"mean={b['mean']:.3f} lo={np.round(b['lo'], 2)} hi={np.round(b['hi'], 2)}")
# ---- MARS ----
Xm = rng.rand(4000, 6); ym = 3 * np.maximum(Xm[:, 2] - 0.45, 0) + 0.1 * rng.randn(4000)
mm = mars._fit(Xm, ym, 1, 0)
first = mm["terms"][0]
check("MARS forward pass picks the planted variable and knot (knot on the quantile grid)", first[1] == 2 and abs(first[2] - 0.45) < 0.06, f"var={first[1]} knot={first[2]:.3f}")
best = min(mm["gcv"], key=mm["gcv"].get)
check("MARS GCV prunes to a small model", best <= 5, f"terms={best}")
S = mm["seq"][best][0]; Bm = mars._basis(Xm, mm["terms"])[:, S]
check("MARS Gram-based RSS == direct least squares RSS", abs(mm["seq"][best][1] - np.sum((ym - Bm @ np.linalg.lstsq(Bm, ym, rcond=None)[0]) ** 2)) < 1e-6 * np.sum(ym ** 2))
Xi = rng.rand(5000, 4); yi = 6 * np.maximum(Xi[:, 0] - 0.5, 0) * np.maximum(Xi[:, 1] - 0.5, 0) + 0.05 * rng.randn(5000)
mi2, mi1 = mars._fit(Xi, yi, 2, 0), mars._fit(Xi, yi, 1, 0)
r2 = min(v[1] for k_, v in mi2["seq"].items() if k_ <= 9); r1 = min(v[1] for k_, v in mi1["seq"].items() if k_ <= 9)
check("MARS with interactions (degree 2) beats additive MARS on a product effect", r2 < 0.5 * r1, f"RSS {r2:.1f} vs {r1:.1f}")
# ---- HME ----
xh = rng.randn(4000, 2); reg1 = xh[:, 0] > 0
yh = np.where(reg1, 2 * xh[:, 1] + 1, -2 * xh[:, 1] - 1) + 0.2 * rng.randn(4000)
Xa = hme._prep_X(xh, xh.mean(0), xh.std(0))
fit = hme._em(Xa, yh, "regression", 1, 2, 25, 0)
check("HME EM log-likelihood is (nearly) non-decreasing", all(fit["hist"][i + 1] >= fit["hist"][i] - 1e-3 for i in range(len(fit["hist"]) - 1)), f"{fit['hist'][0]:.3f} -> {fit['hist'][-1]:.3f}")
pred, _, _ = hme._predict(fit, Xa, "regression")
lin = Xa @ np.linalg.lstsq(Xa, yh, rcond=None)[0]
check("HME (2 soft regimes) beats a single linear model on a two-regime function", np.mean((yh - pred) ** 2) < 0.2 * np.mean((yh - lin) ** 2), f"MSE {np.mean((yh - pred) ** 2):.3f} vs {np.mean((yh - lin) ** 2):.3f}")
yc = (rng.rand(4000) < common.sigmoid(np.where(reg1, 2 * xh[:, 1], -2 * xh[:, 1]))).astype(float)
fc = hme._em(Xa, yc, "classification", 1, 2, 20, 0)
check("HME classification log-likelihood is (nearly) non-decreasing", all(fc["hist"][i + 1] >= fc["hist"][i] - 1e-3 for i in range(len(fc["hist"]) - 1)))
# ---- CART: more pruning -> not more leaves ----
Xc = rng.randn(3000, 5); yc2 = np.where(Xc[:, 0] > 0, 1.0, -1.0) + 0.5 * rng.randn(3000)
lv = [cart._tree("regression", a * np.var(yc2), 0).fit(Xc, yc2).get_n_leaves() for a in (0, 1e-4, 1e-3, 1e-2, 1e-1)]
check("CART cost-complexity pruning: leaves non-increasing in alpha", all(lv[i + 1] <= lv[i] for i in range(len(lv) - 1)), f"{lv}")
# ---- ranking finds the planted variables ----
Xr = rng.randn(5000, 30); yr = 2 * Xr[:, 7] + np.sin(2 * Xr[:, 3]) + rng.randn(5000)
order, uni, imp = selection.rank_features(Xr, yr, "regression", 0, 40, 5000)
check("variable ranking (regression) puts the planted variables first", set(order[:2]) == {3, 7}, f"top={order[:4]}")
yk = (rng.rand(5000) < common.sigmoid(1.5 * Xr[:, 11] - 1.5 * Xr[:, 20] - 1)).astype(float)
order, _, _ = selection.rank_features(Xr, yk, "classification", 0, 40, 5000)
check("variable ranking (classification) puts the planted variables first", set(order[:2]) == {11, 20}, f"top={order[:4]}")
Xd = rng.randn(4000, 8); Xd[:, 5] = 2017 - 3 * Xd[:, 1] + 1e-3 * rng.randn(4000)           # column 5 is a (reversed) near-duplicate of column 1
yd = 2 * Xd[:, 1] + rng.randn(4000)
od, _, _ = selection.rank_features(Xd, yd, "regression", 0, 30, 4000)
pos = {int(c): i for i, c in enumerate(od)}
check("redundancy filter: a near-duplicate of a selected variable is ranked after the independent variables", max(pos[1], pos[5]) - min(pos[1], pos[5]) >= 5 and min(pos[1], pos[5]) == 0, f"positions of the pair: {pos[1]}, {pos[5]}")
# ---- tests ----
e = rng.randn(3000) ** 2
check("Diebold-Mariano: clearly different losses -> tiny p", common.dm_test(e, e + 0.5)[2] < 1e-6)
check("Holm", np.allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
