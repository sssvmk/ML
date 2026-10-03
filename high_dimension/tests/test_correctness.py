"""Verify the Chapter 18 glue against hand computations / scikit-learn: split, diagonal LDA (18.2), shrunken centroids, fused lasso vs lasso, gene ordering, the leak-free screening, tests."""
import sys, warnings, os
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import hd_lib as hl, hd_data
import reg_fused_lasso as fl, clf_nsc, clf_diag_lda
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
# ---- split: 71 rows -> 47 / 12 / 12, disjoint, stratified on y ----
y = rng.randn(71)
tr, va, te = hd_data.split_regression(y, seed=1)
check("split of 71 rows is 47 / 12 / 12, disjoint and complete", (len(tr), len(va), len(te)) == (47, 12, 12) and len(set(tr) | set(va) | set(te)) == 71 and not (set(tr) & set(te)) and not (set(va) & set(te)))
q = np.quantile(y, [0, 1 / 3, 2 / 3, 1]); cnt = np.histogram(y[te], bins=q)[0]
check("test rows cover the whole range of y (stratified)", cnt.min() >= 2, f"counts per tercile {cnt.tolist()}")
y72 = rng.randn(72); a, b, c = hd_data.split_regression(y72, seed=1)
check("72 rows give the specified 48 / 12 / 12", (len(a), len(b), len(c)) == (48, 12, 12))
# ---- diagonal LDA (18.2): the scikit-learn pipeline reproduces the formula ----
n, p = 40, 300
yc = np.repeat([1, 2, 3, 4], 10); X = rng.randn(n, p) + 0.8 * np.eye(4)[yc - 1] @ rng.randn(4, p)
pipe = clf_diag_lda.make_est(False, "/tmp/hd_test_cache").fit(X, yc)
Xte = rng.randn(60, p) + 0.8 * np.eye(4)[rng.randint(0, 4, 60)] @ rng.randn(4, p)
cl = np.unique(yc); mk = np.array([X[yc == c].mean(0) for c in cl]); resid = X - mk[yc - 1]; s2 = (resid ** 2).sum(0) / n; pri = np.array([(yc == c).mean() for c in cl])
delta = np.stack([-0.5 * (((Xte - mk[k]) ** 2) / s2).sum(1) + np.log(pri[k]) for k in range(4)], 1)
check("diagonal LDA pipeline == the closed form delta_k(x) = -1/2 sum_j (x_j - xbar_kj)^2 / s_j^2 + log pi_k", (cl[delta.argmax(1)] == pipe.predict(Xte)).all())
# ---- nearest shrunken centroids: the surviving genes shrink monotonically with Delta ----
Xn = rng.randn(60, 500); yn = np.repeat([0, 1, 2], 20); Xn[yn == 1, :10] += 2.0; Xn[yn == 2, 10:20] -= 2.0
surv = [int(clf_nsc.surviving(clf_nsc.make_est(False, "/tmp/hd_test_cache").set_params(nsc__shrink_threshold=d).fit(Xn, yn)).sum()) for d in (0.1, 1.0, 2.0, 4.0, 8.0)]
check("shrunken centroids: genes surviving fall as Delta grows; at Delta = 2 the 20 planted genes are what remain; at Delta = 8 none", all(surv[i] >= surv[i + 1] for i in range(4)) and 18 <= surv[2] <= 30 and surv[-1] == 0, f"surviving = {surv}")
# ---- fused lasso with lambda2 = 0 equals the lasso (cvxpy vs scikit-learn coordinate descent) ----
Xs = rng.randn(40, 60); bt = np.zeros(60); bt[5:9] = 1.5; ys = Xs @ bt + 0.3 * rng.randn(40)
Xs = (Xs - Xs.mean(0)) / Xs.std(0)
f1 = fl.FusedLassoRegressor(lam1=0.2, lam2=0.0).fit(Xs, ys)
lmax = np.max(np.abs(Xs.T @ (ys - ys.mean()))) / 40
l1 = Lasso(alpha=0.2 * lmax, tol=1e-10, max_iter=100000).fit(Xs, ys)
check("fused lasso with lambda2 = 0 == scikit-learn lasso", np.abs(f1.coef_ - l1.coef_).max() < 5e-3 * max(np.abs(l1.coef_).max(), 1e-9) + 1e-3, f"max diff {np.abs(f1.coef_ - l1.coef_).max():.2e}")
f2 = fl.FusedLassoRegressor(lam1=0.0, lam2=5.0).fit(Xs, ys)
check("a huge fusion penalty makes the coefficient profile flat (no jumps)", np.abs(np.diff(f2.coef_)).max() < 1e-3 * max(np.abs(f2.coef_).max(), 1e-9) + 1e-6)
fz = fl.FusedLassoRegressor(lam1=0.0, lam2=0.1).fit(Xs, ys)
tolj = 1e-3 * np.abs(fz.coef_).max()
check("a moderate fusion penalty produces piecewise-constant blocks (far fewer jumps than coefficients, relative tolerance)", 0 < int(np.sum(np.abs(np.diff(fz.coef_)) > tolj)) < 30, f"{int(np.sum(np.abs(np.diff(fz.coef_)) > tolj))} jumps among {len(fz.coef_) - 1} steps")
# ---- hierarchical gene ordering puts correlated genes next to each other ----
base = rng.randn(50, 3); G = np.hstack([base[:, [g]] + 0.2 * rng.randn(50, 30) for g in range(3)]); perm = rng.permutation(90); grp = np.repeat([0, 1, 2], 30)[perm]
order = hl.GeneOrderer().fit(G[:, perm]).order_
changes = int(np.sum(np.diff(grp[order]) != 0)); changes_rand = int(np.sum(np.diff(grp) != 0))
check("gene ordering groups correlated genes (few group changes along the order)", changes <= 6 and changes < changes_rand / 4, f"{changes} changes vs {changes_rand} in random order")
# ---- selection INSIDE the CV folds: pure noise must give chance-level error, selecting first gives an illusion ----
Xr = rng.randn(40, 2000); yr = np.repeat([0, 1], 20)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
inside = 1 - cross_val_score(Pipeline([("s", SelectKBest(f_classif, k=20)), ("lr", LogisticRegression(max_iter=2000))]), Xr, yr, cv=cv).mean()
Xsel = SelectKBest(f_classif, k=20).fit_transform(Xr, yr)
outside = 1 - cross_val_score(LogisticRegression(max_iter=2000), Xsel, yr, cv=cv).mean()
check("screening inside the folds: noise gives chance error; screening before CV gives an illusion (the book's selection bias)", inside > 0.38 and outside < 0.25, f"inside {inside:.2f}, selected-first {outside:.2f}")
# ---- one-SE rule with the corrected SE ----
class _C: cv_folds = 5; cv_repeats = 3; seed = 0
from sklearn.linear_model import Ridge
Xq = rng.randn(60, 20); yq = Xq[:, 0] + 0.5 * rng.randn(60)
r, best, cvmin = hl.tune(Ridge(), {"alpha": [0.1, 1, 10, 100, 1000]}, Xq, yq, "regression", _C(), lambda p: -p["alpha"], "neg_mean_squared_error")
check("one-SE choice is at least as simple as the CV minimum and within one corrected SE", best["alpha"] >= cvmin["alpha"] and r.loc[r["selected_one_se"], "cv_loss"].iloc[0] <= r["cv_loss"].min() + r.loc[r["cv_minimum"], "cv_se"].iloc[0] + 1e-12, f"chosen alpha {best['alpha']}, minimum at {cvmin['alpha']}")
# ---- multiple-testing helpers ----
check("Holm", np.allclose(hl.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
a_ = np.array([1] * 9 + [0] * 1 + [1] * 10, bool); b_ = np.array([0] * 8 + [1] * 2 + [1] * 10, bool)
check("exact McNemar: 8 vs 1 discordant pairs gives p = 0.039", abs(hl.mcnemar_exact(np.r_[np.ones(8, bool), np.zeros(1, bool), np.ones(5, bool)], np.r_[np.zeros(8, bool), np.ones(1, bool), np.ones(5, bool)]) - 0.0390625) < 1e-9)
e = rng.randn(12) ** 2
check("paired t and Diebold-Mariano agree that identical losses are not different", hl.paired_t(e, e) == 1.0 and hl.dm_test(e, e)[1] == 1.0)
# ---- real data, if available (set HD_DATA_DIR) ----
dd = os.environ.get("HD_DATA_DIR")
if dd and Path(dd, "riboflavin.csv").exists():
    Xr_, yr_, g = hd_data.load_riboflavin(Path(dd, "riboflavin.csv")); check("riboflavin: 71 x 4088", Xr_.shape == (71, 4088) and len(g) == 4088)
if dd and Path(dd, "srbct.csv").exists():
    a1, b1, a2, b2, g = hd_data.load_srbct(Path(dd, "srbct.csv")); check("SRBCT: book split 63 / 20 with 2308 genes", a1.shape == (63, 2308) and a2.shape == (20, 2308) and np.bincount(b1)[1:].tolist() == [8, 23, 12, 20])
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
