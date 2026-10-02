"""Verify FDA / PDA / MDA / SVM glue against independent references (scikit-learn LDA / SVC, planted structure)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common, svmlib as sl, fda, pda, mda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
n, k = 4000, 6
y = (rng.rand(n) < 0.3).astype(int)
X = rng.randn(n, k) + y[:, None] * rng.uniform(0.2, 0.8, k)
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None).fit(X, y)
sl_ = lda.decision_function(X)
# ---- PDA with (almost) no penalty == LDA (optimal scoring on the identity basis) ----
p0 = pda.PDAModel(1e-8).fit(X, y).predict_proba(X)[:, 1]
lg = np.log(p0 / (1 - p0))
check("PDA(lambda -> 0) == LDA: discriminant scores perfectly correlated", np.corrcoef(lg, sl_)[0, 1] > 0.99999, f"corr={np.corrcoef(lg, sl_)[0, 1]:.7f}")
check("PDA(lambda -> 0) == LDA: AUC identical", abs(roc_auc_score(y, lg) - roc_auc_score(y, sl_)) < 1e-9)
# ---- PDA with a huge penalty -> direction of the class-mean difference ----
pl = pda.PDAModel(1e9).fit(X, y).predict_proba(X)[:, 1]
d = X @ (X[y == 1].mean(0) - X[y == 0].mean(0))
check("PDA(lambda -> inf) -> class-mean-difference direction", abs(np.corrcoef(np.log(pl / (1 - pl)), d)[0, 1]) > 0.999, f"{abs(np.corrcoef(np.log(pl / (1 - pl)), d)[0, 1]):.5f}")
check("PDA effective df shrinks with lambda", pda.PDAModel(1e5).fit(X, y).df < pda.PDAModel(1.0).fit(X, y).df)
# ---- MDA with one subclass per class == LDA ----
m1 = mda.MDAModel(1, 1e-12, 0).fit(X, y)
lm = m1.class_logdens(X) + np.log(m1.prior)[None]
lo = lm[:, 1] - lm[:, 0]
slope = np.polyfit(sl_, lo, 1)[0]
check("MDA (R = 1) == LDA with a common covariance", np.corrcoef(lo, sl_)[0, 1] > 0.99999 and abs(slope - 1) < 0.01, f"corr={np.corrcoef(lo, sl_)[0, 1]:.7f} slope={slope:.4f}")
# ---- MDA: EM likelihood monotone; subclasses solve a problem LDA cannot ----
n2 = 4000; y2 = (rng.rand(n2) < 0.4).astype(int)
X2 = rng.randn(n2, 3); sgn = rng.choice([-3, 3], n2); X2[y2 == 1, 0] += sgn[y2 == 1]
mR2 = mda.MDAModel(2, 1e-6, 0).fit(X2, y2); mR1 = mda.MDAModel(1, 1e-6, 0).fit(X2, y2)
h = mR2.ll_hist
check("MDA EM log-likelihood is non-decreasing", all(h[i + 1] >= h[i] - 1e-6 for i in range(len(h) - 1)), f"{h[0]:.3f} -> {h[-1]:.3f} in {len(h)} iterations")
a2, a1 = roc_auc_score(y2, mR2.predict_proba(X2)[:, 1]), roc_auc_score(y2, mR1.predict_proba(X2)[:, 1])
check("MDA with 2 subclasses/class beats the single-Gaussian (LDA) model on a bimodal class", a2 > 0.95 and a1 < 0.7, f"AUC {a2:.3f} vs {a1:.3f}")
check("MDA BIC prefers R = 2 on bimodal data", mR2.bic < mR1.bic, f"{mR2.bic:.0f} < {mR1.bic:.0f}")
# ---- FDA: nonlinear boundary that LDA cannot represent ----
X3 = rng.randn(5000, 2); y3 = ((X3 ** 2).sum(1) > 2).astype(int)
aL = roc_auc_score(y3, LinearDiscriminantAnalysis().fit(X3, y3).decision_function(X3))
fm = fda.FDAModel("spline5", 1.0, 0.3, 0).fit(X3, y3)
aF = roc_auc_score(y3, fm.predict_proba(X3)[:, 1])
fn = fda.FDAModel("nystroem100", 1.0, 0.5, 0).fit(X3, y3)
aN = roc_auc_score(y3, fn.predict_proba(X3)[:, 1])
check("FDA (spline basis) captures a circular boundary; LDA cannot", aF > 0.85 and aL < 0.6, f"AUC spline {aF:.3f}, nystroem {aN:.3f}, LDA {aL:.3f}")
# ---- SVM pieces ----
Xs = rng.randn(1500, 4); ys = (Xs[:, 0] + 0.8 * rng.randn(1500) > 0).astype(int)
lin = LinearSVC(loss="hinge", C=0.05, dual=True, max_iter=20000, tol=1e-6).fit(Xs, ys)
svc = SVC(kernel="linear", C=0.05, tol=1e-6).fit(Xs, ys)
fr_lin = float(np.mean((2 * ys - 1) * lin.decision_function(Xs) <= 1 + 1e-4)); fr_svc = float(svc.n_support_.sum() / len(ys))
check("linear SVC support-point fraction (y f <= 1) == SVC n_support / N", abs(fr_lin - fr_svc) < 0.03, f"{fr_lin:.3f} vs {fr_svc:.3f}")
check("eps-insensitive loss: zero inside the tube, linear outside", sl.eps_loss(np.array([0.0, 0.0, 0.0]), np.array([0.05, -0.1, 0.5]), 0.1) == (0.4 / 3))
cfgs = sl.sample_configs({"frac": sl.FRACS, "C": [1, 2, 3]}, 2, 5)
check("random search always evaluates the keep-all-variables configuration", any(c["frac"] == 1.0 for c in cfgs))
# ---- plumbing: predict_path returns one score row per configuration ----
class _P: pass
grid = sl.with_k([{"frac": 1.0, "C": 0.1, "cw": None}, {"frac": 0.5, "C": 1.0, "cw": "balanced"}], 6)
import svc as svcmod
pp = sl.make_predict_path(svcmod.build, sl.margin_score, diag=svcmod.diag)
ctx = {"task": "classification", "order": np.arange(6), "final": False, "metrics": {}, "yq": ys[:200]}
out = pp(Xs[:, :4].dot(np.ones((4, 6)))[:1000], ys[:1000], grid, Xs[:200, :4].dot(np.ones((4, 6))), common.RunConfig(), ctx)
check("predict_path: one score vector per configuration + diagnostics recorded", out.shape == (2, 200) and "support_fraction" in ctx["metrics"] and "margin" in ctx["metrics"])
e = rng.randn(3000) ** 2
check("Diebold-Mariano: clearly different losses -> tiny p", common.dm_test(e, e + 0.5)[2] < 1e-6)
check("Holm", np.allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
