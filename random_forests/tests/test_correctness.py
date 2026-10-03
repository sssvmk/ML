"""Verify the forest glue (OOB / validation curves, proximity, defaults, selection rule) against scikit-learn and the book's statements."""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common, rflib as rl, rf_entry
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.inspection import permutation_importance

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
X = rng.randn(3000, 8)
y = 2 * X[:, 0] + X[:, 3] ** 2 + 0.5 * rng.randn(3000)
yc = ((X[:, 0] + X[:, 3] ** 2 + 0.7 * rng.randn(3000)) > 1).astype(int)
class _C: seed = 0; n_jobs = 1; rf_trees_cv = 150; rf_trees_final = 150
# ---- OOB error ~ cross-validation (book 15.3.1, Ex. 15.2) ----
rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, oob_score=True, random_state=0, n_jobs=1).fit(X, y)
oob = float(np.mean((y - rf.oob_prediction_) ** 2))
cv = float(np.mean((y - cross_val_predict(RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=0, n_jobs=1), X, y, cv=KFold(10, shuffle=True, random_state=0))) ** 2))
check("OOB MSE is almost identical to 10-fold CV MSE", abs(oob - cv) / cv < 0.08, f"OOB {oob:.4f} vs CV {cv:.4f}")
# ---- curves vs number of trees: running average of the trees == the forest ----
cur = rl.tree_curves(rf, X[:500], y[:500], "regression", X, y, n_points=10)
check("validation curve: last point == forest prediction", abs(cur["eval_mse"].iloc[-1] - float(np.mean((y[:500] - rf.predict(X[:500])) ** 2))) < 1e-9)
check("OOB curve: last point == scikit-learn oob_prediction_ MSE", abs(cur["oob_mse"].iloc[-1] - oob) < 1e-9, f"{cur['oob_mse'].iloc[-1]:.5f} vs {oob:.5f}")
check("OOB curve decreases with more trees (variance reduction, 15.1)", cur["oob_mse"].iloc[-1] < cur["oob_mse"].iloc[0])
rc = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, oob_score=True, random_state=0, n_jobs=1).fit(X, yc)
cc = rl.tree_curves(rc, X[:500], yc[:500], "classification", X, yc, n_points=8)
ok_ = np.isfinite(rc.oob_decision_function_[:, 1])
check("classification OOB AUC at the last tree == AUC of oob_decision_function_", abs(cc["oob_auc"].iloc[-1] - common.auc_score(yc[ok_], rc.oob_decision_function_[ok_, 1])) < 1e-9)
# ---- proximity from apply() ----
rp = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=1).fit(X, yc)
P = rl.proximity_matrix(rp, X[:300], 60)
check("proximity matrix: symmetric, unit diagonal, values in [0, 1]", np.allclose(P, P.T) and np.allclose(np.diag(P), 1) and P.min() >= 0 and P.max() <= 1)
same = (yc[:300][:, None] == yc[:300][None, :]); off = ~np.eye(300, dtype=bool)
check("points of the same class are closer in the forest than points of different classes", P[same & off].mean() > P[~same].mean(), f"{P[same & off].mean():.3f} > {P[~same].mean():.3f}")
# ---- the book's defaults (15.3): m = floor(p/3), node size 5 (regression); floor(sqrt(p)), node size 1 (classification) ----
class _P: task = "regression"; p = 233
g = rf_entry.make("rf_defaults", "rf", "defaults")[0](_C(), _P(), None, None)[0]
_P.task, _P.p = "classification", 240
g2 = rf_entry.make("rf_defaults", "rf", "defaults")[0](_C(), _P(), None, None)[0]
check("rf_defaults regression: m = floor(233/3) = 77, minimum node size 5", g["mf"] == 77 and g["leaf"] == 5)
check("rf_defaults classification: m = floor(sqrt(240)) = 15, minimum node size 1", g2["mf"] == 15 and g2["leaf"] == 1)
fitted = rl.build_forest("rf", {"mf": 77, "leaf": 5}, "regression", _C(), False).fit(rng.randn(300, 233), rng.randn(300))
check("scikit-learn uses exactly m = 77 candidate variables per split", fitted.estimators_[0].max_features_ == 77)
# ---- importance finds the planted variables ----
imp_split = np.argsort(-rf.feature_importances_)[:2]
pi = permutation_importance(rf, X[:1000], y[:1000], scoring="neg_mean_squared_error", n_repeats=3, random_state=0)
check("split-based and permutation importance both rank the planted variables (0 and 3) first", set(imp_split) == {0, 3} and set(np.argsort(-pi.importances_mean)[:2]) == {0, 3})
# ---- permutation importance on threads: works where the default process-based backend must pickle the forest (PicklingError on large forests) ----
class _Unpicklable(RandomForestRegressor):
    def __reduce__(self):
        raise TypeError("this forest cannot be pickled")
big = _Unpicklable(n_estimators=40, min_samples_leaf=5, random_state=0, n_jobs=1).fit(X, y)
try:
    permutation_importance(big, X[:300], y[:300], scoring="neg_mean_squared_error", n_repeats=1, random_state=0, n_jobs=2); plain_fails = False
except Exception:
    plain_fails = True
pt = rl.permutation_importance_threads(big, X[:300], y[:300], "neg_mean_squared_error", 2, 0, 2)
p1 = rl.permutation_importance_threads(big, X[:300], y[:300], "neg_mean_squared_error", 2, 0, 1)
check("default process-based permutation_importance fails on a forest that must be pickled; the thread version does not", plain_fails and np.allclose(pt.importances_mean, p1.importances_mean))
# ---- a failing extra step must not destroy the run ----
import logging
summ = {}
common._safe_extra(lambda c: (_ for _ in ()).throw(RuntimeError("boom")), logging.getLogger("t"), summ, {})
check("a failing extra step is logged and recorded in summary.json instead of aborting the run", summ.get("extra_error", "").startswith("RuntimeError"))
# ---- ExtraTrees: no bootstrap -> no OOB, handled gracefully ----
et = rl.build_forest("et", {"mf": 0.5, "leaf": 5}, "regression", _C(), False).fit(X, y)
check("ExtraTrees entry has no OOB and the OOB diagnostic returns nothing", rl.oob_diag(et, {}, X, y, None, None, "regression") == {})
# ---- keep-all-variables selection rule ----
hp_all, hp_q = {"frac": 1.0, "leaf": 10, "mf": 0.5}, {"frac": 0.5, "leaf": 50, "mf": 0.2}
fm = np.array([[1.00, 0.99, 1.02], [2.00, 1.98, 2.03], [1.50, 1.49, 1.52], [3.00, 2.99, 3.02]])
comp = np.array([rl.drop_penalty(hp_all) + 100 / 10 + 5, rl.drop_penalty(hp_all) + 100 / 5 + 10, rl.drop_penalty(hp_q) + 100 / 50 + 2])
sel, best = common.one_se_paired(fm, comp)
check("one-SE selection keeps all variables when dropping some is not clearly better", comp[sel] < 1e5, f"selected {sel}")
e = rng.randn(3000) ** 2
check("Diebold-Mariano: clearly different losses -> tiny p", common.dm_test(e, e + 0.5)[2] < 1e-6)
check("Holm", np.allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
