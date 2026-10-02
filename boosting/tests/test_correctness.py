"""Verify the glue around the prebuilt libraries (scores, stage handling, selection rules) against independent computations."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common, boostlib as bl
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import lightgbm as lgb, xgboost as xgb

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
X = rng.randn(3000, 6); y = (X[:, 0] + 0.7 * X[:, 1] ** 2 + rng.randn(3000) > 1).astype(int)
# ---- AdaBoost: ESLII f = 1/2 sum alpha_m G_m and p = 1/(1+exp(-2f)) from the library's staged decision function ----
m = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=0), n_estimators=15, random_state=0).fit(X, y)
st = bl.predict_stages(m, "ada", X, [5, 10, 15], "classification")
for M in (5, 10, 15):
    f = 0.5 * sum(a * (2 * e.predict(X) - 1) for a, e in zip(m.estimator_weights_[:M], m.estimators_[:M]))
    check(f"AdaBoost stage {M}: p == 1/(1+exp(-2 f)) with f = 1/2 sum alpha G", np.abs(st[M] - 1 / (1 + np.exp(-2 * f))).max() < 1e-9, f"{np.abs(st[M] - 1 / (1 + np.exp(-2 * f))).max():.1e}")
Xs_ = rng.randn(500, 3); ys_ = (Xs_[:, 0] > 0).astype(int)                                    # perfectly separable by one stump -> AdaBoost stops after 1 stage
msep = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=0).fit(Xs_, ys_)
try:
    stp = bl.predict_stages(msep, "ada", Xs_, [10, 25, 50], "classification"); okk = all(M in stp for M in (10, 25, 50))
except KeyError:
    okk = False
check("AdaBoost that stops early (perfectly separable data): every requested M has a prediction", okk, f"fitted stages = {len(msep.estimators_)}")
# ---- forward stagewise: sklearn GB (lr=1, stumps, squared error) == manual residual fitting ----
Xr = rng.randn(2000, 5); yr = np.sin(2 * Xr[:, 0]) + Xr[:, 1] + 0.3 * rng.randn(2000)
gb = GradientBoostingRegressor(loss="squared_error", learning_rate=1.0, max_depth=1, n_estimators=12, random_state=0).fit(Xr, yr)
f = np.full(len(yr), yr.mean())
for _ in range(12):
    f = f + DecisionTreeRegressor(max_depth=1, random_state=0).fit(Xr, yr - f).predict(Xr)
check("forward stagewise (least squares) == sklearn GB with learning_rate=1 and stumps", np.abs(gb.predict(Xr) - f).max() < 1e-8, f"{np.abs(gb.predict(Xr) - f).max():.1e}")
# ---- stage handling: M-limited prediction == a model trained with M trees ----
for kind, mk in (("lgb", lambda M: lgb.LGBMRegressor(n_estimators=M, num_leaves=7, learning_rate=0.1, verbose=-1, n_jobs=1, random_state=0)),
                 ("xgb", lambda M: xgb.XGBRegressor(n_estimators=M, max_depth=3, learning_rate=0.1, tree_method="hist", n_jobs=1, random_state=0, verbosity=0))):
    big, small = mk(60).fit(Xr, yr), mk(25).fit(Xr, yr)
    p = bl.predict_stages(big, kind, Xr, [25, 60], "regression")
    check(f"{kind}: prediction limited to M=25 iterations == model trained with 25", np.abs(p[25] - small.predict(Xr)).max() < 1e-5, f"{np.abs(p[25] - small.predict(Xr)).max():.1e}")
gbs = GradientBoostingRegressor(n_estimators=30, random_state=0).fit(Xr, yr)
ps = bl.predict_stages(gbs, "sk", Xr, [10, 30], "regression")
check("sklearn staged_predict at M=10 == model trained with 10", np.abs(ps[10] - GradientBoostingRegressor(n_estimators=10, random_state=0).fit(Xr, yr).predict(Xr)).max() < 1e-9)
# ---- the 'keep features unless dropping helps' rule ----
hp_all, hp_half = {"frac": 1.0, "M": 100}, {"frac": 0.5, "M": 20}
check("complexity ranks a dropped-feature model above any all-feature model", bl.drop_penalty(hp_half) + 20 > bl.drop_penalty(hp_all) + 100000)
fm = np.array([[1.00, 0.99, 1.02], [2.00, 1.98, 2.03], [1.50, 1.49, 1.52], [3.00, 2.99, 3.02]])        # col2 (half the features, fewer trees) is NOT clearly better
comp = np.array([bl.drop_penalty(hp_all) + 100, bl.drop_penalty(hp_all) + 200, bl.drop_penalty(hp_half) + 20])
sel, best = common.one_se_paired(fm, comp)
check("one-SE selection keeps all features when dropping some is not clearly better", sel in (0, 1) and comp[sel] < 1e5, f"selected index {sel} (CV-best {best})")
cfgs = bl.sample_configs({"frac": bl.FRACS, "J": [4, 6, 8]}, 2, 3)
check("random search always evaluates the keep-all-features configuration", any(c["frac"] == 1.0 for c in cfgs), f"{cfgs}")
g = bl.expand([{"frac": 0.5, "x": 1}], [10, 20], 100)
check("expand: k = round(frac * p), one entry per M", [h["k"] for h in g] == [50, 50] and [h["M"] for h in g] == [10, 20])
e = rng.randn(3000) ** 2
check("Diebold-Mariano: clearly different losses -> tiny p", common.dm_test(e, e + 0.5)[2] < 1e-6)
check("Holm", np.allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
