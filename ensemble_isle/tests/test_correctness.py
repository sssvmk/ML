"""Verify the Chapter 16 glue against scikit-learn and the book's statements (rules reproduce the tree, dictionary reproduces GBM, lasso path properties, epsilon -> 0, L1 margin)."""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common, isle_lib as il, boost_path as bp
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
X = rng.randn(3000, 8)
y = 2 * X[:, 0] + np.where(X[:, 2] > 0, 1.5, -1.0) * X[:, 3] + 0.5 * rng.randn(3000)
yc = ((X[:, 0] + np.abs(X[:, 2]) + 0.8 * rng.randn(3000)) > 1.2).astype(int)
# ---- (16.14) a linear expansion in the node indicators reproduces the tree (Ex. 16.3) ----
t = DecisionTreeRegressor(max_leaf_nodes=6, random_state=0).fit(X, y)
R = il.rule_matrix([t], X)
tr = t.tree_; v = tr.value[:, 0, 0]
parent = np.full(tr.node_count, -1)
for nd in range(tr.node_count):
    if tr.children_left[nd] != -1:
        parent[tr.children_left[nd]] = nd; parent[tr.children_right[nd]] = nd
coef = np.array([v[nd] - v[parent[nd]] for nd in range(1, tr.node_count)])
check("rules (node indicators from decision_path): v_root + sum (v_node - v_parent) R_node == tree.predict", np.abs(v[0] + R @ coef - t.predict(X)).max() < 1e-9, f"{R.shape[1]} rules")
meta = il.rule_meta([t], [f"x{i}" for i in range(8)])
dp = t.decision_path(X[:50]).toarray()
check("rule metadata: depth = number of conditions; a rule's indicator is 1 exactly when all its conditions hold",
      all(m["depth"] == len(m["text"].split(" & ")) for m in meta) and all(
          bool(R[i, j]) == all(((X[i, int(c.split()[0][1:])] <= float(c.split()[2])) if c.split()[1] == "<=" else (X[i, int(c.split()[0][1:])] > float(c.split()[2]))) for c in meta[j]["text"].split(" & "))
          for i in range(30) for j in range(len(meta))))
# ---- the ISLE dictionary reproduces the boosted model: f = f0 + nu * sum_m T_m ----
gb = GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_leaf_nodes=4, subsample=0.5, random_state=0).fit(X, y)
T = il.tree_matrix(il.trees_of(gb), X)
check("dictionary of tree outputs: init + nu * sum_m T_m(x) == GradientBoosting.predict", np.abs(gb.init_.predict(X) + 0.1 * T.sum(1) - gb.predict(X)).max() < 1e-9)
# ---- lasso path (16.9) ----
from sklearn.preprocessing import StandardScaler
Ts = StandardScaler().fit_transform(T)
lam = il.LAMR_REG
preds, nnz, coefs = il.lasso_path_fit(Ts, y, [Ts], lam)
check("lasso path: no coefficient at lambda_max, sparsity grows as the penalty is relaxed", nnz[0] == 0 and all(nnz[i + 1] >= nnz[i] - 1 for i in range(len(nnz) - 1)), f"nnz = {nnz.tolist()}")
from sklearn.linear_model import LinearRegression
mse_path = np.mean((y - preds[0][-1]) ** 2); mse_ols = np.mean((y - LinearRegression().fit(Ts, y).predict(Ts)) ** 2)
check("at lambda = 0.001 lambda_max the training fit is close to least squares on the dictionary", mse_path < 1.02 * mse_ols + 0.02, f"{mse_path:.4f} vs OLS {mse_ols:.4f}")
# ---- ISLE: fewer trees, about the same accuracy ----
ntr = 2000
gb2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_leaf_nodes=4, subsample=0.5, random_state=0).fit(X[:ntr], y[:ntr])
tr2 = il.trees_of(gb2); sc = StandardScaler().fit(il.tree_matrix(tr2, X[:ntr]))
Tt, Te = sc.transform(il.tree_matrix(tr2, X[:ntr])), sc.transform(il.tree_matrix(tr2, X[ntr:]))
pr, nz, _ = il.lasso_path_fit(Tt, y[:ntr], [Te], lam)
mses = [np.mean((y[ntr:] - p) ** 2) for p in pr[0]]; j = int(np.argmin(mses))
mse_gb = np.mean((y[ntr:] - gb2.predict(X[ntr:])) ** 2)
check("ISLE: lasso post-processing keeps far fewer trees at about the accuracy of the full boosted ensemble", nz[j] < 0.8 * 200 and mses[j] < 1.1 * mse_gb, f"{nz[j]} of 200 trees, MSE {mses[j]:.4f} vs full GBM {mse_gb:.4f}")
# ---- L1-penalised logistic path: classification ISLE ----
gc = GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_leaf_nodes=4, subsample=0.5, random_state=0).fit(X[:ntr], yc[:ntr])
trc = il.trees_of(gc); scc = StandardScaler().fit(il.tree_matrix(trc, X[:ntr]))
pc, nzc, _ = il.logistic_path_fit(scc.transform(il.tree_matrix(trc, X[:ntr])), yc[:ntr], [scc.transform(il.tree_matrix(trc, X[ntr:]))], il.LAMR_CLF)
aucs = [common.auc_score(yc[ntr:], p) for p in pc[0]]
check("L1 logistic path: empty model at C_min, test AUC rises as trees enter", nzc[0] == 0 and max(aucs) > 0.85 and nzc[int(np.argmax(aucs))] > 0, f"AUC {max(aucs):.3f} with {nzc[int(np.argmax(aucs))]} trees")
# ---- boosting with shrinkage as the epsilon-forward-stagewise path: the curve in the arc length t converges as epsilon -> 0 ----
class _C: seed = 0; n_jobs = 1
curves = {}
for nu, M in ((0.4, 25), (0.2, 50), (0.1, 100)):
    g = bp.build_gb({"nu": nu, "J": 2}, M, "regression", _C()).fit(X[:ntr], y[:ntr])
    Tk = il.tree_matrix(il.trees_of(g), X[:ntr]); tcum = np.cumsum(nu * Tk.std(0))
    st = bp.staged_preds(g, X[ntr:], range(1, M + 1), "regression")
    curves[nu] = (tcum, np.array([np.mean((y[ntr:] - st[m]) ** 2) for m in range(1, M + 1)]))
tt = 0.5 * min(c[0][-1] for c in curves.values())
at = {nu: float(np.interp(tt, c[0], c[1])) for nu, c in curves.items()}
check("epsilon -> 0: the MSE-vs-arc-length curves for nu, nu/2, nu/4 converge (spread shrinks)", abs(at[0.1] - at[0.2]) <= abs(at[0.2] - at[0.4]) + 1e-4, f"MSE at t={tt:.2f}: " + ", ".join(f"nu={k}: {v:.4f}" for k, v in at.items()))
# ---- normalised L1 margin (16.7) ----
yy = np.array([1, 0, 1, 0]); f = np.array([2.0, -1.0, 0.5, -3.0])
check("L1 margin = min_i y_i f(x_i) / sum |alpha| (here 0.5 / 2)", abs(il.l1_margin(yy, f, 2.0) - 0.25) < 1e-12 and abs(il.l1_margin(yy, 3 * f, 6.0) - il.l1_margin(yy, f, 2.0)) < 1e-12)
# ---- keep-all-variables rule and tests ----
hp_all, hp_q = {"frac": 1.0}, {"frac": 0.5}
fm = np.array([[1.00, 0.99, 1.02], [2.00, 1.98, 2.03], [1.50, 1.49, 1.52], [3.00, 2.99, 3.02]])
comp = np.array([il.drop_penalty(hp_all) + 5, il.drop_penalty(hp_all) + 9, il.drop_penalty(hp_q) + 1])
sel, _ = common.one_se_paired(fm, comp)
check("one-SE selection keeps all variables when dropping some is not clearly better", comp[sel] < 1e5, f"selected {sel}")
e = rng.randn(3000) ** 2
check("Diebold-Mariano: clearly different losses -> tiny p", common.dm_test(e, e + 0.5)[2] < 1e-6)
check("Holm", np.allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
