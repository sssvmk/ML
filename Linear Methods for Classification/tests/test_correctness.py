"""Check every method / metric against independent references (scikit-learn, brute force, analytic identities)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common
import indicator_regression, lda, qda, rda, reduced_rank_lda, logistic, l1_logistic, perceptron
import compare
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(3)
n, p = 6000, 8
y = (rng.rand(n) < 0.2).astype(int)
S1 = np.diag(rng.uniform(0.5, 2, p)); S0 = np.eye(p) + 0.3
X = np.where(y[:, None] == 1, rng.multivariate_normal(rng.randn(p) * 0.5, S1, n), rng.multivariate_normal(np.zeros(p), S0, n))
X = (X - X.mean(0)) / X.std(0)
cs = common.build_class_stats(X, y)
view = common.TrainView.__new__(common.TrainView); view.X, view.y, view.w, view.cs, view.n, view.p = X, y, np.ones(n), cs, n, p
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

s_lda = lda.fit(view, {}).score(X)
sk = LinearDiscriminantAnalysis(solver="svd").fit(X, y)
check("LDA log-odds vs sklearn", np.abs(s_lda - sk.decision_function(X)).max() < 1e-6, f"{np.abs(s_lda - sk.decision_function(X)).max():.1e}")
s_qda = qda.fit(view, {}).score(X)
skq = QuadraticDiscriminantAnalysis(reg_param=0.0).fit(X, y)
lp = skq.predict_log_proba(X); d = lp[:, 1] - lp[:, 0]
check("QDA log-odds vs sklearn", np.abs(s_qda - d).max() < 1e-5, f"{np.abs(s_qda - d).max():.1e}")
check("RDA(alpha=0,gamma=1) == LDA", np.abs(rda.fit(view, {"alpha": 0.0, "gamma": 1.0}).score(X) - s_lda).max() < 1e-8)
check("RDA(alpha=1,gamma=1) == QDA", np.abs(rda.fit(view, {"alpha": 1.0, "gamma": 1.0}).score(X) - s_qda).max() < 1e-8)
m = reduced_rank_lda.fit(view, {"rank": 1})
check("reduced-rank LDA (K=2) == LDA", np.abs(m.score(X) - s_lda).max() < 1e-7, f"coef rel diff {m.info['relative_max_coef_difference_vs_LDA']:.1e}")
f1 = indicator_regression.fit(view, {})
lr = LinearRegression().fit(X, y)
check("indicator regression == OLS on 0/1", np.abs((f1.score(X) + 1) / 2 - lr.predict(X)).max() < 1e-8)
check("indicator columns sum to one", f1.info["max_abs_f0_plus_f1_minus_1"] < 1e-8)

# logistic (IRLS) vs sklearn unpenalised
logistic._WARM.clear()
ml = logistic.fit(view, {})
check('logistic: no separation flag on overlapping classes', not ml.info['quasi_separation_suspected'])
skl = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(X, y)
check("logistic IRLS vs sklearn", np.abs(ml.coef - skl.coef_.ravel()).max() < 1e-5 and abs(ml.intercept - skl.intercept_[0]) < 1e-5, f"{np.abs(ml.coef - skl.coef_.ravel()).max():.1e}")

# L1 logistic vs sklearn saga (objective -(1/n)loglik + lam|b|  <=>  C = 1/(n lam))
l1_logistic._XF.clear()
grid = l1_logistic.make_grid(type("P", (), {"y_train": y, "X_train": X})(), common.RunConfig(l1_nlambda=12))
models = l1_logistic.lasso_logistic_path(np.asfortranarray(X), y.astype(float), np.ones(n), [h["lam"] for h in grid], max_outer=8)
errs = []
for i in (3, 7, 11):
    lam = grid[i]["lam"]
    skl1 = LogisticRegression(l1_ratio=1, C=1 / (n * lam), solver="saga", tol=1e-10, max_iter=20000).fit(X, y)
    errs.append(max(np.abs(models[i][0] - skl1.coef_.ravel()).max(), abs(models[i][1] - skl1.intercept_[0])))
check("L1 logistic path vs sklearn saga", max(errs) < 2e-4, f"maxerr={max(errs):.1e}")
check("L1 logistic sparsity grows along path", (models[0][0] != 0).sum() <= (models[-1][0] != 0).sum())

# perceptron converges on separable data, flags non-convergence otherwise
perceptron._CFG.update(epochs=200, pocket=False, seed=0)
Xs = rng.randn(500, 4); ysep = (Xs @ np.array([1., -2, .5, 1]) + 0.3 > 0).astype(int)
Xs = Xs[np.abs(Xs @ np.array([1., -2, .5, 1]) + 0.3) > 0.4]; ysep = (Xs @ np.array([1., -2, .5, 1]) + 0.3 > 0).astype(int)
v2 = common.TrainView.__new__(common.TrainView); v2.X, v2.y, v2.w, v2.cs, v2.n, v2.p = Xs, ysep, np.ones(len(Xs)), None, len(Xs), 4
mp_ = perceptron.fit(v2, {})
check("perceptron converges on separable data", mp_.info["converged"] and mp_.info["final_train_error"] == 0.0, f"epochs={mp_.info['epochs_run']}")
perceptron._CFG.update(epochs=15)
mp2 = perceptron.fit(view, {})
check("perceptron flags non-convergence on overlapping classes", not mp2.info["converged"])

# metrics
s = rng.randn(2000) + 0.8 * (np.arange(2000) % 5 == 0); yy = (np.arange(2000) % 5 == 0).astype(int)
check("AUC == sklearn", abs(common.auc_score(yy, s) - roc_auc_score(yy, s)) < 1e-12)
a, v = common.delong_var(yy, s)
boot = [common.auc_score(yy[i], s[i]) for i in (np.random.RandomState(b).randint(0, 2000, 2000) for b in range(400))]
check("DeLong SE ~ bootstrap SE", abs(np.sqrt(v) / np.std(boot) - 1) < 0.2, f"delong={np.sqrt(v):.4f} boot={np.std(boot):.4f}")
t = common.youden_threshold(yy, s); yh = (s > t).astype(int)
best = max(((s > th).astype(int)[yy == 1].mean() - (s > th).astype(int)[yy == 0].mean()) for th in np.unique(s))
check("Youden threshold maximises TPR-FPR", abs((yh[yy == 1].mean() - yh[yy == 0].mean()) - best) < 1e-12)
d_, se_, p_ = compare.delong_pair(yy, s, s + 0.0)
check("DeLong paired test: identical scores -> diff 0, p=1", abs(d_) < 1e-15 and p_ > 0.99)
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
