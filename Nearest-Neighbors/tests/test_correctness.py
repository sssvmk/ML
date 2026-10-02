"""Verify the neighbour glue, tangent distance, LVQ1 and DANN against scikit-learn / hand computations / the book's own example."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import common, nnlib, lvq, dann, knn, kmeans_prototypes as kmp, gmm
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_auc_score

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
Xr, Xq = rng.randn(800, 5), rng.randn(150, 5)
yr = np.sin(Xr[:, 0]) + 0.3 * rng.randn(800); yc = (rng.rand(800) < 0.3).astype(float)
# ---- one neighbour search serves every k: identical to scikit-learn's estimators ----
for w in ("uniform", "distance"):
    for p in (1, 2):
        dist, ind = NearestNeighbors(n_neighbors=60, p=p).fit(Xr).kneighbors(Xq)
        for nn in (1, 5, 20, 60):
            ref = KNeighborsRegressor(n_neighbors=nn, weights=w, p=p).fit(Xr, yr).predict(Xq)
            mine = nnlib.knn_scores(dist, yr[ind], nn, w, "regression", 0.0)
            refc = KNeighborsClassifier(n_neighbors=nn, weights=w, p=p).fit(Xr, yc.astype(int)).predict_proba(Xq)[:, 1]
            minec = nnlib.knn_scores(dist, yc[ind], nn, w, "classification", yc.mean())
            if not (np.abs(ref - mine).max() < 1e-9 and np.corrcoef(refc, minec)[0, 1] > 0.99999 and np.abs((refc * nn + yc.mean()) / (nn + 1) - minec).max() < 1e-9):
                check(f"prefix k-NN == scikit-learn ({w}, p={p}, k={nn})", False); break
        else:
            continue
        break
else:
    check("k-NN from ONE neighbour search == KNeighborsRegressor / Classifier (uniform & distance, Euclidean & Manhattan, k = 1..60)", True)
# ---- leave-one-out via the same search == brute force ----
Xs, ys = rng.randn(120, 3), rng.randn(120)
loo = nnlib.loo_curve(Xs, ys, [3, 10], "uniform", 2)
brute = {nn: np.array([ys[np.argsort(((Xs - Xs[i]) ** 2).sum(1))[1:nn + 1]].mean() for i in range(120)]) for nn in (3, 10)}
check("leave-one-out k-NN predictions == brute force", all(np.abs(loo[nn] - brute[nn]).max() < 1e-9 for nn in (3, 10)))
# ---- tangent distance ----
t = np.linspace(0, 1, 200); f = lambda s: np.exp(-((t - 0.5 - s) / 0.08) ** 2)             # a bump shifted by s
x0, x1 = f(0.0), f(0.004)
T = ((f(0.001) - f(-0.001)) / 0.002)[:, None]                                              # tangent vector of the shift transformation
d_e, d_t = np.linalg.norm(x0 - x1), nnlib.tangent_distance(x0, x1, T, None)
check("tangent distance ignores a small shift that Euclidean distance penalises", d_t < 0.15 * d_e, f"tangent {d_t:.4f} vs Euclidean {d_e:.4f}")
a, b = rng.randn(6), rng.randn(6)
check("empty tangent set: tangent distance == Euclidean distance", abs(nnlib.tangent_distance(a, b) - np.linalg.norm(a - b)) < 1e-12 and nnlib.tangent_projection(None) is None)
Xsh = np.array([f(s) for s in np.linspace(-0.004, 0.004, 40)]); Pj = nnlib.tangent_projection(T)
check("projecting the tangent out collapses shifted copies (one-sided shared-tangent distance)", np.linalg.norm(Pj(Xsh) - Pj(Xsh).mean(0), axis=1).max() < 0.2 * np.linalg.norm(Xsh - Xsh.mean(0), axis=1).max())
# ---- LVQ1 update rule ----
pr = np.array([[0.0, 0.0], [5.0, 5.0]]); lb = np.array([0, 1])
j = lvq.lvq_step(pr, lb, np.array([1.0, 0.0]), 0, 0.1)
check("LVQ1: same-class point attracts the nearest prototype by eps*(x - m)", j == 0 and np.allclose(pr[0], [0.1, 0.0]))
pr2 = np.array([[0.0, 0.0], [5.0, 5.0]]); lvq.lvq_step(pr2, lb, np.array([1.0, 0.0]), 1, 0.1)
check("LVQ1: different-class point repels the nearest prototype by eps*(x - m)", np.allclose(pr2[0], [-0.1, 0.0]))
Xb = np.r_[rng.randn(300, 2) + [-1.5, 0], rng.randn(300, 2) + [1.5, 0], rng.randn(300, 2) + [0, 3]]; yb = np.r_[np.zeros(300), np.ones(300), np.zeros(300)].astype(int)
mL = lvq.LVQ1(4, 0.05, 0).fit(Xb, yb)
check("LVQ1 prototypes move (after the K-means start) and keep their labels", np.linalg.norm(mL.protos - mL.start, axis=1).max() > 1e-3 and (mL.labels == np.r_[np.zeros(4), np.ones(4)]).all())
auc_l, auc_k = roc_auc_score(yb, mL.decision_function(Xb)), roc_auc_score(yb, mL.start_scores(Xb))
check("LVQ1 does not hurt the K-means solution on blobs", auc_l >= auc_k - 0.01, f"AUC {auc_l:.4f} vs K-means start {auc_k:.4f}")
# ---- K-means prototype classifier with R=1 == nearest class centroid ----
Xk, yk = rng.randn(600, 4) + np.r_[np.zeros(4), np.ones(4)][:4] * 0, (rng.rand(600) < 0.4).astype(int)
Xk[yk == 1] += 1.0
mk = kmp.KMeansProto(1, 0).fit(Xk, yk)
c0, c1 = Xk[yk == 0].mean(0), Xk[yk == 1].mean(0)
sc = np.linalg.norm(Xk - c0, axis=1) - np.linalg.norm(Xk - c1, axis=1)
check("K-means prototype classifier (R=1) == nearest class centroid", np.abs(mk.decision_function(Xk) - sc).max() < 1e-6)
# ---- DANN local metric ----
Xn = rng.randn(60, 3); yn = (Xn[:, 0] > 0).astype(int)
S = dann.local_metric(Xn, yn, 1.0)
check("DANN metric (13.8) is symmetric positive definite", S is not None and np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > 0)
check("DANN metric weights the discriminating direction (x0) more: neighbourhoods shrink there and stretch along the others", S[0, 0] / S[1, 1] > 1.0, f"S00/S11 = {S[0, 0] / S[1, 1]:.3f}")
check("DANN falls back to Euclidean when a neighbourhood has one class", dann.local_metric(Xn, np.zeros(60, int), 1.0) is None)
# ---- the book's 10-D example (13.4.1): class 1 almost surrounds class 2; DANN must beat 5-NN ----
def book(n, rng):
    X1 = []
    while len(X1) < n:
        z = rng.randn(10)
        if 22.4 < (z ** 2).sum() < 40:
            X1.append(z)
    return np.r_[np.array(X1), rng.randn(n, 10)], np.r_[np.zeros(n), np.ones(n)].astype(int)
Xtr, ytr = book(250, np.random.RandomState(1)); Xte, yte = book(500, np.random.RandomState(2))
e5 = float(np.mean(KNeighborsClassifier(5).fit(Xtr, ytr).predict(Xte) != yte))
class _C: n_jobs = 1; seed = 0
grid = [{"k": 10, "KM": 50, "eps": 1.0, "nn": 5}]
p = dann.predict_path(Xtr, ytr.astype(float), grid, Xte, _C(), {"order": np.arange(10), "final": False})[0]
ed = float(np.mean((p > 0.5).astype(int) != yte))
check("DANN beats 5-NN on the book's 10-D nested-class example (Fig. 13.15)", ed < e5, f"DANN {ed:.3f} vs 5-NN {e5:.3f}")
# ---- misc ----
check("GMM classifier posteriors are proper probabilities", np.all((gmm.GMMClassifier(2, "diag", 0).fit(Xk, yk).predict_proba(Xk) >= 0) & (gmm.GMMClassifier(2, "diag", 0).fit(Xk, yk).predict_proba(Xk) <= 1)))
e = rng.randn(3000) ** 2
check("Diebold-Mariano: clearly different losses -> tiny p", common.dm_test(e, e + 0.5)[2] < 1e-6)
check("Holm", np.allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
