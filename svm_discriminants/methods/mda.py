"""
Mixture discriminant analysis (ESLII 12.7) - CLASSIFICATION. Each class k is a mixture of R_k Gaussian subclasses with a COMMON covariance:
  P(x | k) = sum_r pi_kr phi(x; mu_kr, Sigma);   posterior by Bayes' rule with the training class proportions.
Loss: negative mixture log-likelihood, fitted by EM. The class labels are known, so the E-step only distributes each observation over the subclasses of ITS class; the M-step updates mu_kr, pi_kr and
the pooled covariance Sigma. No package implements this model, so this small EM loop is the only hand-written numerical routine of the chapter (initialised with scikit-learn KMeans).
Hyper-parameters (CV): k = number of top-ranked variables (a full common covariance needs k << N, so 2-20 % of the pool), subclasses per class R, covariance shrinkage. CV selects on AUC.
Metrics: test error +- SE, AUC, log-loss, held-out log-likelihood log p(x, class), BIC vs R, CV error / AUC vs R, canonical-coordinate plot with the subclass centres.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

import common
import svmlib as sl

NAME = "mda"
SPACE = {"frac": [0.02, 0.05, 0.1, 0.2], "R": [1, 2, 3, 4], "reg": [1e-3, 1e-1]}


class MDAModel:
    def __init__(self, R, reg, seed, iters=40):
        self.R, self.reg, self.seed, self.iters = R, reg, seed, iters

    def _maha(self, X, mu):
        return np.sum(solve_triangular(self.L, (X - mu).T, lower=True) ** 2, axis=0)

    def fit(self, X, y):
        from sklearn.cluster import KMeans
        n, k = X.shape
        self.k, self.classes = k, [0, 1]
        Xc = {c: X[y == c] for c in self.classes}
        Nc = {c: len(Xc[c]) for c in self.classes}
        self.prior = np.array([Nc[c] / n for c in self.classes])
        resp = {}
        for c in self.classes:
            lab = KMeans(self.R, n_init=2, random_state=self.seed).fit_predict(Xc[c]) if self.R > 1 else np.zeros(Nc[c], int)
            resp[c] = np.eye(self.R)[lab]
        self.mu = {c: np.zeros((self.R, k)) for c in self.classes}
        self.pi = {c: np.zeros(self.R) for c in self.classes}
        self.ll_hist = []
        prev = -np.inf
        for it in range(self.iters):
            S = np.zeros((k, k))
            for c in self.classes:                                                  # M-step
                for r in range(self.R):
                    w = resp[c][:, r]
                    sw = max(w.sum(), 1e-9)
                    self.mu[c][r] = (w @ Xc[c]) / sw
                    self.pi[c][r] = sw / Nc[c]
                    D = Xc[c] - self.mu[c][r]
                    S += (D * w[:, None]).T @ D
            Sigma = S / n
            Sigma += self.reg * np.trace(Sigma) / k * np.eye(k)
            self.L = cholesky(Sigma, lower=True)
            self.logdet = 2.0 * np.sum(np.log(np.diag(self.L)))
            tot = 0.0
            for c in self.classes:                                                  # E-step (within the class of each observation)
                lp = np.stack([np.log(self.pi[c][r] + 1e-300) - 0.5 * self._maha(Xc[c], self.mu[c][r]) for r in range(self.R)], 1)
                lse = logsumexp(lp, axis=1)
                resp[c] = np.exp(lp - lse[:, None])
                tot += float(lse.sum()) - 0.5 * Nc[c] * (self.logdet + k * np.log(2 * np.pi)) + Nc[c] * np.log(self.prior[c])
            self.ll_hist.append(tot / n)
            if abs(tot - prev) < 1e-7 * abs(tot):
                break
            prev = tot
        self.n_params = int(k * 2 * self.R + 2 * (self.R - 1) + k * (k + 1) / 2)
        self.loglik_total = tot
        self.n = n
        return self

    def class_logdens(self, X):
        out = []
        for c in self.classes:
            lp = np.stack([np.log(self.pi[c][r] + 1e-300) - 0.5 * self._maha(X, self.mu[c][r]) for r in range(self.R)], 1)
            out.append(logsumexp(lp, axis=1) - 0.5 * (self.logdet + self.k * np.log(2 * np.pi)))
        return np.stack(out, 1)

    def predict_proba(self, X):
        ld = self.class_logdens(X) + np.log(self.prior)[None]
        p1 = common.sigmoid(np.clip(ld[:, 1] - ld[:, 0], -30, 30))
        return np.c_[1 - p1, p1]

    def heldout_ll(self, X, y):
        ld = self.class_logdens(X) + np.log(self.prior)[None]
        return float(np.mean(ld[np.arange(len(y)), y]))

    @property
    def bic(self):
        return -2 * self.loglik_total + self.n_params * np.log(self.n)


def build(hp, task, cfg, y):
    return MDAModel(hp["R"], hp["reg"], cfg.seed)


def make_grid(cfg, prep, Xs, ys):
    return sl.with_k(sl.sample_configs(SPACE, cfg.n_configs_svm, cfg.seed), prep.p)


def complexity(hp):
    return hp["R"] * hp["k"]


def diag(model, hp, Xr, y, Xq, yq, task):
    d = {"bic_train": model.bic}
    if yq is not None:
        d["heldout_loglik"] = model.heldout_ll(Xq, yq.astype(int))
    return d


def extra(ctx):
    import json as _j
    S = ctx["store"]
    model, cols, hp = S["model"], S["cols"], S["hp"]
    Xs = np.asarray(ctx["Xs"][:, cols], np.float64)
    ys = ctx["ys_ev"].astype(int)
    bic = {}
    for R in (1, 2, 3, 4):
        m = MDAModel(R, hp["reg"], ctx["cfg"].seed).fit(Xs, ys)
        bic[R] = float(m.bic)
    ctx["summary"]["mda"] = {"R": hp["R"], "k": hp["k"], "bic_by_R_on_tuning_subsample": bic, "em_iterations": len(model.ll_hist), "fit_rows": int(len(S["yfit"]))}
    (ctx["out"] / "mda_extra_metrics.json").write_text(_j.dumps(common._jsonable(ctx["summary"]["mda"]), indent=2))
    # canonical coordinates: whiten with the common covariance, then PCA of the subclass centres (class-1 centre minus class-0 centre is the first coordinate)
    cent = np.vstack([model.mu[0], model.mu[1]])
    labs = np.r_[np.zeros(model.R), np.ones(model.R)]
    W = solve_triangular(model.L, np.vstack([cent, np.asarray(ctx["X_te"][:3000, cols], np.float64)]).T, lower=True).T
    Wc, Wt = W[:len(cent)], W[len(cent):]
    u, s, vt = np.linalg.svd(Wc - Wc.mean(0), full_matrices=False)
    P = vt[:2].T if vt.shape[0] >= 2 else np.c_[vt[0], np.zeros(len(vt[0]))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    yt = ctx["y_te"][:3000]
    ax[0].scatter((Wt - Wc.mean(0)) @ P[:, 0], (Wt - Wc.mean(0)) @ P[:, 1] if P.shape[1] > 1 else 0 * yt, c=yt, s=4, alpha=0.35, cmap="coolwarm")
    ax[0].scatter((Wc - Wc.mean(0)) @ P[:, 0], (Wc - Wc.mean(0)) @ P[:, 1] if P.shape[1] > 1 else 0 * labs, c=labs, s=120, marker="*", edgecolors="k", cmap="coolwarm")
    ax[0].set_title(f"MDA: test points and subclass centres (R = {model.R}) in canonical coordinates")
    ax[1].plot(list(bic), [bic[r] - min(bic.values()) for r in bic], "o-"); ax[1].set_xlabel("subclasses per class R"); ax[1].set_ylabel("BIC - min"); ax[1].set_title("MDA: BIC vs R (tuning subsample)")
    fig.tight_layout(); fig.savefig(ctx["out"] / "mda_canonical_coordinates.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("MDA is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=sl.make_predict_path(build, sl.margin_score, diag=diag), complexity=complexity, extra=extra,
                             select_by="auc", notes="EM for a common-covariance Gaussian mixture per class (numpy), KMeans initialisation; all training rows.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
