"""
Gaussian mixture classifier (ESLII 6.8 / Ch. 12) - CLASSIFICATION on ALL 200 Santander features (rank-gaussed).
Loss: per-class negative log-likelihood of a diagonal-covariance Gaussian mixture, -sum_i log sum_m alpha_m phi(x_i; mu_m, Sigma_m) (6.32), fitted by EM
(scikit-learn GaussianMixture); posterior by Bayes' theorem with the training class proportions. M (components per class) is the smoothing parameter.
Metrics: held-out log-likelihood per class and BIC / AIC vs M (reported in extra tables), test error +- SE, log-loss, AUC, calibration; M chosen by CV log-loss (one-SE rule).
"""
import _bootstrap  # noqa: F401
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common

NAME, TASK, INPUT = "gaussian_mixture_classifier", "classification", "x200"
MS = [1, 2, 3, 5, 8]


def make_grid(cfg, prep, Xs, ys):
    return [{"M": M} for M in MS]


def complexity(hp):
    return float(hp["M"])


def _fit(X, M, seed):
    from sklearn.mixture import GaussianMixture
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianMixture(n_components=M, covariance_type="diag", reg_covar=1e-4, max_iter=60, n_init=1, random_state=seed).fit(X)


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    prior = np.log(yref.mean() / (1 - yref.mean()))
    out = np.empty((len(grid), len(Xq)))
    for h, hp in enumerate(grid):
        g1, g0 = _fit(Xref[yref == 1], hp["M"], cfg.seed), _fit(Xref[yref == 0], hp["M"], cfg.seed)
        out[h] = common.sigmoid(np.clip(g1.score_samples(Xq) - g0.score_samples(Xq) + prior, -30, 30))
    return out


def extra(ctx):
    Xs, ys, folds, cfg = ctx["Xs"], ctx["ys_ev"], ctx["folds"], ctx["cfg"]
    rows = []
    for M in MS:
        r = {"M": M}
        for c in (0, 1):
            g = _fit(Xs[ys == c], M, cfg.seed)
            r[f"bic_class{c}"], r[f"aic_class{c}"] = g.bic(Xs[ys == c]), g.aic(Xs[ys == c])
            ll = []
            for idx in folds:
                tr = np.ones(len(ys), bool); tr[idx] = False
                held = Xs[idx][ys[idx] == c]
                ll.append(_fit(Xs[tr & (ys == c)], M, cfg.seed).score_samples(held).mean())
            r[f"heldout_loglik_class{c}"], r[f"heldout_loglik_se_class{c}"] = float(np.mean(ll)), float(np.std(ll, ddof=1) / np.sqrt(len(ll)))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(ctx["out"] / "mixture_components_selection.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for c in (0, 1):
        ax[0].errorbar(df["M"], df[f"heldout_loglik_class{c}"], yerr=df[f"heldout_loglik_se_class{c}"], fmt="o-", capsize=2, label=f"class {c}")
        ax[1].plot(df["M"], df[f"bic_class{c}"] - df[f"bic_class{c}"].min(), "o-", label=f"BIC class {c}"); ax[1].plot(df["M"], df[f"aic_class{c}"] - df[f"aic_class{c}"].min(), "s--", label=f"AIC class {c}")
    ax[0].set_xlabel("components M"); ax[0].set_ylabel("held-out log-likelihood"); ax[0].legend(); ax[1].set_xlabel("components M"); ax[1].set_ylabel("IC - min"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx["out"] / "mixture_components_selection.png", dpi=120); plt.close(fig)
    ctx["summary"]["M_by_bic"] = {f"class{c}": int(df.loc[df[f"bic_class{c}"].idxmin(), "M"]) for c in (0, 1)}
    ctx["summary"]["M_by_heldout_loglik"] = {f"class{c}": int(df.loc[df[f"heldout_loglik_class{c}"].idxmax(), "M"]) for c in (0, 1)}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    complexity=complexity, extra=extra, notes="Diagonal covariances; M = 1 is Gaussian naive Bayes with class-specific variances.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
