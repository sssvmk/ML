"""
Kernel density classification (ESLII 6.6.1-6.6.2) - CLASSIFICATION on the top-8 Santander features.
No single loss: each class density is a Gaussian-kernel estimate f_j(x) = (1/N_j) sum_i phi_h(x - x_i), then Pr(G = 1 | x) = pi_1 f_1 / (pi_0 f_0 + pi_1 f_1)  (6.25),
priors = training proportions. Bandwidth h (isotropic, in the rank-gaussed unit scale) is the smoothing parameter, selected by CV log-loss (one-SE rule: smoothest);
the per-class HELD-OUT DENSITY LOG-LIKELIHOOD curves (the criterion for density estimation) are reported alongside.
Metrics: test error +- SE, log-loss, AUC, calibration, held-out density log-likelihood per class, CV error.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
import kern

NAME, TASK, INPUT = "kernel_density_classifier", "classification", "x8"
HS = [0.15, 0.2, 0.3, 0.4, 0.55, 0.8]


def make_grid(cfg, prep, Xs, ys):
    return [{"h": h} for h in HS]


def complexity(hp):
    return 1.0 / hp["h"]


def log_density(Xc, Xq, hs, max_elems=2.5e7):
    """log f(x) for every h in hs: Gaussian product-free isotropic kernel, exact (chunked)."""
    n, p = Xc.shape
    out = np.empty((len(hs), len(Xq)))
    Xc32, sq = Xc.astype(np.float32), (Xc.astype(np.float32) ** 2).sum(1)
    q = int(max(1, max_elems // n))
    for a in range(0, len(Xq), q):
        xq = Xq[a:a + q].astype(np.float32)
        d2 = np.maximum((xq ** 2).sum(1)[:, None] + sq[None] - 2 * xq @ Xc32.T, 0)
        for i, h in enumerate(hs):
            e = -d2 / (2 * h * h)
            m = e.max(1, keepdims=True)
            out[i, a:a + len(xq)] = (m[:, 0] + np.log(np.exp(e - m).sum(1, dtype=np.float64))) - np.log(n) - 0.5 * p * np.log(2 * np.pi * h * h)
    return out


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    hs = [hp["h"] for hp in grid]
    f1 = log_density(Xref[yref == 1], Xq, hs)
    f0 = log_density(Xref[yref == 0], Xq, hs)
    prior = np.log(yref.mean() / (1 - yref.mean()))
    return kern.sigmoid(np.clip(f1 - f0 + prior, -30, 30))


def extra(ctx):
    Xs, ys, folds = ctx["Xs"], ctx["ys_ev"], ctx["folds"]
    hs = [hp["h"] for hp in ctx["grid"]]
    ll = {c: np.zeros((len(folds), len(hs))) for c in (0, 1)}
    for k, idx in enumerate(folds):
        tr = np.ones(len(ys), bool); tr[idx] = False
        for c in (0, 1):
            held = Xs[idx][ys[idx] == c]
            ll[c][k] = log_density(Xs[tr & (ys == c)], held, hs).mean(1)
    rows = pd.DataFrame({"h": hs, "heldout_loglik_class0": ll[0].mean(0), "se0": ll[0].std(0, ddof=1) / np.sqrt(len(folds)),
                         "heldout_loglik_class1": ll[1].mean(0), "se1": ll[1].std(0, ddof=1) / np.sqrt(len(folds))})
    rows.to_csv(ctx["out"] / "class_density_heldout_loglik.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in (0, 1):
        ax.errorbar(hs, rows[f"heldout_loglik_class{c}"], yerr=rows[f"se{c}"], fmt="o-", capsize=2, label=f"class {c}")
    ax.set_xlabel("bandwidth h"); ax.set_ylabel("held-out density log-likelihood"); ax.legend(); ax.set_title("KDE bandwidth: held-out log-likelihood")
    fig.tight_layout(); fig.savefig(ctx["out"] / "class_density_heldout_loglik.png", dpi=120); plt.close(fig)
    ctx["summary"]["best_h_by_density_loglik"] = {"class0": hs[int(np.argmax(rows.heldout_loglik_class0))], "class1": hs[int(np.argmax(rows.heldout_loglik_class1))]}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    complexity=complexity, extra=extra, notes="Exact Gaussian KDE per class in R^8, isotropic bandwidth.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
