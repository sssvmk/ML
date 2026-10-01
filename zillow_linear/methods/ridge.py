"""Ridge regression.  Objective: RSS + lam * sum b_j^2.   Metrics: 10-fold CV MSE vs lam (one-SE), GCV, effective df."""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

import common
from common import Path_

NAME = "ridge"


def make_grid(bundle, prep, cfg):
    # lam is on the RSS scale, so scale the grid with n (large lam first = least complex first)
    return bundle.tr_fit.n * np.geomspace(10.0, 1e-7, cfg.n_lambda)


def fit_path(st, lams, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    w, V = np.linalg.eigh(Gc)
    w = np.clip(w, 0, None)
    z = V.T @ cc
    B = V @ (z[:, None] / (w[:, None] + lams[None, :]))
    df = (w[:, None] / (w[:, None] + lams[None, :])).sum(0)
    return Path_(B, ybar - xbar @ B, -np.log(lams), lam=lams, df=df)


def extra_curve(curve, path, bundle):
    n = bundle.tr_fit.n
    df_eff = curve["df"].values + 1.0
    curve["gcv"] = (curve["train_rss"].values / n) / (1 - df_eff / n) ** 2
    curve["comp_plot"] = curve["lam"]
    return curve


def final_extras(beta, b0, st, bundle):
    # effective df of the final model's lambda is reported in summary; GCV minimiser reported from the curve
    return {}


def extra_plot(out, curve, summary):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(curve["lam"], curve["gcv"], "o-", ms=3)
    ax[0].set_xscale("log"); ax[0].set_xlabel("lambda"); ax[0].set_ylabel("GCV"); ax[0].set_title("Generalised CV")
    ax[1].plot(curve["lam"], curve["df"], "o-", ms=3)
    ax[1].set_xscale("log"); ax[1].set_xlabel("lambda"); ax[1].set_ylabel("effective df"); ax[1].set_title("df(lambda)")
    fig.tight_layout(); fig.savefig(out / "gcv_and_df.png", dpi=130); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="lambda (small = complex)", comp_log=True, extra_curve=extra_curve,
                                  final_extras=final_extras, extra_plot=extra_plot,
                                  notes="Complexity ordering = -log(lambda); x-axis of CV plot is lambda.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
