"""
Elastic net.  Objective: RSS + lam * [ alpha * sum b_j^2 + (1 - alpha) * sum |b_j| ]   (ESLII 3.91; alpha weights L2).
Solver: pathwise coordinate descent. Metrics: 10-fold CV MSE over the (alpha, lam) grid.
Selection: global CV minimum fixes alpha*; then the one-SE rule picks the largest lam at that alpha.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

import common
from common import Path_
from solvers import enet_path_abs, lam_max_abs

NAME = "elastic_net"


def make_grid(bundle, prep, cfg):
    return {"alphas": list(cfg.enet_alphas),
            "lams": {a: lam_max_abs(bundle.tr_fit, a) * np.geomspace(1.0, 1e-4, cfg.n_lambda) for a in cfg.enet_alphas}}


def fit_path(st, grid, cfg):
    Bs, b0s, al, lm, comp = [], [], [], [], []
    for a in grid["alphas"]:
        B, b0 = enet_path_abs(st, grid["lams"][a], a)
        Bs.append(B); b0s.append(b0)
        al.append(np.full(B.shape[1], a)); lm.append(grid["lams"][a])
        l1 = np.abs(B).sum(0)
        comp.append(l1 / max(l1[-1], 1e-300))
    return Path_(np.hstack(Bs), np.concatenate(b0s), np.concatenate(comp), alpha=np.concatenate(al), lam=np.concatenate(lm))


def select(curve, path):
    m, se = curve["cv_mse"].values, curve["cv_se"].values
    i_min = int(np.nanargmin(m))
    a = curve["alpha"].values[i_min]
    ok = np.where((curve["alpha"].values == a) & (m <= m[i_min] + se[i_min]))[0]
    return int(ok[np.argmax(curve["lam"].values[ok])]), i_min


def fit_one(st, grid, cfg, i, path):
    a, lam = float(path.extra["alpha"][i]), float(path.extra["lam"][i])
    lams = grid["lams"][a]
    j = int(np.argmin(np.abs(lams - lam)))
    B, b0 = enet_path_abs(st, lams[: j + 1], a)
    return B[:, -1], float(b0[-1])


def extra_plot(out, curve, summary):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for a, g in curve.groupby("alpha"):
        ax.plot(g["lam"], g["cv_mse"], label=f"alpha={a}")
    ax.set_xscale("log"); ax.set_xlabel("lambda"); ax.set_ylabel("10-fold CV MSE"); ax.legend(fontsize=8)
    ax.set_title("Elastic net: CV MSE over the (alpha, lambda) grid")
    fig.tight_layout(); fig.savefig(out / "cv_by_alpha.png", dpi=130); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="shrinkage factor s (within alpha)", select=select, fit_one=fit_one,
                                  extra_plot=extra_plot, notes=f"alphas={list(cfg.enet_alphas)} (alpha weights the L2 term).",
                                  console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
