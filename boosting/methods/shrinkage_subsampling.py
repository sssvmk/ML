"""
Shrinkage and subsampling = stochastic gradient boosting (ESLII 10.12), via scikit-learn GradientBoosting*:
  update f_m = f_{m-1} + nu * sum_j gamma_jm 1(x in R_jm)   (shrinkage nu in (0, 1]);  every tree is grown on a random fraction eta of the rows, without replacement (eta ~ 1/2).
  loss: squared error (regression) / binomial deviance (classification).
Hyper-parameters (CV, random search): nu in {0.5, 0.2, 0.1, 0.05}, eta in {1, 0.5, 0.3}, J, M, fraction of the ranked variables kept (all unless dropping helps).
Metrics: test MSE (deviance, error, AUC) vs M for several (nu, eta), CV over (nu, eta, M) with the one-SE rule, training time per configuration.
"""
import _bootstrap  # noqa: F401
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
import boostlib as bl

NAME = "shrinkage_subsampling"
M_GRID = [20, 50, 100, 150, 200]


def build(conf, M, task, cfg, y):
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    kw = dict(learning_rate=conf["nu"], subsample=conf["eta"], n_estimators=M, max_leaf_nodes=conf["J"], random_state=cfg.seed)
    return GradientBoostingRegressor(loss="squared_error", **kw) if task == "regression" else GradientBoostingClassifier(loss="log_loss", **kw)


def make_grid(cfg, prep, Xs, ys):
    space = {"frac": bl.FRACS, "nu": [0.5, 0.2, 0.1, 0.05], "eta": [1.0, 0.5, 0.3], "J": [4, 8]}
    return bl.expand(bl.sample_configs(space, cfg.n_configs_sk, cfg.seed), M_GRID, prep.p, cfg.max_iter)


def complexity(hp):
    return bl.drop_penalty(hp) + hp["M"] * hp["J"] * hp["nu"]


def extra(ctx):
    bl.standard_extra(ctx, group_col="nu")
    S, cfg = ctx["store"], ctx["cfg"]
    cols, task, conf = S["cols"], S["task"], S["conf"]
    r0 = bl.cap_rows(len(ctx["ys_fit"]), cfg.diag_rows)
    Xs = np.asarray(ctx["Xs"][r0][:, cols], np.float32)
    ys = ctx["ys_fit"][r0] if task == "regression" else ctx["ys_ev"][r0].astype(int)
    rows = np.random.RandomState(0).choice(len(ctx["y_te"]), min(10000, len(ctx["y_te"])), replace=False)
    Xte, yte = np.asarray(ctx["X_te"][rows][:, cols], np.float32), ctx["y_te"][rows]
    Mmax = max(bl.grid_m(M_GRID, cfg.max_iter))
    out, fig, ax = [], *plt.subplots(figsize=(7.5, 4.6))
    for nu, eta in [(1.0, 1.0), (0.1, 1.0), (0.1, 0.5), (0.05, 0.5)]:
        c = {**conf, "nu": nu, "eta": eta}
        m = build(c, Mmax, task, cfg, ys)
        t0 = time.time(); m.fit(Xs, ys); secs = time.time() - t0
        Ms = list(range(5, Mmax + 1, 5))
        pr = bl.predict_stages(m, "sk", Xte, Ms, task)
        met = [float(np.mean((yte - pr[M]) ** 2)) if task == "regression" else float(common.logloss_vec(yte, pr[M]).mean()) for M in Ms]
        ax.plot(Ms, met, label=f"nu={nu}, eta={eta} ({secs:.0f}s)")
        out += [{"nu": nu, "eta": eta, "M": M, "test_metric": v, "fit_seconds": secs} for M, v in zip(Ms, met)]
    ax.set_xlabel("iterations M"); ax.set_ylabel("test MSE" if task == "regression" else "test deviance (log-loss)"); ax.legend(fontsize=8); ax.set_title("Shrinkage and subsampling (J = %d)" % conf["J"])
    fig.tight_layout(); fig.savefig(ctx["out"] / "shrinkage_subsampling_curves.png", dpi=120); plt.close(fig)
    pd.DataFrame(out).to_csv(ctx["out"] / "shrinkage_subsampling_curves.csv", index=False)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=bl.make_predict_path(build, "sk"), complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "auc", notes="scikit-learn GradientBoosting with learning_rate = nu and subsample = eta.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
