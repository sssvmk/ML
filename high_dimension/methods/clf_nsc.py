"""
Nearest shrunken centroids (classification) - scikit-learn NearestCentroid(shrink_threshold=Delta), the Tibshirani et al. method: class centroids are standardised by the pooled within-class sd plus the median s0,
soft-thresholded by Delta (equivalent to a lasso-penalised naive-Bayes model, 18.55) and a sample goes to the nearest shrunken centroid. Genes whose centroid deviations are all shrunk to zero are dropped.
Loss: squared standardised distance to the centroid (with log-prior offsets). Tuning: 25 values of Delta by repeated stratified CV, one-SE rule (largest Delta). Probabilities are scikit-learn's softmax of the negative distances.
Metrics: test error +- SE, CV error vs Delta, number of genes surviving (vs Delta), log-loss, macro one-vs-rest AUC, the surviving genes with their centroid shifts.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "nsc", "classification"
GRID = {"nsc__shrink_threshold": [round(v, 3) for v in np.linspace(0.1, 6.0, 25)]}


def make_est(wins, out):
    from sklearn.neighbors import NearestCentroid
    return hl.make_pipeline(hl.prefix_steps(wins, scale=False) + [("nsc", NearestCentroid())], out)


def surviving(model):
    nc = model.named_steps["nsc"]
    return np.ptp(nc.centroids_, axis=0) > 1e-12                              # a gene survives iff its shrunken class centroids still differ (all shrunk to the overall centroid otherwise)


def n_features(model, p):
    return int(surviving(model).sum())


def extra(ctx):
    from sklearn.base import clone
    m, out = ctx["model"], ctx["out"]
    keep = np.where(m.named_steps["var"].get_support())[0]
    surv = surviving(m)
    nc = m.named_steps["nsc"]
    shift = nc.centroids_ - m[:-1].transform(ctx["Xf"]).mean(0)               # shrunken centroid minus the overall (training) centroid
    genes = np.array(ctx["genes"])[keep]
    df = pd.DataFrame(shift.T, columns=[f"centroid_shift_class{c}" for c in nc.classes_]); df.insert(0, "gene", genes)
    df = df[surv].assign(max_abs_shift=lambda d: d.filter(like="centroid_shift").abs().max(1)).sort_values("max_abs_shift", ascending=False)
    df.to_csv(out / "surviving_genes_and_centroid_shifts.csv", index=False)
    ng, tr = [], []
    for dlt in GRID["nsc__shrink_threshold"]:
        mm = clone(ctx["est"]).set_params(nsc__shrink_threshold=dlt).fit(ctx["Xf"], ctx["yf"])
        ng.append(int(surviving(mm).sum())); tr.append(float(np.mean(mm.predict(ctx["Xf"]) != ctx["yf"])))
    cv = ctx["cv"].assign(delta=[p["nsc__shrink_threshold"] for p in ctx["cv"]["params"]]).sort_values("delta")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(cv["delta"], cv["cv_loss"], yerr=cv["cv_se"], marker="o", ms=3, label="CV error"); ax[0].plot(GRID["nsc__shrink_threshold"], tr, "s--", ms=3, label="training error")
    ax[0].axvline(ctx["best"]["nsc__shrink_threshold"], color="k", ls=":"); ax[0].set_xlabel("shrinkage threshold Delta"); ax[0].set_ylabel("misclassification error"); ax[0].legend(); ax[0].set_title("NSC: error vs Delta (dotted: one-SE choice)")
    ax[1].plot(GRID["nsc__shrink_threshold"], ng, "o-", ms=3); ax[1].set_yscale("log"); ax[1].set_xlabel("Delta"); ax[1].set_ylabel("genes surviving"); ax[1].set_title("NSC: genes used vs Delta")
    fig.tight_layout(); fig.savefig(out / "nsc_error_and_genes_vs_delta.png", dpi=120); plt.close(fig)
    ctx["summary"]["nsc"] = {"delta": ctx["best"]["nsc__shrink_threshold"], "genes_surviving": int(surv.sum())}


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, lambda p: -p["nsc__shrink_threshold"], n_features, extra, notes="scikit-learn NearestCentroid(shrink_threshold)")
