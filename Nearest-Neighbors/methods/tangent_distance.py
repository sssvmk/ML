"""
Tangent distance (ESLII 13.3.3) - CLASSIFICATION, as agreed INCLUDED AS WRITTEN: Santander (and Zillow) variables have no known small transformations (the rotations / shifts that tangent distance makes the
metric invariant to for images), so the tangent set is EMPTY and the metric reduces to the Euclidean distance. The entry therefore reproduces the Euclidean k-NN row; this is by construction, not a coincidence.
The real tangent-distance machinery is implemented and unit-tested in nnlib.py (tangent_distance: min over a, b of ||(x + T_x a) - (x' + T_x' b)||; tangent_projection: projecting a global tangent space out
of the data) - it is exercised in tests on shifted signals - and plugs into this entry through `TANGENTS` if a tangent matrix (p, m) is ever available.
Loss: the implicit 0-1 loss of k-NN with the invariant metric. Hyper-parameters (CV): k, number of top-ranked variables, set of transformations (none) and tangent-space dimension (0). CV selects on AUC.
Metrics: test error +- SE vs plain Euclidean k-NN (identical here: the run asserts that the tangent metric equals the Euclidean one on the data), CV error over (k, transformations, tangent dimension).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np

import common
import nnlib
import knn

NAME = "tangent"
TANGENTS = None                                  # tangent matrix (p, m) of the known small transformations; None for tabular data
PROJECT = nnlib.tangent_projection(TANGENTS)     # None = identity


def make_grid(cfg, prep, Xs, ys):
    return [{**g, "tdim": 0, "transforms": "none"} for g in knn.make_grid(cfg, prep, Xs, ys)]


def extra(ctx):
    S = ctx["store"]
    X = S["Xr"][:200]
    d_t = [nnlib.tangent_distance(X[i], X[i + 1], None, None) for i in range(100)]
    d_e = [float(np.linalg.norm(X[i + 1] - X[i])) for i in range(100)]
    same = bool(np.allclose(d_t, d_e))
    ctx["summary"]["tangent"] = {"n_tangent_vectors": 0, "tangent_metric_equals_euclidean_on_this_data": same, "chosen_k": S["hp"]["nn"], "n_variables": S["hp"]["k"],
                                 "note": "empty tangent set: identical to Euclidean k-NN by construction"}
    assert same, "empty tangent set must reproduce the Euclidean distance"
    knn.extra(ctx)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("Tangent-distance k-NN is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=nnlib.make_predict_path("distance", 2, project=PROJECT), complexity=knn.complexity, extra=extra,
                             select_by="auc", notes="k-NN with the tangent metric; the tangent set is empty on tabular data -> Euclidean.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
