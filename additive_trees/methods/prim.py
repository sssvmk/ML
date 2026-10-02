"""
PRIM - patient rule induction / bump hunting (ESLII 9.3) - REGRESSION and CLASSIFICATION (0/1 response).
Objective (not squared error): find boxes B = {lo_j < x_j < hi_j} with a HIGH MEAN RESPONSE (regression: mean y; classification: class-1 proportion).
Top-down PEELING (remove the alpha-fraction slice, along one variable, that maximises the remaining mean) until support < beta0; the box on the peeling trajectory is chosen
by its mean on a HOLD-OUT 30 % of the training rows; then bottom-up PASTING; sequential covering: boxes are found one after another on the points not yet covered.
Induced predictor: x in the first box containing it -> that box's mean (estimated on the training rows), else the mean of the uncovered rows (piecewise constant).
Hyper-parameters (CV): k top variables, peel fraction alpha, minimum support beta0, number of boxes.
Metrics: box mean, support, lift (regression: mean - overall mean; classification: rate / base rate), sensitivity (classification), train/validation/test box means, number of rules and variables;
the piecewise-constant predictor's test MSE (regression) / error, AUC, log-loss (classification) are secondary.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common

NAME = "prim"
KS = [10, 25]
ALPHAS = [0.07, 0.15]
BETA0S = [0.05, 0.15]
NBOXES = [1, 2, 3, 5]
MAX_BOXES = max(NBOXES)


def make_grid(cfg, prep, Xs, ys):
    return [{"k": min(k, prep.p), "alpha": a, "beta0": b, "n_boxes": nb} for k in KS for a in ALPHAS for b in BETA0S for nb in NBOXES]


def complexity(hp):
    return hp["n_boxes"] + 0.001 * hp["k"]


def _peel(Xp, yp, alpha, beta0, max_remove=0.5):
    n, k = Xp.shape
    lo, hi, inbox = np.full(k, -np.inf), np.full(k, np.inf), np.ones(n, bool)
    traj = [(lo.copy(), hi.copy())]
    while True:
        idx = np.flatnonzero(inbox)
        m = len(idx)
        if m * (1 - alpha) < beta0 * n or m < 20:
            break
        sub, ys = Xp[idx], yp[idx]
        best, move = -np.inf, None
        for j in range(k):
            col = sub[:, j]
            ql, qh = np.quantile(col, [alpha, 1 - alpha])
            for side, q, keep in (("lo", ql, col > ql), ("hi", qh, col < qh)):
                c = int(keep.sum())
                if 0 < c < m and (m - c) / m <= max_remove:
                    mean = ys[keep].mean()
                    if mean > best:
                        best, move = mean, (j, side, q)
        if move is None:
            break
        j, side, q = move
        if side == "lo":
            lo[j] = q; inbox &= Xp[:, j] > q
        else:
            hi[j] = q; inbox &= Xp[:, j] < q
        traj.append((lo.copy(), hi.copy()))
    return traj


def _inside(X, lo, hi):
    return np.all((X > lo) & (X < hi), axis=1)


def _paste(Xp, yp, lo, hi, alpha, iters=15):
    n, k = Xp.shape
    lo, hi = lo.copy(), hi.copy()
    for _ in range(iters):
        ins = (Xp > lo) & (Xp < hi)
        oc = (~ins).sum(1)
        mask = oc == 0
        m = int(mask.sum())
        if m == 0:
            break
        s0 = yp[mask].sum()
        base, best = s0 / m, None
        add = max(1, int(np.ceil(alpha * m)))
        for j in range(k):
            cand = (~ins[:, j]) & (oc == 1)
            if not cand.any():
                continue
            v, yv = Xp[cand, j], yp[cand]
            for side in ("lo", "hi"):
                sel = v <= lo[j] if side == "lo" else v >= hi[j]
                if not sel.any():
                    continue
                vs, ys_ = v[sel], yv[sel]
                o = np.argsort(-vs if side == "lo" else vs)[:add]
                mean = (s0 + ys_[o].sum()) / (m + len(o))
                if mean > base and (best is None or mean > best[0]):
                    best = (mean, j, side, vs[o].min() if side == "lo" else vs[o].max())
        if best is None:
            break
        _, j, side, val = best
        if side == "lo":
            lo[j] = val - 1e-9
        else:
            hi[j] = val + 1e-9
    return lo, hi


def _fit(X, y, alpha, beta0, nbox, seed):
    rng = np.random.RandomState(seed)
    n = len(y)
    remaining = np.ones(n, bool)
    boxes, rest = [], []
    for b in range(nbox):
        idx = np.flatnonzero(remaining)
        if len(idx) < 60:
            break
        perm = rng.permutation(idx)
        nh = int(0.3 * len(perm))
        hold, peel = perm[:nh], perm[nh:]
        traj = _peel(X[peel], y[peel], alpha, beta0)
        hs, ok = [], []
        for lo, hi in traj:
            m = _inside(X[hold], lo, hi)
            ok.append(m.sum() >= max(10, 0.5 * beta0 * nh))
            hs.append(y[hold][m].mean() if m.any() else -np.inf)
        hs = np.where(ok, hs, -np.inf)
        lo, hi = traj[int(np.argmax(hs))] if np.isfinite(hs).any() else traj[0]
        lo, hi = _paste(X[peel], y[peel], lo, hi, alpha)
        m = remaining & _inside(X, lo, hi)
        if m.sum() < 10:
            break
        boxes.append({"lo": lo, "hi": hi, "mean": float(y[m].mean()), "support_of_remaining": float(m.sum() / len(idx)), "n": int(m.sum())})
        remaining &= ~m
        rest.append(float(y[remaining].mean()) if remaining.any() else float(y.mean()))
    return boxes, rest, float(y.mean())


def _predict(boxes, rest, base, X, nb):
    nb = min(nb, len(boxes))
    pred = np.full(len(X), base if nb == 0 else rest[nb - 1])
    done = np.zeros(len(X), bool)
    for b in range(nb):
        m = _inside(X, boxes[b]["lo"], boxes[b]["hi"]) & ~done
        pred[m] = boxes[b]["mean"]; done |= m
    return pred


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    order = ctx["order"]
    out = np.empty((len(grid), len(Xq)))
    keys = sorted({(hp["k"], hp["alpha"], hp["beta0"]) for hp in grid})
    for (k, a, b0) in keys:
        cols = order[:k]
        Xk, Xqk = np.asarray(Xref[:, cols], float), np.asarray(Xq[:, cols], float)
        boxes, rest, base = _fit(Xk, yref, a, b0, MAX_BOXES, cfg.seed)
        for h, hp in enumerate(grid):
            if (hp["k"], hp["alpha"], hp["beta0"]) == (k, a, b0):
                out[h] = _predict(boxes, rest, base, Xqk, hp["n_boxes"])
                if ctx.get("final"):
                    ctx["store"].update({"boxes": boxes[:hp["n_boxes"]], "rest": rest, "base": base, "cols": cols, "hp": hp})
                    ctx["store"]["summary"] = {"n_boxes_found": len(boxes), "box_means": [bx["mean"] for bx in boxes[:hp["n_boxes"]]]}
    return out


def extra(ctx):
    S = ctx["store"]
    boxes, cols, task = S["boxes"], S["cols"], ctx["prep"].task
    names = np.array(ctx["prep"].feature_names)[cols]
    reg = task == "regression"
    splits = {"train": (np.asarray(ctx["X_tr"][:, cols], float), ctx["y_tr"].astype(float)), "val": (np.asarray(ctx["X_va"][:, cols], float), ctx["y_va"].astype(float)),
              "test": (np.asarray(ctx["X_te"][:, cols], float), ctx["y_te"].astype(float))}
    rows = []
    covered = {s: np.zeros(len(v[1]), bool) for s, v in splits.items()}
    base = {s: float(v[1].mean()) for s, v in splits.items()}
    for b, bx in enumerate(boxes):
        r = {"box": b + 1, "n_rules": int(np.sum(np.isfinite(bx["lo"])) + np.sum(np.isfinite(bx["hi"]))), "n_variables": int(np.sum(np.isfinite(bx["lo"]) | np.isfinite(bx["hi"])))}
        for s, (X, y) in splits.items():
            m = _inside(X, bx["lo"], bx["hi"]) & ~covered[s]
            covered[s] |= m
            r[f"support_{s}"] = float(m.mean()); r[f"mean_{s}"] = float(y[m].mean()) if m.any() else np.nan
            r[f"lift_{s}"] = r[f"mean_{s}"] - base["train"] if reg else (r[f"mean_{s}"] / base["train"] if base["train"] > 0 else np.nan)
            if not reg:
                r[f"sensitivity_{s}"] = float(y[m].sum() / max(y.sum(), 1))
        r["rule"] = " AND ".join(f"{names[j]} > {bx['lo'][j]:.3g}" if np.isfinite(bx["lo"][j]) and not np.isfinite(bx["hi"][j]) else
                                 f"{names[j]} < {bx['hi'][j]:.3g}" if np.isfinite(bx["hi"][j]) and not np.isfinite(bx["lo"][j]) else f"{bx['lo'][j]:.3g} < {names[j]} < {bx['hi'][j]:.3g}"
                                 for j in range(len(cols)) if np.isfinite(bx["lo"][j]) or np.isfinite(bx["hi"][j]))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(ctx["out"] / "prim_boxes.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.25
    for i, s in enumerate(("train", "val", "test")):
        ax.bar(df["box"] + (i - 1) * w, df[f"mean_{s}"], w, label=f"box mean ({s})")
    ax.axhline(base["train"], color="k", ls="--", label="overall train mean"); ax.set_xlabel("box (sequential covering)"); ax.set_ylabel("mean response" if reg else "class-1 rate"); ax.legend(fontsize=8)
    ax.set_title("PRIM boxes: train vs validation vs test")
    fig.tight_layout(); fig.savefig(ctx["out"] / "prim_box_means.png", dpi=120); plt.close(fig)
    ctx["summary"]["prim_boxes"] = df.drop(columns=["rule"]).to_dict(orient="records")


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path, complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "auc", notes="Peeling + pasting + sequential covering; box chosen on a 30 % hold-out of the training rows.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
