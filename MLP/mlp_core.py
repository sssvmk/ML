"""
MLP core: data loading, model, metrics and the training loop.

Design follows Deep Learning book, Chapter 6:
  hidden units  : fully connected + ReLU
  output unit   : MNIST -> 10 linear logits (softmax lives inside CrossEntropyLoss)
                  Housing -> 1 linear unit
  cost function : MNIST -> cross-entropy ; Housing -> mean squared error
  optimizer     : minibatch SGD + momentum (or AdamW), exponentially decaying LR
  initialization: He (Kaiming) normal for hidden layers, small positive bias

This module has no MLflow dependency; tracking.py / run.py attach to it through the
`on_epoch` callback of fit(). infer.py and the packaged MLflow model import it too.
"""
import copy
import hashlib
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

DEFAULTS = {
    "mnist": dict(hidden=[256, 128], optimizer="sgd", lr=0.05, momentum=0.9,
                  batch_size=128, epochs=30, gamma=0.95, weight_decay=1e-4,
                  dropout=0.0, patience=5),
    "housing": dict(hidden=[64, 32], optimizer="sgd", lr=0.01, momentum=0.9,
                    batch_size=64, epochs=100, gamma=0.97, weight_decay=1e-5,
                    dropout=0.0, patience=10),
    # any tabular data source (CSV, Parquet, a database, ... see datasources.py)
    "tabular": dict(hidden=[128, 64], optimizer="sgd", lr=0.01, momentum=0.9,
                batch_size=256, epochs=60, gamma=0.97, weight_decay=1e-5,
                dropout=0.0, patience=8),
}
PRIMARY = {"classification": "accuracy", "regression": "rmse"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def subset(ds, n):
    """First n rows of a TensorDataset (splits are already shuffled)."""
    if n is None or n >= len(ds):
        return ds
    return TensorDataset(*[t[:n] for t in ds.tensors])


def fingerprint(splits, rows=5000):
    """Short hash of split sizes + the first `rows` rows of each split (lineage tag)."""
    h = hashlib.sha256()
    for name in sorted(splits):
        h.update(name.encode())
        h.update(str(len(splits[name])).encode())
        for t in splits[name].tensors:
            h.update(np.ascontiguousarray(t[:rows].cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def _load_mnist(root, seed):
    """MNIST from torchvision: 60k train -> 55k train + 5k validation; 10k test."""
    from torchvision import datasets

    train_full = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)

    def prep(ds):  # uint8 [N,28,28] -> normalized float [N,784]
        x = ds.data.float().div(255.0).sub(MNIST_MEAN).div(MNIST_STD)
        return x.view(len(ds), -1), ds.targets.long()

    from splits import carve_validation

    x, y = prep(train_full)
    x_te, y_te = prep(test)
    tr_idx, va_idx = carve_validation(len(x), y.numpy(), val_size=5000, seed=seed, stratify=True)
    tr_idx, va_idx = torch.from_numpy(tr_idx), torch.from_numpy(va_idx)
    splits = {
        "train": TensorDataset(x[tr_idx], y[tr_idx]),
        "val": TensorDataset(x[va_idx], y[va_idx]),
        "test": TensorDataset(x_te, y_te),
    }
    split_info = {
        "presplit": True,
        "strategy": (f"PRE-SPLIT by the dataset: official {len(x):,} train / {len(x_te):,} test (test set "
                     f"kept untouched). No validation set ships with MNIST, so {len(va_idx):,} rows were "
                     f"carved out of the official train set (stratified by digit, seed {seed}); "
                     f"{len(tr_idx):,} rows remain for training."),
    }
    info = dict(
        task="classification", in_dim=784, out_dim=10, y_mean=0.0, y_std=1.0,
        baseline={}, class_names=[str(i) for i in range(10)],
        preproc={"type": "mnist", "mean": MNIST_MEAN, "std": MNIST_STD},
        source="torchvision.datasets.MNIST", split_info=split_info,
    )
    return splits, info


def _load_housing(seed):
    """California Housing from scikit-learn: 60/20/20 split, scalers fit on train only."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from splits import split_three_way

    try:
        data = fetch_california_housing()
    except OSError as e:  # urllib's HTTPError / URLError are OSErrors
        raise RuntimeError(
            f"could not download California Housing ({type(e).__name__}: {e}). The host scikit-learn downloads "
            f"it from (figshare) has been refusing scripted downloads with HTTP 403. Options: (1) use another "
            f"copy of the data as your own file, e.g. --data housing.csv --task regression --target "
            f"median_house_value (a different column layout from scikit-learn's version, so results are not "
            f"comparable); (2) copy scikit-learn's cache file cal_housing_py3.pkz into ~/scikit_learn_data "
            f"(Windows: C:\\Users\\<you>\\scikit_learn_data), but only from a source you trust, because it is "
            f"a pickle file and loading an untrusted pickle can run arbitrary code.") from e
    X, y = data.data, data.target  # 20,640 x 8 ; target in units of $100k
    names = list(data.feature_names)
    tr_i, va_i, te_i = split_three_way(len(X), y, val_size=0.2, test_size=0.2, seed=seed)
    X_tr, X_va, X_te = X[tr_i], X[va_i], X[te_i]
    y_tr, y_va, y_te = y[tr_i], y[va_i], y[te_i]
    split_info = {
        "presplit": False,
        "strategy": (f"NOT pre-split: the dataset is a single table of {len(X):,} rows. The pipeline splits it "
                     f"{len(tr_i):,} train / {len(va_i):,} validation / {len(te_i):,} test "
                     f"(60/20/20, random, seed {seed}). Feature and target scalers are fit on the "
                     f"training rows only."),
    }

    xs = StandardScaler().fit(X_tr)
    ys = StandardScaler().fit(y_tr.reshape(-1, 1))

    # Reference points on the same split (test RMSE, original units).
    lin = LinearRegression().fit(xs.transform(X_tr), y_tr)
    lin_pred = lin.predict(xs.transform(X_te))
    baseline = {
        "linear_regression_test_mse": float(np.mean((lin_pred - y_te) ** 2)),
        "linear_regression_test_rmse": float(np.sqrt(np.mean((lin_pred - y_te) ** 2))),
        "predict_mean_test_rmse": float(np.sqrt(np.mean((y_tr.mean() - y_te) ** 2))),
    }

    def tens(Xa, ya):
        xt = torch.tensor(xs.transform(Xa), dtype=torch.float32)
        yt = torch.tensor(ys.transform(ya.reshape(-1, 1)), dtype=torch.float32)
        return TensorDataset(xt, yt)

    splits = {"train": tens(X_tr, y_tr), "val": tens(X_va, y_va), "test": tens(X_te, y_te)}
    info = dict(
        task="regression", in_dim=X.shape[1], out_dim=1,
        y_mean=float(ys.mean_[0]), y_std=float(ys.scale_[0]),
        baseline=baseline, class_names=None,
        preproc={
            "type": "housing", "feature_names": names,
            "x_mean": xs.mean_.tolist(), "x_scale": xs.scale_.tolist(),
            "y_mean": float(ys.mean_[0]), "y_std": float(ys.scale_[0]),
        },
        source="sklearn.datasets.fetch_california_housing", split_info=split_info,
        target_label="median house value ($100k)", money_scale=1e5,
        baseline_test_pred=lin_pred,  # original units, same row order as the test split
    )
    return splits, info


def make_loaders(splits, batch_size):
    return {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(splits["val"], batch_size=1024),
        "test": DataLoader(splits["test"], batch_size=1024),
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_mlp(in_dim, hidden, out_dim, dropout=0.0):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))  # linear output; the loss supplies the nonlinearity
    model = nn.Sequential(*layers)

    linears = [m for m in model if isinstance(m, nn.Linear)]
    for m in linears[:-1]:  # hidden layers: He init + small positive bias (keeps ReLUs active)
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.constant_(m.bias, 0.01)
    nn.init.xavier_uniform_(linears[-1].weight)
    nn.init.zeros_(linears[-1].bias)
    return model


def make_optimizer(model, cfg):
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                               momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError(f"unknown optimizer {cfg['optimizer']!r}")


def make_loss(task):
    return nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()


# --------------------------------------------------------------------------- #
# Metrics and epoch loop
# --------------------------------------------------------------------------- #
def compute_metrics(task, outs, targets, y_mean, y_std, with_auc=False):
    """Cheap per-epoch metrics from raw model outputs.

    classification: outs are logits -> accuracy (and macro one-vs-rest AUC if with_auc)
    regression    : outs/targets are standardized -> RMSE, MAE, R2 in original units
    The full test-set report (AUC with CI, MSE +- SE, ...) lives in metrics.py.
    """
    if task == "classification":
        m = {"accuracy": (outs.argmax(1) == targets).float().mean().item()}
        if with_auc:
            from metrics import macro_auc
            m["auc"] = macro_auc(torch.softmax(outs, 1).numpy(), targets.numpy())[0]
        return m
    preds = outs
    p = preds * y_std + y_mean  # back to original units ($100k)
    t = targets * y_std + y_mean
    err = p - t
    sse = err.pow(2).sum().item()
    sst = (t - t.mean()).pow(2).sum().item()
    return {"rmse": math.sqrt(sse / len(t)), "mae": err.abs().mean().item(),
            "r2": 1.0 - sse / sst}


def run_epoch(model, loader, loss_fn, device, task, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    P, T = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = x.size(0)
            total += loss.item() * bs
            n += bs
            P.append(out.detach().cpu())  # logits (classification) or standardized predictions
            T.append(y.cpu())
    return total / n, torch.cat(P), torch.cat(T)


def fit(model, loaders, cfg, info, device, trial=None, on_epoch=None, verbose=True):
    """Train with early stopping on validation loss.

    on_epoch(row)  : optional callback invoked after every epoch (used for MLflow logging).
    trial          : optional Optuna trial; validation loss is reported each epoch and the
                     trial is pruned when it falls behind earlier trials.
    """
    task, key = info["task"], PRIMARY[info["task"]]
    loss_fn = make_loss(task)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["gamma"])

    rows, bad, diverged = [], 0, False
    best = dict(val_loss=float("inf"), epoch=0, state=None)
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tp, tt = run_epoch(model, loaders["train"], loss_fn, device, task, optimizer)
        va_loss, vp, vt = run_epoch(model, loaders["val"], loss_fn, device, task)
        if not (math.isfinite(tr_loss) and math.isfinite(va_loss)):
            diverged = True  # loss blew up (learning rate too high)
            break
        tr_m = compute_metrics(task, tp, tt, info["y_mean"], info["y_std"])
        va_m = compute_metrics(task, vp, vt, info["y_mean"], info["y_std"], with_auc=True)
        scheduler.step()

        row = {"epoch": epoch, "lr": lr, "train_loss": tr_loss, "val_loss": va_loss}
        row.update({f"train_{k}": v for k, v in tr_m.items()})
        row.update({f"val_{k}": v for k, v in va_m.items()})
        rows.append(row)
        if on_epoch is not None:
            on_epoch(row)
        if verbose:
            print(f"epoch {epoch:3d} | lr {lr:.5f} | train loss {tr_loss:.4f} {key} {tr_m[key]:.4f} "
                  f"| val loss {va_loss:.4f} {key} {va_m[key]:.4f} | {time.time() - t0:.1f}s")

        if va_loss < best["val_loss"] - 1e-6:
            best.update(val_loss=va_loss, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best epoch {best['epoch']})")
                break

        if trial is not None:
            trial.report(va_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    return dict(rows=rows, best_epoch=best["epoch"], best_val_loss=best["val_loss"],
                best_state=best["state"], diverged=diverged)
