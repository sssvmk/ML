"""
hd_data.py - loading, the splits and the preparation bundle.
regression    Riboflavin CSV (column y first, then the genes; written from the R package hdi). Random split into train / validation / test = n-24 / 12 / 12 (48/12/12 for 72 rows, 47/12/12 for the
              71 rows of the real data), stratified on quantile bins of y so that every part covers the range of the response.
classification SRBCT: either the CSV (genes + `class` + `split` columns) or the keyword ISLP (pip install ISLP; Khan et al. data). The book's split is used: 63 training and 20 test samples
              (no validation set: the training samples are tuned by repeated stratified CV).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_riboflavin(path):
    df = pd.read_csv(path)
    ycol = "y" if "y" in df.columns else df.columns[0]
    y = df[ycol].values.astype(float)
    X = df.drop(columns=[ycol]).select_dtypes("number")
    return X.values.astype(float), y, list(X.columns)


def load_srbct(path=None):
    if path in (None, "ISLP", "islp"):
        from ISLP import load_data
        K = load_data("Khan")
        Xtr, Xte = np.asarray(K["xtrain"], float), np.asarray(K["xtest"], float)
        ytr, yte = np.asarray(K["ytrain"]).ravel().astype(int), np.asarray(K["ytest"]).ravel().astype(int)
        genes = [f"g{j + 1}" for j in range(Xtr.shape[1])]
        return Xtr, ytr, Xte, yte, genes
    df = pd.read_csv(path)
    ycol = next(c for c in ("class", "Class", "Y", "y", "label", "target") if c in df.columns)
    tr, te = df["split"].astype(str).str.lower().eq("train").values, df["split"].astype(str).str.lower().eq("test").values
    X = df.drop(columns=[ycol, "split"]).select_dtypes("number")
    y = df[ycol].values.astype(int)
    return X.values[tr].astype(float), y[tr], X.values[te].astype(float), y[te], list(X.columns)


def split_regression(y, n_val=12, n_test=12, seed=0, stratify=True):
    n = len(y)
    rng = np.random.RandomState(seed)
    if stratify:
        bins = pd.qcut(pd.Series(y).rank(method="first"), q=n_val, labels=False).values     # n_val bins of ~equal size: one validation and one test point per bin
        va, te = [], []
        for b in range(n_val):
            idx = rng.permutation(np.where(bins == b)[0])
            te.append(idx[0]); va.append(idx[1])
        va, te = np.array(va), np.array(te)
        if len(te) != n_test:
            te = te[:n_test]
    else:
        perm = rng.permutation(n)
        te, va = perm[:n_test], perm[n_test:n_test + n_val]
    tr = np.setdiff1d(np.arange(n), np.r_[va, te])
    return np.sort(tr), np.sort(va), np.sort(te)


def build_bundle(task, data_path, out_dir, cfg, eda_summary=None, logger=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if task == "regression":
        X, y, genes = load_riboflavin(data_path)
        tr, va, te = split_regression(y, seed=cfg.seed)
        b = dict(X_tr=X[tr], y_tr=y[tr], X_va=X[va], y_va=y[va], X_te=X[te], y_te=y[te], idx_tr=tr, idx_va=va, idx_te=te)
        logger(f"riboflavin: {X.shape[0]} samples x {X.shape[1]} genes | split train/val/test = {len(tr)}/{len(va)}/{len(te)} (random, stratified on y bins)")
    else:
        Xtr, ytr, Xte, yte, genes = load_srbct(data_path)
        b = dict(X_tr=Xtr, y_tr=ytr, X_va=np.zeros((0, Xtr.shape[1])), y_va=np.zeros(0, int), X_te=Xte, y_te=yte)
        logger(f"SRBCT: {Xtr.shape[1]} genes | book split: train {len(ytr)} (classes {np.bincount(ytr)[1:].tolist()}), test {len(yte)} (classes {np.bincount(yte)[1:].tolist()})")
    wins = {"on": True, "off": False}.get(cfg.winsorize, bool((eda_summary or {}).get("recommend_winsorize", False)))
    np.savez_compressed(out / "bundle.npz", **b, genes=np.array(genes), winsorize=np.array(wins), task=np.array(task))
    return b, genes, wins


def load_bundle(out_dir):
    z = np.load(Path(out_dir) / "bundle.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["winsorize"] = bool(d["winsorize"])
    d["genes"] = [str(g) for g in d["genes"]]
    d["task"] = str(d["task"])
    return d
