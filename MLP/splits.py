"""
Train / validation / test splitting and a data audit that runs before any training.

Two situations, both handled here:
  * dataset ships PRE-SPLIT into train/test only (MNIST)  -> keep the official test set,
    carve a validation set out of the training part  (carve_validation)
  * dataset ships as ONE table (California Housing)       -> split all three ways
    (split_three_way)

Splits are seeded (reproducible) and, for classification, stratified by label.
"""
import hashlib

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def _count(size, n):
    """A size given as a fraction (0 < s < 1) or an absolute row count."""
    return int(round(size * n)) if 0 < size < 1 else int(size)


def carve_validation(n, y=None, val_size=0.1, seed=42, stratify=False):
    """Split n training rows into (train_idx, val_idx). Stratified on y if requested."""
    n_val = _count(val_size, n)
    if not 0 < n_val < n:
        raise ValueError(f"validation size {val_size!r} is not usable for {n} rows")
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx, test_size=n_val, random_state=seed,
        stratify=np.asarray(y) if (stratify and y is not None) else None)
    return np.sort(train_idx), np.sort(val_idx)


def split_three_way(n, y=None, val_size=0.2, test_size=0.2, seed=42, stratify=False):
    """Split n rows into (train_idx, val_idx, test_idx). Sizes are fractions or counts."""
    n_test, n_val = _count(test_size, n), _count(val_size, n)
    if n_test <= 0 or n_val <= 0 or n_test + n_val >= n:
        raise ValueError(f"val_size={val_size!r}, test_size={test_size!r} are not usable for {n} rows")
    y = np.asarray(y) if (stratify and y is not None) else None
    idx = np.arange(n)
    rest, test_idx = train_test_split(idx, test_size=n_test, random_state=seed,
                                      stratify=None if y is None else y)
    train_idx, val_idx = train_test_split(rest, test_size=n_val, random_state=seed,
                                          stratify=None if y is None else y[rest])
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _row_hashes(x):
    arr = np.ascontiguousarray(x.cpu().numpy())
    return {hashlib.blake2b(row.tobytes(), digest_size=8).digest() for row in arr}


def audit_splits(splits, info, shift_tol=0.02):
    """Checks on the three splits. Returns a JSON-serialisable report.

    errors   (stop the pipeline) : empty split, NaN/Inf in features or target, a class
                                   missing from a split
    warnings (reported only)     : identical rows shared between splits (possible leakage),
                                   class proportions or target mean that differ across splits
    """
    task = info["task"]
    rep = {"sizes": {}, "fractions": {}, "nonfinite": {}, "overlap_identical_rows": {},
           "warnings": [], "errors": []}
    total = sum(len(v) for v in splits.values())
    hashes = {}
    for name, ds in splits.items():
        x, y = ds.tensors
        rep["sizes"][name] = len(ds)
        rep["fractions"][name] = round(len(ds) / total, 4)
        bad_x = int((~torch.isfinite(x)).sum())
        bad_y = int((~torch.isfinite(y.float())).sum())
        rep["nonfinite"][name] = {"features": bad_x, "target": bad_y}
        if len(ds) == 0:
            rep["errors"].append(f"split {name!r} is empty")
        if bad_x or bad_y:
            rep["errors"].append(f"split {name!r} has non-finite values "
                                 f"({bad_x} in features, {bad_y} in target)")
        hashes[name] = _row_hashes(x)

    if task == "classification":
        k = info["out_dim"]
        counts, props = {}, {}
        for name, ds in splits.items():
            c = torch.bincount(ds.tensors[1].long(), minlength=k)
            counts[name] = c.tolist()
            props[name] = (c.float() / max(int(c.sum()), 1)).tolist()
            missing = [i for i in range(k) if int(c[i]) == 0]
            if missing:
                rep["errors"].append(f"class(es) {missing} missing from split {name!r}")
        rep["class_counts"] = counts
        if "train" in props:
            for name in props:
                gap = max(abs(a - b) for a, b in zip(props[name], props["train"]))
                rep.setdefault("max_class_proportion_gap_vs_train", {})[name] = round(gap, 4)
                if gap > shift_tol:
                    rep["warnings"].append(f"class proportions in {name!r} differ from train "
                                           f"by up to {gap:.3f}")
    else:
        stats = {}
        for name, ds in splits.items():
            t = ds.tensors[1].float() * info["y_std"] + info["y_mean"]  # original units
            stats[name] = {"target_mean": float(t.mean()), "target_std": float(t.std())}
        rep["target_stats"] = stats
        if "train" in stats:
            for name in stats:
                shift = abs(stats[name]["target_mean"] - stats["train"]["target_mean"]) \
                    / max(stats["train"]["target_std"], 1e-12)
                if shift > 0.15:
                    rep["warnings"].append(f"target mean in {name!r} differs from train by "
                                           f"{shift:.2f} standard deviations")

    names = [n for n in ("train", "val", "test") if n in hashes]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = len(hashes[a] & hashes[b])
            rep["overlap_identical_rows"][f"{a}&{b}"] = shared
            if shared:
                rep["warnings"].append(f"{shared} identical feature rows appear in both {a!r} and "
                                       f"{b!r} (possible leakage or genuine duplicates)")
    rep["ok"] = not rep["errors"]
    return rep
