"""Synthetic stand-in for santander_train.csv (same layout: ID_code, Target, var_0..var_199). Smoke tests only."""
import sys
import numpy as np
import pandas as pd


def make(n=20000, seed=0, path="synthetic_santander.csv"):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < 0.10).astype(int)
    cols = {}
    for j in range(200):
        kind = j % 4
        loc, sc = rng.uniform(-10, 10), rng.uniform(1, 6)
        shift = rng.randn() * 0.5 if j < 60 else 0.0
        sc_pos = sc * (1.25 if 60 <= j < 90 else 1.0)
        z = rng.randn(n)
        base = np.where(y == 1, loc + shift * sc + sc_pos * z, loc + sc * z)
        if kind == 1:
            base = np.exp(0.3 * (base - loc) / sc) * sc + loc
        if kind == 2:
            base = np.round(base, 1)
        cols[f"var_{j}"] = np.round(base, 4)
    df = pd.DataFrame({"ID_code": [f"train_{i}" for i in range(n)], "Target": y, **cols})
    df.to_csv(path, index=False)
    print("wrote", path, n)


if __name__ == "__main__":
    make(int(sys.argv[1]) if len(sys.argv) > 1 else 20000, path=sys.argv[2] if len(sys.argv) > 2 else "synthetic_santander.csv")
