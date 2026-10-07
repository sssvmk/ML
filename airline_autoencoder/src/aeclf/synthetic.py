"""Synthetic data with the SAME schema as the real airline-satisfaction file.

Only for smoke tests, CI and demos. Numbers produced from it say nothing about real-world performance.
The generator plants realistic structure: business travel/class, digital + comfort ratings and delays drive
satisfaction, with an interaction (business travellers punish poor in-flight service harder) and label noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
               "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
               "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]


def make_synthetic(n: int = 20000, seed: int = 0, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(40, 15, n).round(), 7, 85).astype(int)
    gender = rng.choice(["Female", "Male"], n)
    loyal = rng.choice(["Loyal Customer", "disloyal Customer"], n, p=[0.82, 0.18])
    travel = rng.choice(["Business travel", "Personal Travel"], n, p=[0.69, 0.31])
    cls = np.where(travel == "Business travel", rng.choice(["Business", "Eco", "Eco Plus"], n, p=[0.6, 0.3, 0.1]),
                   rng.choice(["Business", "Eco", "Eco Plus"], n, p=[0.07, 0.8, 0.13]))
    dist = np.clip(rng.lognormal(6.7, 0.9, n), 31, 4983).round().astype(int)
    dep = np.where(rng.random(n) < 0.55, 0, rng.exponential(30, n).round()).astype(float)
    arr = np.where(rng.random(n) < 0.97, np.maximum(0, dep + rng.normal(0, 8, n).round()), rng.exponential(20, n).round())
    latent = rng.normal(0, 1, n)                                    # passenger's overall mood
    data = {"id": np.arange(n), "Age": age, "Flight Distance": dist}
    for c in RATING_COLS:
        base = 3 + 0.7 * latent + rng.normal(0, 1.1, n) + (0.4 if "Online boarding" in c else 0) * (travel == "Business travel")
        r = np.clip(np.round(base), 1, 5)
        na = rng.random(n) < (0.04 if c in ("Inflight wifi service", "Gate location") else 0.002)
        data[c] = np.where(na, 0, r).astype(int)
    data["Departure Delay in Minutes"] = dep
    arr = arr.astype(float)
    arr[rng.random(n) < 0.0006] = np.nan
    data["Arrival Delay in Minutes"] = arr
    data["Gender"], data["Customer Type"], data["Type of Travel"], data["Class"] = gender, loyal, travel, cls
    df = pd.DataFrame(data)
    if with_target:
        rated = df[RATING_COLS].replace(0, np.nan)
        score = (-2.2 + 0.9 * (rated.mean(axis=1).fillna(3) - 3) + 0.9 * (df["Online boarding"] - 3) * 0.4
                 + 1.1 * (travel == "Business travel") + 0.7 * (cls == "Business") - 0.9 * (loyal != "Loyal Customer")
                 - 0.012 * np.nan_to_num(arr, nan=0) + 0.9 * latent
                 + 0.35 * (travel == "Business travel") * (rated["On-board service"].fillna(3) - 3)
                 + rng.normal(0, 0.6, n))
        df["satisfaction"] = rng.random(n) < 1 / (1 + np.exp(-score))
    return df


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-target", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = make_synthetic(a.rows, a.seed, not a.no_target)
    if "satisfaction" in d:
        d["satisfaction"] = d["satisfaction"].map({True: "TRUE", False: "FALSE"})
    d.to_csv(a.out, index=False)
    print(f"wrote {len(d)} rows to {a.out}")
