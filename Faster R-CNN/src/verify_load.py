"""Load-and-predict check, executed in a FRESH process from a neutral directory by register.py.

Exit code 0 = the model loads from its URI (code and dependencies bundled) and reproduces the expected detections.
"""
import json
import sys

import mlflow
import numpy as np
import pandas as pd


def main(uri: str, example_csv: str, expected_json: str, atol: float) -> int:
    df = pd.read_csv(example_csv)
    want = json.load(open(expected_json))
    model = mlflow.pyfunc.load_model(uri)
    got = json.loads(model.predict(df, params={"score_floor": 0.0})["detections"].iloc[0])["detections"]
    if len(got) != len(want):
        print(f"FAIL detections {len(got)} != {len(want)}")
        return 1
    if not got:
        print("OK (no detections on either side)")
        return 0
    gb, wb = np.array([d["box"] for d in got]), np.array([d["box"] for d in want])
    gs, ws = np.array([d["score"] for d in got]), np.array([d["score"] for d in want])
    err = max(float(np.abs(gb - wb).max()), float(np.abs(gs - ws).max()))
    print(f"max abs diff {err:.2e} (atol {atol}) over {len(got)} detections")
    return 0 if err <= atol else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])))
