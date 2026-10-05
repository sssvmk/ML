"""Load-and-predict check, executed in a FRESH process from a neutral directory by register.py.

Exit code 0 = model loads from its URI (code and dependencies bundled) and reproduces the expected output.
"""
import sys

import mlflow
import numpy as np


def main(uri: str, example: str, expected: str, atol: float) -> int:
    x, want = np.load(example), np.load(expected)
    model = mlflow.pyfunc.load_model(uri)
    got = np.asarray(model.predict(x))
    if got.shape != want.shape:
        print(f"FAIL shape {got.shape} != {want.shape}")
        return 1
    err = float(np.abs(got - want).max())
    print(f"max abs diff {err:.2e} (atol {atol})")
    return 0 if err <= atol else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])))
