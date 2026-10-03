"""Bagging (m = p) - see rf_entry.py (loss, tuning protocol and metrics) and rflib.py (importance, curves, proximity)."""
import _bootstrap  # noqa: F401
import common
import rf_entry

NAME = "bagging"
make_grid, complexity, run = rf_entry.make(NAME, "rf", "bagging")

if __name__ == "__main__":
    common.method_main(NAME, run)
