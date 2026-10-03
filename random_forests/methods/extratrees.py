"""ExtraTrees (tuned) - see rf_entry.py (loss, tuning protocol and metrics) and rflib.py (importance, curves, proximity)."""
import _bootstrap  # noqa: F401
import common
import rf_entry

NAME = "extratrees"
make_grid, complexity, run = rf_entry.make(NAME, "et", "tuned")

if __name__ == "__main__":
    common.method_main(NAME, run)
