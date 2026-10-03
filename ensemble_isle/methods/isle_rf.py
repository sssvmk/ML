"""ISLE_with_random_forest_dictionary - see isle_entry.py for the loss, tuning protocol and metrics."""
import _bootstrap  # noqa: F401
import common
import isle_entry

NAME = "isle_rf"
make_grid, complexity, run = isle_entry.make(NAME, "isle_rf")

if __name__ == "__main__":
    common.method_main(NAME, run)
