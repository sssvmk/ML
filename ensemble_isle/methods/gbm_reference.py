"""Unprocessed_gradient_boosting_reference - see boost_path.py for the loss, tuning protocol and metrics."""
import _bootstrap  # noqa: F401
import common
import boost_path

NAME = "gbm_reference"
make_grid, complexity, run = boost_path.make(NAME, "gbm_reference")

if __name__ == "__main__":
    common.method_main(NAME, run)
