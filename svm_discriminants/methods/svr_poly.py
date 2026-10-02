"""SVM for regression, poly kernel (ESLII 12.3.6): see svr_common.py for the loss (12.36-12.37), hyper-parameters and metrics."""
import _bootstrap  # noqa: F401
import common
import svr_common

NAME = "svr_poly"
make_grid, run = svr_common.make("poly", NAME)

if __name__ == "__main__":
    common.method_main(NAME, run)
