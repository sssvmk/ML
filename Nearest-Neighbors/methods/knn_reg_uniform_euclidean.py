"""k-NN regression, uniform weights, euclidean distance: see knn_reg_common.py for the loss, hyper-parameters and metrics."""
import _bootstrap  # noqa: F401
import common
import knn_reg_common

NAME = "knn_reg_uniform_euclidean"
make_grid, run = knn_reg_common.make("uniform", 2, NAME)

if __name__ == "__main__":
    common.method_main(NAME, run)
