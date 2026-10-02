"""k-NN regression, distance weights, manhattan distance: see knn_reg_common.py for the loss, hyper-parameters and metrics."""
import _bootstrap  # noqa: F401
import common
import knn_reg_common

NAME = "knn_reg_distance_manhattan"
make_grid, run = knn_reg_common.make("distance", 1, NAME)

if __name__ == "__main__":
    common.method_main(NAME, run)
