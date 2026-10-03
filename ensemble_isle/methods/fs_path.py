"""Forward_stagewise_path_view - see boost_path.py for the loss, tuning protocol and metrics."""
import _bootstrap  # noqa: F401
import common
import boost_path

NAME = "fs_path"
make_grid, complexity, run = boost_path.make(NAME, "fs_path")

if __name__ == "__main__":
    common.method_main(NAME, run)
