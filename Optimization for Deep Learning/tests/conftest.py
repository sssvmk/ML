import pytest
import torch

from optpipe.config import load_config
from optpipe.core import Ctx, set_l1_smoothing
from optpipe.data import load_raw, make_splits

SMALL = ["data.source=synthetic", "data.val_size=400", "data.synthetic_train=2400", "data.synthetic_test=400",
         "model.hidden=[32,16]", "train.epochs=3", "tune.epochs=1", "tune.n_trials=2", "tune.fullbatch_iters=3",
         "full_batch.iterations=4", "method_settings.newton.curvature_batch=300", "method_settings.bfgs.hidden=[6]"]


@pytest.fixture(scope="session")
def cfg():
    c = load_config(None, SMALL)
    set_l1_smoothing(c["regularization"]["l1_smoothing"])
    return c


@pytest.fixture(scope="session")
def data(cfg):
    return make_splits(load_raw(cfg["data"], 0), cfg["data"]["val_size"], 0)


@pytest.fixture()
def ctx(cfg):
    return Ctx(cfg, torch.device("cpu"), 0, epochs=3)


@pytest.fixture()
def f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)
