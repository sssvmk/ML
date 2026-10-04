import pytest
import torch

from regpipe.config import load_config
from regpipe.data import load_raw, make_splits
from regpipe.methods.base import Ctx

SMALL = ["data.source=synthetic", "data.val_size=400", "data.synthetic_train=2400", "data.synthetic_test=400",
         "model.hidden=[32,16]", "train.epochs=2", "tune.epochs=1", "tune.n_trials=2",
         "method_settings.under_constrained.n_train=200", "method_settings.semi_supervised.n_labeled=200",
         "method_settings.bagging.n_members=2", "method_settings.bagging.tune_members=2",
         "method_settings.tangent.n_prototypes=200"]


@pytest.fixture(scope="session")
def cfg():
    return load_config(None, SMALL)


@pytest.fixture(scope="session")
def data(cfg):
    return make_splits(load_raw(cfg["data"], 0), cfg["data"]["val_size"], 0)


@pytest.fixture()
def ctx(cfg):
    return Ctx(cfg, torch.device("cpu"), 0, epochs=2)
