import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def restore_global_state():
    """setup_environment() changes env vars, the temp dir and the torch hub dir: undo that after every test."""
    import torch
    env, tmp, pyc, hub = dict(os.environ), tempfile.tempdir, sys.pycache_prefix, torch.hub._hub_dir
    yield
    os.environ.clear()
    os.environ.update(env)
    tempfile.tempdir, sys.pycache_prefix, torch.hub._hub_dir = tmp, pyc, hub
