"""
Loader for the vendored official ETSformer (Salesforce, BSD-3-Clause; decision D-12, gap G-36).

The upstream package is called `models.etsformer`, a top-level name that would collide with anything else called `models`,
so it is registered under a private name (`stc_vendored_etsformer`) and its relative imports resolve inside that name.
Nothing outside algorithms/etsformer.py should import it.
"""
from __future__ import annotations
import importlib
import importlib.util
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "etsformer"
PKG_DIR = VENDOR_DIR / "src" / "models" / "etsformer"
PKG_NAME = "stc_vendored_etsformer"


def load_etsformer():
    """Returns the vendored package's `model` module (class `ETSformer`)."""
    if PKG_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(PKG_NAME, PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[PKG_NAME] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{PKG_NAME}.model")
