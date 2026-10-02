import os as _os
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))   # Windows: stops joblib/loky from calling the removed `wmic` tool
try:                                                         # loky asks `wmic` for the physical core count on Windows (removed from Windows 11): answer it ourselves, silently
    from joblib.externals.loky.backend import context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda _n=(_os.cpu_count() or 1): (_n, None)
except Exception:
    pass
import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
_ME = str(Path(__file__).resolve().parent)
for _p in (_ROOT, _ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)
