import os as _os
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))
import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
_ME = str(Path(__file__).resolve().parent)
for _p in (_ROOT, _ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)
