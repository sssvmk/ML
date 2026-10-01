"""
Prebuilt (pretrained) model resolution -- config-driven, path-based, offline.

Every pretrained model the system may use has an entry under config.json -> "prebuilt_models":

    "patchtsmixer": {
        "enabled": true,
        "path": "/models/granite-timeseries-patchtsmixer",   # directory holding the model files (HF save_pretrained layout)
        "fine_tune": true,                                    # true: pretrained weights are the starting point, fitted per segment
        "expected_fingerprint": null,                         # optional: sha256 to verify; when null the fingerprint is only recorded
        "licence_id": "Apache-2.0"
    }

Expectation on the code: it reads the model ONLY from `path` (never downloads at run time), fails with a message naming the
config key when the entry is disabled / has no path / the path does not exist, and records a fingerprint of the files it
loaded on every trained artifact, so a result can always be traced to the exact weights that produced it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib


class PrebuiltModelUnavailable(RuntimeError):
    """The configured pretrained model cannot be used (disabled, no path, path missing, fingerprint mismatch)."""


@dataclass
class PrebuiltSpec:
    key: str
    path: Path
    fine_tune: bool
    fingerprint: str
    fingerprint_verified: bool
    entry: dict = field(default_factory=dict)


_FP_CACHE: dict = {}


def fingerprint_path(path: str | Path) -> str:
    """
    sha256 over (relative file name, size, content) of every file under `path` (or of the single file). Cached per
    (path, file names, sizes, mtimes): a backtest creates a fresh module per fold and must not re-read large weights each time,
    yet any change to the files invalidates the cache.
    """
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    sig = (str(p.resolve()), tuple((str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files))
    if sig in _FP_CACHE:
        return _FP_CACHE[sig]
    h = hashlib.sha256()
    for f in files:
        h.update((f.name if p.is_file() else str(f.relative_to(p))).encode())
        h.update(str(f.stat().st_size).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    _FP_CACHE[sig] = h.hexdigest()
    return _FP_CACHE[sig]


def resolve_prebuilt(entry: dict | None, key: str, *, default_fine_tune: bool, environment: str | None = None) -> PrebuiltSpec:
    where = f"config.json -> prebuilt_models.{key}"
    if not entry:
        raise PrebuiltModelUnavailable(f"{where} is missing; add an entry with the path of the pretrained model")
    if not entry.get("enabled", False):
        raise PrebuiltModelUnavailable(f"{where}.enabled is false; enable it (and set its path) to use the pretrained model")
    if entry.get("use_scope") == "non_commercial":
        # licence guard (decision D-8): weights that may only be used non-commercially / non-production are never used by accident
        if not entry.get("non_commercial_use_acknowledged"):
            raise PrebuiltModelUnavailable(f"{where}: these weights are licensed for NON-COMMERCIAL, NON-PRODUCTION use only "
                                           f"({entry.get('licence_id')}); set {where}.non_commercial_use_acknowledged to true to confirm this deployment qualifies")
        allowed = entry.get("allowed_environments")
        if allowed is not None and environment not in allowed:
            raise PrebuiltModelUnavailable(f"{where}: use is restricted to environments {list(allowed)}, but config.json -> deployment.environment "
                                           f"is {environment!r}; these weights must not be used in a production run")
    raw = entry.get("path")
    if not raw:
        raise PrebuiltModelUnavailable(f"{where}.path is not set; point it at the directory holding the pretrained model files")
    path = Path(raw).expanduser()
    if not path.exists():
        raise PrebuiltModelUnavailable(f"{where}.path = {str(path)!r} does not exist")
    if path.is_dir() and not any(path.iterdir()):
        raise PrebuiltModelUnavailable(f"{where}.path = {str(path)!r} is an empty directory")
    fp = fingerprint_path(path)
    expected = entry.get("expected_fingerprint")
    if expected and expected != fp:
        raise PrebuiltModelUnavailable(f"{where}: files at {str(path)!r} have fingerprint {fp[:16]}..., expected {str(expected)[:16]}...")
    return PrebuiltSpec(key=key, path=path, fine_tune=bool(entry.get("fine_tune", default_fine_tune)), fingerprint=fp,
                        fingerprint_verified=bool(expected), entry=dict(entry))
