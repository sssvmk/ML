"""
Loader for the vendored, Ray-optional NeuralForecast 3.2.2 (decision D-9, strategy S2; gap G-41).

Upstream `import neuralforecast` needs Ray importable in every released version. The copy under
vendor/nf_patched carries a small documented patch (vendor/nf_patched/STC_PATCH.md) that makes the ray import
optional, so NeuralForecast-backed modules work where Ray is not installed (e.g. Windows-native).

`neuralforecast()` always resolves to the vendored copy: a pip-installed neuralforecast that was already imported
would re-introduce the Ray requirement and diverge from the audited source, so that is refused, not tolerated.
"""
from __future__ import annotations
import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "nf_patched"


def ensure_vendored_neuralforecast():
    """Puts the vendored copy first on sys.path and returns the imported `neuralforecast` package."""
    root = str(VENDOR_ROOT)
    already = sys.modules.get("neuralforecast")
    if already is not None:
        loaded_from = Path(getattr(already, "__file__", "") or "").resolve()
        if VENDOR_ROOT.resolve() not in loaded_from.parents:
            raise ImportError(
                f"neuralforecast was already imported from {loaded_from}, not from the vendored copy {VENDOR_ROOT}. "
                "Import STC modules before any direct `import neuralforecast`, or uninstall the pip package.")
        return already
    if root not in sys.path:
        sys.path.insert(0, root)
    import neuralforecast  # noqa: WPS433
    return neuralforecast
