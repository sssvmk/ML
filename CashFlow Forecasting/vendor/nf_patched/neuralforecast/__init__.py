import logging
from importlib.metadata import version

try:
    __version__ = version("neuralforecast")
except Exception:  # vendored copy, not pip-installed (STC PATCH)
    __version__ = "3.2.2+stc-lazy-ray"

__all__ = ['NeuralForecast']

# Suppress PyTorch Lightning's "💡 Tip:" promos for LitLogger / cloud uploads,
# emitted via rank_zero_info on the `pytorch_lightning.utilities.rank_zero`
# logger every time a Trainer is constructed.
class _DropLightningTips(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("\U0001f4a1 Tip:")

logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    _DropLightningTips()
)

from .common._base_model import DistributedConfig  # noqa: F401, E402
from .core import NeuralForecast  # noqa: E402
