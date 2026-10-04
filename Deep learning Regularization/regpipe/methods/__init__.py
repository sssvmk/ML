"""Registry: one module per method, in the order of Deep Learning book chapter 7."""
from .adversarial import AdversarialTraining
from .augmentation import Augmentation
from .bagging import Bagging
from .baseline import Baseline
from .constrained_norm import ConstrainedNorm
from .dropout import Dropout
from .early_stopping import EarlyStopping
from .input_noise import InputNoise
from .l1 import L1
from .l2_weight_decay import L2WeightDecay
from .label_smoothing import LabelSmoothing
from .multitask import Multitask
from .noise_robustness import NoiseRobustness
from .parameter_sharing import ParameterSharing
from .semi_supervised import SemiSupervised
from .sparse_representations import SparseRepresentations
from .tangent import Tangent
from .under_constrained import UnderConstrained
from .weight_noise import WeightNoise

ORDER = [Baseline, L2WeightDecay, L1, ConstrainedNorm, UnderConstrained, Augmentation, NoiseRobustness,
         InputNoise, WeightNoise, LabelSmoothing, SemiSupervised, Multitask, EarlyStopping,
         ParameterSharing, SparseRepresentations, Bagging, Dropout, AdversarialTraining, Tangent]

REGISTRY = {cls.name: cls for cls in ORDER}


def get_methods(names):
    """names: 'all' or a list of method names; baseline is always included first."""
    if names in (None, "all", ["all"]):
        return [cls() for cls in ORDER]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown methods {unknown}; available: {sorted(REGISTRY)}")
    names = ["baseline"] + [n for n in names if n != "baseline"]
    return [REGISTRY[n]() for n in names]
