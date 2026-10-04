"""Registry: one module per technique, in the order of chapter 8."""
from .adagrad import AdaGrad
from .adam import Adam
from .base import Method
from .baseline import Baseline
from .batch_gd import BatchGD
from .batch_norm import BatchNorm
from .bfgs import BFGS
from .bias_gate import BiasGate
from .bias_output_marginal import BiasOutputMarginal
from .bias_relu_positive import BiasReluPositive
from .bias_zero import BiasZero
from .clip_backprop import ClipBackprop
from .clip_norm import ClipNorm
from .clip_value import ClipValue
from .conjugate_gradients import ConjugateGradients
from .continuation import Continuation
from .coordinate_descent import CoordinateDescent
from .curriculum import Curriculum
from .design_activations import DesignActivations
from .design_aux_heads import DesignAuxHeads
from .design_skip import DesignSkip
from .greedy_pretraining import GreedyPretraining
from .init_fixed_scale import InitFixedScale
from .init_glorot import InitGlorot
from .init_orthogonal import InitOrthogonal
from .init_random import InitRandom
from .init_random_walk import InitRandomWalk
from .init_scale_search import InitScaleSearch
from .init_sparse import InitSparse
from .init_supervised_pretrain import InitSupervisedPretrain
from .init_unsupervised_pretrain import InitUnsupervisedPretrain
from .lbfgs import LBFGS
from .lr_decay import LRDecay
from .momentum import Momentum
from .nesterov import Nesterov
from .newton import Newton
from .polyak_averaging import PolyakAveraging
from .rmsprop import RMSProp
from .rmsprop_nesterov import RMSPropNesterov
from .sgd import MinibatchSGD
from .variance_precision import VariancePrecision

ORDER = [Baseline,
         # gradient-based update rules (8.3, 8.1.3)
         BatchGD, MinibatchSGD, Momentum, Nesterov, LRDecay,
         # adaptive learning rates (8.5)
         AdaGrad, RMSProp, RMSPropNesterov, Adam,
         # second-order and approximate second-order (8.6)
         Newton, ConjugateGradients, BFGS, LBFGS,
         # initialisation (8.4)
         InitRandom, InitFixedScale, InitGlorot, InitOrthogonal, InitRandomWalk, InitSparse, InitScaleSearch,
         BiasZero, BiasOutputMarginal, BiasReluPositive, BiasGate, VariancePrecision,
         InitUnsupervisedPretrain, InitSupervisedPretrain,
         # gradient clipping (8.2.4)
         ClipValue, ClipNorm, ClipBackprop,
         # meta-algorithms and strategies (8.7)
         BatchNorm, CoordinateDescent, PolyakAveraging, GreedyPretraining,
         DesignActivations, DesignSkip, DesignAuxHeads, Continuation, Curriculum]

REGISTRY = {cls.name: cls for cls in ORDER}


def get_methods(names):
    """names: 'all' or a list of method names; the baseline is always included first."""
    if names in (None, "all", ["all"]):
        return [cls() for cls in ORDER]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown methods {unknown}; available: {sorted(REGISTRY)}")
    return [REGISTRY[n]() for n in ["baseline"] + [n for n in names if n != "baseline"]]
