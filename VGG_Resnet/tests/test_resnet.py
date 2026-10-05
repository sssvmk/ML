import math

import pytest
import torch
import torch.nn as nn

from src.model import build_model, count_params
from src.resnet import BLOCKS_PER_STAGE, CifarResNet


@pytest.mark.parametrize("arch,expected", [("resnet20", 0.27e6), ("resnet32", 0.46e6), ("resnet56", 0.85e6)])
def test_param_counts_match_paper(arch, expected):
    n = count_params(CifarResNet(arch, 10))
    assert abs(n - expected) / expected < 0.03  # paper: 0.27M / 0.46M / 0.85M


@pytest.mark.parametrize("arch", ["resnet20", "resnet32", "resnet56"])
def test_depth_is_6n_plus_2_and_forward_shape(arch):
    m = CifarResNet(arch, 10, width_mult=0.5).eval()
    convs = [x for name, x in m.named_modules() if isinstance(x, nn.Conv2d) and "shortcut" not in name]
    n = BLOCKS_PER_STAGE[arch]
    assert len(convs) + 1 == 6 * n + 2  # weight layers = convs + the fully connected layer
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_shortcut_options_and_factory_dispatch():
    a = CifarResNet("resnet20", 10, shortcut="A")
    b = CifarResNet("resnet20", 10, shortcut="B")
    assert count_params(b) > count_params(a)  # B adds 1x1 projections
    m = build_model({"arch": "resnet20", "width_mult": 0.5}, 7)
    assert isinstance(m, CifarResNet) and m(torch.randn(1, 3, 32, 32)).shape == (1, 7)
    with pytest.raises(ValueError):
        CifarResNet("resnet20", 10, shortcut="C")
    with pytest.raises(ValueError):
        CifarResNet("resnet21", 10)


def test_initial_loss_close_to_ln_k_and_zero_init_residual():
    torch.manual_seed(0)
    m = CifarResNet("resnet20", 10, zero_init_residual=True).train()
    assert all(float(b.bn2.weight.abs().sum()) == 0 for b in m.blocks)
    loss = nn.functional.cross_entropy(m(torch.randn(64, 3, 32, 32)), torch.randint(0, 10, (64,))).item()
    assert abs(loss - math.log(10)) < 0.5


def test_gradients_flow_to_every_parameter():
    m = CifarResNet("resnet20", 10, width_mult=0.5).train()
    nn.functional.cross_entropy(m(torch.randn(8, 3, 32, 32)), torch.randint(0, 10, (8,))).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
