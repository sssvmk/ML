import math

import torch

from src.model import VGG, count_params


def test_forward_shape_all_archs():
    for arch in ("vgg11", "vgg13", "vgg16", "vgg19"):
        m = VGG(arch, 100, width_mult=0.25).eval()
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, 100)


def test_initial_loss_close_to_ln_k():
    torch.manual_seed(0)
    m = VGG("vgg11", 100, width_mult=0.25).train()
    x, y = torch.randn(64, 3, 32, 32), torch.randint(0, 100, (64,))
    loss = torch.nn.functional.cross_entropy(m(x), y).item()
    assert abs(loss - math.log(100)) < 0.5


def test_vgg16_param_count_in_expected_range():
    n = count_params(VGG("vgg16", 100))
    assert 14e6 < n < 16e6
