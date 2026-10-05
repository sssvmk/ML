"""Packaging check: a wrong-shaped or wrong-typed input is rejected by the serving wrapper."""
import numpy as np
import pytest

from src.serve import to_uint8_array


def test_to_uint8_array_validates():
    assert to_uint8_array(np.zeros((32, 32, 3), np.uint8)).shape == (32, 32, 3)
    assert to_uint8_array(np.zeros((64, 48, 3), np.uint8)).shape == (32, 32, 3)
    with pytest.raises(ValueError):
        to_uint8_array(np.zeros((32, 32, 3), np.float32))
    with pytest.raises(ValueError):
        to_uint8_array(np.zeros((32, 32), np.uint8))
