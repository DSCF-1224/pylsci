"""Tests for the PyTensor backend."""

import numpy as np
import pytest

from pytensor import tensor as pt

from pylsci.pytensor_backend import fit as fit_lsci


@pytest.mark.parametrize("n", range(4, 361))
def test_fit_unit_circle(n):
    """Fit points on a unit circle."""

    theta = 2 * np.pi * pt.arange(n) / float(n)

    x = pt.cos(theta)
    y = pt.sin(theta)

    result = fit_lsci(x, y)

    assert np.isclose(result.radius.eval(), 1.0)
    assert np.isclose(result.center.x.eval(), 0.0)
    assert np.isclose(result.center.y.eval(), 0.0)
    assert np.isclose(result.roundness.eval(), 0.0)


def test_mismatched_length():
    with pytest.raises(ValueError):
        fit_lsci(
            pt.as_tensor([1.0, 0.0, -1.0, 0.0]),
            pt.as_tensor([0.0, 1.0, 0.0])
        )
