"""Tests for the PyTensor backend."""

import numpy as np

from pytensor import tensor as pt

from pylsci.pytensor_backend import fit as fit_lsci


def test_fit_unit_circle():
    """Fit points on a unit circle."""

    x = pt.stack([1.0, 0.0, -1.0, 0.0])
    y = pt.stack([0.0, 1.0, 0.0, -1.0])

    result = fit_lsci(x, y)

    assert np.isclose(result.radius.eval(), 1.0)
    assert np.isclose(result.center.x.eval(), 0.0)
    assert np.isclose(result.center.y.eval(), 0.0)
    assert np.isclose(result.roundness.eval(), 0.0)
