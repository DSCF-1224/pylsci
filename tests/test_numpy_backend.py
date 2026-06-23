"""Tests for the NumPy backend."""

import numpy as np

from pylsci.numpy_backend import fit as fit_lsci


def test_fit_unit_circle():
    """Fit points on a unit circle."""

    x = np.array([1.0, 0.0, -1.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, -1.0])

    result = fit_lsci(x, y)

    assert np.isclose(result.radius, 1.0)
    assert np.isclose(result.center.x, 0.0)
    assert np.isclose(result.center.y, 0.0)
    assert np.isclose(result.roundness, 0.0)
