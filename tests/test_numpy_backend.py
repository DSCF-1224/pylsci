"""Tests for the NumPy backend."""

import numpy as np
import pytest

from pylsci.numpy_backend import fit as fit_lsci


@pytest.mark.parametrize("n", range(4, 361))
def test_fit_unit_circle(n):
    """Fit points on a unit circle."""

    theta = 2 * np.pi * np.arange(n) / n

    x = np.cos(theta)
    y = np.sin(theta)

    result = fit_lsci(x, y)

    assert np.isclose(result.radius, 1.0)
    assert np.isclose(result.center.x, 0.0)
    assert np.isclose(result.center.y, 0.0)
    assert np.isclose(result.roundness, 0.0)


def test_mismatched_length():
    """Reject points with mismatched coordinate lengths."""

    with pytest.raises(ValueError):
        fit_lsci(
            np.array([1.0, 0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        )

    with pytest.raises(ValueError):
        fit_lsci(
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, -1.0, 0.0])
        )


def test_requires_at_least_three_points():
    """Reject fewer than three points."""

    with pytest.raises(ValueError):
        fit_lsci(
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0])
        )
