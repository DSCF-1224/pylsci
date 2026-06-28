"""Tests for the PyTensor backend."""

import numpy as np
import pytest

from pytensor import tensor as pt

from pylsci.pytensor_backend import fit as fit_lsci


@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_random_circle(seed):
    """Fit points on a circle with a random center and radius."""

    rng = np.random.default_rng(seed)

    x = rng.uniform(low=-1.0, high=1.0)
    y = rng.uniform(low=-1.0, high=1.0)
    r = 10 ** rng.uniform(low=-1.0, high=1.0)
    n = rng.integers(low=4, high=361)

    theta = 2 * np.pi * pt.arange(n) / float(n)

    result = fit_lsci(
        x=x + r * pt.cos(theta),
        y=y + r * pt.sin(theta)
    )

    assert np.isclose(result.radius.eval(), r)
    assert np.isclose(result.center.x.eval(), x)
    assert np.isclose(result.center.y.eval(), y)
    assert np.isclose(result.roundness.eval(), 0.0)


@pytest.mark.parametrize("n", range(3, 361))
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
    """Reject points with mismatched coordinate lengths."""

    with pytest.raises(ValueError):
        fit_lsci(
            pt.as_tensor([1.0, 0.0, -1.0, 0.0]),
            pt.as_tensor([0.0, 1.0, 0.0])
        )

    with pytest.raises(ValueError):
        fit_lsci(
            pt.as_tensor([0.0, 1.0, 0.0]),
            pt.as_tensor([1.0, 0.0, -1.0, 0.0])
        )


def test_requires_at_least_three_points():
    """Reject fewer than three points."""

    with pytest.raises(ValueError):
        fit_lsci(
            pt.as_tensor([1.0, 0.0]),
            pt.as_tensor([0.0, 1.0])
        )
