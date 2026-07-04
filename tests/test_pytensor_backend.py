"""Tests for the PyTensor backend."""

import numpy as np
import pytest

from pytensor import tensor as pt

from utils import atol

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

    cos_theta = pt.cos(theta)
    sin_theta = pt.sin(theta)

    result = fit_lsci(
        x=x + r * cos_theta,
        y=y + r * sin_theta
    )

    np.testing.assert_allclose(actual=result.radius.eval(), desired=r)
    np.testing.assert_allclose(actual=result.center.x.eval(), desired=x)
    np.testing.assert_allclose(actual=result.center.y.eval(), desired=y)

    val_roundness = result.roundness.eval()
    np.testing.assert_allclose(
        actual=val_roundness,
        desired=0.0,
        atol=atol(x=val_roundness)
    )

    result = fit_lsci(
        x=x + r * cos_theta
        + 0.1 * r * rng.normal(loc=0.0, scale=1.0, size=n),
        y=y + r * sin_theta
        + 0.1 * r * rng.normal(loc=0.0, scale=1.0, size=n)
    )

    assert result.roundness.eval() > 0.0


@pytest.mark.parametrize("n", range(3, 361))
def test_fit_unit_circle(n):
    """Fit points on a unit circle."""

    theta = 2 * np.pi * pt.arange(n) / float(n)

    x = pt.cos(theta)
    y = pt.sin(theta)

    result = fit_lsci(x, y)

    np.testing.assert_allclose(actual=result.radius.eval(), desired=1.0)

    val_center_x = result.center.x.eval()
    np.testing.assert_allclose(
        actual=val_center_x,
        desired=0.0,
        atol=atol(val_center_x)
    )

    val_center_y = result.center.y.eval()
    np.testing.assert_allclose(
        actual=val_center_y,
        desired=0.0,
        atol=atol(val_center_y)
    )

    val_roundness = result.roundness.eval()
    np.testing.assert_allclose(
        actual=val_roundness,
        desired=0.0,
        atol=atol(val_roundness)
    )


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
