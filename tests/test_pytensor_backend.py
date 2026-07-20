"""Tests for the PyTensor backend."""

import numpy as np
import pytest

from pytensor import tensor as pt

import utils

from pylsci.pytensor_backend import fit as fit_lsci


@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_random_circle(seed):
    """Fit points on a circle with a random center and radius."""

    rng = np.random.default_rng(seed)

    center, r, n = utils.sample_random_circle_parameters(rng)

    theta = 2 * np.pi * pt.arange(n) / float(n)

    cos_theta = pt.cos(theta)
    sin_theta = pt.sin(theta)

    result = fit_lsci(
        x=center.x + r * cos_theta,
        y=center.y + r * sin_theta
    )

    np.testing.assert_allclose(actual=result.radius.eval(), desired=r)
    np.testing.assert_allclose(actual=result.center.x.eval(), desired=center.x)
    np.testing.assert_allclose(actual=result.center.y.eval(), desired=center.y)

    val_roundness = result.roundness.eval()
    np.testing.assert_allclose(
        actual=val_roundness,
        desired=0.0,
        atol=utils.atol(x=val_roundness)
    )

    result = fit_lsci(
        x=center.x
        + r * cos_theta
        + 0.1 * r * rng.normal(loc=0.0, scale=1.0, size=n),
        y=center.y
        + r * sin_theta
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
        atol=utils.atol(val_center_x)
    )

    val_center_y = result.center.y.eval()
    np.testing.assert_allclose(
        actual=val_center_y,
        desired=0.0,
        atol=utils.atol(val_center_y)
    )

    val_roundness = result.roundness.eval()
    np.testing.assert_allclose(
        actual=val_roundness,
        desired=0.0,
        atol=utils.atol(val_roundness)
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
