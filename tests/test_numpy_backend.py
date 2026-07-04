"""Tests for the NumPy backend."""

import numpy as np
import pytest

from utils import atol, sample_random_circle_parameters

from pylsci.numpy_backend import fit as fit_lsci


@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_random_circle(seed):
    """Fit points on a circle with a random center and radius."""

    rng = np.random.default_rng(seed)

    center, r, n = sample_random_circle_parameters(rng)

    theta = 2 * np.pi * np.arange(n) / n

    result = fit_lsci(
        x=center.x + r * np.cos(theta),
        y=center.y + r * np.sin(theta)
    )

    np.testing.assert_allclose(actual=result.radius, desired=r)
    np.testing.assert_allclose(actual=result.center.x, desired=center.x)
    np.testing.assert_allclose(actual=result.center.y, desired=center.y)

    np.testing.assert_allclose(
        actual=result.roundness,
        desired=0.0,
        atol=atol(result.roundness)
    )

    result = fit_lsci(
        x=center.x
        + r * np.cos(theta)
        + 0.1 * r * rng.normal(loc=0.0, scale=1.0, size=n),
        y=center.y
        + r * np.sin(theta)
        + 0.1 * r * rng.normal(loc=0.0, scale=1.0, size=n)
    )
    assert result.roundness > 0.0


@pytest.mark.parametrize("n", range(3, 361))
def test_fit_unit_circle(n):
    """Fit points on a unit circle."""

    theta = 2 * np.pi * np.arange(n) / n

    x = np.cos(theta)
    y = np.sin(theta)

    result = fit_lsci(x, y)

    np.testing.assert_allclose(result.radius, 1.0)

    np.testing.assert_allclose(
        actual=result.center.x,
        desired=0.0,
        atol=atol(result.center.x)
    )

    np.testing.assert_allclose(
        actual=result.center.y,
        desired=0.0,
        atol=atol(result.center.y)
    )

    np.testing.assert_allclose(
        actual=result.roundness,
        desired=0.0,
        atol=atol(result.roundness)
    )


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
