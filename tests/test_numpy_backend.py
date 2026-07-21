"""Tests for the NumPy backend."""

import numpy as np
import pytest

import utils

from pylsci.numpy_backend import fit as fit_lsci


@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_random_circle(seed):
    """Fit points on a circle with a random center and radius."""

    rng = np.random.default_rng(seed)

    center, r, n = utils.sample_random_circle_parameters(rng)

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
        atol=utils.atol(result.roundness)
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

    x, y = utils.make_unit_circle_coords(n)

    result = fit_lsci(x, y)

    assert result.center.x == pytest.approx(0.0)
    assert result.center.y == pytest.approx(0.0)

    assert result.radius == pytest.approx(1.0)

    assert result.roundness == pytest.approx(0.0)


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
