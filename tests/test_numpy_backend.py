"""Tests for the NumPy backend."""

import numpy as np
import pytest

import utils

from pylsci.numpy_backend import fit as fit_lsci


@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_random_circle(seed):
    """Fit points on a circle with a random center and radius."""

    rng = np.random.default_rng(seed)

    desired_circle, x, y = utils.make_random_circle_case(rng)

    result = fit_lsci(x=x, y=y)

    assert result.center.x == pytest.approx(desired_circle.center.x)
    assert result.center.y == pytest.approx(desired_circle.center.y)

    assert result.radius == pytest.approx(desired_circle.radius)

    assert result.roundness == pytest.approx(0.0)

    noisy_x = x + \
        0.1 * desired_circle.radius * \
        rng.normal(loc=0.0, scale=1.0, size=x.size)

    noisy_y = y + \
        0.1 * desired_circle.radius * \
        rng.normal(loc=0.0, scale=1.0, size=y.size)

    result = fit_lsci(x=noisy_x, y=noisy_y)

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
