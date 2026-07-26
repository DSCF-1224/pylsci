"""Tests for the NumPy backend."""

import numpy as np
import pytest

import pylsci._messages as msg
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


@pytest.mark.parametrize("n", range(3, 361))
def test_fit_unit_circle(n):
    """Fit points on a unit circle."""

    x, y = utils.make_unit_circle_coords(n)

    result = fit_lsci(x, y)

    assert result.center.x == pytest.approx(0.0)
    assert result.center.y == pytest.approx(0.0)

    assert result.radius == pytest.approx(1.0)

    assert result.roundness == pytest.approx(0.0)


@pytest.mark.parametrize("x_len, y_len", utils.MISMATCHED_LENGTH_CASES)
def test_mismatched_length(x_len: int, y_len: int):
    """Reject points with mismatched coordinate lengths."""

    with pytest.raises(ValueError, match=msg.MSG_SAME_LENGTH):
        fit_lsci(x=np.zeros(x_len), y=np.zeros(y_len))


@pytest.mark.parametrize("x_dim, y_dim", utils.NON_1D_SHAPE_CASES)
def test_rejects_non_1d_input(x_dim: int, y_dim: int):
    """A non-1-dimensional x or y should raise ValueError immediately."""

    with pytest.raises(ValueError, match=msg.MSG_NOT_1D):
        fit_lsci(
            x=np.zeros((3,) * x_dim),
            y=np.zeros((3,) * y_dim)
        )


@pytest.mark.parametrize("num_points", range(0, 3))
def test_requires_at_least_three_points(num_points: int):
    """Reject fewer than three points."""

    with pytest.raises(ValueError, match=msg.MSG_MIN_POINTS):
        fit_lsci(x=np.zeros(num_points), y=np.zeros(num_points))


@pytest.mark.parametrize("seed", range(0, 10))
def test_roundness_is_positive_for_noisy_circle(seed):
    """Noisy circle samples should have positive roundness."""

    rng = np.random.default_rng(seed)

    _, noisy_x, noisy_y = utils.make_noisy_random_circle_case(rng)

    result = fit_lsci(x=noisy_x, y=noisy_y)

    assert result.roundness > 0.0


@pytest.mark.parametrize("seed", range(0, 10))
def test_roundness_matches_known_value(seed):
    """
    For a point set with a known roundness by construction,
    the fitted roundness should match.
    """

    rng = np.random.default_rng(seed)

    desired_roundness, x, y = utils.make_known_roundness_case(rng)

    result = fit_lsci(x=x, y=y)

    assert result.roundness == pytest.approx(desired_roundness)
