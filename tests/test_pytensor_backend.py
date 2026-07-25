"""Tests for the PyTensor backend."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytensor.tensor.type as ptt
import pytest

import pylsci._messages as msg
import utils

from pylsci.pytensor_backend import fit as fit_lsci


@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_random_circle(seed):
    """Fit points on a circle with a random center and radius."""

    rng = np.random.default_rng(seed)

    desired_circle, x, y = utils.make_random_circle_case(rng)

    result = fit_lsci(x=x, y=y)

    center_x, center_y, radius, roundness = \
        pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
            [],
            [result.center.x, result.center.y, result.radius, result.roundness]
        )()

    assert center_x == pytest.approx(desired_circle.center.x)
    assert center_y == pytest.approx(desired_circle.center.y)

    assert radius == pytest.approx(desired_circle.radius)

    assert roundness == pytest.approx(0.0)


@pytest.mark.parametrize("n", range(3, 361))
def test_fit_unit_circle(n):
    """Fit points on a unit circle."""

    x, y = utils.make_unit_circle_coords(n)

    result = fit_lsci(x, y)

    center_x, center_y, radius, roundness = \
        pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
            [],
            [result.center.x, result.center.y, result.radius, result.roundness]
        )()

    assert center_x == pytest.approx(0.0)
    assert center_y == pytest.approx(0.0)

    assert radius == pytest.approx(1.0)

    assert roundness == pytest.approx(0.0)


@pytest.mark.parametrize("x_len, y_len", utils.MISMATCHED_LENGTH_CASES)
def test_mismatched_length_dynamic(x_len: int, y_len: int):
    """
    When lengths are not statically known, a mismatch should not raise
    at fit() call time, but should raise when the graph is evaluated.
    """

    x = ptt.vector("x")
    y = ptt.vector("y")

    # should not raise here
    result = fit_lsci(x=x, y=y)

    fn = pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
        [x, y],
        [result.center.x, result.radius, result.roundness]
    )

    with pytest.raises(AssertionError, match="must have the same length"):
        fn(np.zeros(x_len), np.zeros(y_len))


@pytest.mark.parametrize("x_len, y_len", utils.MISMATCHED_LENGTH_CASES)
def test_mismatched_length_static(x_len: int, y_len: int):
    """Reject points with mismatched coordinate lengths."""

    with pytest.raises(ValueError, match=msg.MSG_SAME_LENGTH):
        fit_lsci(x=np.zeros(x_len), y=np.zeros(y_len))


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

    assert result.roundness.eval() > 0.0
