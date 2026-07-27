"""Tests for the PyTensor backend."""

import numpy as np
import pytensor
import pytensor.tensor.variable as ptv
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


@pytest.mark.parametrize("x_dim, y_dim", utils.NON_1D_SHAPE_CASES)
def test_rejects_non_1d_input_constant(x_dim: int, y_dim: int):
    """
    A non-1-dimensional x or y should raise ValueError immediately,
    when given as NumPy arrays (converted internally to TensorConstant).
    """

    with pytest.raises(ValueError, match=msg.MSG_NOT_1D):
        fit_lsci(
            x=np.zeros((3,) * x_dim),
            y=np.zeros((3,) * y_dim)
        )


@pytest.mark.parametrize("x_dim, y_dim", utils.NON_1D_SHAPE_CASES)
def test_rejects_non_1d_input_symbolic(x_dim: int, y_dim: int):
    """
    A non-1-dimensional x or y should raise ValueError immediately,
    when given as symbolic (shapeless) PyTensor variables.
    """

    def _make(dim: int, name: str) -> ptv.TensorVariable:

        if dim == 1:
            return ptt.vector(name)
        if dim == 2:
            return ptt.matrix(name)
        if dim == 3:
            return ptt.tensor3(name)

        raise ValueError("`dim` must be less than 4")

    x = _make(dim=x_dim, name="x")
    y = _make(dim=y_dim, name="y")

    with pytest.raises(ValueError, match=msg.MSG_NOT_1D):
        fit_lsci(x=x, y=y)


# pylint: disable=duplicate-code
@pytest.mark.parametrize("num_points", range(0, 3))
def test_requires_at_least_three_points_dynamic(num_points: int) -> None:
    """
    When the point count is not statically known, a count below 3
    should not raise at fit() call time, but should raise when the
    graph is evaluated.
    """

    x = ptt.vector("x")
    y = ptt.vector("y")

    # should not raise here
    result = fit_lsci(x=x, y=y)

    fn = pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
        [x, y],
        [result.center.x, result.radius, result.roundness]
    )

    with pytest.raises(AssertionError, match=msg.MSG_MIN_POINTS):
        fn(np.zeros(num_points), np.zeros(num_points))


@pytest.mark.parametrize("num_points", range(0, 3))
def test_requires_at_least_three_points_static(num_points: int):
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


@pytest.mark.parametrize("seed", range(0, 10))
def test_roundness_matches_known_value(seed):
    """
    For a point set with a known roundness by construction,
    the fitted roundness should match.
    """

    rng = np.random.default_rng(seed)

    desired_roundness, x, y = utils.make_known_roundness_case(rng)

    result = fit_lsci(x=x, y=y)

    assert result.roundness.eval() == pytest.approx(desired_roundness)
