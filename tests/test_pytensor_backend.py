"""Tests for the PyTensor backend."""

import numpy as np
import pytensor
import pytest

from pytensor import tensor as pt

import utils

from pylsci.pytensor_backend import fit as fit_lsci


@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_random_circle(seed):
    """Fit points on a circle with a random center and radius."""

    rng = np.random.default_rng(seed)

    desired_circle, num_points = utils.sample_random_circle(rng)

    x, y = utils.make_random_circle_coords(
        circle=desired_circle, num_points=num_points
    )

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

    noisy_x = x + \
        0.1 * desired_circle.radius * \
        rng.normal(loc=0.0, scale=1.0, size=num_points)

    noisy_y = y + \
        0.1 * desired_circle.radius * \
        rng.normal(loc=0.0, scale=1.0, size=num_points)

    result = fit_lsci(x=noisy_x, y=noisy_y)

    assert result.roundness.eval() > 0.0


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
