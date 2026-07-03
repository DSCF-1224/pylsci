"""Tests for consistency between the NumPy and PyTensor backends."""

import numpy as np
import pytest

from pytensor import tensor as pt

from pylsci import numpy_backend, pytensor_backend


@pytest.mark.parametrize("seed", range(50))
def test_fit_produces_consistent_results(seed):
    """Ensure that both backends produce equivalent fitting results."""

    rng = np.random.default_rng(seed)

    num_points = rng.integers(low=4, high=361)

    center_x = rng.uniform(low=-1.0, high=1.0)
    center_y = rng.uniform(low=-1.0, high=1.0)

    radius = 10 ** rng.uniform(low=-1.0, high=1.0)

    theta = np.linspace(
        start=0.0,
        stop=2 * np.pi,
        num=num_points,
        endpoint=False
    )

    points_x = center_x + (radius * np.cos(theta))
    points_y = center_y + (radius * np.sin(theta))

    points_x += (0.01 * radius
                 * rng.normal(loc=0.0, scale=1.0, size=num_points))
    points_y += (0.01 * radius
                 * rng.normal(loc=0.0, scale=1.0, size=num_points))

    np_result = numpy_backend.fit(x=points_x, y=points_y)

    pt_result = pytensor_backend.fit(
        x=pt.as_tensor(points_x),
        y=pt.as_tensor(points_y)
    )

    assert np.isclose(np_result.radius, pt_result.radius.eval())
    assert np.isclose(np_result.center.x, pt_result.center.x.eval())
    assert np.isclose(np_result.center.y, pt_result.center.y.eval())
    assert np.isclose(np_result.roundness, pt_result.roundness.eval())
