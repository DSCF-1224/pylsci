"""Tests for consistency between the NumPy and PyTensor backends."""

import numpy as np
import pytensor
import pytest
import utils

from pylsci import numpy_backend, pytensor_backend


@pytest.mark.parametrize("seed", range(50))
def test_fit_produces_consistent_results(seed):
    """Ensure that both backends produce equivalent fitting results."""

    rng = np.random.default_rng(seed)

    _, x, y = utils.make_noisy_random_circle_case(rng)

    np_result = numpy_backend.fit(x=x, y=y)
    pt_result = pytensor_backend.fit(x=x, y=y)

    pt_center_x, pt_center_y, pt_radius, pt_roundness = \
        pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
            [],
            [
                pt_result.center.x,
                pt_result.center.y,
                pt_result.radius,
                pt_result.roundness
            ]
        )()

    assert pt_center_x == pytest.approx(np_result.center.x)
    assert pt_center_y == pytest.approx(np_result.center.y)

    assert pt_radius == pytest.approx(np_result.radius)

    assert pt_roundness == pytest.approx(np_result.roundness)
