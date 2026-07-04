"""Test utilities."""

import numpy as np

from pylsci.result import Center


def atol(x, factor=100):
    """Return an absolute tolerance based on the dtype of x."""
    # Work around a Pylint E1101 false positive with NumPy 2.4+.
    # See https://github.com/pylint-dev/pylint/issues/10806.
    return factor * getattr(np.finfo(np.asarray(x).dtype), 'eps')


def sample_random_circle_parameters(rng: np.random.Generator) -> tuple[Center, float, np.integer]:
    """Sample a random circle and the number of points."""

    center = Center(x=rng.uniform(low=-1.0, high=1.0),
                    y=rng.uniform(low=-1.0, high=1.0))

    radius = 10 ** rng.uniform(low=-1.0, high=1.0)

    num_points = rng.integers(low=4, high=361)

    return center, radius, num_points
