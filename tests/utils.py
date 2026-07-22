"""Test utilities."""

import numpy as np

from pylsci.result import Center, Circle


def atol(x, factor=100):
    """Return an absolute tolerance based on the dtype of x."""
    # Work around a Pylint E1101 false positive with NumPy 2.4+.
    # See https://github.com/pylint-dev/pylint/issues/10806.
    return factor * getattr(np.finfo(np.asarray(x).dtype), 'eps')


def make_unit_circle_coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameter
    ---------
    n
        the number of points on the unit circle

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x,y coordinates of the points on the unit circle.
        The first point is (x,y)=(1,0)
    """

    theta = 2 * np.pi * np.arange(n) / n

    return np.cos(theta), np.sin(theta)


def sample_random_circle(rng: np.random.Generator) -> tuple[Circle, np.integer]:
    """Sample a random circle and the number of points."""

    center = Center(x=rng.uniform(low=-1.0, high=1.0),
                    y=rng.uniform(low=-1.0, high=1.0))

    radius = 10 ** rng.uniform(low=-1.0, high=1.0)

    num_points = rng.integers(low=4, high=361)

    return Circle(center=center, radius=radius), num_points
