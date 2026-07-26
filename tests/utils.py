"""Test utilities."""

import itertools

import numpy as np

from pylsci.result import Center, Circle


MISMATCHED_LENGTH_CASES = [(3, 4), (4, 3)]

NON_1D_SHAPE_CASES = [
    (x_dim, y_dim)
    for x_dim, y_dim in itertools.product([1, 2, 3], repeat=2)
    if not (x_dim == 1 and y_dim == 1)
]


def atol(x, factor=100):
    """Return an absolute tolerance based on the dtype of x."""
    # Work around a Pylint E1101 false positive with NumPy 2.4+.
    # See https://github.com/pylint-dev/pylint/issues/10806.
    return factor * getattr(np.finfo(np.asarray(x).dtype), 'eps')


def make_known_roundness_case(
        rng: np.random.Generator
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Generate a point set with a known roundness.

    Samples are placed on two concentric circles with radii
    r ± roundness / 2, so the expected roundness is known exactly.
    """

    base_radius = 10 ** rng.uniform(low=-1.0, high=1.0)

    # Number of samples on each circle.
    num_points = rng.integers(low=4, high=181)

    x_base, y_base = make_unit_circle_coords(int(num_points))

    roundness = rng.uniform(low=0.0, high=0.5 * base_radius)

    outer_radius = base_radius + 0.5 * roundness
    inner_radius = base_radius - 0.5 * roundness

    x = np.concatenate([outer_radius * x_base, inner_radius * x_base])
    y = np.concatenate([outer_radius * y_base, inner_radius * y_base])

    return roundness, x, y


def make_noisy_random_circle_case(
        rng: np.random.Generator
) -> tuple[Circle, np.ndarray, np.ndarray]:
    """
    Generate a random circle and noisy sample coordinates on its circumference.
    """

    circle, x_base, y_base = make_random_circle_case(rng)

    noisy_x = x_base + \
        0.1 * circle.radius * \
        rng.normal(loc=0.0, scale=1.0, size=x_base.size)

    noisy_y = y_base + \
        0.1 * circle.radius * \
        rng.normal(loc=0.0, scale=1.0, size=y_base.size)

    return circle, noisy_x, noisy_y


def make_random_circle_case(rng: np.random.Generator) -> tuple[Circle, np.ndarray, np.ndarray]:
    """
    Generate a random circle and coordinates on its circumference.
    """

    circle, num_points = sample_random_circle(rng)

    x, y = make_random_circle_coords(circle=circle, num_points=num_points)

    return circle, x, y


def make_random_circle_coords(circle: Circle, num_points: np.integer) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate coordinates uniformly distributed on a circle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x,y coordinates on a random circle.
    """

    x_base, y_base = make_unit_circle_coords(int(num_points))

    x = circle.radius * x_base + circle.center.x
    y = circle.radius * y_base + circle.center.y

    return x, y


def make_unit_circle_coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameter
    ---------
    n
        the number of points on the unit circle

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x,y coordinates uniformly distributed on the unit circle.
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
