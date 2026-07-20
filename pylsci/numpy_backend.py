"""NumPy implementation of LSCI circle fitting."""

import numpy as np

from ._messages import MSG_MIN_POINTS, MSG_NOT_1D, MSG_SAME_LENGTH
from .result import Center, FittedCircle


def _construct_normal_equation(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the normal equation for LSCI fitting.

    Parameters
    ----------
    x
        X coordinates.
    y
        Y coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normal-equation matrix and right-hand-side vector.
    """

    x2 = x * x
    y2 = y * y

    r2 = x2 + y2

    sum_x1_y0 = np.sum(x)
    sum_x0_y1 = np.sum(y)
    sum_x1_y1 = np.sum(x * y)
    sum_x2_y0 = np.sum(x2)
    sum_x0_y2 = np.sum(y2)

    matrix = np.array(
        [
            [sum_x2_y0, sum_x1_y1, sum_x1_y0],
            [sum_x1_y1, sum_x0_y2, sum_x0_y1],
            [sum_x1_y0, sum_x0_y1, float(x.size)]
        ]
    )

    vector = np.array([np.sum(x * r2), np.sum(y * r2), np.sum(r2)])

    return matrix, vector


def fit(x: np.ndarray, y: np.ndarray) -> FittedCircle:
    """
    Fit a least-squares reference circle (LSCI) from a set of points.

    Parameters
    ----------
    x
        X coordinates.
    y
        Y coordinates.

    Returns
    -------
    FittedCircle
        Fitted circle and evaluated roundness.

    Raises
    ------
    ValueError
        If x or y is not 1-dimensional,
        if x and y have different lengths or
        fewer than three points are provided.
    numpy.linalg.LinAlgError
        If the normal equation matrix is singular
    """

    if np.ndim(x) != 1 or np.ndim(y) != 1:
        raise ValueError(MSG_NOT_1D)

    size_x = np.size(x)

    if size_x != np.size(y):
        raise ValueError(MSG_SAME_LENGTH)

    if size_x < 3:
        raise ValueError(MSG_MIN_POINTS)

    centroid = Center(x=np.mean(x), y=np.mean(y))

    x_offset = x - centroid.x
    y_offset = y - centroid.y

    matrix, vector = _construct_normal_equation(x=x_offset, y=y_offset)

    solution = np.linalg.solve(matrix, vector)

    center_offset = Center(x=0.5 * solution[0], y=0.5 * solution[1])

    dx = x_offset - center_offset.x
    dy = y_offset - center_offset.y
    dr = np.sqrt((dx * dx) + (dy * dy))

    return FittedCircle(
        center=Center(x=center_offset.x + centroid.x,
                      y=center_offset.y + centroid.y),
        radius=np.sqrt(center_offset.sum_of_squares() + solution[2]),
        roundness=np.max(dr) - np.min(dr)
    )
