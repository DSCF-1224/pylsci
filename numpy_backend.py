"""NumPy implementation of LSCI circle fitting."""

import numpy as np

from .result import Center, FittedCircle


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
        If x and y have different lengths or fewer than
        three points are provided.
    """

    size_x = np.size(x)

    if size_x != np.size(y):
        raise ValueError("x and y must have the same length")

    if size_x < 3:
        raise ValueError("at least 3 points are required")

    x2 = x * x
    y2 = y * y

    r2 = x2 + y2

    sum_x1_y0 = np.sum(x)
    sum_x0_y1 = np.sum(y)
    sum_x1_y1 = np.sum(x * y)
    sum_x2_y0 = np.sum(x2)
    sum_x0_y2 = np.sum(y2)

    matrix = np.array(
        [[sum_x2_y0, sum_x1_y1, sum_x1_y0],
         [sum_x1_y1, sum_x0_y2, sum_x0_y1],
         [sum_x1_y0, sum_x0_y1, float(size_x)]]
    )

    vector = np.array([np.sum(x * r2), np.sum(y * r2), np.sum(r2)])

    solution = np.linalg.solve(matrix, vector)

    center = Center(x=0.5 * solution[0], y=0.5 * solution[1])

    center_r2 = (center.x * center.x) + (center.y * center.y)

    dx = x - center.x
    dy = y - center.y
    dr = np.sqrt((dx * dx) + (dy * dy))

    return FittedCircle(
        center=center,
        radius=np.sqrt(center_r2 + solution[2]),
        roundness=np.max(dr) - np.min(dr)
    )
