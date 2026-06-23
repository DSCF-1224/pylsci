"""PyTensor implementation of LSCI circle fitting."""

from pytensor import tensor as pt
from pytensor.tensor import TensorVariable

from .result import Center, FittedCircle


def _construct_normal_equation(x: TensorVariable, y: TensorVariable) -> tuple[TensorVariable, TensorVariable]:
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
    tuple[TensorVariable, TensorVariable]
        Normal-equation matrix and right-hand-side vector.
    """

    x2 = x * x
    y2 = y * y

    r2 = x2 + y2

    sum_x1_y0 = pt.sum(x)
    sum_x0_y1 = pt.sum(y)
    sum_x1_y1 = pt.sum(x * y)
    sum_x2_y0 = pt.sum(x2)
    sum_x0_y2 = pt.sum(y2)

    n = pt.cast(x.shape[0], x.dtype)

    matrix = pt.stack(
        [
            pt.stack([sum_x2_y0, sum_x1_y1, sum_x1_y0]),
            pt.stack([sum_x1_y1, sum_x0_y2, sum_x0_y1]),
            pt.stack([sum_x1_y0, sum_x0_y1, n])
        ]
    )

    vector = pt.stack([pt.sum(x * r2), pt.sum(y * r2), pt.sum(r2)])

    return matrix, vector


def fit(x: TensorVariable, y: TensorVariable) -> FittedCircle:
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

    Notes
    -----
    No explicit shape validation.
    Shape compatibility is checked by PyTensor during graph execution.
    """

    matrix, vector = _construct_normal_equation(x, y)

    solution = pt.linalg.solve(matrix, vector)

    center = Center(x=0.5 * solution[0], y=0.5 * solution[1])

    center_r2 = (center.x * center.x) + (center.y * center.y)

    dx = x - center.x
    dy = y - center.y
    dr = pt.sqrt((dx * dx) + (dy * dy))

    return FittedCircle(
        center=center,
        radius=pt.sqrt(center_r2 + solution[2]),
        roundness=pt.max(dr) - pt.min(dr)
    )
