"""PyTensor implementation of LSCI circle fitting."""

from pytensor import tensor as pt
from pytensor.tensor import TensorVariable

from .result import Center, FittedCircle


def _construct_normal_equation(
    x: TensorVariable,
    y: TensorVariable
) -> tuple[TensorVariable, TensorVariable]:
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
    When input lengths are statically known, mismatched lengths and fewer than
    three points raise ``ValueError`` immediately. Otherwise, shape compatibility
    is checked by PyTensor during graph execution.
    """

    size_x = x.type.shape[0]
    size_y = y.type.shape[0]

    if (size_x is not None) and (size_y is not None):

        if size_x != size_y:
            raise ValueError("x and y must have the same length")

        if size_x < 3:
            raise ValueError("at least 3 points are required")

    centroid = Center(x=pt.mean(x), y=pt.mean(y))

    x_offset = x - centroid.x
    y_offset = y - centroid.y

    matrix, vector = _construct_normal_equation(x=x_offset, y=y_offset)

    solution = pt.linalg.solve(matrix, vector)

    center_offset = Center(x=0.5 * solution[0], y=0.5 * solution[1])

    dx = x_offset - center_offset.x
    dy = y_offset - center_offset.y
    dr = pt.sqrt((dx * dx) + (dy * dy))

    return FittedCircle(
        center=Center(x=center_offset.x + centroid.x,
                      y=center_offset.y + centroid.y),
        radius=pt.sqrt(center_offset.sum_of_squares() + solution[2]),
        roundness=pt.max(dr) - pt.min(dr)
    )
