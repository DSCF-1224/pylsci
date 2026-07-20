"""PyTensor implementation of LSCI circle fitting."""

from typing import cast

import pytensor.raise_op as pr
import pytensor.tensor as pt
import pytensor.tensor.basic as ptb
import pytensor.tensor.linalg as ptl
import pytensor.tensor.math as ptm
import pytensor.tensor.variable as ptv

from ._messages import MSG_MIN_POINTS, MSG_NOT_1D, MSG_SAME_LENGTH
from .result import Center, FittedCircle

_assert_min_points = pr.Assert(MSG_MIN_POINTS)
_assert_same_length = pr.Assert(MSG_SAME_LENGTH)


def _construct_normal_equation(
    x: ptv.TensorVariable, y: ptv.TensorVariable
) -> tuple[ptv.TensorVariable, ptv.TensorVariable]:
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

    sum_x1_y0 = ptm.sum(x)
    sum_x0_y1 = ptm.sum(y)
    sum_x1_y1 = ptm.sum(x * y)
    sum_x2_y0 = ptm.sum(x2)
    sum_x0_y2 = ptm.sum(y2)

    n = ptb.cast(x.shape[0], x.type.dtype)

    matrix = cast(
        ptv.TensorVariable,
        ptb.stack(
            [
                ptb.stack([sum_x2_y0, sum_x1_y1, sum_x1_y0]),
                ptb.stack([sum_x1_y1, sum_x0_y2, sum_x0_y1]),
                ptb.stack([sum_x1_y0, sum_x0_y1, n])
            ]
        )
    )

    vector = cast(
        ptv.TensorVariable,
        ptb.stack([ptm.sum(x * r2), ptm.sum(y * r2), ptm.sum(r2)])
    )

    return matrix, vector


def _validate_xy_shapes(
    x: ptv.TensorVariable, y: ptv.TensorVariable
) -> tuple[ptv.TensorVariable, ptv.TensorVariable]:
    """
    Validate x and y shapes.

    x and y must each be 1-dimensional; since ndim is always
    statically known in PyTensor, this is checked unconditionally.
    Length mismatches and point-count violations raise ValueError
    immediately when shapes are statically known; otherwise the
    checks are embedded in the computation graph and only raise
    when the graph is evaluated (e.g. via `.eval()` or a compiled
    function).

    Raises
    ------
    ValueError
        If x or y is not 1-dimensional, if x or y have different
        lengths, or if fewer than three points are provided
        (the latter two only when shapes are statically known).
    AssertionError
        If x or y have different lengths, or if fewer than
        three points are provided, and this could not be
        determined statically (raised when the graph is
        evaluated, via the embedded pytensor.raise_op.Assert
        checks).
    """

    if x.type.ndim != 1 or y.type.ndim != 1:
        raise ValueError(MSG_NOT_1D)

    size_x = x.type.shape[0]
    size_y = y.type.shape[0]

    if (size_x is not None) and (size_y is not None):

        if size_x != size_y:
            raise ValueError(MSG_SAME_LENGTH)

        if size_x < 3:
            raise ValueError(MSG_MIN_POINTS)

        return x, y

    length_x = x.shape[0]

    x_checked = cast(
        ptv.TensorVariable,
        _assert_same_length(x, ptm.eq(length_x, y.shape[0]))
    )

    x_checked = cast(
        ptv.TensorVariable,
        _assert_min_points(x_checked, length_x >= 3)
    )

    return x_checked, y


def fit(x: pt.TensorLike, y: pt.TensorLike) -> FittedCircle:
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
        three points are provided (only when shapes are
        statically known).
    LinAlgError
        If the normal equation matrix is singular. Because PyTensor
        builds a symbolic computation graph, this error is raised when
        the graph is evaluated (e.g. via `.eval()` or a compiled function),
        rather than when `fit` is called.
    """

    x_tensor, y_tensor = _validate_xy_shapes(
        x=ptb.as_tensor_variable(x),
        y=ptb.as_tensor_variable(y)
    )

    centroid = Center(x=ptm.mean(x_tensor), y=ptm.mean(y_tensor))

    x_offset = x_tensor - centroid.x
    y_offset = y_tensor - centroid.y

    matrix, vector = _construct_normal_equation(x=x_offset, y=y_offset)

    solution = cast(ptv.TensorVariable, ptl.solve(matrix, vector))

    center_offset = Center(x=0.5 * solution[0], y=0.5 * solution[1])

    dx = x_offset - center_offset.x
    dy = y_offset - center_offset.y
    dr = ptm.sqrt((dx * dx) + (dy * dy))

    return FittedCircle(
        center=Center(x=center_offset.x + centroid.x,
                      y=center_offset.y + centroid.y),
        radius=ptm.sqrt(center_offset.sum_of_squares() + solution[2]),
        roundness=ptm.max(dr) - ptm.min(dr)
    )
