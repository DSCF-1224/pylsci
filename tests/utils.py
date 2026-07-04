"""Test utilities."""

import numpy as np


def atol(x, factor=100):
    """Return an absolute tolerance based on the dtype of x."""
    # Work around a Pylint E1101 false positive with NumPy 2.4+.
    # See https://github.com/pylint-dev/pylint/issues/10806.
    return factor * getattr(np.finfo(np.asarray(x).dtype), 'eps')
