"""Result models for circle fitting algorithms."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Center:
    """Center of a circle."""

    x: Any
    y: Any


@dataclass(frozen=True)
class FittedCircle:
    """Circle obtained from a fitting algorithm."""

    center: Center
    radius: Any
    roundness: Any
