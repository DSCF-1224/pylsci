"""Result models for circle fitting algorithms."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Center:
    """Center of a circle in 2D (xy) plane."""

    x: Any
    y: Any

    def sum_of_squares(self) -> Any:
        """Return x^2 + y^2."""
        return (self.x * self.x) + (self.y * self.y)


@dataclass(frozen=True)
class Circle:
    """A circle in 2D (xy) plane"""

    center: Center
    radius: Any


@dataclass(frozen=True)
class FittedCircle(Circle):
    """Circle obtained from a fitting algorithm."""

    roundness: Any
