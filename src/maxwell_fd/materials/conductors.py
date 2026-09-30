"""Staircase PEC and PMC objects.

A volumetric law paints ε, μ, and σ. A conductor does not. PEC removes every
electric sample whose coordinate lies in one of its regions, and PMC removes
every magnetic sample in its regions. The leapfrog writes the removed samples
back to zero. Setting ε to infinity, or μ to zero, is a different model and
is not what these objects do.

Regions use the same tests as the staircase media. A band is the half-open
interval ``y0 <= y < y1``. A circle is closed. A sample on the circle is
inside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee2d import Polarization, YeeGrid2D

BoolArray = NDArray[np.bool_]


class ConductorRegion(Protocol):
    """Geometry test sampled at a Yee coordinate."""

    def contains(self, x: NDArray, y: NDArray) -> NDArray:
        """True where the sample coordinate lies in the region."""


@dataclass(frozen=True)
class YBand:
    """Horizontal band ``y0 <= y < y1``."""

    y0: float
    y1: float

    def __post_init__(self) -> None:
        if not self.y1 > self.y0:
            raise ValueError(f"band needs y1 > y0, got {self.y0}, {self.y1}")

    def contains(self, x: NDArray, y: NDArray) -> BoolArray:
        del x
        return (y >= self.y0) & (y < self.y1)


@dataclass(frozen=True)
class Circle:
    """Closed circle. A sample is inside when its own coordinate is inside."""

    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        if not self.radius > 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def contains(self, x: NDArray, y: NDArray) -> BoolArray:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius**2


@dataclass(frozen=True)
class ConductorMasks:
    """Boolean sample masks. True means the sample is not an unknown."""

    pec: dict[str, BoolArray]
    pmc: dict[str, BoolArray]


class Conductors:
    """PEC and PMC staircase objects on one grid."""

    def __init__(
        self,
        pec: tuple[ConductorRegion, ...] = (),
        pmc: tuple[ConductorRegion, ...] = (),
    ) -> None:
        self.pec = tuple(pec)
        self.pmc = tuple(pmc)
        for region in self.pec + self.pmc:
            contains = getattr(region, "contains", None)
            if not callable(contains):
                raise TypeError("a conductor region needs contains(x, y)")

    @property
    def active(self) -> bool:
        """True when at least one PEC or PMC region is present."""
        return bool(self.pec or self.pmc)

    def masks(self, grid: YeeGrid2D) -> ConductorMasks:
        """Sample masks for ``grid``. PEC is electric. PMC is magnetic."""
        if grid.polarization is Polarization.TMZ:
            electric = ("ez",)
            magnetic = ("hx", "hy")
        else:
            electric = ("ex", "ey")
            magnetic = ("hz",)
        return ConductorMasks(
            pec={name: _union(grid, name, self.pec) for name in electric},
            pmc={name: _union(grid, name, self.pmc) for name in magnetic},
        )


def _union(
    grid: YeeGrid2D, component: str, regions: tuple[ConductorRegion, ...]
) -> BoolArray:
    x, y = grid.coordinates(component)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = np.zeros(xx.shape, dtype=bool)
    for region in regions:
        inside = np.asarray(region.contains(xx, yy), dtype=bool)
        if inside.shape != mask.shape:
            raise ValueError(f"{component} mask shape {inside.shape} != {mask.shape}")
        mask |= inside
    return mask
