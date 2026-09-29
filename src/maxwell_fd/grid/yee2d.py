"""Uniform 2D Yee grid shared by the FDTD and FDFD drivers.

A grid is ``nx`` by ``ny`` cells of size ``dx`` by ``dy``. The domain is
``[0, nx*dx] × [0, ny*dy]``. Array axis 0 is ``x`` and axis 1 is ``y``.

PEC stores the boundary electric samples and holds them at zero. Periodic
identification drops the duplicate far side, so every stored sample is an
unknown and curls wrap. Locations and shapes are Table 1 of
``FD_LaTeX_Reference.tex``.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Polarization(Enum):
    """Out-of-plane field that the 2D reduction keeps."""

    TMZ = "tmz"
    TEZ = "tez"


class Boundary(Enum):
    """Outer boundary treatment on all four sides."""

    PEC = "pec"
    PERIODIC = "periodic"


class YeeGrid2D:
    """Cartesian Yee grid for one 2D polarization and one boundary kind."""

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        polarization: Polarization,
        boundary: Boundary,
    ) -> None:
        if int(nx) != nx or int(ny) != ny or nx < 1 or ny < 1:
            raise ValueError(f"nx and ny must be positive integers, got {nx}, {ny}")
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError(f"dx and dy must be positive, got {dx}, {dy}")
        if boundary is Boundary.PEC and (nx < 2 or ny < 2):
            raise ValueError("A PEC grid needs at least 2 cells in each direction")
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.polarization = polarization
        self.boundary = boundary

    @property
    def a(self) -> float:
        """Domain length along x."""
        return self.nx * self.dx

    @property
    def b(self) -> float:
        """Domain length along y."""
        return self.ny * self.dy

    def shapes(self) -> dict[str, tuple[int, int]]:
        """Full stored shape of each field component, including PEC boundaries."""
        nx, ny = self.nx, self.ny
        periodic = self.boundary is Boundary.PERIODIC
        if self.polarization is Polarization.TMZ:
            if periodic:
                return {"ez": (nx, ny), "hx": (nx, ny), "hy": (nx, ny)}
            return {"ez": (nx + 1, ny + 1), "hx": (nx + 1, ny), "hy": (nx, ny + 1)}
        if periodic:
            return {"hz": (nx, ny), "ex": (nx, ny), "ey": (nx, ny)}
        return {"hz": (nx, ny), "ex": (nx, ny + 1), "ey": (nx + 1, ny)}

    def coordinates(
        self, component: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One-dimensional x and y coordinates of ``component`` samples."""
        shapes = self.shapes()
        if component not in shapes:
            known = ", ".join(sorted(shapes))
            raise KeyError(
                f"{component} is not a {self.polarization.value} component ({known})"
            )
        x, y = _axis_coords(self, component)
        if (x.size, y.size) != shapes[component]:
            raise RuntimeError(f"coordinate shape mismatch for {component}")
        return x, y


def _axis_coords(
    grid: YeeGrid2D, component: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy
    periodic = grid.boundary is Boundary.PERIODIC
    if grid.polarization is Polarization.TMZ:
        if component == "ez":
            return _nodes(nx, dx, periodic), _nodes(ny, dy, periodic)
        if component == "hx":
            return _nodes(nx, dx, periodic), _mids(ny, dy)
        if component == "hy":
            return _mids(nx, dx), _nodes(ny, dy, periodic)
    else:
        if component == "hz":
            return _mids(nx, dx), _mids(ny, dy)
        if component == "ex":
            return _mids(nx, dx), _nodes(ny, dy, periodic)
        if component == "ey":
            return _nodes(nx, dx, periodic), _mids(ny, dy)
    raise KeyError(component)


def _nodes(n: int, h: float, periodic: bool) -> NDArray[np.float64]:
    count = n if periodic else n + 1
    return np.arange(count, dtype=np.float64) * h


def _mids(n: int, h: float) -> NDArray[np.float64]:
    """Coordinates of the n samples that sit in the middle of each cell."""
    return (np.arange(n, dtype=np.float64) + 0.5) * h
