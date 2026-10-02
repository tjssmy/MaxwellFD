"""Uniform 3D Yee grid shared by the FDTD and FDFD drivers.

A grid is ``nx`` by ``ny`` by ``nz`` cells of size ``dx``, ``dy``, ``dz``.
The domain is ``[0, nx*dx] × [0, ny*dy] × [0, nz*dz]``. Array axis 0 is
``x``, axis 1 is ``y``, and axis 2 is ``z``.

Each component sits on the edge or face of the Yee cell. PEC stores the
boundary electric samples and holds the tangential ones at zero. Periodic
identification drops the duplicate far side, so every stored sample is an
unknown and curls wrap. ``Boundary`` is the same enum as the 2D grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee2d import Boundary

_COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")


class YeeGrid3D:
    """Cartesian Yee grid for one boundary kind, with all six field components."""

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float,
        dz: float,
        boundary: Boundary,
    ) -> None:
        if (
            int(nx) != nx
            or int(ny) != ny
            or int(nz) != nz
            or nx < 1
            or ny < 1
            or nz < 1
        ):
            raise ValueError(
                f"nx, ny, and nz must be positive integers, got {nx}, {ny}, {nz}"
            )
        if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
            raise ValueError(f"dx, dy, and dz must be positive, got {dx}, {dy}, {dz}")
        if boundary is Boundary.PERIODIC_X:
            raise ValueError("periodic-x is a 2D boundary")
        if boundary is Boundary.PEC and (nx < 2 or ny < 2 or nz < 2):
            raise ValueError("A PEC grid needs at least 2 cells in each direction")
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.boundary = boundary

    @property
    def a(self) -> float:
        """Domain length along x."""
        return self.nx * self.dx

    @property
    def b(self) -> float:
        """Domain length along y."""
        return self.ny * self.dy

    @property
    def c(self) -> float:
        """Domain length along z."""
        return self.nz * self.dz

    def shapes(self) -> dict[str, tuple[int, int, int]]:
        """Full stored shape of each field component, including PEC boundaries."""
        nx, ny, nz = self.nx, self.ny, self.nz
        if self.boundary is Boundary.PERIODIC:
            cube = (nx, ny, nz)
            return {name: cube for name in _COMPONENTS}
        return {
            "ex": (nx, ny + 1, nz + 1),
            "ey": (nx + 1, ny, nz + 1),
            "ez": (nx + 1, ny + 1, nz),
            "hx": (nx + 1, ny, nz),
            "hy": (nx, ny + 1, nz),
            "hz": (nx, ny, nz + 1),
        }

    def coordinates(
        self, component: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """One-dimensional x, y, and z coordinates of ``component`` samples."""
        shapes = self.shapes()
        if component not in shapes:
            known = ", ".join(_COMPONENTS)
            raise KeyError(f"{component} is not a 3D component ({known})")
        x, y, z = _axis_coords(self, component)
        if (x.size, y.size, z.size) != shapes[component]:
            raise RuntimeError(f"coordinate shape mismatch for {component}")
        return x, y, z


def _axis_coords(
    grid: YeeGrid3D, component: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    periodic = grid.boundary is Boundary.PERIODIC
    node = {
        "x": _nodes(grid.nx, grid.dx, periodic),
        "y": _nodes(grid.ny, grid.dy, periodic),
        "z": _nodes(grid.nz, grid.dz, periodic),
    }
    mid = {
        "x": _mids(grid.nx, grid.dx),
        "y": _mids(grid.ny, grid.dy),
        "z": _mids(grid.nz, grid.dz),
    }
    # E is staggered along its own axis. H is staggered along the other two.
    on_midpoint = {
        "ex": {"x"},
        "ey": {"y"},
        "ez": {"z"},
        "hx": {"y", "z"},
        "hy": {"x", "z"},
        "hz": {"x", "y"},
    }[component]
    axes = tuple(
        mid[axis] if axis in on_midpoint else node[axis] for axis in ("x", "y", "z")
    )
    return axes[0], axes[1], axes[2]


def _nodes(n: int, h: float, periodic: bool) -> NDArray[np.float64]:
    count = n if periodic else n + 1
    return np.arange(count, dtype=np.float64) * h


def _mids(n: int, h: float) -> NDArray[np.float64]:
    """Coordinates of the n samples that sit in the middle of each cell."""
    return (np.arange(n, dtype=np.float64) + 0.5) * h
