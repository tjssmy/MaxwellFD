"""Semi-discrete resonant frequencies of a 2D PEC Yee cavity.

With ``a = nx Δx``, ``b = ny Δy``, ``k_x = m π / a`` and ``k_y = n π / b``,

    (ω / c)² = (2 sin(k_x Δx / 2) / Δx)² + (2 sin(k_y Δy / 2) / Δy)².

TMz modes need ``m ≥ 1`` and ``n ≥ 1``. TEz modes also allow ``m = 0`` or
``n = 0``, but not both. This is eq. (3.2) of ``FD_LaTeX_Reference.tex``.
The mode shapes are (3.4) and (3.5).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.utils.constants import C0


def semi_discrete_omega(
    *,
    m: int,
    n: int,
    a: float,
    b: float,
    dx: float,
    dy: float,
    c: float = C0,
) -> float:
    """Continuous-time Yee frequency of cavity indices ``(m, n)``, eq. (3.2)."""
    if m < 0 or n < 0 or (m == 0 and n == 0):
        raise ValueError(
            f"mode indices must be non-negative and not both zero, got {(m, n)}"
        )
    kx = m * np.pi / a
    ky = n * np.pi / b
    spatial = (2.0 * np.sin(kx * dx / 2.0) / dx) ** 2 + (
        2.0 * np.sin(ky * dy / 2.0) / dy
    ) ** 2
    return float(c * np.sqrt(spatial))


def tmz_ez_mode(grid: YeeGrid2D, m: int, n: int) -> NDArray[np.float64]:
    """PEC TMz mode ``sin(m π x / a) sin(n π y / b)`` on ``Ez`` nodes, eq. (3.4)."""
    _require_pec(grid, Polarization.TMZ)
    if m < 1 or n < 1:
        raise ValueError(f"TMz cavity modes need m ≥ 1 and n ≥ 1, got {(m, n)}")
    x, y = grid.coordinates("ez")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.sin(m * np.pi * xx / grid.a) * np.sin(n * np.pi * yy / grid.b)


def tez_hz_mode(grid: YeeGrid2D, m: int, n: int) -> NDArray[np.float64]:
    """PEC TEz mode ``cos(m π x / a) cos(n π y / b)`` on ``Hz`` centers, eq. (3.5)."""
    _require_pec(grid, Polarization.TEZ)
    if m < 0 or n < 0 or (m == 0 and n == 0):
        raise ValueError(
            f"TEz cavity modes need non-negative indices, not both zero, got {(m, n)}"
        )
    x, y = grid.coordinates("hz")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.cos(m * np.pi * xx / grid.a) * np.cos(n * np.pi * yy / grid.b)


def _require_pec(grid: YeeGrid2D, polarization: Polarization) -> None:
    if grid.boundary is not Boundary.PEC or grid.polarization is not polarization:
        raise ValueError(
            f"expected PEC {polarization.value}, got {grid.boundary.value} {grid.polarization.value}"
        )
