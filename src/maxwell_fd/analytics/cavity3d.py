"""Semi-discrete resonant frequencies of a 3D PEC Yee cavity.

With ``a = nx Δx`` and the same lengths on y and z,

    (ω / c)² = Σ (2 sin(k_w Δw / 2) / Δw)²,

where ``k_x = m π / a``, ``k_y = n π / b``, and ``k_z = p π / c``. This is
eq. (3.2) of ``FD_LaTeX_Reference.tex`` with the third centered difference.
Indices are non-negative, and at least two of them are positive: a single
nonzero index does not satisfy the tangential PEC condition.

A mode with exactly one index equal to zero is a single Cartesian
component. ``(m, n, 0)`` is ``E_z``, ``(m, 0, p)`` is ``E_y``, and
``(0, n, p)`` is ``E_x``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.utils.constants import C0

FloatArray = NDArray[np.float64]


def semi_discrete_omega(
    *,
    m: int,
    n: int,
    p: int,
    a: float,
    b: float,
    c: float,
    dx: float,
    dy: float,
    dz: float,
    speed: float = C0,
) -> float:
    """Continuous-time Yee frequency of cavity indices ``(m, n, p)``."""
    _require_indices(m, n, p)
    spatial = _term(m, a, dx) + _term(n, b, dy) + _term(p, c, dz)
    return float(speed * np.sqrt(spatial))


def cavity_mode(grid: YeeGrid3D, m: int, n: int, p: int) -> dict[str, FloatArray]:
    """PEC mode with exactly one index zero, sampled on the electric locations."""
    if grid.boundary is not Boundary.PEC:
        raise ValueError(f"cavity mode expects a PEC grid, got {grid.boundary.value}")
    _require_indices(m, n, p)
    if (m > 0) + (n > 0) + (p > 0) != 2:
        raise ValueError(
            f"single-component mode needs exactly one zero index, got {(m, n, p)}"
        )
    shapes = grid.shapes()
    field = {name: np.zeros(shapes[name]) for name in ("ex", "ey", "ez")}
    if p == 0:
        field["ez"] = _product(grid, "ez", (m, "x"), (n, "y"))
    elif n == 0:
        field["ey"] = _product(grid, "ey", (m, "x"), (p, "z"))
    else:
        field["ex"] = _product(grid, "ex", (n, "y"), (p, "z"))
    return field


def _require_indices(m: int, n: int, p: int) -> None:
    if m < 0 or n < 0 or p < 0 or (m > 0) + (n > 0) + (p > 0) < 2:
        raise ValueError(
            f"mode indices must be non-negative with two positive, got {(m, n, p)}"
        )


def _term(index: int, length: float, spacing: float) -> float:
    if index == 0:
        return 0.0
    wavenumber = index * np.pi / length
    return float((2.0 * np.sin(wavenumber * spacing / 2.0) / spacing) ** 2)


def _product(
    grid: YeeGrid3D,
    component: str,
    first: tuple[int, str],
    second: tuple[int, str],
) -> FloatArray:
    x, y, z = grid.coordinates(component)
    lengths = {"x": grid.a, "y": grid.b, "z": grid.c}
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    samples = {"x": xx, "y": yy, "z": zz}
    values = np.ones(xx.shape, dtype=np.float64)
    for index, axis in (first, second):
        phase = index * np.pi * samples[axis] / lengths[axis]
        values = values * np.sin(phase)
    return values
