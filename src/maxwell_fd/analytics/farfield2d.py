"""Two-dimensional TMz near-to-far transform, ``e^{+jωt}``.

A closed contour in the unstretched region carries the tangential fields.
With ``n`` the outward unit normal and ``K_z = (n × H)_z``, the far-zone
electric field at distance ``ρ`` and angle ``φ`` from ``+x`` is

    E_z = (j/4) √(2 / (π k ρ)) exp(-j (k ρ - π/4))
          ∫ exp(j k ρ-hat · r') [j ω μ K_z - j k (n · ρ-hat) E_z] dl'.

``k = ω √(μ ε)``. The leading Hankel factor is the same one that takes
``H_0^{(2)}(kρ)`` to infinity, so a line current at the origin returns
``-(ω μ I / 4)`` times that factor. On a Yee rectangle, ``E_z`` is read
on the nodes and the tangential ``H`` is the average of the two faces
that touch the node row.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee2d import Polarization, YeeGrid2D
from maxwell_fd.operators.curl2d import array_curl_e
from maxwell_fd.utils.constants import EPS0, MU0

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def line_current_far_ez(
    phi: NDArray,
    rho: float | NDArray,
    *,
    omega: float,
    current: float = 1.0,
    mu: float = MU0,
    eps: float = EPS0,
) -> ComplexArray:
    """Far-zone ``E_z`` of a line current at the origin.

    ``phi`` is measured from ``+x``. ``rho`` is one distance, or one
    distance per angle. The kernel is the leading ``H_0^{(2)}`` asymptotic.
    """
    wavenumber, _angles, distance = _radiation(phi, rho, omega, mu, eps)
    factor = np.sqrt(2.0 / (np.pi * wavenumber * distance))
    amplitude = -0.25 * float(omega) * float(mu) * current * factor
    return np.asarray(
        amplitude * np.exp(-1j * (wavenumber * distance - np.pi / 4.0)),
        dtype=np.complex128,
    )


def tmz_far_ez(
    x: NDArray,
    y: NDArray,
    ez: NDArray,
    hx: NDArray,
    hy: NDArray,
    nx: NDArray,
    ny: NDArray,
    dl: NDArray,
    *,
    phi: NDArray,
    rho: float | NDArray,
    omega: float,
    mu: float = MU0,
    eps: float = EPS0,
) -> ComplexArray:
    """Far-zone ``E_z`` of one TMz contour.

    Every contour array is one-dimensional and the same length. ``(nx, ny)``
    is the outward unit normal. ``phi`` is measured from ``+x``.
    """
    wavenumber, angles, distance = _radiation(phi, rho, omega, mu, eps)
    samples = _contour(x, y, ez, hx, hy, nx, ny, dl)
    xx, yy, electric, magnetic_x, magnetic_y, normal_x, normal_y, length = samples
    direction_x = np.cos(angles)[:, None]
    direction_y = np.sin(angles)[:, None]
    phase = np.exp(1j * wavenumber * (direction_x * xx + direction_y * yy))
    current = normal_x * magnetic_y - normal_y * magnetic_x
    projected = normal_x * direction_x + normal_y * direction_y
    density = (
        1j * float(omega) * float(mu) * current - 1j * wavenumber * projected * electric
    )
    pattern = np.sum(phase * density * length, axis=1)
    prefactor = (1j / 4.0) * np.sqrt(2.0 / (np.pi * wavenumber * distance))
    prefactor = prefactor * np.exp(-1j * (wavenumber * distance - np.pi / 4.0))
    return np.asarray(prefactor * pattern, dtype=np.complex128)


def tmz_magnetic(
    grid: YeeGrid2D, ez: NDArray, omega: float, mu: float = MU0
) -> tuple[ComplexArray, ComplexArray]:
    """Physical ``H_x`` and ``H_y`` from ``E_z`` through the unstretched curl.

    ``H = -(∇ × E) / (j ω μ)``. A sample inside a PML needs the stretched
    curl instead, so a near-to-far contour stays off that band.
    """
    if grid.polarization is not Polarization.TMZ:
        raise ValueError("magnetic field is TMz")
    if not np.isfinite(omega) or omega == 0.0:
        raise ValueError(f"omega must be a nonzero finite number, got {omega}")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    electric = np.asarray(ez)
    if electric.shape != grid.shapes()["ez"]:
        raise ValueError(f"ez shape {electric.shape} != {grid.shapes()['ez']}")
    curl = array_curl_e(grid, {"ez": np.asarray(electric, dtype=np.complex128)})
    scale = -1.0 / (1j * float(omega) * float(mu))
    return (
        np.asarray(curl["hx"] * scale, dtype=np.complex128),
        np.asarray(curl["hy"] * scale, dtype=np.complex128),
    )


def tmz_rectangle(
    grid: YeeGrid2D,
    ez: NDArray,
    hx: NDArray,
    hy: NDArray,
    i_lo: int,
    i_hi: int,
    j_lo: int,
    j_hi: int,
) -> dict[str, NDArray]:
    """Outward TMz rectangle on ``E_z`` nodes, corners counted once.

    ``i_lo < i < i_hi`` is not the test; the edges are the node lines
    ``i_lo``, ``i_hi``, ``j_lo``, and ``j_hi``. Tangential ``H`` on an edge
    is the average of the two faces that touch that line. The indices have
    to leave those faces inside the array.
    """
    if grid.polarization is not Polarization.TMZ:
        raise ValueError("rectangle contour is TMz")
    if grid.periodic_axes() != (False, False):
        raise ValueError("rectangle contour needs a PEC wall")
    electric = np.asarray(ez)
    magnetic_x = np.asarray(hx)
    magnetic_y = np.asarray(hy)
    shapes = grid.shapes()
    if electric.shape != shapes["ez"]:
        raise ValueError(f"ez shape {electric.shape} != {shapes['ez']}")
    if magnetic_x.shape != shapes["hx"]:
        raise ValueError(f"hx shape {magnetic_x.shape} != {shapes['hx']}")
    if magnetic_y.shape != shapes["hy"]:
        raise ValueError(f"hy shape {magnetic_y.shape} != {shapes['hy']}")
    i0 = _index("i_lo", i_lo)
    i1 = _index("i_hi", i_hi)
    j0 = _index("j_lo", j_lo)
    j1 = _index("j_hi", j_hi)
    if i0 >= i1 or j0 >= j1:
        raise ValueError("rectangle edges must have a positive span")
    if i0 < 1 or j0 < 1 or i1 > grid.nx - 1 or j1 > grid.ny - 1:
        raise ValueError("contour leaves the grid")
    x, y = grid.coordinates("ez")
    bottom = np.arange(i0, i1)
    right = np.arange(j0, j1)
    top = np.arange(i1, i0, -1)
    left = np.arange(j1, j0, -1)
    sides = (
        _side(
            x[bottom],
            np.full(bottom.shape, y[j0]),
            electric[bottom, j0],
            0.5 * (magnetic_x[bottom, j0 - 1] + magnetic_x[bottom, j0]),
            np.zeros(bottom.shape, dtype=np.complex128),
            0.0,
            -1.0,
            grid.dx,
        ),
        _side(
            np.full(right.shape, x[i1]),
            y[right],
            electric[i1, right],
            np.zeros(right.shape, dtype=np.complex128),
            0.5 * (magnetic_y[i1 - 1, right] + magnetic_y[i1, right]),
            1.0,
            0.0,
            grid.dy,
        ),
        _side(
            x[top],
            np.full(top.shape, y[j1]),
            electric[top, j1],
            0.5 * (magnetic_x[top, j1 - 1] + magnetic_x[top, j1]),
            np.zeros(top.shape, dtype=np.complex128),
            0.0,
            1.0,
            grid.dx,
        ),
        _side(
            np.full(left.shape, x[i0]),
            y[left],
            electric[i0, left],
            np.zeros(left.shape, dtype=np.complex128),
            0.5 * (magnetic_y[i0 - 1, left] + magnetic_y[i0, left]),
            -1.0,
            0.0,
            grid.dy,
        ),
    )
    keys = ("x", "y", "ez", "hx", "hy", "nx", "ny", "dl")
    return {key: np.concatenate([side[key] for side in sides]) for key in keys}


def _radiation(
    phi: NDArray, rho: float | NDArray, omega: float, mu: float, eps: float
) -> tuple[float, FloatArray, FloatArray]:
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega}")
    if not np.isfinite(mu) or not np.isfinite(eps) or mu <= 0.0 or eps <= 0.0:
        raise ValueError(f"mu and eps must be positive, got {mu}, {eps}")
    angles = np.asarray(phi, dtype=np.float64).reshape(-1)
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("phi must be finite angles")
    distance = np.asarray(rho, dtype=np.float64)
    if distance.ndim == 0:
        distance = np.full(angles.shape, float(distance))
    if (
        distance.shape != angles.shape
        or not np.all(np.isfinite(distance))
        or np.any(distance <= 0.0)
    ):
        raise ValueError("rho must be a positive distance or one distance per angle")
    wavenumber = float(omega) * float(np.sqrt(mu * eps))
    return wavenumber, angles, distance


def _contour(
    x: NDArray,
    y: NDArray,
    ez: NDArray,
    hx: NDArray,
    hy: NDArray,
    nx: NDArray,
    ny: NDArray,
    dl: NDArray,
) -> tuple[NDArray, ...]:
    arrays = [
        np.asarray(
            array, dtype=np.complex128 if name in {"ez", "hx", "hy"} else np.float64
        )
        for name, array in (
            ("x", x),
            ("y", y),
            ("ez", ez),
            ("hx", hx),
            ("hy", hy),
            ("nx", nx),
            ("ny", ny),
            ("dl", dl),
        )
    ]
    if arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("contour must be a non-empty line")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("contour arrays must share one length")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("contour samples must be finite")
    return tuple(arrays)


def _index(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an index, got {value}")
    return int(value)


def _side(
    x: FloatArray,
    y: FloatArray,
    ez: ComplexArray,
    hx: ComplexArray,
    hy: ComplexArray,
    nx: float,
    ny: float,
    spacing: float,
) -> dict[str, NDArray]:
    count = x.size
    return {
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "ez": np.asarray(ez, dtype=np.complex128),
        "hx": np.asarray(hx, dtype=np.complex128),
        "hy": np.asarray(hy, dtype=np.complex128),
        "nx": np.full(count, nx, dtype=np.float64),
        "ny": np.full(count, ny, dtype=np.float64),
        "dl": np.full(count, spacing, dtype=np.float64),
    }
