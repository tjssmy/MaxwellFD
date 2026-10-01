"""Convolutional PML on a PEC-backed band.

The frequency-domain stretch is eq:stretch of ``FD_LaTeX_Reference.tex``:
each ``∂_w`` in the curls is replaced by ``s_w^{-1} ∂_w``, with

    s_w = κ_w + σ_w / (α_w + j ω ε0).

Inside a band of thickness ``d``, depth ``ρ`` is measured from the inner face
toward the PEC wall,

    σ = σ_max (ρ/d)^m,
    σ_max = -(m+1) ln R(0) / (2 η0 d),
    κ = 1 + (κ_max - 1) (ρ/d)^m,
    α = α_max (1 - ρ/d)^m.

``κ_max = 1`` and ``α_max = 0`` leave the polynomial conductivity of
eq:sigma-max. A larger ``κ_max`` and a positive ``α_max`` are the grading
described in that section. Samples on the inner face, where ``ρ = 0``, stay
unstretched.

The leapfrog keeps one auxiliary per stretched derivative. With ``e^{+jωt}``,
eq:stretch-inv is the causal ODE ``∂t ψ + β ψ = -γ ∂_w u``, where
``β = α/ε0 + σ/(κ ε0)`` and ``γ = σ/(κ² ε0)``. Holding the derivative fixed
over one step integrates that ODE by

    ψ ← e^{-β Δt} ψ - (γ/β) (1 - e^{-β Δt}) ∂_w u,

and the stretched derivative is ``κ^{-1} ∂_w u + ψ``. The update uses the
auxiliary at the new time. ``Δt`` is the leapfrog step.

``PMLSpec3`` and ``PMLProfile3`` put the same stretch on a 3D PEC box.
That profile feeds the frequency-domain curls. A 3D leapfrog auxiliary
is not built here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.operators.curl2d import CurlScale, array_curl_e, array_curl_h
from maxwell_fd.operators.curl3d import CurlScale3
from maxwell_fd.utils.constants import EPS0, ETA0

FloatArray = NDArray[np.float64]
FieldArray = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True)
class PMLSpec:
    """Cell counts and CFS parameters for the four PEC-backed bands.

    ``x_lo`` is the number of cells on the low-``x`` side, and likewise for
    the other three. Zero cells leaves that side a plain PEC wall. ``m`` is
    the polynomial order, ``r0`` is the normal-incidence target ``R(0)``,
    and ``kappa_max`` and ``alpha_max`` are the outer ``κ`` and the inner
    ``α``.
    """

    x_lo: int = 0
    x_hi: int = 0
    y_lo: int = 0
    y_hi: int = 0
    m: int = 3
    r0: float = 1e-6
    kappa_max: float = 1.0
    alpha_max: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_lo", "x_hi", "y_lo", "y_hi", "m"):
            value = getattr(self, name)
            if isinstance(value, (bool, float)) or int(value) != value:
                raise ValueError(f"{name} must be an integer, got {value}")
            object.__setattr__(self, name, int(value))
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        if min(self.x_lo, self.x_hi, self.y_lo, self.y_hi) < 0:
            raise ValueError("PML thicknesses must be non-negative")
        if not 0.0 < float(self.r0) < 1.0:
            raise ValueError(f"r0 must lie in (0, 1), got {self.r0}")
        if float(self.kappa_max) < 1.0:
            raise ValueError(f"kappa_max must be >= 1, got {self.kappa_max}")
        if float(self.alpha_max) < 0.0:
            raise ValueError(f"alpha_max must be >= 0, got {self.alpha_max}")
        object.__setattr__(self, "r0", float(self.r0))
        object.__setattr__(self, "kappa_max", float(self.kappa_max))
        object.__setattr__(self, "alpha_max", float(self.alpha_max))

    @property
    def active(self) -> bool:
        """True when at least one side has a positive thickness."""
        return self.x_lo + self.x_hi + self.y_lo + self.y_hi > 0

    @classmethod
    def box(cls, cells: int, **kwargs: float | int) -> PMLSpec:
        """The same cell count on every side."""
        return cls(x_lo=cells, x_hi=cells, y_lo=cells, y_hi=cells, **kwargs)


@dataclass(frozen=True)
class PMLSpec3:
    """Cell counts and CFS parameters for the six PEC-backed bands.

    ``z_lo`` and ``z_hi`` are the thicknesses along z. The polynomial, the
    target reflection, and the CFS grading match ``PMLSpec``.
    """

    x_lo: int = 0
    x_hi: int = 0
    y_lo: int = 0
    y_hi: int = 0
    z_lo: int = 0
    z_hi: int = 0
    m: int = 3
    r0: float = 1e-6
    kappa_max: float = 1.0
    alpha_max: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi", "m"):
            value = getattr(self, name)
            if isinstance(value, (bool, float)) or int(value) != value:
                raise ValueError(f"{name} must be an integer, got {value}")
            object.__setattr__(self, name, int(value))
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        if min(self.x_lo, self.x_hi, self.y_lo, self.y_hi, self.z_lo, self.z_hi) < 0:
            raise ValueError("PML thicknesses must be non-negative")
        if not 0.0 < float(self.r0) < 1.0:
            raise ValueError(f"r0 must lie in (0, 1), got {self.r0}")
        if float(self.kappa_max) < 1.0:
            raise ValueError(f"kappa_max must be >= 1, got {self.kappa_max}")
        if float(self.alpha_max) < 0.0:
            raise ValueError(f"alpha_max must be >= 0, got {self.alpha_max}")
        object.__setattr__(self, "r0", float(self.r0))
        object.__setattr__(self, "kappa_max", float(self.kappa_max))
        object.__setattr__(self, "alpha_max", float(self.alpha_max))

    @property
    def active(self) -> bool:
        """True when at least one side has a positive thickness."""
        return self.x_lo + self.x_hi + self.y_lo + self.y_hi + self.z_lo + self.z_hi > 0

    @classmethod
    def box(cls, cells: int, **kwargs: float | int) -> PMLSpec3:
        """The same cell count on every side."""
        return cls(
            x_lo=cells,
            x_hi=cells,
            y_lo=cells,
            y_hi=cells,
            z_lo=cells,
            z_hi=cells,
            **kwargs,
        )


@dataclass(frozen=True)
class Grade:
    """``σ``, ``κ``, and ``α`` along one staggered coordinate line."""

    sigma: FloatArray
    kappa: FloatArray
    alpha: FloatArray


class PMLProfile:
    """CFS coefficients at the node and midpoint lines of one PEC grid."""

    def __init__(self, grid: YeeGrid2D, spec: PMLSpec) -> None:
        if grid.boundary is not Boundary.PEC:
            raise ValueError("PML is backed by the PEC wall")
        _require_room(spec.x_lo, spec.x_hi, grid.nx, "x")
        _require_room(spec.y_lo, spec.y_hi, grid.ny, "y")
        self.grid = grid
        self.spec = spec
        self.x_node = _grade(
            _coords(grid.nx, grid.dx, mid=False),
            grid.a,
            grid.dx,
            spec.x_lo,
            spec.x_hi,
            spec,
        )
        self.x_mid = _grade(
            _coords(grid.nx, grid.dx, mid=True),
            grid.a,
            grid.dx,
            spec.x_lo,
            spec.x_hi,
            spec,
        )
        self.y_node = _grade(
            _coords(grid.ny, grid.dy, mid=False),
            grid.b,
            grid.dy,
            spec.y_lo,
            spec.y_hi,
            spec,
        )
        self.y_mid = _grade(
            _coords(grid.ny, grid.dy, mid=True),
            grid.b,
            grid.dy,
            spec.y_lo,
            spec.y_hi,
            spec,
        )

    def scale(self, omega: float) -> CurlScale:
        """``1/s_w(ω)`` on every derivative this polarization centers."""
        if omega == 0.0:
            raise ValueError("omega must be nonzero")
        if self.grid.polarization is Polarization.TMZ:
            return CurlScale(
                tm_y_on_hx=_expand_y(_inv_s(self.y_mid, omega), self.grid.nx + 1),
                tm_x_on_hy=_expand_x(_inv_s(self.x_mid, omega), self.grid.ny + 1),
                tm_x_on_ez=_expand_x(_inv_s(self.x_node, omega), self.grid.ny + 1),
                tm_y_on_ez=_expand_y(_inv_s(self.y_node, omega), self.grid.nx + 1),
            )
        return CurlScale(
            te_x_on_hz=_expand_x(_inv_s(self.x_mid, omega), self.grid.ny),
            te_y_on_hz=_expand_y(_inv_s(self.y_mid, omega), self.grid.nx),
            te_y_on_ex=_expand_y(_inv_s(self.y_node, omega), self.grid.nx),
            te_x_on_ey=_expand_x(_inv_s(self.x_node, omega), self.grid.ny),
        )


class PMLProfile3:
    """CFS coefficients on the node and midpoint lines of one 3D PEC grid."""

    def __init__(self, grid: YeeGrid3D, spec: PMLSpec3) -> None:
        if grid.boundary is not Boundary.PEC:
            raise ValueError("PML is backed by the PEC wall")
        _require_room(spec.x_lo, spec.x_hi, grid.nx, "x")
        _require_room(spec.y_lo, spec.y_hi, grid.ny, "y")
        _require_room(spec.z_lo, spec.z_hi, grid.nz, "z")
        self.grid = grid
        self.spec = spec
        self.x_node = _grade(
            _coords(grid.nx, grid.dx, mid=False),
            grid.a,
            grid.dx,
            spec.x_lo,
            spec.x_hi,
            spec,
        )
        self.x_mid = _grade(
            _coords(grid.nx, grid.dx, mid=True),
            grid.a,
            grid.dx,
            spec.x_lo,
            spec.x_hi,
            spec,
        )
        self.y_node = _grade(
            _coords(grid.ny, grid.dy, mid=False),
            grid.b,
            grid.dy,
            spec.y_lo,
            spec.y_hi,
            spec,
        )
        self.y_mid = _grade(
            _coords(grid.ny, grid.dy, mid=True),
            grid.b,
            grid.dy,
            spec.y_lo,
            spec.y_hi,
            spec,
        )
        self.z_node = _grade(
            _coords(grid.nz, grid.dz, mid=False),
            grid.c,
            grid.dz,
            spec.z_lo,
            spec.z_hi,
            spec,
        )
        self.z_mid = _grade(
            _coords(grid.nz, grid.dz, mid=True),
            grid.c,
            grid.dz,
            spec.z_lo,
            spec.z_hi,
            spec,
        )

    def scale(self, omega: float) -> CurlScale3:
        """``1/s_w(ω)`` on every 3D derivative."""
        if omega == 0.0:
            raise ValueError("omega must be nonzero")
        shapes = self.grid.shapes()
        x_node = _inv_s(self.x_node, omega)
        x_mid = _inv_s(self.x_mid, omega)
        y_node = _inv_s(self.y_node, omega)
        y_mid = _inv_s(self.y_mid, omega)
        z_node = _inv_s(self.z_node, omega)
        z_mid = _inv_s(self.z_mid, omega)
        return CurlScale3(
            y_on_hx=_along(y_mid, shapes["hx"], 1),
            z_on_hx=_along(z_mid, shapes["hx"], 2),
            z_on_hy=_along(z_mid, shapes["hy"], 2),
            x_on_hy=_along(x_mid, shapes["hy"], 0),
            x_on_hz=_along(x_mid, shapes["hz"], 0),
            y_on_hz=_along(y_mid, shapes["hz"], 1),
            y_on_ex=_along(y_node, shapes["ex"], 1),
            z_on_ex=_along(z_node, shapes["ex"], 2),
            z_on_ey=_along(z_node, shapes["ey"], 2),
            x_on_ey=_along(x_node, shapes["ey"], 0),
            x_on_ez=_along(x_node, shapes["ez"], 0),
            y_on_ez=_along(y_node, shapes["ez"], 1),
        )


class ConvolutionalPML:
    """Leapfrog auxiliaries for one grid, one spec, and one ``Δt``."""

    def __init__(self, grid: YeeGrid2D, spec: PMLSpec, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.grid = grid
        self.spec = spec
        self.dt = float(dt)
        self.profile = PMLProfile(grid, spec)
        shapes = grid.shapes()
        if grid.polarization is Polarization.TMZ:
            self._b_hx, self._a_hx, self._k_hx = _field_y(
                self.profile.y_mid, dt, grid.nx + 1
            )
            self._b_hy, self._a_hy, self._k_hy = _field_x(
                self.profile.x_mid, dt, grid.ny + 1
            )
            self._b_ez_x, self._a_ez_x, self._k_ez_x = _field_x(
                self.profile.x_node, dt, grid.ny + 1
            )
            self._b_ez_y, self._a_ez_y, self._k_ez_y = _field_y(
                self.profile.y_node, dt, grid.nx + 1
            )
            self.psi_hx = np.zeros(shapes["hx"])
            self.psi_hy = np.zeros(shapes["hy"])
            self.psi_ez_x = np.zeros(shapes["ez"])
            self.psi_ez_y = np.zeros(shapes["ez"])
        else:
            self._b_hz_x, self._a_hz_x, self._k_hz_x = _field_x(
                self.profile.x_mid, dt, grid.ny
            )
            self._b_hz_y, self._a_hz_y, self._k_hz_y = _field_y(
                self.profile.y_mid, dt, grid.nx
            )
            self._b_ex, self._a_ex, self._k_ex = _field_y(
                self.profile.y_node, dt, grid.nx
            )
            self._b_ey, self._a_ey, self._k_ey = _field_x(
                self.profile.x_node, dt, grid.ny
            )
            self.psi_hz_x = np.zeros(shapes["hz"])
            self.psi_hz_y = np.zeros(shapes["hz"])
            self.psi_ex = np.zeros(shapes["ex"])
            self.psi_ey = np.zeros(shapes["ey"])

    def faraday(
        self, grid: YeeGrid2D, fields: dict[str, FloatArray], dt: float
    ) -> dict[str, FloatArray]:
        """Stretched ``∇×E`` at the magnetic samples. Advances those auxiliaries."""
        self._check(grid, dt)
        if grid.polarization is Polarization.TMZ:
            return self._faraday_tm(fields["ez"])
        return self._faraday_te(fields["ex"], fields["ey"])

    def ampere(
        self, grid: YeeGrid2D, fields: dict[str, FloatArray], dt: float
    ) -> dict[str, FloatArray]:
        """Stretched ``∇×H`` at the electric samples. Advances those auxiliaries."""
        self._check(grid, dt)
        if grid.polarization is Polarization.TMZ:
            return {"ez": self._ampere_tm(fields["hx"], fields["hy"])}
        return self._ampere_te(fields["hz"])

    def _faraday_tm(self, ez: FloatArray) -> dict[str, FloatArray]:
        raw = array_curl_e(self.grid, {"ez": ez})
        d_ez_dy = raw["hx"]
        d_ez_dx = -raw["hy"]
        self.psi_hx = self._b_hx * self.psi_hx + self._a_hx * d_ez_dy
        self.psi_hy = self._b_hy * self.psi_hy + self._a_hy * d_ez_dx
        return {
            "hx": self._k_hx * d_ez_dy + self.psi_hx,
            "hy": -(self._k_hy * d_ez_dx + self.psi_hy),
        }

    def _ampere_tm(self, hx: FloatArray, hy: FloatArray) -> FloatArray:
        d_hy_dx, d_hx_dy = _tm_ampere_terms(self.grid, hx, hy)
        self.psi_ez_x = self._b_ez_x * self.psi_ez_x + self._a_ez_x * d_hy_dx
        self.psi_ez_y = self._b_ez_y * self.psi_ez_y + self._a_ez_y * d_hx_dy
        return (self._k_ez_x * d_hy_dx + self.psi_ez_x) - (
            self._k_ez_y * d_hx_dy + self.psi_ez_y
        )

    def _faraday_te(self, ex: FloatArray, ey: FloatArray) -> dict[str, FloatArray]:
        d_ey_dx, d_ex_dy = _te_faraday_terms(self.grid, ex, ey)
        self.psi_hz_x = self._b_hz_x * self.psi_hz_x + self._a_hz_x * d_ey_dx
        self.psi_hz_y = self._b_hz_y * self.psi_hz_y + self._a_hz_y * d_ex_dy
        return {
            "hz": (self._k_hz_x * d_ey_dx + self.psi_hz_x)
            - (self._k_hz_y * d_ex_dy + self.psi_hz_y)
        }

    def _ampere_te(self, hz: FloatArray) -> dict[str, FloatArray]:
        raw = array_curl_h(self.grid, {"hz": hz})
        d_hz_dy = raw["ex"]
        d_hz_dx = -raw["ey"]
        self.psi_ex = self._b_ex * self.psi_ex + self._a_ex * d_hz_dy
        self.psi_ey = self._b_ey * self.psi_ey + self._a_ey * d_hz_dx
        return {
            "ex": self._k_ex * d_hz_dy + self.psi_ex,
            "ey": -(self._k_ey * d_hz_dx + self.psi_ey),
        }

    def _check(self, grid: YeeGrid2D, dt: float) -> None:
        if float(dt) != self.dt:
            raise ValueError(f"PML coefficients were built for dt={self.dt}, got {dt}")
        same = (
            grid.nx == self.grid.nx
            and grid.ny == self.grid.ny
            and grid.dx == self.grid.dx
            and grid.dy == self.grid.dy
            and grid.polarization is self.grid.polarization
            and grid.boundary is self.grid.boundary
        )
        if not same:
            raise ValueError("PML was built for a different grid")


def sigma_max(m: int, r0: float, thickness: float) -> float:
    """Peak conductivity ``-(m+1) ln R(0) / (2 η0 d)``, eq:sigma-max."""
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    return -(m + 1) * float(np.log(r0)) / (2.0 * ETA0 * thickness)


def _require_room(n_lo: int, n_hi: int, n: int, axis: str) -> None:
    if n_lo + n_hi >= n:
        raise ValueError(
            f"PML thicknesses on {axis} ({n_lo} + {n_hi}) leave no interior cell in {n}"
        )


def _coords(count: int, spacing: float, *, mid: bool) -> FloatArray:
    if mid:
        return (np.arange(count, dtype=np.float64) + 0.5) * spacing
    return np.arange(count + 1, dtype=np.float64) * spacing


def _grade(
    coords: FloatArray,
    length: float,
    spacing: float,
    n_lo: int,
    n_hi: int,
    spec: PMLSpec | PMLSpec3,
) -> Grade:
    sigma = np.zeros(coords.shape, dtype=np.float64)
    kappa = np.ones(coords.shape, dtype=np.float64)
    alpha = np.zeros(coords.shape, dtype=np.float64)
    _paint(sigma, kappa, alpha, coords, n_lo * spacing, n_lo * spacing, spec, low=True)
    _paint(
        sigma,
        kappa,
        alpha,
        coords,
        length - n_hi * spacing,
        n_hi * spacing,
        spec,
        low=False,
    )
    return Grade(sigma=sigma, kappa=kappa, alpha=alpha)


def _paint(
    sigma: FloatArray,
    kappa: FloatArray,
    alpha: FloatArray,
    coords: FloatArray,
    face: float,
    thickness: float,
    spec: PMLSpec | PMLSpec3,
    *,
    low: bool,
) -> None:
    if thickness <= 0.0:
        return
    rho = face - coords if low else coords - face
    mask = rho > 0.0
    if not np.any(mask):
        return
    unit = rho[mask] / thickness
    peak = sigma_max(spec.m, spec.r0, thickness)
    power = unit**spec.m
    sigma[mask] = peak * power
    kappa[mask] = 1.0 + (spec.kappa_max - 1.0) * power
    alpha[mask] = spec.alpha_max * (1.0 - unit) ** spec.m


def _inv_s(grade: Grade, omega: float) -> FieldArray:
    return 1.0 / (grade.kappa + grade.sigma / (grade.alpha + 1j * float(omega) * EPS0))


def _recurrence(grade: Grade, dt: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    beta = grade.alpha / EPS0 + grade.sigma / (grade.kappa * EPS0)
    gamma = grade.sigma / (grade.kappa**2 * EPS0)
    beta_dt = beta * dt
    ratio = np.ones(beta.shape, dtype=np.float64)
    large = np.abs(beta_dt) >= 1e-8
    np.divide(np.expm1(-beta_dt), -beta_dt, out=ratio, where=large)
    return np.exp(-beta_dt), -gamma * dt * ratio, 1.0 / grade.kappa


def _along(line: NDArray, shape: tuple[int, int, int], axis: int) -> NDArray:
    if line.shape != (shape[axis],):
        raise ValueError(
            f"PML line length {line.shape[0]} does not match axis {axis} of {shape}"
        )
    view = [1, 1, 1]
    view[axis] = line.shape[0]
    return np.broadcast_to(np.reshape(line, view), shape).copy()


def _expand_x(values: NDArray, n_y: int) -> NDArray:
    return np.repeat(np.asarray(values)[:, None], n_y, axis=1)


def _expand_y(values: NDArray, n_x: int) -> NDArray:
    return np.repeat(np.asarray(values)[None, :], n_x, axis=0)


def _field_x(
    grade: Grade, dt: float, n_y: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    b, a, inv_kappa = _recurrence(grade, dt)
    return _expand_x(b, n_y), _expand_x(a, n_y), _expand_x(inv_kappa, n_y)


def _field_y(
    grade: Grade, dt: float, n_x: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    b, a, inv_kappa = _recurrence(grade, dt)
    return _expand_y(b, n_x), _expand_y(a, n_x), _expand_y(inv_kappa, n_x)


def _tm_ampere_terms(
    grid: YeeGrid2D, hx: FloatArray, hy: FloatArray
) -> tuple[FloatArray, FloatArray]:
    d_hy_dx = np.zeros(grid.shapes()["ez"], dtype=np.result_type(hy, np.float64))
    d_hx_dy = np.zeros(grid.shapes()["ez"], dtype=np.result_type(hx, np.float64))
    d_hy_dx[1:-1, 1:-1] = (hy[1:, 1:-1] - hy[:-1, 1:-1]) / grid.dx
    d_hx_dy[1:-1, 1:-1] = (hx[1:-1, 1:] - hx[1:-1, :-1]) / grid.dy
    return d_hy_dx, d_hx_dy


def _te_faraday_terms(
    grid: YeeGrid2D, ex: FloatArray, ey: FloatArray
) -> tuple[FloatArray, FloatArray]:
    d_ey_dx = (ey[1:, :] - ey[:-1, :]) / grid.dx
    d_ex_dy = (ex[:, 1:] - ex[:, :-1]) / grid.dy
    return d_ey_dx, d_ex_dy
