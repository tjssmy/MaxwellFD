"""Volumetric ε, μ, and σ sampled on Yee locations.

``UniformIsotropic`` fills every sample of a component with one constant.
``StaircaseIsotropic`` fills from the sample coordinate, eq:staircase in
``FD_LaTeX_Reference.tex``. ``MappedPermittivity`` scales ε by a prescribed
array passed as ``state``. ``omega`` is accepted so a dispersive law can
return a complex permittivity; this module's laws ignore it. The contract
is Sec. 1.3 of the note.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee2d import Polarization, YeeGrid2D
from maxwell_fd.utils.constants import EPS0, MU0

FloatArray = NDArray[np.float64]
CoeffArray = NDArray[np.float64] | NDArray[np.complex128]


def _eps_array(values: NDArray) -> CoeffArray:
    """Keep a real permittivity real. A complex input is the full ε(ω)."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        return np.asarray(array, dtype=np.complex128)
    return np.asarray(array, dtype=np.float64)


def _real_array(values: NDArray, name: str) -> FloatArray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    return np.asarray(array, dtype=np.float64)


class TMComponents:
    """TMz constitutive samples, shaped like ``Ez``, ``Hx``, and ``Hy``."""

    def __init__(
        self,
        eps_z: NDArray,
        mu_x: NDArray,
        mu_y: NDArray,
        sigma_z: NDArray,
    ) -> None:
        self.eps_z = _eps_array(eps_z)
        self.mu_x = _real_array(mu_x, "mu_x")
        self.mu_y = _real_array(mu_y, "mu_y")
        self.sigma_z = _real_array(sigma_z, "sigma_z")


class TEComponents:
    """TEz constitutive samples, shaped like ``Ex``, ``Ey``, and ``Hz``."""

    def __init__(
        self,
        eps_x: NDArray,
        eps_y: NDArray,
        mu_z: NDArray,
        sigma_x: NDArray,
        sigma_y: NDArray,
    ) -> None:
        self.eps_x = _eps_array(eps_x)
        self.eps_y = _eps_array(eps_y)
        self.mu_z = _real_array(mu_z, "mu_z")
        self.sigma_x = _real_array(sigma_x, "sigma_x")
        self.sigma_y = _real_array(sigma_y, "sigma_y")


class ConstitutiveLaw(Protocol):
    """Samples bulk constitutive values onto a grid."""

    def sample_tm(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TMComponents:
        """Return TMz samples. ``omega`` and ``state`` may be ignored."""

    def sample_te(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TEComponents:
        """Return TEz samples. ``omega`` and ``state`` may be ignored."""


class UniformIsotropic:
    """One real ε, μ, and σ everywhere. σ is an electric conductivity."""

    def __init__(self, eps: float = EPS0, mu: float = MU0, sigma: float = 0.0) -> None:
        if eps <= 0.0 or mu <= 0.0:
            raise ValueError(f"eps and mu must be positive, got eps={eps}, mu={mu}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.eps = float(eps)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def sample_tm(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TMComponents:
        del omega, state
        _require_polarization(grid, Polarization.TMZ)
        shape = grid.shapes()
        return TMComponents(
            eps_z=np.full(shape["ez"], self.eps),
            mu_x=np.full(shape["hx"], self.mu),
            mu_y=np.full(shape["hy"], self.mu),
            sigma_z=np.full(shape["ez"], self.sigma),
        )

    def sample_te(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TEComponents:
        del omega, state
        _require_polarization(grid, Polarization.TEZ)
        shape = grid.shapes()
        return TEComponents(
            eps_x=np.full(shape["ex"], self.eps),
            eps_y=np.full(shape["ey"], self.eps),
            mu_z=np.full(shape["hz"], self.mu),
            sigma_x=np.full(shape["ex"], self.sigma),
            sigma_y=np.full(shape["ey"], self.sigma),
        )


def _require_polarization(grid: YeeGrid2D, polarization: Polarization) -> None:
    if grid.polarization is not polarization:
        raise ValueError(
            f"grid polarization is {grid.polarization.value}, expected {polarization.value}"
        )


def _positive_material(eps: float, mu: float, sigma: float) -> None:
    if eps <= 0.0 or mu <= 0.0:
        raise ValueError(f"eps and mu must be positive, got eps={eps}, mu={mu}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")


@dataclass(frozen=True)
class SlabY:
    """Horizontal slab ``y0 <= y < y1``. Later regions overwrite earlier ones."""

    y0: float
    y1: float
    eps: float
    mu: float = MU0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.y1 <= self.y0:
            raise ValueError(f"slab needs y1 > y0, got {self.y0}, {self.y1}")
        _positive_material(self.eps, self.mu, self.sigma)

    def contains(self, x: NDArray, y: NDArray) -> NDArray:
        del x
        return (y >= self.y0) & (y < self.y1)


@dataclass(frozen=True)
class Disk:
    """Closed disk. A sample is inside when its own coordinate is inside."""

    x: float
    y: float
    radius: float
    eps: float
    mu: float = MU0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        _positive_material(self.eps, self.mu, self.sigma)

    def contains(self, x: NDArray, y: NDArray) -> NDArray:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius**2


class StaircaseIsotropic:
    """Piecewise-constant ε, μ, and σ on the sample coordinates, eq:staircase."""

    def __init__(
        self,
        background: UniformIsotropic,
        regions: tuple[SlabY | Disk, ...] = (),
    ) -> None:
        self.background = background
        self.regions = tuple(regions)

    def sample_tm(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TMComponents:
        del omega, state
        _require_polarization(grid, Polarization.TMZ)
        eps_z, sigma_z = _paint(grid, "ez", self, electric=True)
        mu_x = _paint(grid, "hx", self, electric=False)
        mu_y = _paint(grid, "hy", self, electric=False)
        return TMComponents(eps_z=eps_z, mu_x=mu_x, mu_y=mu_y, sigma_z=sigma_z)

    def sample_te(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TEComponents:
        del omega, state
        _require_polarization(grid, Polarization.TEZ)
        eps_x, sigma_x = _paint(grid, "ex", self, electric=True)
        eps_y, sigma_y = _paint(grid, "ey", self, electric=True)
        mu_z = _paint(grid, "hz", self, electric=False)
        return TEComponents(
            eps_x=eps_x, eps_y=eps_y, mu_z=mu_z, sigma_x=sigma_x, sigma_y=sigma_y
        )


class MappedPermittivity:
    """``ε = eps * (1 + scale * state)`` on electric samples. ``state`` may be an array or a dict."""

    def __init__(
        self,
        eps: float = EPS0,
        mu: float = MU0,
        sigma: float = 0.0,
        scale: float = 0.0,
    ) -> None:
        _positive_material(eps, mu, sigma)
        self.eps = float(eps)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.scale = float(scale)

    def sample_tm(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TMComponents:
        del omega
        _require_polarization(grid, Polarization.TMZ)
        shape = grid.shapes()
        return TMComponents(
            eps_z=_scaled(self, shape["ez"], state, "ez"),
            mu_x=np.full(shape["hx"], self.mu),
            mu_y=np.full(shape["hy"], self.mu),
            sigma_z=np.full(shape["ez"], self.sigma),
        )

    def sample_te(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TEComponents:
        del omega
        _require_polarization(grid, Polarization.TEZ)
        shape = grid.shapes()
        return TEComponents(
            eps_x=_scaled(self, shape["ex"], state, "ex"),
            eps_y=_scaled(self, shape["ey"], state, "ey"),
            mu_z=np.full(shape["hz"], self.mu),
            sigma_x=np.full(shape["ex"], self.sigma),
            sigma_y=np.full(shape["ey"], self.sigma),
        )


def _paint(
    grid: YeeGrid2D, component: str, law: StaircaseIsotropic, *, electric: bool
) -> FloatArray | tuple[FloatArray, FloatArray]:
    x, y = grid.coordinates(component)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if electric:
        eps = np.full(xx.shape, law.background.eps)
        sigma = np.full(xx.shape, law.background.sigma)
        for region in law.regions:
            mask = region.contains(xx, yy)
            eps[mask] = region.eps
            sigma[mask] = region.sigma
        return eps, sigma
    mu = np.full(xx.shape, law.background.mu)
    for region in law.regions:
        mu[region.contains(xx, yy)] = region.mu
    return mu


def _scaled(
    law: MappedPermittivity,
    shape: tuple[int, int],
    state: object | None,
    name: str,
) -> FloatArray:
    eps = np.full(shape, law.eps)
    if state is None:
        return eps
    if isinstance(state, dict):
        factor = np.asarray(state[name], dtype=np.float64)
    else:
        factor = np.asarray(state, dtype=np.float64)
    return eps * (1.0 + law.scale * factor)
