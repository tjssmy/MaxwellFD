"""Leapfrog updates on the 2D Yee grid.

One step replaces ``H`` at the half-step ``n-1/2`` with ``H`` at ``n+1/2``,
then ``E`` at ``n`` with ``E`` at ``n+1``. The TMz update is
``FD_LaTeX_Reference.tex`` (2.4)--(2.7) and the TEz magnetic update is (2.8).
An impressed current and a polarization current are subtracted inside the
electric update, eq:jp. Both default to zero, which is the milestone-1 step.
The electric update uses the lossy coefficients even when ``σ = 0``. A
resistive sheet adds ``1/(Z_s Δy)`` to that ``σ`` before ``C_a`` and ``C_b``
are formed. PEC
boundaries are written back to zero after the electric update. An optional
convolutional PML replaces each derivative with the stretched derivative
from eq:stretch-inv before that update. Embedded PEC samples are cleared
with the wall. Embedded PMC samples are cleared after the magnetic update,
before Ampere's law reads them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.conductors import Conductors
from maxwell_fd.materials.volume import TEComponents, TMComponents
from maxwell_fd.operators.curl2d import array_curl_e, array_curl_h
from maxwell_fd.operators.pml import ConvolutionalPML
from maxwell_fd.utils.constants import C0

FloatArray = NDArray[np.float64]


@dataclass
class TMzState:
    """TMz fields. ``ez`` is at integer time, ``hx`` and ``hy`` at the previous half-step."""

    ez: FloatArray
    hx: FloatArray
    hy: FloatArray


@dataclass
class TEzState:
    """TEz fields. ``ex`` and ``ey`` are at integer time, ``hz`` at the previous half-step."""

    hz: FloatArray
    ex: FloatArray
    ey: FloatArray


def cfl_timestep(dx: float, dy: float, c: float = C0) -> float:
    """Largest stable ``dt``, ``1/(c √(Δx⁻²+Δy⁻²))``, eq. (3.1) of the note."""
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")
    if c <= 0.0:
        raise ValueError("c must be positive")
    return 1.0 / (c * float(np.sqrt(1.0 / dx**2 + 1.0 / dy**2)))


def electric_update_coefficients(
    eps: FloatArray,
    sigma: FloatArray,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Lossy leapfrog pair ``(C_a, C_b)`` for ``E ← C_a E + C_b (∇×H)``, eq. (2.7)."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if np.iscomplexobj(eps) or np.iscomplexobj(sigma):
        raise ValueError("leapfrog eps and sigma must be real")
    eps_a = np.asarray(eps, dtype=np.float64)
    sigma_a = np.asarray(sigma, dtype=np.float64)
    if np.any(eps_a <= 0.0):
        raise ValueError("eps must be positive")
    factor = sigma_a * dt / (2.0 * eps_a)
    denom = 1.0 + factor
    c_a = (1.0 - factor) / denom
    c_b = (dt / eps_a) / denom
    return c_a, c_b


def zeros_tmz(grid: YeeGrid2D) -> TMzState:
    """Allocate a zero TMz state with the grid's stored shapes."""
    _require(grid, Polarization.TMZ)
    shape = grid.shapes()
    return TMzState(
        ez=np.zeros(shape["ez"]),
        hx=np.zeros(shape["hx"]),
        hy=np.zeros(shape["hy"]),
    )


def zeros_tez(grid: YeeGrid2D) -> TEzState:
    """Allocate a zero TEz state with the grid's stored shapes."""
    _require(grid, Polarization.TEZ)
    shape = grid.shapes()
    return TEzState(
        hz=np.zeros(shape["hz"]),
        ex=np.zeros(shape["ex"]),
        ey=np.zeros(shape["ey"]),
    )


def step_tmz(
    grid: YeeGrid2D,
    components: TMComponents,
    state: TMzState,
    dt: float,
    impressed_j: FloatArray | None = None,
    polarization_current: FloatArray | Callable[[FloatArray], FloatArray] | None = None,
    *,
    pml: ConvolutionalPML | None = None,
    conductors: Conductors | None = None,
) -> None:
    """Advance a TMz state by one full time step, in place.

    ``polarization_current`` is either ``J_p`` at the new half-step or a
    callable of ``∇×H`` invoked after the magnetic update. A callable is how
    the implicit Debye auxiliary sees the Ampere term. ``pml`` stretches the
    derivatives and advances its own auxiliaries. ``conductors`` clears
    embedded PEC and PMC samples.
    """
    _require(grid, Polarization.TMZ)
    _match_tm(grid, components, state)
    masks = _masks(grid, conductors)
    if pml is None:
        curls = array_curl_e(grid, {"ez": state.ez})
    else:
        curls = pml.faraday(grid, {"ez": state.ez}, dt)
    state.hx -= (dt / components.mu_x) * curls["hx"]
    state.hy -= (dt / components.mu_y) * curls["hy"]
    if masks is not None:
        state.hx[masks.pmc["hx"]] = 0.0
        state.hy[masks.pmc["hy"]] = 0.0
    c_a, c_b = electric_update_coefficients(components.eps_z, components.sigma_z, dt)
    if pml is None:
        ampere = array_curl_h(grid, {"hx": state.hx, "hy": state.hy})["ez"]
    else:
        ampere = pml.ampere(grid, {"hx": state.hx, "hy": state.hy}, dt)["ez"]
    extra = _electric_current(ampere, impressed_j, polarization_current)
    state.ez *= c_a
    state.ez += c_b * (ampere - extra)
    _enforce_tm_pec(grid, state)
    if masks is not None:
        state.ez[masks.pec["ez"]] = 0.0


def step_tez(
    grid: YeeGrid2D,
    components: TEComponents,
    state: TEzState,
    dt: float,
    impressed_j: dict[str, FloatArray] | None = None,
    polarization_current: (
        dict[str, FloatArray]
        | Callable[[dict[str, FloatArray]], dict[str, FloatArray]]
        | None
    ) = None,
    *,
    pml: ConvolutionalPML | None = None,
    conductors: Conductors | None = None,
) -> None:
    """Advance a TEz state by one full time step, in place."""
    _require(grid, Polarization.TEZ)
    _match_te(grid, components, state)
    masks = _masks(grid, conductors)
    if pml is None:
        curl_z = array_curl_e(grid, {"ex": state.ex, "ey": state.ey})["hz"]
    else:
        curl_z = pml.faraday(grid, {"ex": state.ex, "ey": state.ey}, dt)["hz"]
    state.hz -= (dt / components.mu_z) * curl_z
    if masks is not None:
        state.hz[masks.pmc["hz"]] = 0.0
    if pml is None:
        curls = array_curl_h(grid, {"hz": state.hz})
    else:
        curls = pml.ampere(grid, {"hz": state.hz}, dt)
    currents = _te_currents(curls, impressed_j, polarization_current)
    for name, eps, sigma, field in (
        ("ex", components.eps_x, components.sigma_x, state.ex),
        ("ey", components.eps_y, components.sigma_y, state.ey),
    ):
        c_a, c_b = electric_update_coefficients(eps, sigma, dt)
        updated = c_a * field + c_b * (curls[name] - currents[name])
        field[...] = updated
    _enforce_te_pec(grid, state)
    if masks is not None:
        state.ex[masks.pec["ex"]] = 0.0
        state.ey[masks.pec["ey"]] = 0.0


def tmz_energy(
    components: TMComponents, state: TMzState, dx: float, dy: float
) -> float:
    """Sum of electric and magnetic sample energies at the stored time levels.

    ``E`` and ``H`` are a half-step apart, so this value oscillates even when
    the scheme is lossless. Use :func:`tmz_leapfrog_invariant` for the quadratic
    form the leapfrog update conserves.
    """
    electric = np.sum(components.eps_z * state.ez**2)
    magnetic = np.sum(components.mu_x * state.hx**2) + np.sum(
        components.mu_y * state.hy**2
    )
    return 0.5 * float(electric + magnetic) * dx * dy


def tmz_leapfrog_invariant(
    components: TMComponents,
    ez: FloatArray,
    hx_new: FloatArray,
    hx_old: FloatArray,
    hy_new: FloatArray,
    hy_old: FloatArray,
    dx: float,
    dy: float,
) -> float:
    """Lossless leapfrog invariant, eq. (3.6) of ``FD_LaTeX_Reference.tex``.

    Pass ``E`` from before the step and ``H`` from after it, together with the
    ``H`` that entered the step. Conductivity removes this conservation law.
    """
    electric = np.sum(components.eps_z * ez**2)
    magnetic = np.sum(components.mu_x * hx_new * hx_old) + np.sum(
        components.mu_y * hy_new * hy_old
    )
    return 0.5 * float(electric + magnetic) * dx * dy


def _electric_current(
    ampere: FloatArray,
    impressed: FloatArray | None,
    polarization: FloatArray | Callable[[FloatArray], FloatArray] | None,
) -> FloatArray:
    if callable(polarization):
        polarization = polarization(ampere)
    total = np.zeros(ampere.shape, dtype=np.float64)
    if impressed is not None:
        total = total + impressed
    if polarization is not None:
        total = total + polarization
    return total


def _te_currents(
    ampere: dict[str, FloatArray],
    impressed: dict[str, FloatArray] | None,
    polarization: (
        dict[str, FloatArray]
        | Callable[[dict[str, FloatArray]], dict[str, FloatArray]]
        | None
    ),
) -> dict[str, FloatArray]:
    if callable(polarization):
        polarization = polarization(ampere)
    return {
        name: _electric_current(
            ampere[name],
            None if impressed is None else impressed[name],
            None if polarization is None else polarization[name],
        )
        for name in ampere
    }


def _masks(grid: YeeGrid2D, conductors: Conductors | None):
    if conductors is None or not conductors.active:
        return None
    return conductors.masks(grid)


def _enforce_tm_pec(grid: YeeGrid2D, state: TMzState) -> None:
    if grid.boundary is Boundary.PEC:
        state.ez[0, :] = 0.0
        state.ez[-1, :] = 0.0
        state.ez[:, 0] = 0.0
        state.ez[:, -1] = 0.0
    elif grid.boundary is Boundary.PERIODIC_X:
        state.ez[:, 0] = 0.0
        state.ez[:, -1] = 0.0


def _enforce_te_pec(grid: YeeGrid2D, state: TEzState) -> None:
    if grid.boundary is Boundary.PEC:
        state.ex[:, 0] = 0.0
        state.ex[:, -1] = 0.0
        state.ey[0, :] = 0.0
        state.ey[-1, :] = 0.0
    elif grid.boundary is Boundary.PERIODIC_X:
        state.ex[:, 0] = 0.0
        state.ex[:, -1] = 0.0


def _require(grid: YeeGrid2D, polarization: Polarization) -> None:
    if grid.polarization is not polarization:
        raise ValueError(
            f"grid polarization is {grid.polarization.value}, expected {polarization.value}"
        )


def _match_tm(grid: YeeGrid2D, components: TMComponents, state: TMzState) -> None:
    shapes = grid.shapes()
    _shape(state.ez, shapes["ez"], "ez")
    _shape(state.hx, shapes["hx"], "hx")
    _shape(state.hy, shapes["hy"], "hy")
    _shape(components.eps_z, shapes["ez"], "eps_z")
    _shape(components.mu_x, shapes["hx"], "mu_x")
    _shape(components.mu_y, shapes["hy"], "mu_y")
    _shape(components.sigma_z, shapes["ez"], "sigma_z")


def _match_te(grid: YeeGrid2D, components: TEComponents, state: TEzState) -> None:
    shapes = grid.shapes()
    _shape(state.hz, shapes["hz"], "hz")
    _shape(state.ex, shapes["ex"], "ex")
    _shape(state.ey, shapes["ey"], "ey")
    _shape(components.eps_x, shapes["ex"], "eps_x")
    _shape(components.eps_y, shapes["ey"], "eps_y")
    _shape(components.mu_z, shapes["hz"], "mu_z")
    _shape(components.sigma_x, shapes["ex"], "sigma_x")
    _shape(components.sigma_y, shapes["ey"], "sigma_y")


def _shape(field: NDArray, shape: tuple[int, int], name: str) -> None:
    if field.shape != shape:
        raise ValueError(f"{name} shape {field.shape} != {shape}")
