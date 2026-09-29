"""Uniform Debye and Lorentz dielectrics, and their leapfrog auxiliaries.

Frequency responses are eq:debye and eq:lorentz-bulk in
``FD_LaTeX_Reference.tex``. The ODEs are eq:debye-ode and eq:lorentz-ode.
``J_p = ∂t P`` enters the electric update through eq:jp. A call without
``omega`` returns the instantaneous permittivity used by the leapfrog.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.drivers.fdtd2d import electric_update_coefficients
from maxwell_fd.grid.yee2d import Polarization, YeeGrid2D
from maxwell_fd.materials.volume import (
    TEComponents,
    TMComponents,
    _require_polarization,
)
from maxwell_fd.utils.constants import EPS0, MU0

FloatArray = NDArray[np.float64]
FieldMap = dict[str, FloatArray]


class DebyeDielectric:
    """One Debye pole, ``ε(ω) = ε_inf + ε0 Δε / (1 + jωτ)``."""

    def __init__(
        self,
        eps_inf: float = EPS0,
        delta_eps: float = 1.0,
        tau: float = 1.0,
        mu: float = MU0,
    ) -> None:
        if eps_inf <= 0.0 or mu <= 0.0:
            raise ValueError(f"eps_inf and mu must be positive, got {eps_inf}, {mu}")
        if delta_eps <= 0.0 or tau <= 0.0:
            raise ValueError(
                f"delta_eps and tau must be positive, got {delta_eps}, {tau}"
            )
        self.eps_inf = float(eps_inf)
        self.delta_eps = float(delta_eps)
        self.tau = float(tau)
        self.mu = float(mu)

    def permittivity(self, omega: float) -> complex:
        """Complex permittivity at one real frequency, eq:debye."""
        return self.eps_inf + EPS0 * self.delta_eps / (
            1.0 + 1j * float(omega) * self.tau
        )

    def sample_tm(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TMComponents:
        del state
        _require_polarization(grid, Polarization.TMZ)
        return _uniform_tm(grid, _eps(self, omega), self.mu)

    def sample_te(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TEComponents:
        del state
        _require_polarization(grid, Polarization.TEZ)
        return _uniform_te(grid, _eps(self, omega), self.mu)


class LorentzDielectric:
    """One Lorentz pole, ``ε(ω) = ε_inf + ε0 ω_p² / (ω0² - ω² + jγω)``."""

    def __init__(
        self,
        eps_inf: float = EPS0,
        omega_p: float = 1.0,
        omega0: float = 1.0,
        gamma: float = 0.0,
        mu: float = MU0,
    ) -> None:
        if eps_inf <= 0.0 or mu <= 0.0:
            raise ValueError(f"eps_inf and mu must be positive, got {eps_inf}, {mu}")
        if omega_p <= 0.0 or omega0 <= 0.0 or gamma < 0.0:
            raise ValueError(
                f"need omega_p > 0, omega0 > 0, gamma >= 0, got {omega_p}, {omega0}, {gamma}"
            )
        self.eps_inf = float(eps_inf)
        self.omega_p = float(omega_p)
        self.omega0 = float(omega0)
        self.gamma = float(gamma)
        self.mu = float(mu)

    def permittivity(self, omega: float) -> complex:
        """Complex permittivity at one real frequency, eq:lorentz-bulk."""
        w = float(omega)
        denom = self.omega0**2 - w**2 + 1j * self.gamma * w
        return self.eps_inf + EPS0 * self.omega_p**2 / denom

    def sample_tm(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TMComponents:
        del state
        _require_polarization(grid, Polarization.TMZ)
        return _uniform_tm(grid, _eps(self, omega), self.mu)

    def sample_te(
        self,
        grid: YeeGrid2D,
        omega: float | None = None,
        state: object | None = None,
    ) -> TEComponents:
        del state
        _require_polarization(grid, Polarization.TEZ)
        return _uniform_te(grid, _eps(self, omega), self.mu)


class LorentzAuxiliary:
    """Explicit integer-step Lorentz polarization. ``current`` returns ``J_p``."""

    def __init__(
        self, law: LorentzDielectric, shapes: dict[str, tuple[int, int]]
    ) -> None:
        self.law = law
        self.p = {name: np.zeros(shape) for name, shape in shapes.items()}
        self.p_prev = {name: np.zeros(shape) for name, shape in shapes.items()}

    def set_phasor(self, fields: dict[str, NDArray], omega: float, dt: float) -> None:
        """Store ``P`` at ``n = 0`` and ``n = -1`` for a harmonic electric phasor."""
        factor = self.law.permittivity(omega) - self.law.eps_inf
        for name, electric in fields.items():
            phasor = factor * np.asarray(electric)
            self.p[name] = np.real(phasor)
            self.p_prev[name] = np.real(phasor * np.exp(-1j * omega * dt))

    def current(self, fields: FieldMap, dt: float) -> FieldMap:
        """Advance ``P`` from ``E^n`` and return ``J_p`` at the following half-step."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        inv_dt2 = 1.0 / dt**2
        half_gamma = self.law.gamma / (2.0 * dt)
        coeff = inv_dt2 + half_gamma
        drive = EPS0 * self.law.omega_p**2
        out: FieldMap = {}
        for name, electric in fields.items():
            previous = self.p_prev[name]
            current = self.p[name]
            nxt = (
                drive * electric
                - self.law.omega0**2 * current
                + 2.0 * inv_dt2 * current
                - inv_dt2 * previous
                + half_gamma * previous
            ) / coeff
            out[name] = (nxt - current) / dt
            self.p_prev[name] = current
            self.p[name] = nxt
        return out


class DebyeAuxiliary:
    """Locally implicit Debye polarization. The hook is called after the H update."""

    def __init__(
        self, law: DebyeDielectric, shapes: dict[str, tuple[int, int]]
    ) -> None:
        self.law = law
        self.p = {name: np.zeros(shape) for name, shape in shapes.items()}

    def set_phasor(self, fields: dict[str, NDArray], omega: float) -> None:
        """Store ``P^0 = Re[(ε(ω) - ε_inf) E]``."""
        factor = self.law.permittivity(omega) - self.law.eps_inf
        for name, electric in fields.items():
            self.p[name] = np.real(factor * np.asarray(electric))

    def hook(
        self,
        electric: FieldMap,
        eps: FieldMap,
        sigma: FieldMap,
        impressed: FieldMap | None,
        dt: float,
    ) -> Callable[[NDArray | FieldMap], NDArray | FieldMap]:
        """Return ``J_p(∇×H)`` consistent with the centered Debye average."""

        def polarization_current(ampere: NDArray | FieldMap) -> NDArray | FieldMap:
            if isinstance(ampere, dict):
                return {
                    name: self._one(
                        name,
                        electric[name],
                        ampere[name],
                        None if impressed is None else impressed[name],
                        eps[name],
                        sigma[name],
                        dt,
                    )
                    for name in ampere
                }
            name = next(iter(electric))
            jz = None if impressed is None else impressed[name]
            return self._one(
                name, electric[name], ampere, jz, eps[name], sigma[name], dt
            )

        return polarization_current

    def _one(
        self,
        name: str,
        electric: FloatArray,
        ampere: NDArray,
        impressed: NDArray | None,
        eps: FloatArray,
        sigma: FloatArray,
        dt: float,
    ) -> FloatArray:
        source = ampere if impressed is None else ampere - impressed
        c_a, c_b = electric_update_coefficients(eps, sigma, dt)
        alpha = self.law.tau / dt + 0.5
        beta = self.law.tau / dt - 0.5
        gain = EPS0 * self.law.delta_eps / (2.0 * alpha)
        coupling = c_b * gain / dt
        rhs = (
            c_a * electric
            + c_b * source
            - coupling * electric
            + (c_b / (dt * alpha)) * self.p[name]
        )
        updated = rhs / (1.0 + coupling)
        nxt = gain * (updated + electric) + (beta / alpha) * self.p[name]
        current = (nxt - self.p[name]) / dt
        self.p[name] = nxt
        return current


def _eps(
    law: DebyeDielectric | LorentzDielectric, omega: float | None
) -> float | complex:
    if omega is None:
        return law.eps_inf
    return law.permittivity(omega)


def _uniform_tm(grid: YeeGrid2D, eps: float | complex, mu: float) -> TMComponents:
    shape = grid.shapes()
    return TMComponents(
        eps_z=np.full(shape["ez"], eps),
        mu_x=np.full(shape["hx"], mu),
        mu_y=np.full(shape["hy"], mu),
        sigma_z=np.zeros(shape["ez"]),
    )


def _uniform_te(grid: YeeGrid2D, eps: float | complex, mu: float) -> TEComponents:
    shape = grid.shapes()
    return TEComponents(
        eps_x=np.full(shape["ex"], eps),
        eps_y=np.full(shape["ey"], eps),
        mu_z=np.full(shape["hz"], mu),
        sigma_x=np.zeros(shape["ex"]),
        sigma_y=np.zeros(shape["ey"]),
    )
