"""Debye and Lorentz leapfrog against the complex-permittivity phasor."""

import numpy as np

from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.drivers.fdtd2d import (
    cfl_timestep,
    step_tez,
    step_tmz,
    tmz_leapfrog_invariant,
    zeros_tez,
    zeros_tmz,
)
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.dispersion import (
    DebyeAuxiliary,
    DebyeDielectric,
    LorentzAuxiliary,
    LorentzDielectric,
)
from maxwell_fd.materials.volume import Disk, StaircaseIsotropic, UniformIsotropic
from maxwell_fd.operators.curl2d import array_curl_e
from maxwell_fd.utils.constants import EPS0, MU0


def test_lossless_staircase_keeps_the_leapfrog_invariant() -> None:
    grid = YeeGrid2D(20, 16, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    law = StaircaseIsotropic(
        UniformIsotropic(),
        (Disk(6.0, 5.0, 3.5, eps=4.0 * EPS0, mu=1.5 * MU0),),
    )
    components = law.sample_tm(grid)
    dt = 0.5 * cfl_timestep(grid.dx, grid.dy)
    rng = np.random.default_rng(1)
    state = zeros_tmz(grid)
    state.ez[1:-1, 1:-1] = rng.normal(size=(grid.nx - 1, grid.ny - 1))
    state.hx[:] = rng.normal(size=state.hx.shape)
    state.hy[:] = rng.normal(size=state.hy.shape)
    values = []
    for _ in range(500):
        ez, hx, hy = state.ez.copy(), state.hx.copy(), state.hy.copy()
        step_tmz(grid, components, state, dt)
        values.append(
            tmz_leapfrog_invariant(
                components, ez, state.hx, hx, state.hy, hy, grid.dx, grid.dy
            )
        )
    assert max(values) / values[0] < 1.01
    assert min(values) / values[0] > 0.99


def test_debye_and_lorentz_hold_a_harmonic_phasor() -> None:
    for law in (_debye(), _lorentz()):
        for pol in (Polarization.TMZ, Polarization.TEZ):
            error = _one_period_error(law, pol)
            assert error < 3.0e-3, f"{law.__class__.__name__} {pol.value} error {error}"


def _debye() -> DebyeDielectric:
    return DebyeDielectric(eps_inf=EPS0, delta_eps=1.5, tau=1.0)


def _lorentz() -> LorentzDielectric:
    return LorentzDielectric(eps_inf=EPS0, omega_p=1.0, omega0=1.0, gamma=0.1)


def _one_period_error(
    law: DebyeDielectric | LorentzDielectric, pol: Polarization
) -> float:
    grid = YeeGrid2D(12, 10, 1.0, 1.0, pol, Boundary.PEC)
    dt = 0.4 * cfl_timestep(grid.dx, grid.dy)
    # An integer step count lands on one period. 2π/126 is 0.04987, the plan's ω Δt = 0.05.
    n_period = 126
    omega = 2.0 * np.pi / (n_period * dt)
    if isinstance(law, DebyeDielectric):
        law.tau = 1.0 / omega
    else:
        law.omega0 = 2.0 * omega
        law.omega_p = 0.5 * law.omega0
        law.gamma = 0.1 * law.omega0
    components = law.sample_tm(grid) if pol is Polarization.TMZ else law.sample_te(grid)
    dispersive = (
        law.sample_tm(grid, omega=omega)
        if pol is Polarization.TMZ
        else law.sample_te(grid, omega=omega)
    )
    operator = FDFDOperator(grid, dispersive)
    rng = np.random.default_rng(3)
    free = rng.normal(size=operator.layout.n_e) + 1j * rng.normal(
        size=operator.layout.n_e
    )
    current = -(operator.system_matrix(omega) @ free)
    phasor = _scatter_electric(operator, free)
    real_fields = {name: np.real(values) for name, values in phasor.items()}
    if pol is Polarization.TMZ:
        return _run_tmz(
            grid,
            components,
            law,
            phasor,
            real_fields,
            current,
            operator,
            omega,
            dt,
            n_period,
        )
    return _run_tez(
        grid,
        components,
        law,
        phasor,
        real_fields,
        current,
        operator,
        omega,
        dt,
        n_period,
    )


def _run_tmz(
    grid, components, law, phasor, real_fields, current, operator, omega, dt, n_period
) -> float:
    state = zeros_tmz(grid)
    state.ez[:] = real_fields["ez"]
    hx, hy = _magnetic_tm(grid, phasor["ez"], omega, components.mu_x, components.mu_y)
    phase_h = np.exp(-1j * omega * dt / 2.0)
    state.hx[:] = np.real(hx * phase_h)
    state.hy[:] = np.real(hy * phase_h)
    impressed = operator.layout.e[0].scatter(current)
    if isinstance(law, LorentzDielectric):
        aux = LorentzAuxiliary(law, {"ez": state.ez.shape})
        aux.set_phasor(phasor, omega, dt)
        for n in range(n_period):
            jz = np.real(impressed * np.exp(1j * omega * (n + 0.5) * dt))
            jp = aux.current({"ez": state.ez}, dt)["ez"]
            step_tmz(
                grid, components, state, dt, impressed_j=jz, polarization_current=jp
            )
    else:
        aux = DebyeAuxiliary(law, {"ez": state.ez.shape})
        aux.set_phasor(phasor, omega)
        drive = {"ez": np.zeros(state.ez.shape)}
        hook = aux.hook(
            {"ez": state.ez},
            {"ez": components.eps_z},
            {"ez": components.sigma_z},
            drive,
            dt,
        )
        for n in range(n_period):
            drive["ez"] = np.real(impressed * np.exp(1j * omega * (n + 0.5) * dt))
            step_tmz(
                grid,
                components,
                state,
                dt,
                impressed_j=drive["ez"],
                polarization_current=hook,
            )
    advanced = np.real(phasor["ez"] * np.exp(1j * omega * n_period * dt))
    return float(np.linalg.norm(state.ez - advanced) / np.linalg.norm(advanced))


def _run_tez(
    grid, components, law, phasor, real_fields, current, operator, omega, dt, n_period
) -> float:
    state = zeros_tez(grid)
    state.ex[:] = real_fields["ex"]
    state.ey[:] = real_fields["ey"]
    hz = _magnetic_te(grid, phasor["ex"], phasor["ey"], omega, components.mu_z)
    state.hz[:] = np.real(hz * np.exp(-1j * omega * dt / 2.0))
    n_ex = operator.layout.e[0].i.size
    impressed = {
        "ex": operator.layout.e[0].scatter(current[:n_ex]),
        "ey": operator.layout.e[1].scatter(current[n_ex:]),
    }
    shapes = {"ex": state.ex.shape, "ey": state.ey.shape}
    if isinstance(law, LorentzDielectric):
        aux = LorentzAuxiliary(law, shapes)
        aux.set_phasor(phasor, omega, dt)
        for n in range(n_period):
            phase = np.exp(1j * omega * (n + 0.5) * dt)
            jz = {name: np.real(values * phase) for name, values in impressed.items()}
            jp = aux.current({"ex": state.ex, "ey": state.ey}, dt)
            step_tez(
                grid, components, state, dt, impressed_j=jz, polarization_current=jp
            )
    else:
        aux = DebyeAuxiliary(law, shapes)
        aux.set_phasor(phasor, omega)
        drive = {name: np.zeros(shapes[name]) for name in shapes}
        hook = aux.hook(
            {"ex": state.ex, "ey": state.ey},
            {"ex": components.eps_x, "ey": components.eps_y},
            {"ex": components.sigma_x, "ey": components.sigma_y},
            drive,
            dt,
        )
        for n in range(n_period):
            phase = np.exp(1j * omega * (n + 0.5) * dt)
            for name in drive:
                drive[name] = np.real(impressed[name] * phase)
            step_tez(
                grid,
                components,
                state,
                dt,
                impressed_j=drive,
                polarization_current=hook,
            )
    advanced_x = np.real(phasor["ex"] * np.exp(1j * omega * n_period * dt))
    advanced_y = np.real(phasor["ey"] * np.exp(1j * omega * n_period * dt))
    num = np.concatenate([state.ex.ravel(), state.ey.ravel()])
    ref = np.concatenate([advanced_x.ravel(), advanced_y.ravel()])
    return float(np.linalg.norm(num - ref) / np.linalg.norm(ref))


def _scatter_electric(
    operator: FDFDOperator, free: np.ndarray
) -> dict[str, np.ndarray]:
    if len(operator.layout.e) == 1:
        return {"ez": operator.layout.e[0].scatter(free)}
    n_ex = operator.layout.e[0].i.size
    return {
        "ex": operator.layout.e[0].scatter(free[:n_ex]),
        "ey": operator.layout.e[1].scatter(free[n_ex:]),
    }


def _magnetic_tm(grid, ez, omega, mu_x, mu_y):
    curls = array_curl_e(grid, {"ez": ez})
    scale = -1.0 / (1j * omega)
    return curls["hx"] * scale / mu_x, curls["hy"] * scale / mu_y


def _magnetic_te(grid, ex, ey, omega, mu_z):
    curls = array_curl_e(grid, {"ex": ex, "ey": ey})
    return -curls["hz"] / (1j * omega * mu_z)
