"""Leapfrog CFL behavior and a PEC-cavity modal frequency."""

import numpy as np

from maxwell_fd.analytics.cavity2d import tez_hz_mode, tmz_ez_mode
from maxwell_fd.analytics.dispersion import (
    fully_discrete_omega,
    leapfrog_omega_from_samples,
)
from maxwell_fd.drivers.fdtd2d import (
    cfl_timestep,
    electric_update_coefficients,
    step_tez,
    step_tmz,
    tmz_energy,
    tmz_leapfrog_invariant,
    zeros_tez,
    zeros_tmz,
)
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import UniformIsotropic


def test_lossy_coefficients_reduce_to_the_lossless_update() -> None:
    eps = np.array([2.0])
    sigma = np.array([0.0])
    c_a, c_b = electric_update_coefficients(eps, sigma, dt=0.1)
    np.testing.assert_allclose(c_a, 1.0)
    np.testing.assert_allclose(c_b, 0.05)
    c_a, c_b = electric_update_coefficients(np.array([4.0]), np.array([2.0]), dt=0.5)
    factor = 2.0 * 0.5 / (2.0 * 4.0)
    denom = 1.0 + factor
    np.testing.assert_allclose(c_a, (1.0 - factor) / denom)
    np.testing.assert_allclose(c_b, (0.5 / 4.0) / denom)


def test_subcfl_energy_stays_bounded_and_supercfl_grows() -> None:
    grid = YeeGrid2D(20, 16, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    components = UniformIsotropic().sample_tm(grid)
    dt_cfl = cfl_timestep(grid.dx, grid.dy)
    invariant = _run_invariant(grid, components, 0.99 * dt_cfl, n_steps=2000)
    assert max(invariant) / invariant[0] < 1.01
    assert min(invariant) / invariant[0] > 0.99
    unstable = _run_physical_energy(
        grid, components, 1.01 * dt_cfl, n_steps=2000, stop_ratio=10.0
    )
    assert max(unstable) / unstable[0] > 10.0


def test_tmz_cavity_frequency_matches_dispersion() -> None:
    grid = YeeGrid2D(16, 12, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    components = UniformIsotropic().sample_tm(grid)
    mode = tmz_ez_mode(grid, 1, 1)
    state = zeros_tmz(grid)
    state.ez[:] = mode
    dt = 0.5 * cfl_timestep(grid.dx, grid.dy)
    samples = []
    for _ in range(3):
        samples.append(float(np.sum(state.ez * mode)))
        step_tmz(grid, components, state, dt)
    omega = leapfrog_omega_from_samples(*samples, dt)
    expected = fully_discrete_omega(
        kx=np.pi / grid.a,
        ky=np.pi / grid.b,
        dx=grid.dx,
        dy=grid.dy,
        dt=dt,
    )
    np.testing.assert_allclose(omega, expected, rtol=1e-6)


def test_tez_cavity_frequency_matches_dispersion() -> None:
    grid = YeeGrid2D(16, 12, 1.0, 1.0, Polarization.TEZ, Boundary.PEC)
    components = UniformIsotropic().sample_te(grid)
    mode = tez_hz_mode(grid, 1, 0)
    state = zeros_tez(grid)
    state.hz[:] = mode
    dt = 0.5 * cfl_timestep(grid.dx, grid.dy)
    samples = []
    for _ in range(3):
        samples.append(float(np.sum(state.hz * mode)))
        step_tez(grid, components, state, dt)
    omega = leapfrog_omega_from_samples(*samples, dt)
    expected = fully_discrete_omega(
        kx=np.pi / grid.a, ky=0.0, dx=grid.dx, dy=grid.dy, dt=dt
    )
    np.testing.assert_allclose(omega, expected, rtol=1e-6)


def _initial_state(grid):
    rng = np.random.default_rng(0)
    state = zeros_tmz(grid)
    state.ez[:] = rng.normal(size=state.ez.shape)
    state.hx[:] = rng.normal(size=state.hx.shape)
    state.hy[:] = rng.normal(size=state.hy.shape)
    state.ez[0, :] = state.ez[-1, :] = state.ez[:, 0] = state.ez[:, -1] = 0.0
    return state


def _run_invariant(grid, components, dt: float, n_steps: int) -> list[float]:
    state = _initial_state(grid)
    values: list[float] = []
    for _ in range(n_steps):
        ez = state.ez.copy()
        hx = state.hx.copy()
        hy = state.hy.copy()
        step_tmz(grid, components, state, dt)
        values.append(
            tmz_leapfrog_invariant(
                components, ez, state.hx, hx, state.hy, hy, grid.dx, grid.dy
            )
        )
    return values


def _run_physical_energy(
    grid, components, dt: float, n_steps: int, stop_ratio: float
) -> list[float]:
    state = _initial_state(grid)
    energies = [tmz_energy(components, state, grid.dx, grid.dy)]
    for _ in range(n_steps):
        step_tmz(grid, components, state, dt)
        energies.append(tmz_energy(components, state, grid.dx, grid.dy))
        if not np.isfinite(energies[-1]) or energies[-1] > stop_ratio * energies[0]:
            break
    return energies
