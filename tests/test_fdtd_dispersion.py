"""Periodic free-space standing wave against the fully discrete dispersion relation."""

import numpy as np

from maxwell_fd.analytics.dispersion import (
    fully_discrete_omega,
    leapfrog_omega_from_samples,
)
from maxwell_fd.drivers.fdtd2d import cfl_timestep, step_tmz, zeros_tmz
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import UniformIsotropic


def test_periodic_standing_wave_matches_dispersion() -> None:
    grid = YeeGrid2D(32, 8, 1.0, 1.0, Polarization.TMZ, Boundary.PERIODIC)
    components = UniformIsotropic().sample_tm(grid)
    i = np.arange(grid.nx)[:, None]
    mode = np.cos(2.0 * np.pi * i / grid.nx) * np.ones((1, grid.ny))
    state = zeros_tmz(grid)
    state.ez[:] = mode
    dt = 0.5 * cfl_timestep(grid.dx, grid.dy)
    samples = []
    for _ in range(3):
        samples.append(float(np.sum(state.ez * mode)))
        step_tmz(grid, components, state, dt)
    omega = leapfrog_omega_from_samples(*samples, dt)
    expected = fully_discrete_omega(
        kx=2.0 * np.pi / grid.a,
        ky=0.0,
        dx=grid.dx,
        dy=grid.dy,
        dt=dt,
    )
    np.testing.assert_allclose(omega, expected, rtol=1e-6)
