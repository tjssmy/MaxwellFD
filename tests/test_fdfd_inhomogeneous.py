"""Spatially varying real media in the sparse frequency-domain system."""

import numpy as np
from scipy import linalg
from scipy.optimize import brentq

from maxwell_fd.analytics.cavity2d import semi_discrete_omega, tmz_ez_mode
from maxwell_fd.analytics.dispersion import (
    fully_discrete_omega,
    leapfrog_omega_from_samples,
)
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.drivers.fdtd2d import cfl_timestep, step_tmz, zeros_tmz
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import (
    MappedPermittivity,
    SlabY,
    StaircaseIsotropic,
    TEComponents,
    TMComponents,
    UniformIsotropic,
)
from maxwell_fd.utils.constants import C0, EPS0, MU0


def test_conductivity_and_varying_coefficients_are_recovered() -> None:
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(16, 12, 0.05, 0.04, pol, Boundary.PEC)
        law = UniformIsotropic(sigma=1.0e-3)
        components = (
            law.sample_tm(grid) if pol is Polarization.TMZ else law.sample_te(grid)
        )
        _recover(grid, components)
        varying = _varying_components(grid)
        _recover(grid, varying)


def test_mapped_permittivity_matches_the_prescribed_state() -> None:
    grid = YeeGrid2D(12, 10, 0.05, 0.05, Polarization.TMZ, Boundary.PEC)
    x, y = grid.coordinates("ez")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    state = 0.2 * np.sin(np.pi * xx / grid.a) * np.sin(np.pi * yy / grid.b)
    sampled = MappedPermittivity(scale=0.5).sample_tm(grid, state=state)
    expected = EPS0 * (1.0 + 0.5 * state)
    np.testing.assert_allclose(sampled.eps_z, expected)
    hx_x, hx_y = grid.coordinates("hx")
    hy_x, hy_y = grid.coordinates("hy")
    mu_x = MU0 * (
        1.5 + 0.1 * np.sin(np.pi * np.meshgrid(hx_x, hx_y, indexing="ij")[0] / grid.a)
    )
    mu_y = MU0 * (
        1.2 + 0.1 * np.cos(np.pi * np.meshgrid(hy_x, hy_y, indexing="ij")[1] / grid.b)
    )
    components = TMComponents(sampled.eps_z, mu_x, mu_y, sampled.sigma_z)
    _recover(grid, components)

    te = YeeGrid2D(12, 10, 0.05, 0.05, Polarization.TEZ, Boundary.PEC)
    state_x = np.full(te.shapes()["ex"], 0.25)
    state_y = np.full(te.shapes()["ey"], -0.1)
    sampled_te = MappedPermittivity(scale=0.5).sample_te(
        te, state={"ex": state_x, "ey": state_y}
    )
    np.testing.assert_allclose(sampled_te.eps_x, EPS0 * (1.0 + 0.5 * state_x))
    np.testing.assert_allclose(sampled_te.eps_y, EPS0 * (1.0 + 0.5 * state_y))


def test_dirichlet_trace_is_lifted_onto_the_right_hand_side() -> None:
    rng = np.random.default_rng(7)
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(10, 8, 0.05, 0.04, pol, Boundary.PEC)
        components = (
            UniformIsotropic(sigma=2.0e-4).sample_tm(grid)
            if pol is Polarization.TMZ
            else UniformIsotropic(sigma=2.0e-4).sample_te(grid)
        )
        operator = FDFDOperator(grid, components)
        omega = 0.3 * C0 * np.pi / max(grid.a, grid.b)
        full = {
            dof.name: rng.normal(size=dof.shape) + 1j * rng.normal(size=dof.shape)
            for dof in operator.layout.e
        }
        free = operator.layout.pack_e(full)
        load = operator._array_lhs(omega, full)
        current = -operator.layout.pack_e(load)
        matrix_load = operator.system_matrix(omega) @ free
        # Zero the boundary of a copy and check the matrix against the array form.
        interior = {name: values.copy() for name, values in full.items()}
        for dof in operator.layout.e:
            mask = np.ones(dof.shape, dtype=bool)
            mask[dof.i, dof.j] = False
            interior[dof.name][mask] = 0.0
        np.testing.assert_allclose(
            operator.layout.pack_e(operator._array_lhs(omega, interior)),
            matrix_load,
            rtol=1e-10,
            atol=1e-10,
        )
        recovered = operator.solve(omega, current, dirichlet=full)
        np.testing.assert_allclose(recovered, free, rtol=1e-10, atol=1e-10)


def test_uniform_dielectric_cavity_scales_with_the_index() -> None:
    grid_tm = YeeGrid2D(16, 12, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    law = UniformIsotropic(eps=4.0 * EPS0)
    omega = _lowest_omega(grid_tm, law.sample_tm(grid_tm))
    expected = semi_discrete_omega(
        m=1, n=1, a=grid_tm.a, b=grid_tm.b, dx=grid_tm.dx, dy=grid_tm.dy, c=C0 / 2.0
    )
    np.testing.assert_allclose(omega, expected, rtol=1e-10)

    grid_te = YeeGrid2D(16, 12, 1.0, 1.0, Polarization.TEZ, Boundary.PEC)
    positive = _positive_omegas(grid_te, law.sample_te(grid_te))
    expected_te = semi_discrete_omega(
        m=1, n=0, a=grid_te.a, b=grid_te.b, dx=grid_te.dx, dy=grid_te.dy, c=C0 / 2.0
    )
    np.testing.assert_allclose(positive[0], expected_te, rtol=1e-10)

    state = zeros_tmz(grid_tm)
    shape = tmz_ez_mode(grid_tm, 1, 1)
    state.ez[:] = shape
    components = law.sample_tm(grid_tm)
    dt = 0.5 * cfl_timestep(grid_tm.dx, grid_tm.dy)
    samples = []
    for _ in range(3):
        samples.append(float(np.sum(state.ez * shape)))
        step_tmz(grid_tm, components, state, dt)
    fitted = leapfrog_omega_from_samples(*samples, dt)
    discrete = fully_discrete_omega(
        kx=np.pi / grid_tm.a,
        ky=np.pi / grid_tm.b,
        dx=grid_tm.dx,
        dy=grid_tm.dy,
        dt=dt,
        c=C0 / 2.0,
    )
    np.testing.assert_allclose(fitted, discrete, rtol=1e-6)


def test_slab_loaded_cavity_tracks_the_transcendental_root() -> None:
    grid = YeeGrid2D(24, 32, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    # Halfway between Ez nodes, so the half-open slab does not own the face.
    height = (grid.ny // 2 + 0.5) * grid.dy
    law = StaircaseIsotropic(
        UniformIsotropic(),
        (SlabY(0.0, height, eps=4.0 * EPS0),),
    )
    components = law.sample_tm(grid)
    operator = FDFDOperator(grid, components)
    stiffness, mass = operator.spatial_operator()
    values, vectors = linalg.eigh(
        stiffness.toarray(), mass.toarray(), check_finite=False
    )
    omega0 = float(np.sqrt(values[0]))
    root = _slab_wavenumber(grid.a, grid.b, height, eps_r=4.0)
    np.testing.assert_allclose(omega0, root * C0, rtol=1e-2)

    mode = operator.layout.e[0].scatter(vectors[:, 0])
    state = zeros_tmz(grid)
    state.ez[:] = mode
    dt = 0.5 * cfl_timestep(grid.dx, grid.dy)
    samples = []
    for _ in range(3):
        samples.append(float(np.sum(state.ez * mode)))
        step_tmz(grid, components, state, dt)
    fitted = leapfrog_omega_from_samples(*samples, dt)
    expected = (2.0 / dt) * np.arcsin(omega0 * dt / 2.0)
    np.testing.assert_allclose(fitted, expected, rtol=1e-6)


def _recover(grid: YeeGrid2D, components: TMComponents | TEComponents) -> None:
    operator = FDFDOperator(grid, components)
    omega = 0.25 * C0 * np.pi / max(grid.a, grid.b)
    rng = np.random.default_rng(4)
    electric = rng.normal(size=operator.layout.n_e) + 1j * rng.normal(
        size=operator.layout.n_e
    )
    current = -(operator.system_matrix(omega) @ electric)
    recovered = operator.solve(omega, current)
    np.testing.assert_allclose(recovered, electric, rtol=1e-10, atol=1e-10)


def _varying_components(grid: YeeGrid2D) -> TMComponents | TEComponents:
    def wave(component: str, scale: float, shift: float) -> np.ndarray:
        x, y = grid.coordinates(component)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return scale * (
            1.5
            + 0.4 * np.sin(np.pi * xx / grid.a + shift) * np.cos(np.pi * yy / grid.b)
        )

    if grid.polarization is Polarization.TMZ:
        return TMComponents(
            eps_z=wave("ez", EPS0, 0.0),
            mu_x=wave("hx", MU0, 0.3),
            mu_y=wave("hy", MU0, 0.6),
            sigma_z=np.zeros(grid.shapes()["ez"]),
        )
    return TEComponents(
        eps_x=wave("ex", EPS0, 0.2),
        eps_y=wave("ey", EPS0, 0.4),
        mu_z=wave("hz", MU0, 0.8),
        sigma_x=np.zeros(grid.shapes()["ex"]),
        sigma_y=np.zeros(grid.shapes()["ey"]),
    )


def _lowest_omega(grid: YeeGrid2D, components: TMComponents | TEComponents) -> float:
    stiffness, mass = FDFDOperator(grid, components).spatial_operator()
    values = linalg.eigh(stiffness.toarray(), mass.toarray(), eigvals_only=True)
    return float(np.sqrt(values[0]))


def _positive_omegas(grid: YeeGrid2D, components: TEComponents) -> np.ndarray:
    stiffness, mass = FDFDOperator(grid, components).spatial_operator()
    values = linalg.eigh(stiffness.toarray(), mass.toarray(), eigvals_only=True)
    omega = np.sqrt(np.clip(values, 0.0, None))
    return omega[omega > 1e-6 * omega[-1]]


def _slab_wavenumber(a: float, b: float, height: float, eps_r: float) -> float:
    kx = np.pi / a

    def residual(k0: float) -> float:
        s1 = eps_r * k0**2 - kx**2
        s2 = k0**2 - kx**2
        if s1 <= 1e-18:
            return np.nan
        ky1 = np.sqrt(s1)
        if abs(np.sin(ky1 * height)) < 1e-8:
            return np.nan
        left = ky1 * np.cos(ky1 * height) / np.sin(ky1 * height)
        length = b - height
        if s2 >= 0.0:
            ky2 = np.sqrt(s2)
            if abs(np.sin(ky2 * length)) < 1e-8:
                return np.nan
            right = -ky2 * np.cos(ky2 * length) / np.sin(ky2 * length)
        else:
            kappa = np.sqrt(-s2)
            right = -kappa / np.tanh(kappa * length)
        return float(left - right)

    k_lo = kx / np.sqrt(eps_r) * 1.01
    k_hi = np.pi * np.sqrt(1.0 / a**2 + 1.0 / b**2)
    grid = np.linspace(k_lo, k_hi, 4000)
    values = np.array([residual(k0) for k0 in grid])
    for left, right, k_left, k_right in zip(
        values, values[1:], grid, grid[1:], strict=False
    ):
        if np.isnan(left) or np.isnan(right) or left * right > 0.0:
            continue
        return float(
            brentq(
                lambda k0: 0.0 if np.isnan(residual(k0)) else residual(k0),
                k_left,
                k_right,
            )
        )
    raise RuntimeError("no root of the slab characteristic equation")
