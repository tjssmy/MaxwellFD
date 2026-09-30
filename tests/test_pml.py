"""CFS profile, stretched curls, line-current absorption, and the leapfrog auxiliary."""

import numpy as np
from scipy.special import hankel2

from maxwell_fd.analytics.green2d import line_current_ez
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.drivers.fdtd2d import (
    cfl_timestep,
    step_tez,
    step_tmz,
    zeros_tez,
    zeros_tmz,
)
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.curl2d import array_curl_e, array_curl_h, build_curls
from maxwell_fd.operators.pml import ConvolutionalPML, PMLProfile, PMLSpec, sigma_max
from maxwell_fd.utils.constants import C0, EPS0, ETA0, MU0


def _uniform(grid: YeeGrid2D):
    law = UniformIsotropic()
    if grid.polarization is Polarization.TMZ:
        return law.sample_tm(grid)
    return law.sample_te(grid)


def test_conductivity_profile_matches_the_note() -> None:
    dx = 0.25
    grid = YeeGrid2D(16, 12, dx, dx, Polarization.TMZ, Boundary.PEC)
    spec = PMLSpec(
        x_lo=4, x_hi=3, y_lo=2, y_hi=5, m=4, r0=1e-7, kappa_max=2.5, alpha_max=0.05
    )
    profile = PMLProfile(grid, spec)
    x = np.arange(grid.nx + 1) * dx
    _expect_side(profile.x_node, x, grid.a, 4 * dx, low=True, spec=spec)
    _expect_side(profile.x_node, x, grid.a, 3 * dx, low=False, spec=spec)
    y_mid = (np.arange(grid.ny) + 0.5) * dx
    _expect_side(profile.y_mid, y_mid, grid.b, 2 * dx, low=True, spec=spec)
    _expect_side(profile.y_mid, y_mid, grid.b, 5 * dx, low=False, spec=spec)
    interior = (x > 4 * dx) & (x < grid.a - 3 * dx)
    assert np.all(profile.x_node.sigma[interior] == 0.0)
    assert np.all(profile.x_node.kappa[interior] == 1.0)
    assert np.all(profile.x_node.alpha[interior] == 0.0)


def test_frequency_scale_is_the_reciprocal_stretch() -> None:
    grid = YeeGrid2D(10, 8, 0.2, 0.2, Polarization.TMZ, Boundary.PEC)
    spec = PMLSpec.box(3, m=3, r0=1e-6, kappa_max=1.4, alpha_max=0.02)
    profile = PMLProfile(grid, spec)
    omega = 2.0 * np.pi * 1.0e9
    scale = profile.scale(omega)
    y = (np.arange(grid.ny) + 0.5) * grid.dy
    expected = 1.0 / (
        profile.y_mid.kappa
        + profile.y_mid.sigma / (profile.y_mid.alpha + 1j * omega * EPS0)
    )
    np.testing.assert_allclose(scale.tm_y_on_hx[0], expected)
    np.testing.assert_allclose(scale.tm_y_on_hx[4], expected)
    # The inner face is unstretched, so a sample outside the band has 1/s = 1.
    outside = (y > spec.y_lo * grid.dy) & (y < grid.b - spec.y_hi * grid.dy)
    np.testing.assert_allclose(expected[outside], 1.0)


def test_stretched_matrix_matches_the_array_curl() -> None:
    rng = np.random.default_rng(7)
    omega = 1.7e8
    spec = PMLSpec(
        x_lo=2, x_hi=3, y_lo=2, y_hi=2, m=3, r0=1e-5, kappa_max=2.0, alpha_max=0.04
    )
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(9, 8, 0.3, 0.25, pol, Boundary.PEC)
        scale = PMLProfile(grid, spec).scale(omega)
        ops = build_curls(grid, scale=scale)
        electric = {dof.name: rng.normal(size=dof.shape) for dof in ops.layout.e}
        magnetic = {dof.name: rng.normal(size=dof.shape) for dof in ops.layout.h}
        if pol is Polarization.TMZ:
            electric["ez"][0, :] = electric["ez"][-1, :] = 0.0
            electric["ez"][:, 0] = electric["ez"][:, -1] = 0.0
        else:
            electric["ex"][:, 0] = electric["ex"][:, -1] = 0.0
            electric["ey"][0, :] = electric["ey"][-1, :] = 0.0
        got_e = ops.curl_e @ ops.layout.pack_e(electric)
        ref_e = ops.layout.pack_h(array_curl_e(grid, electric, scale=scale))
        got_h = ops.curl_h @ ops.layout.pack_h(magnetic)
        ref_h = ops.layout.pack_e(array_curl_h(grid, magnetic, scale=scale))
        np.testing.assert_allclose(got_e, ref_e, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got_h, ref_h, rtol=1e-12, atol=1e-12)


def test_dirichlet_load_uses_the_same_stretch() -> None:
    omega = 3.0e8
    spec = PMLSpec.box(2, m=3, r0=1e-6, kappa_max=1.5, alpha_max=0.01)
    rng = np.random.default_rng(4)
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(8, 7, 0.2, 0.2, pol, Boundary.PEC)
        operator = FDFDOperator(grid, _uniform(grid), spec)
        fields = {
            dof.name: rng.normal(size=dof.shape) + 1j * rng.normal(size=dof.shape)
            for dof in operator.layout.e
        }
        free = operator.layout.pack_e(fields)
        load = operator._dirichlet_load(omega, fields)
        full = operator.layout.pack_e(operator._array_lhs(omega, fields))
        np.testing.assert_allclose(
            operator.system_matrix(omega) @ free + load,
            full,
            rtol=1e-10,
            atol=1e-10,
        )


def test_zero_thickness_pml_reproduces_the_unstretched_step() -> None:
    rng = np.random.default_rng(1)
    dt_scale = 0.4
    spec = PMLSpec()
    for pol, stepper, zeros in (
        (Polarization.TMZ, step_tmz, zeros_tmz),
        (Polarization.TEZ, step_tez, zeros_tez),
    ):
        grid = YeeGrid2D(6, 5, 0.2, 0.25, pol, Boundary.PEC)
        components = _uniform(grid)
        dt = dt_scale * cfl_timestep(grid.dx, grid.dy)
        plain = zeros(grid)
        stretched = zeros(grid)
        _seed(plain, rng)
        _copy_state(stretched, plain)
        pml = ConvolutionalPML(grid, spec, dt)
        for _ in range(4):
            if pol is Polarization.TMZ:
                current = rng.normal(size=grid.shapes()["ez"])
                stepper(grid, components, plain, dt, impressed_j=current)
                stepper(grid, components, stretched, dt, impressed_j=current, pml=pml)
            else:
                current = {
                    name: rng.normal(size=shape)
                    for name, shape in grid.shapes().items()
                    if name.startswith("e")
                }
                stepper(grid, components, plain, dt, impressed_j=current)
                stepper(grid, components, stretched, dt, impressed_j=current, pml=pml)
        for name in plain.__dataclass_fields__:
            np.testing.assert_allclose(
                getattr(stretched, name), getattr(plain, name), rtol=0.0, atol=0.0
            )


def test_auxiliary_step_matches_the_exponential_integrator() -> None:
    dx = 0.2
    grid = YeeGrid2D(8, 8, dx, dx, Polarization.TMZ, Boundary.PEC)
    spec = PMLSpec.box(3, m=3, r0=1e-6, kappa_max=1.8, alpha_max=0.03)
    dt = 0.2 * cfl_timestep(dx, dx)
    pml = ConvolutionalPML(grid, spec, dt)
    y = np.arange(grid.ny + 1) * dx
    _, yy = np.meshgrid(np.arange(grid.nx + 1) * dx, y, indexing="ij")
    pml.faraday(grid, {"ez": yy.copy()}, dt)
    # ∂Ez/∂y = 1, so the new auxiliary at Hx is the one-step coefficient a.
    grade = pml.profile.y_mid
    beta = grade.alpha / EPS0 + grade.sigma / (grade.kappa * EPS0)
    gamma = grade.sigma / (grade.kappa**2 * EPS0)
    a = np.zeros_like(beta)
    nonzero = beta > 0.0
    a[nonzero] = -(gamma[nonzero] / beta[nonzero]) * (1.0 - np.exp(-beta[nonzero] * dt))
    a[~nonzero] = -gamma[~nonzero] * dt
    np.testing.assert_allclose(pml.psi_hx[2], a, rtol=1e-12, atol=1e-18)
    # Outside the band γ = 0, so the auxiliary stays zero and κ = 1.
    outside = (grade.sigma == 0.0) & (grade.alpha == 0.0)
    assert np.all(pml.psi_hx[:, outside] == 0.0)


def test_line_current_agrees_with_hankel_inside_the_pml() -> None:
    coarse = _hankel_error(dx=0.05, cells=8)
    fine = _hankel_error(dx=0.025, cells=16)
    bare = _hankel_error(dx=0.05, cells=0, observe=8)
    assert coarse < 0.04, coarse
    assert fine < coarse / 2.5, (fine, coarse)
    assert bare > 0.5, bare


def test_leapfrog_pml_matches_fdfd() -> None:
    for pol in (Polarization.TMZ, Polarization.TEZ):
        error = _phasor_error(pol, steps_per_period=48)
        assert error < 0.015, (pol, error)


def test_pml_rejects_a_periodic_grid_and_a_filled_axis() -> None:
    grid = YeeGrid2D(8, 8, 0.2, 0.2, Polarization.TMZ, Boundary.PERIODIC)
    try:
        PMLProfile(grid, PMLSpec.box(2))
    except ValueError as exc:
        assert "PEC" in str(exc)
    else:
        raise AssertionError("periodic PML was accepted")
    pec = YeeGrid2D(8, 8, 0.2, 0.2, Polarization.TMZ, Boundary.PEC)
    try:
        PMLProfile(pec, PMLSpec(x_lo=4, x_hi=4))
    except ValueError as exc:
        assert "no interior" in str(exc)
    else:
        raise AssertionError("overlapping PML was accepted")
    operator = FDFDOperator(pec, _uniform(pec), PMLSpec.box(2))
    try:
        operator.spatial_operator()
    except ValueError as exc:
        assert "unstretched" in str(exc)
    else:
        raise AssertionError("eigenproblem accepted a PML")


def _expect_side(grade, coords, length, thickness, *, low: bool, spec: PMLSpec) -> None:
    if thickness == 0.0:
        return
    face = thickness if low else length - thickness
    rho = face - coords if low else coords - face
    mask = rho > 0.0
    unit = rho[mask] / thickness
    peak = sigma_max(spec.m, spec.r0, thickness)
    power = unit**spec.m
    np.testing.assert_allclose(grade.sigma[mask], peak * power, rtol=1e-12)
    np.testing.assert_allclose(grade.kappa[mask], 1.0 + (spec.kappa_max - 1.0) * power)
    np.testing.assert_allclose(
        grade.alpha[mask], spec.alpha_max * (1.0 - unit) ** spec.m
    )
    wall = 0 if low else -1
    if np.isclose(coords[wall], 0.0 if low else length):
        np.testing.assert_allclose(grade.sigma[wall], peak)
        np.testing.assert_allclose(grade.kappa[wall], spec.kappa_max)
        np.testing.assert_allclose(grade.alpha[wall], 0.0)
    hand = -(spec.m + 1) * np.log(spec.r0) / (2.0 * ETA0 * thickness)
    np.testing.assert_allclose(peak, hand)


def _hankel_error(*, dx: float, cells: int, observe: int | None = None) -> float:
    """Relative Ez error of a unit line current on a 3-wavelength PEC square."""
    wavelength = 1.0
    side = 3.0
    n = int(round(side / dx))
    grid = YeeGrid2D(n, n, dx, dx, Polarization.TMZ, Boundary.PEC)
    spec = None if cells == 0 else PMLSpec.box(cells)
    window = cells if observe is None else observe
    operator = FDFDOperator(grid, _uniform(grid), spec)
    omega = 2.0 * np.pi * C0 / wavelength
    current = 1.0
    jz = np.zeros(grid.shapes()["ez"])
    i0 = n // 2
    jz[i0, i0] = current / (dx * dx)
    solved = operator.solve(omega, operator.layout.pack_e({"ez": jz}))
    ez = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    dof = operator.layout.e[0]
    ez[dof.i, dof.j] = solved
    x, y = grid.coordinates("ez")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    exact = line_current_ez(xx - x[i0], yy - y[i0], omega=omega, current=current)
    inset = window * dx
    rho = np.hypot(xx - x[i0], yy - y[i0])
    mask = (xx >= inset) & (xx <= x[-1] - inset) & (yy >= inset) & (yy <= y[-1] - inset)
    mask &= rho >= 0.5 * wavelength
    return float(np.linalg.norm(ez[mask] - exact[mask]) / np.linalg.norm(exact[mask]))


def _phasor_error(pol: Polarization, steps_per_period: int) -> float:
    dx = 1.0
    wavelength = 12.0
    n = 28
    cells = 6
    grid = YeeGrid2D(n, n, dx, dx, pol, Boundary.PEC)
    spec = PMLSpec.box(cells)
    components = _uniform(grid)
    omega = 2.0 * np.pi * C0 / wavelength
    operator = FDFDOperator(grid, components, spec)
    i0 = n // 2
    if pol is Polarization.TMZ:
        jz = np.zeros(grid.shapes()["ez"])
        jz[i0, i0] = 1.0
        solved = operator.solve(omega, operator.layout.pack_e({"ez": jz}))
        reference = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
        dof = operator.layout.e[0]
        reference[dof.i, dof.j] = solved
        drive_name = "ez"
    else:
        jx = np.zeros(grid.shapes()["ex"])
        jx[i0, i0] = 1.0
        jy = np.zeros(grid.shapes()["ey"])
        solved = operator.solve(omega, operator.layout.pack_e({"ex": jx, "ey": jy}))
        reference = np.zeros(grid.shapes()["ex"], dtype=np.complex128)
        dof = operator.layout.e[0]
        reference[dof.i, dof.j] = solved[: dof.i.size]
        drive_name = "ex"
    period = 2.0 * np.pi / omega
    dt = period / steps_per_period
    assert dt < cfl_timestep(dx, dx)
    state = zeros_tmz(grid) if pol is Polarization.TMZ else zeros_tez(grid)
    pml = ConvolutionalPML(grid, spec, dt)
    settle = 6 * steps_per_period
    for n_step in range(settle):
        _drive_step(
            pol, grid, components, state, pml, drive_name, i0, omega, dt, n_step
        )
    acc = np.zeros(reference.shape, dtype=np.complex128)
    for n_step in range(settle, settle + steps_per_period):
        sample = state.ez if pol is Polarization.TMZ else state.ex
        acc += sample * np.exp(-1j * omega * n_step * dt)
        _drive_step(
            pol, grid, components, state, pml, drive_name, i0, omega, dt, n_step
        )
    phasor = 2.0 * acc / steps_per_period
    mask = _vacuum_mask(reference.shape, cells)
    return float(
        np.linalg.norm((phasor - reference)[mask]) / np.linalg.norm(reference[mask])
    )


def _drive_step(pol, grid, components, state, pml, name, i0, omega, dt, n_step) -> None:
    del name
    amplitude = np.cos(omega * (n_step + 0.5) * dt)
    if pol is Polarization.TMZ:
        jz = np.zeros(grid.shapes()["ez"])
        jz[i0, i0] = amplitude
        step_tmz(grid, components, state, dt, impressed_j=jz, pml=pml)
    else:
        jx = np.zeros(grid.shapes()["ex"])
        jx[i0, i0] = amplitude
        jy = np.zeros(grid.shapes()["ey"])
        step_tez(grid, components, state, dt, impressed_j={"ex": jx, "ey": jy}, pml=pml)


def _vacuum_mask(shape: tuple[int, int], cells: int) -> np.ndarray:
    ix = np.arange(shape[0])[:, None]
    iy = np.arange(shape[1])[None, :]
    return (
        (ix >= cells)
        & (ix < shape[0] - cells)
        & (iy >= cells)
        & (iy < shape[1] - cells)
    )


def _seed(state, rng) -> None:
    for name in state.__dataclass_fields__:
        field = getattr(state, name)
        field[:] = rng.normal(size=field.shape)


def _copy_state(dest, src) -> None:
    for name in src.__dataclass_fields__:
        getattr(dest, name)[:] = getattr(src, name)


def test_green_function_is_outgoing() -> None:
    omega = 2.0 * np.pi * C0
    rho = np.array([1.0, 1.25])
    field = line_current_ez(rho, np.zeros_like(rho), omega=omega)
    k = omega * np.sqrt(MU0 * EPS0)
    phase = np.angle(field[1] / field[0])
    advance = np.angle(np.exp(-1j * k * (rho[1] - rho[0])))
    np.testing.assert_allclose(phase, advance, atol=0.05)
    # The small-argument singularity is the outgoing Hankel, not its conjugate.
    tiny = line_current_ez(np.array([1e-3]), np.array([0.0]), omega=omega)[0]
    reference = -0.25 * omega * MU0 * hankel2(0, k * 1e-3)
    np.testing.assert_allclose(tiny, reference)
