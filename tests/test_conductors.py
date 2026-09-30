"""Embedded PEC and PMC samples, and an open PEC cylinder on the PML."""

import numpy as np
from scipy import linalg

from maxwell_fd.analytics.cavity2d import semi_discrete_omega
from maxwell_fd.analytics.mie2d import pec_cylinder_ez, pec_cylinder_te_electric
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.drivers.fdtd2d import (
    cfl_timestep,
    step_tez,
    step_tmz,
    zeros_tez,
    zeros_tmz,
)
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.conductors import Circle, Conductors, YBand
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.curl2d import array_curl_e, array_curl_h, build_curls
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0, EPS0, MU0

# 64 cells, Δ = 1, a = 8, eight PML cells, ka = 1.
# Measured TMz 0.042 and TEz 0.056 on that grid.
OPEN_TMZ_BAR = 0.08
OPEN_TEZ_BAR = 0.10


def test_band_is_half_open_and_circle_is_closed() -> None:
    band = YBand(1.0, 2.0)
    y = np.array([0.9, 1.0, 1.5, 2.0])
    inside = band.contains(np.zeros_like(y), y)
    assert list(inside) == [False, True, True, False]
    circle = Circle(0.0, 0.0, 1.0)
    pts = np.array([1.0, 1.0 + 1e-12])
    hit = circle.contains(pts, np.zeros_like(pts))
    assert bool(hit[0])
    assert not bool(hit[1])


def test_pec_band_matches_the_short_cavity() -> None:
    # Filling y >= b_short on a taller grid leaves the short PEC cavity.
    nx, n_short, n_tall = 8, 8, 16
    dx = dy = 1.0
    for pol in (Polarization.TMZ, Polarization.TEZ):
        short = _omegas(nx, n_short, dx, dy, pol, conductors=None)
        tall = _omegas(
            nx,
            n_tall,
            dx,
            dy,
            pol,
            conductors=Conductors(pec=(YBand(n_short * dy, n_tall * dy + dy),)),
        )
        short_pos = _positive(short)
        tall_pos = _positive(tall)
        np.testing.assert_allclose(tall_pos, short_pos, rtol=1e-10)
        if pol is Polarization.TMZ:
            expected = semi_discrete_omega(
                m=1, n=1, a=nx * dx, b=n_short * dy, dx=dx, dy=dy
            )
            np.testing.assert_allclose(short_pos[0], expected, rtol=1e-10)
        else:
            expected = semi_discrete_omega(
                m=1, n=0, a=nx * dx, b=n_short * dy, dx=dx, dy=dy
            )
            np.testing.assert_allclose(short_pos[0], expected, rtol=1e-10)


def test_pmc_quarter_wave_matches_the_sine_formula() -> None:
    # Hx at y = (J+0.5) Δy is the tangential sample on the cut. Removing it
    # is the discrete Neumann condition. ky = (n+1/2) π / y_face is the mode
    # (m, 2n+1) of a PEC cavity of height 2 y_face.
    nx, ny, j_cut = 12, 16, 8
    dx = dy = 1.0
    y_face = (j_cut + 0.5) * dy
    grid = YeeGrid2D(nx, ny, dx, dy, Polarization.TMZ, Boundary.PEC)
    conductors = Conductors(pmc=(YBand(y_face, grid.b + dy),))
    omega = _omegas(nx, ny, dx, dy, Polarization.TMZ, conductors=conductors)
    expected = semi_discrete_omega(m=1, n=1, a=grid.a, b=2.0 * y_face, dx=dx, dy=dy)
    np.testing.assert_allclose(_positive(omega)[0], expected, rtol=1e-10)


def test_reduced_curls_match_arrays_and_stay_transposes() -> None:
    rng = np.random.default_rng(1)
    cases = (
        (Polarization.TMZ, Boundary.PEC),
        (Polarization.TEZ, Boundary.PEC),
        (Polarization.TMZ, Boundary.PERIODIC),
        (Polarization.TEZ, Boundary.PERIODIC),
    )
    for pol, boundary in cases:
        grid = YeeGrid2D(10, 8, 0.3, 0.4, pol, boundary)
        conductors = Conductors(pec=(Circle(1.5, 1.2, 0.7),), pmc=(YBand(2.0, 2.6),))
        masks = conductors.masks(grid)
        ops = build_curls(grid, fixed_e=masks.pec, fixed_h=masks.pmc)
        np.testing.assert_allclose(
            ops.curl_h.toarray(), ops.curl_e.toarray().T, atol=1e-12
        )
        electric = {dof.name: rng.normal(size=dof.shape) for dof in ops.layout.e}
        magnetic = {dof.name: rng.normal(size=dof.shape) for dof in ops.layout.h}
        for name, mask in masks.pec.items():
            electric[name][mask] = 0.0
        for name, mask in masks.pmc.items():
            magnetic[name][mask] = 0.0
        if boundary is Boundary.PEC and pol is Polarization.TMZ:
            electric["ez"][0, :] = electric["ez"][-1, :] = 0.0
            electric["ez"][:, 0] = electric["ez"][:, -1] = 0.0
        if boundary is Boundary.PEC and pol is Polarization.TEZ:
            electric["ex"][:, 0] = electric["ex"][:, -1] = 0.0
            electric["ey"][0, :] = electric["ey"][-1, :] = 0.0
        got_e = ops.curl_e @ ops.layout.pack_e(electric)
        ref_e = ops.layout.pack_h(array_curl_e(grid, electric))
        got_h = ops.curl_h @ ops.layout.pack_h(magnetic)
        ref_h = ops.layout.pack_e(array_curl_h(grid, magnetic))
        np.testing.assert_allclose(got_e, ref_e, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got_h, ref_h, rtol=1e-12, atol=1e-12)


def test_leapfrog_clears_conductor_samples() -> None:
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(12, 14, 1.0, 1.0, pol, Boundary.PEC)
        conductors = Conductors(pec=(Circle(3.0, 3.0, 1.2),), pmc=(YBand(9.5, 30.0),))
        masks = conductors.masks(grid)
        dt = 0.5 * cfl_timestep(grid.dx, grid.dy)
        if pol is Polarization.TMZ:
            components = UniformIsotropic().sample_tm(grid)
            state = zeros_tmz(grid)
            state.ez[masks.pec["ez"]] = 0.3
            state.hx[masks.pmc["hx"]] = -0.4
            state.hy[masks.pmc["hy"]] = 0.2
            current = np.zeros(grid.shapes()["ez"])
            current[6, 6] = 1.0
            for _ in range(4):
                step_tmz(grid, components, state, dt, current, conductors=conductors)
            assert np.all(state.ez[masks.pec["ez"]] == 0.0)
            assert np.all(state.hx[masks.pmc["hx"]] == 0.0)
            assert np.all(state.hy[masks.pmc["hy"]] == 0.0)
            assert np.all(state.ez[0, :] == 0.0) and np.all(state.ez[:, 0] == 0.0)
            assert np.any(state.ez != 0.0)
        else:
            components = UniformIsotropic().sample_te(grid)
            state = zeros_tez(grid)
            state.ex[masks.pec["ex"]] = 0.3
            state.ey[masks.pec["ey"]] = -0.2
            state.hz[masks.pmc["hz"]] = 0.5
            current = {
                "ex": np.zeros(grid.shapes()["ex"]),
                "ey": np.zeros(grid.shapes()["ey"]),
            }
            current["ey"][6, 6] = 1.0
            for _ in range(4):
                step_tez(grid, components, state, dt, current, conductors=conductors)
            assert np.all(state.ex[masks.pec["ex"]] == 0.0)
            assert np.all(state.ey[masks.pec["ey"]] == 0.0)
            assert np.all(state.hz[masks.pmc["hz"]] == 0.0)
            assert np.all(state.ex[:, 0] == 0.0) and np.all(state.ey[0, :] == 0.0)
            assert np.any(state.ey != 0.0)


def test_embedded_trace_drives_the_free_field() -> None:
    omega_scale = 0.25 * np.pi
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(12, 10, 1.0, 1.0, pol, Boundary.PEC)
        conductors = Conductors(
            pec=(Circle(4.0, 3.5, 1.5),),
            pmc=(YBand(7.5, grid.b + grid.dy),),
        )
        law = UniformIsotropic()
        components = (
            law.sample_tm(grid) if pol is Polarization.TMZ else law.sample_te(grid)
        )
        operator = FDFDOperator(grid, components, conductors=conductors)
        omega = omega_scale * C0 / max(grid.a, grid.b)
        masks = conductors.masks(grid)
        if pol is Polarization.TMZ:
            boundary = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
            boundary[masks.pec["ez"]] = 0.2 - 0.1j
            solved = operator.solve(
                omega, np.zeros(operator.layout.n_e), dirichlet={"ez": boundary}
            )
            assert np.linalg.norm(solved) > 0.0
            kept = boundary.copy()
            dof = operator.layout.e[0]
            kept[dof.i, dof.j] = 0.0
            curls = array_curl_e(grid, {"ez": kept})
            scaled = {
                "hx": curls["hx"] / (1j * omega * components.mu_x),
                "hy": curls["hy"] / (1j * omega * components.mu_y),
            }
            scaled["hx"][masks.pmc["hx"]] = 0.0
            scaled["hy"][masks.pmc["hy"]] = 0.0
            lhs = (
                array_curl_h(grid, scaled)["ez"] + 1j * omega * components.eps_z * kept
            )
            residual = operator.system_matrix(omega) @ solved + dof.pack(lhs)
        else:
            boundary = {
                "ex": np.zeros(grid.shapes()["ex"], dtype=np.complex128),
                "ey": np.zeros(grid.shapes()["ey"], dtype=np.complex128),
            }
            boundary["ex"][masks.pec["ex"]] = 0.3
            boundary["ey"][masks.pec["ey"]] = -0.2j
            solved = operator.solve(
                omega, np.zeros(operator.layout.n_e), dirichlet=boundary
            )
            assert np.linalg.norm(solved) > 0.0
            kept = {name: values.copy() for name, values in boundary.items()}
            for dof in operator.layout.e:
                kept[dof.name][dof.i, dof.j] = 0.0
            curls = array_curl_e(grid, kept)
            hz = curls["hz"] / (1j * omega * components.mu_z)
            hz[masks.pmc["hz"]] = 0.0
            ampere = array_curl_h(grid, {"hz": hz})
            lhs = {
                "ex": ampere["ex"] + 1j * omega * components.eps_x * kept["ex"],
                "ey": ampere["ey"] + 1j * omega * components.eps_y * kept["ey"],
            }
            residual = operator.system_matrix(omega) @ solved + operator.layout.pack_e(
                lhs
            )
        scale = max(float(np.linalg.norm(residual)), 1.0)
        # The load itself sets the scale. A missed conductor trace leaves a finite residual.
        load_norm = float(np.linalg.norm(operator.system_matrix(omega) @ solved))
        assert load_norm > 0.0
        np.testing.assert_allclose(residual, 0.0, atol=1e-8 * max(load_norm, scale))


def test_conductors_reject_a_bad_region_and_a_filled_grid() -> None:
    def missing_contains() -> None:
        Conductors(pec=(object(),))

    def inverted_band() -> None:
        YBand(2.0, 1.0)

    def filled() -> None:
        grid = YeeGrid2D(4, 4, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
        FDFDOperator(
            grid,
            UniformIsotropic().sample_tm(grid),
            conductors=Conductors(pec=(YBand(0.0, 10.0),)),
        )

    _raises(missing_contains, TypeError, "contains")
    _raises(inverted_band, ValueError, "y1")
    _raises(filled, ValueError, "electric")


def test_inactive_conductors_keep_the_free_space_layout() -> None:
    grid = YeeGrid2D(8, 6, 0.2, 0.25, Polarization.TEZ, Boundary.PEC)
    components = UniformIsotropic().sample_te(grid)
    plain = FDFDOperator(grid, components)
    empty = FDFDOperator(grid, components, conductors=Conductors())
    assert plain.layout.n_e == empty.layout.n_e
    assert plain.layout.n_h == empty.layout.n_h
    periodic = YeeGrid2D(8, 8, 1.0, 1.0, Polarization.TMZ, Boundary.PERIODIC)
    bare = FDFDOperator(periodic, UniformIsotropic().sample_tm(periodic))
    buried = FDFDOperator(
        periodic,
        UniformIsotropic().sample_tm(periodic),
        conductors=Conductors(pec=(Circle(4.0, 4.0, 1.5),)),
    )
    assert buried.layout.n_e < bare.layout.n_e


def test_open_pec_cylinder_matches_mie() -> None:
    tmz = _open_pec_error(Polarization.TMZ)
    tez = _open_pec_error(Polarization.TEZ)
    assert tmz < OPEN_TMZ_BAR, tmz
    assert tez < OPEN_TEZ_BAR, tez


def _omegas(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    pol: Polarization,
    conductors: Conductors | None,
) -> np.ndarray:
    grid = YeeGrid2D(nx, ny, dx, dy, pol, Boundary.PEC)
    law = UniformIsotropic()
    components = law.sample_tm(grid) if pol is Polarization.TMZ else law.sample_te(grid)
    stiffness, mass = FDFDOperator(
        grid, components, conductors=conductors
    ).spatial_operator()
    eigenvalues = linalg.eigh(stiffness.toarray(), mass.toarray(), eigvals_only=True)
    return np.sqrt(np.clip(eigenvalues, 0.0, None))


def _positive(omega: np.ndarray) -> np.ndarray:
    return omega[omega > 1e-6 * omega[-1]]


def _open_pec_error(pol: Polarization) -> float:
    cells, dx, radius, pml_cells, ka = 64, 1.0, 8.0, 8, 1.0
    length = cells * dx
    center = (0.5 * length, 0.5 * length)
    omega = (ka / radius) * C0
    k = float(omega * np.sqrt(MU0 * EPS0))
    grid = YeeGrid2D(cells, cells, dx, dx, pol, Boundary.PEC)
    conductors = Conductors(pec=(Circle(center[0], center[1], radius),))
    masks = conductors.masks(grid)
    band = pml_cells * dx
    if pol is Polarization.TMZ:
        components = UniformIsotropic().sample_tm(grid)
        operator = FDFDOperator(
            grid, components, PMLSpec.box(pml_cells), conductors=conductors
        )
        x, y = grid.coordinates("ez")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        incident = np.exp(-1j * k * (xx - center[0]))
        trace = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
        trace[masks.pec["ez"]] = -incident[masks.pec["ez"]]
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": trace}
        )
        total = incident.copy()
        dof = operator.layout.e[0]
        total[dof.i, dof.j] = solved + incident[dof.i, dof.j]
        total[masks.pec["ez"]] = 0.0
        total[0, :] = total[-1, :] = total[:, 0] = total[:, -1] = 0.0
        exact = pec_cylinder_ez(xx, yy, k=k, radius=radius, center=center)
        mask = _observe(xx, yy, length, band, center, radius, dx)
        assert int(np.count_nonzero(mask)) > 100
        return float(
            np.linalg.norm(total[mask] - exact[mask]) / np.linalg.norm(exact[mask])
        )
    components = UniformIsotropic().sample_te(grid)
    operator = FDFDOperator(
        grid, components, PMLSpec.box(pml_cells), conductors=conductors
    )
    ex_x, ex_y = grid.coordinates("ex")
    ey_x, ey_y = grid.coordinates("ey")
    ex_xx, ex_yy = np.meshgrid(ex_x, ex_y, indexing="ij")
    ey_xx, ey_yy = np.meshgrid(ey_x, ey_y, indexing="ij")
    ex_inc = np.zeros(grid.shapes()["ex"], dtype=np.complex128)
    ey_inc = (k / (omega * EPS0)) * np.exp(-1j * k * (ey_xx - center[0]))
    trace_ex = np.zeros_like(ex_inc)
    trace_ey = np.zeros_like(ey_inc)
    trace_ex[masks.pec["ex"]] = -ex_inc[masks.pec["ex"]]
    trace_ey[masks.pec["ey"]] = -ey_inc[masks.pec["ey"]]
    solved = operator.solve(
        omega,
        np.zeros(operator.layout.n_e),
        dirichlet={"ex": trace_ex, "ey": trace_ey},
    )
    n_ex = operator.layout.e[0].i.size
    ex = ex_inc.copy()
    ey = ey_inc.copy()
    ex[operator.layout.e[0].i, operator.layout.e[0].j] = (
        solved[:n_ex] + ex_inc[operator.layout.e[0].i, operator.layout.e[0].j]
    )
    ey[operator.layout.e[1].i, operator.layout.e[1].j] = (
        solved[n_ex:] + ey_inc[operator.layout.e[1].i, operator.layout.e[1].j]
    )
    ex[masks.pec["ex"]] = 0.0
    ey[masks.pec["ey"]] = 0.0
    ex[:, 0] = ex[:, -1] = 0.0
    ey[0, :] = ey[-1, :] = 0.0
    exact_ex, _exact_ey = pec_cylinder_te_electric(
        ex_xx, ex_yy, k=k, radius=radius, omega=omega, eps0=EPS0, center=center
    )
    _exact_ex, exact_ey = pec_cylinder_te_electric(
        ey_xx, ey_yy, k=k, radius=radius, omega=omega, eps0=EPS0, center=center
    )
    mask_ex = _observe(ex_xx, ex_yy, length, band, center, radius, dx)
    mask_ey = _observe(ey_xx, ey_yy, length, band, center, radius, dx)
    assert int(np.count_nonzero(mask_ex)) > 100
    assert int(np.count_nonzero(mask_ey)) > 100
    numerical = np.concatenate((ex[mask_ex], ey[mask_ey]))
    exact = np.concatenate((exact_ex[mask_ex], exact_ey[mask_ey]))
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def _observe(
    xx: np.ndarray,
    yy: np.ndarray,
    length: float,
    band: float,
    center: tuple[float, float],
    radius: float,
    dx: float,
) -> np.ndarray:
    rho = np.hypot(xx - center[0], yy - center[1])
    return (
        (xx >= band)
        & (xx <= length - band)
        & (yy >= band)
        & (yy <= length - band)
        & (rho >= radius + 2.0 * dx)
    )


def _raises(operation, exc_type: type[Exception], text: str) -> None:
    try:
        operation()
    except exc_type as exc:
        assert text in str(exc)
    else:
        raise AssertionError(f"{exc_type.__name__} was not raised")
