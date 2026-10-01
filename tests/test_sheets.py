"""Resistive sheet: closed-form R and T, the Ampere-cell conductivity, and FDFD."""

import numpy as np

from maxwell_fd.analytics.sheets import (
    chi_ee_from_zs,
    resistive_coefficients,
    resistive_sheet_fields,
    sheet_coefficients,
)
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.drivers.fdtd2d import cfl_timestep, step_tmz, zeros_tmz
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.sheets import ResistiveSheet
from maxwell_fd.materials.volume import TMComponents, UniformIsotropic
from maxwell_fd.utils.constants import C0, EPS0, ETA0, MU0

# MaxwellMOM rt_from_chi(chi_ee_from_Zs(100, k, eta0=ETA0), 0, k) at k = 2π/0.3.
# Both polarizations agree at normal incidence. Stored, not imported.
_MOM_K = 2.0 * np.pi / 0.3
_MOM_ZS = 100.0
_MOM_CHI = -0.17987547479999969j
_MOM_R = -0.6532174651970024 + 0.0j
_MOM_T = 0.34678253480299764 + 0.0j

# 8×80 box, Δ = 1, sheet on the electric node y = 40, λ = 46, Z_s = η0.
# Measured packed electric error 1.819e-4 (TMz) and 4.197e-3 (TEz).
SHEET_TMZ_BAR = 6.0e-4
SHEET_TEZ_BAR = 1.2e-2


def test_chi_map_matches_the_impedance_and_a_maxwellmom_value() -> None:
    k = _MOM_K
    omega = k * C0
    chi = chi_ee_from_zs(_MOM_ZS, k)
    np.testing.assert_allclose(chi, ETA0 / (1j * k * _MOM_ZS), atol=1e-12)
    # The stored η0, c0, and ε0 agree with η0 = 1/(c0 ε0) to about 1e-9.
    np.testing.assert_allclose(chi, 1.0 / (1j * omega * EPS0 * _MOM_ZS), rtol=1e-9)
    np.testing.assert_allclose(chi, _MOM_CHI, atol=1e-12)
    reflected, transmitted = resistive_coefficients(_MOM_ZS, k)
    np.testing.assert_allclose(reflected, -ETA0 / (ETA0 + 2.0 * _MOM_ZS), atol=1e-12)
    np.testing.assert_allclose(
        transmitted, 2.0 * _MOM_ZS / (ETA0 + 2.0 * _MOM_ZS), atol=1e-12
    )
    np.testing.assert_allclose(reflected, _MOM_R, atol=1e-12)
    np.testing.assert_allclose(transmitted, _MOM_T, atol=1e-12)
    np.testing.assert_allclose(transmitted - reflected, 1.0, atol=1e-12)
    same = resistive_coefficients(_MOM_ZS, k, te=True)
    np.testing.assert_allclose(same[0], reflected, atol=1e-12)
    try:
        chi_ee_from_zs(0.0, k)
    except ValueError as exc:
        assert "nonzero" in str(exc)
    else:
        raise AssertionError("zero impedance was accepted")


def test_oblique_coefficients_follow_the_angle_factors() -> None:
    k = 1.7
    theta = np.deg2rad(30.0)
    chi_ee = 0.3 - 0.1j
    chi_mm = 0.2 + 0.0j
    chi_nn = 0.4 + 0.0j
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    alpha0 = 1j * k * chi_ee / 2.0
    beta0 = 1j * k * chi_mm / 2.0
    alpha = alpha0 / cosine + 1j * k * sine**2 * chi_nn / (2.0 * cosine)
    beta = beta0 * cosine
    electric = (1.0 - alpha) / (1.0 + alpha)
    magnetic = (1.0 - beta) / (1.0 + beta)
    reflected, transmitted = sheet_coefficients(
        chi_ee, chi_mm, k, theta=theta, chi_mm_nn=chi_nn
    )
    np.testing.assert_allclose(reflected, 0.5 * (electric - magnetic), atol=1e-12)
    np.testing.assert_allclose(transmitted, 0.5 * (electric + magnetic), atol=1e-12)
    without = sheet_coefficients(chi_ee, chi_mm, k, theta=theta)
    assert abs(reflected - without[0]) > 1e-6
    normal = sheet_coefficients(chi_ee, chi_mm, k, chi_mm_nn=chi_nn)
    plain = sheet_coefficients(chi_ee, chi_mm, k)
    np.testing.assert_allclose(normal[0], plain[0], atol=1e-12)
    tez = sheet_coefficients(chi_ee, chi_mm, k, theta=theta, te=True, chi_mm_nn=chi_nn)
    alpha_te = alpha0 * cosine
    beta_te = beta0 / cosine
    electric_te = (1.0 - alpha_te) / (1.0 + alpha_te)
    magnetic_te = (1.0 - beta_te) / (1.0 + beta_te)
    np.testing.assert_allclose(tez[0], 0.5 * (electric_te - magnetic_te), atol=1e-12)
    try:
        sheet_coefficients(chi_ee, chi_mm, k, theta=0.5 * np.pi)
    except ValueError as exc:
        assert "theta" in str(exc)
    else:
        raise AssertionError("grazing incidence was accepted")


def test_sheet_field_is_continuous_in_e_and_jumps_in_h() -> None:
    omega = 2.0 * np.pi * C0 / 46.0
    zs = 125.0
    y = np.array([0.0])
    below = resistive_sheet_fields(y, y_sheet=0.0, omega=omega, zs=zs, branch="below")
    above = resistive_sheet_fields(y, y_sheet=0.0, omega=omega, zs=zs, branch="above")
    ez_b, hx_b, hz_b, ex_b = below
    ez_a, hx_a, hz_a, ex_a = above
    np.testing.assert_allclose(ez_a, ez_b, atol=1e-12)
    np.testing.assert_allclose(ex_a, ex_b, atol=1e-12)
    np.testing.assert_allclose(-(hx_a - hx_b), ez_a / zs, atol=1e-12)
    np.testing.assert_allclose(hz_a - hz_b, ex_a / zs, atol=1e-12)
    line = np.linspace(-2.0, 3.0, 11)
    ez, hx, hz, ex = resistive_sheet_fields(line, y_sheet=0.0, omega=omega, zs=zs)
    step = 1e-6
    ez_hi, _hx_hi, hz_hi, _ex_hi = resistive_sheet_fields(
        line + step, y_sheet=0.0, omega=omega, zs=zs
    )
    ez_lo, _hx_lo, hz_lo, _ex_lo = resistive_sheet_fields(
        line - step, y_sheet=0.0, omega=omega, zs=zs
    )
    off = np.abs(line) > 0.05
    d_ez = (ez_hi[off] - ez_lo[off]) / (2.0 * step)
    d_hz = (hz_hi[off] - hz_lo[off]) / (2.0 * step)
    np.testing.assert_allclose(
        hx[off], -d_ez / (1j * omega * MU0), rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        ex[off], d_hz / (1j * omega * EPS0), rtol=1e-7, atol=1e-7
    )
    del ez, hz


def test_sheet_conductivity_uses_the_ampere_cell() -> None:
    grid = YeeGrid2D(6, 8, 0.2, 0.25, Polarization.TMZ, Boundary.PEC)
    sheet = ResistiveSheet(0.5, 50.0)
    bare = UniformIsotropic(sigma=0.25).sample_tm(grid)
    painted = sheet.apply(grid, bare)
    expected = np.array(bare.sigma_z, copy=True)
    expected[:, 2] += 1.0 / (50.0 * grid.dy)
    np.testing.assert_allclose(painted.sigma_z, expected)
    upper = ResistiveSheet(0.5 + 0.5 * grid.dy, 50.0).apply(grid, bare)
    assert np.argmax(upper.sigma_z[0] - bare.sigma_z[0]) == 3
    face = 0.5 + 0.5 * grid.dy
    just_below = ResistiveSheet(face - 1e-12, 50.0).apply(grid, bare)
    assert np.argmax(just_below.sigma_z[0] - bare.sigma_z[0]) == 2
    te = YeeGrid2D(6, 8, 0.2, 0.25, Polarization.TEZ, Boundary.PEC)
    te_bare = UniformIsotropic(sigma=0.25).sample_te(te)
    te_painted = ResistiveSheet(0.5, 50.0).apply(te, te_bare)
    np.testing.assert_allclose(te_painted.sigma_y, te_bare.sigma_y)
    assert te_painted.sigma_x[0, 2] == te_bare.sigma_x[0, 2] + 1.0 / (50.0 * te.dy)
    try:
        ResistiveSheet(0.0, 50.0).apply(grid, bare)
    except ValueError as exc:
        assert "PEC" in str(exc)
    else:
        raise AssertionError("a sheet on the wall was accepted")
    try:
        ResistiveSheet(10.0, 50.0).apply(grid, bare)
    except ValueError as exc:
        assert "misses" in str(exc)
    else:
        raise AssertionError("a sheet outside the grid was accepted")
    try:
        ResistiveSheet(0.5, 0.0)
    except ValueError as exc:
        assert "zs" in str(exc)
    else:
        raise AssertionError("zero impedance was accepted")
    initial = zeros_tmz(grid)
    rng = np.random.default_rng(1)
    initial.ez[...] = rng.normal(size=initial.ez.shape)
    initial.hx[...] = rng.normal(size=initial.hx.shape)
    initial.hy[...] = rng.normal(size=initial.hy.shape)
    hand = TMComponents(bare.eps_z, bare.mu_x, bare.mu_y, expected)
    dt = 0.5 * cfl_timestep(grid.dx, grid.dy)
    left = zeros_tmz(grid)
    right = zeros_tmz(grid)
    vacuum = zeros_tmz(grid)
    for state in (left, right, vacuum):
        state.ez[...] = initial.ez
        state.hx[...] = initial.hx
        state.hy[...] = initial.hy
    step_tmz(grid, painted, left, dt)
    step_tmz(grid, hand, right, dt)
    step_tmz(grid, bare, vacuum, dt)
    np.testing.assert_allclose(left.ez, right.ez, atol=1e-12)
    assert np.max(np.abs(left.ez - vacuum.ez)) > 1e-8


def test_resistive_sheet_matches_the_analytic_field() -> None:
    tmz = _sheet_error(Polarization.TMZ)
    tez = _sheet_error(Polarization.TEZ)
    assert tmz < SHEET_TMZ_BAR, tmz
    assert tez < SHEET_TEZ_BAR, tez


def _sheet_error(pol: Polarization) -> float:
    nx, ny, dx = 8, 80, 1.0
    y_sheet = 40.0
    wavelength = 46.0
    zs = float(ETA0)
    omega = 2.0 * np.pi * C0 / wavelength
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PEC)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    operator = FDFDOperator(grid, ResistiveSheet(y_sheet, zs).apply(grid, bare))
    if pol is Polarization.TMZ:
        _x, y = grid.coordinates("ez")
        ez, _hx, _hz, _ex = resistive_sheet_fields(
            y, y_sheet=y_sheet, omega=omega, zs=zs
        )
        field = np.broadcast_to(ez, grid.shapes()["ez"]).copy()
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": field}
        )
        exact = operator.layout.pack_e({"ez": field})
    else:
        _x, y = grid.coordinates("ex")
        _ez, _hx, _hz, ex = resistive_sheet_fields(
            y, y_sheet=y_sheet, omega=omega, zs=zs
        )
        field = np.broadcast_to(ex, grid.shapes()["ex"]).copy()
        ey = np.zeros(grid.shapes()["ey"], dtype=np.complex128)
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ex": field, "ey": ey}
        )
        exact = operator.layout.pack_e({"ex": field, "ey": ey})
    return float(np.linalg.norm(solved - exact) / np.linalg.norm(exact))
