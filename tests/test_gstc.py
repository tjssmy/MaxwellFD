"""Symmetric GSTC cell: closed-form R and T, the split magnetic face, and FDFD."""

import numpy as np

from maxwell_fd.analytics.sheets import sheet_coefficients, uniform_sheet_fields
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.conductors import Circle, Conductors
from maxwell_fd.materials.sheets import SymmetricSheet
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.gstc import _cuts, magnetic_face
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0, EPS0, MU0

# MaxwellMOM rt_from_chi(0.5, 0.25, k) at k = 2π/0.3, normal incidence.
# Both polarizations agree. Stored, not imported.
_MOM_K = 2.0 * np.pi / 0.3
_MOM_R = -0.09213343174368521 + 0.149072380640055j
_MOM_T = -0.8374826148041825 - 0.5176018991330957j

# 8×80 box, Δ = 1, sheet on the magnetic face y = 40.5, λ = 46,
# χ_ee = 0.5, χ_mm = 0.25.
# Measured packed electric error 1.388e-2 (TMz) and 6.387e-2 (TEz).
GSTC_TMZ_BAR = 4.2e-2
GSTC_TEZ_BAR = 1.9e-1


def test_sheet_coefficients_match_a_stored_maxwellmom_value() -> None:
    reflected, transmitted = sheet_coefficients(0.5, 0.25, _MOM_K)
    np.testing.assert_allclose(reflected, _MOM_R, atol=1e-12)
    np.testing.assert_allclose(transmitted, _MOM_T, atol=1e-12)
    other = sheet_coefficients(0.5, 0.25, _MOM_K, te=True)
    np.testing.assert_allclose(other[0], reflected, atol=1e-12)
    np.testing.assert_allclose(other[1], transmitted, atol=1e-12)


def test_symmetric_sheet_matches_the_analytic_field() -> None:
    coarse = _errors(0.5, 0.25)
    assert coarse[Polarization.TMZ] < GSTC_TMZ_BAR, coarse[Polarization.TMZ]
    assert coarse[Polarization.TEZ] < GSTC_TEZ_BAR, coarse[Polarization.TEZ]
    fine = _errors(0.5, 0.25, ny=240, dx=1.0 / 3.0)
    assert fine[Polarization.TMZ] < coarse[Polarization.TMZ] / 2.0
    assert fine[Polarization.TEZ] < coarse[Polarization.TEZ] / 2.0
    complex_chi = _errors(0.3 - 0.1j, 0.2)
    assert complex_chi[Polarization.TMZ] < GSTC_TMZ_BAR
    assert complex_chi[Polarization.TEZ] < GSTC_TEZ_BAR
    electric_only = _errors(0.5, 0.0)
    assert electric_only[Polarization.TMZ] < GSTC_TMZ_BAR
    assert electric_only[Polarization.TEZ] < GSTC_TEZ_BAR


def test_cut_coefficients_follow_the_jumps() -> None:
    omega = 2.0 * np.pi * C0 / 46.0
    chi_ee, chi_mm = 0.5, 0.25
    alpha = 1j * omega * EPS0 * chi_ee / 2.0
    beta = 1j * omega * MU0 * chi_mm / 2.0
    _assert_cut(Polarization.TMZ, omega, chi_ee, chi_mm, alpha, beta)
    _assert_cut(Polarization.TEZ, omega, chi_ee, chi_mm, alpha, beta)


def test_symmetric_sheet_rejects_the_wall_a_pml_and_a_conductor() -> None:
    grid = YeeGrid2D(8, 80, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    bare = UniformIsotropic().sample_tm(grid)
    sheet = SymmetricSheet(40.5, 0.5, 0.25)
    try:
        SymmetricSheet(40.5, np.nan, 0.25)
    except ValueError as exc:
        assert "chi" in str(exc)
    else:
        raise AssertionError("a non-finite susceptibility was accepted")
    try:
        FDFDOperator(grid, bare, sheet=SymmetricSheet(40.0, 0.5, 0.25))
    except ValueError as exc:
        assert "magnetic face" in str(exc)
    else:
        raise AssertionError("a sheet on an electric node was accepted")
    try:
        FDFDOperator(grid, bare, sheet=SymmetricSheet(0.5, 0.5, 0.25))
    except ValueError as exc:
        assert "PEC" in str(exc)
    else:
        raise AssertionError("a sheet on the wall face was accepted")
    try:
        FDFDOperator(grid, bare, sheet=SymmetricSheet(100.0, 0.5, 0.25))
    except ValueError as exc:
        assert "magnetic face" in str(exc)
    else:
        raise AssertionError("a sheet outside the grid was accepted")
    try:
        FDFDOperator(grid, bare, PMLSpec.box(2), sheet=sheet)
    except ValueError as exc:
        assert "unstretched" in str(exc)
    else:
        raise AssertionError("a sheet on a PML was accepted")
    try:
        FDFDOperator(
            grid, bare, conductors=Conductors((Circle(4.0, 40.0, 1.0),)), sheet=sheet
        )
    except ValueError as exc:
        assert "conductor" in str(exc)
    else:
        raise AssertionError("a sheet on a conductor was accepted")
    operator = FDFDOperator(grid, bare, sheet=sheet)
    try:
        operator.spatial_operator()
    except ValueError as exc:
        assert "GSTC" in str(exc)
    else:
        raise AssertionError("the eigenproblem accepted a sheet")
    vacuum = FDFDOperator(grid, bare, sheet=SymmetricSheet(40.5, 0.0, 0.0))
    assert vacuum.system_matrix(2.0 * np.pi * C0 / 46.0).shape[0] > operator.layout.n_e
    idle = FDFDOperator(grid, bare, PMLSpec(), conductors=Conductors(), sheet=sheet)
    assert (
        idle.system_matrix(2.0 * np.pi * C0 / 46.0).shape
        == operator.system_matrix(2.0 * np.pi * C0 / 46.0).shape
    )


def _errors(
    chi_ee: complex,
    chi_mm: complex,
    *,
    ny: int = 80,
    dx: float = 1.0,
    y_sheet: float = 40.5,
) -> dict[Polarization, float]:
    return {
        pol: _gstc_error(pol, chi_ee, chi_mm, ny=ny, dx=dx, y_sheet=y_sheet)
        for pol in (Polarization.TMZ, Polarization.TEZ)
    }


def _gstc_error(
    pol: Polarization,
    chi_ee: complex,
    chi_mm: complex,
    *,
    ny: int,
    dx: float,
    y_sheet: float,
) -> float:
    nx = int(round(8.0 / dx))
    wavelength = 46.0
    omega = 2.0 * np.pi * C0 / wavelength
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PEC)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    reflected, transmitted = sheet_coefficients(
        chi_ee, chi_mm, omega / C0, te=pol is Polarization.TEZ
    )
    operator = FDFDOperator(grid, bare, sheet=SymmetricSheet(y_sheet, chi_ee, chi_mm))
    if pol is Polarization.TMZ:
        _x, y = grid.coordinates("ez")
        ez, _hx, _hz, _ex = uniform_sheet_fields(
            y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
        )
        field = np.broadcast_to(ez, grid.shapes()["ez"]).copy()
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": field}
        )
        exact = operator.layout.pack_e({"ez": field})
    else:
        _x, y = grid.coordinates("ex")
        _ez, _hx, _hz, ex = uniform_sheet_fields(
            y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
        )
        field = np.broadcast_to(ex, grid.shapes()["ex"]).copy()
        ey = np.zeros(grid.shapes()["ey"], dtype=np.complex128)
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ex": field, "ey": ey}
        )
        exact = operator.layout.pack_e({"ex": field, "ey": ey})
    return float(np.linalg.norm(solved - exact) / np.linalg.norm(exact))


def _assert_cut(
    pol: Polarization,
    omega: float,
    chi_ee: complex,
    chi_mm: complex,
    alpha: complex,
    beta: complex,
) -> None:
    grid = YeeGrid2D(8, 80, 1.0, 1.0, pol, Boundary.PEC)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    operator = FDFDOperator(grid, bare, sheet=SymmetricSheet(40.5, chi_ee, chi_mm))
    matrix = operator.system_matrix(omega).tocsr()
    face = magnetic_face(grid, 40.5)
    cuts, _index = _cuts(grid, operator.layout, face)
    chosen = None
    for k, cut in enumerate(cuts):
        if pol is Polarization.TMZ or (cut.ey_here >= 0 and cut.ey_right >= 0):
            chosen = (k, cut)
            break
    assert chosen is not None
    k, cut = chosen
    hm = operator.layout.n_e + 2 * k
    hp = hm + 1
    assert matrix.shape == (operator.layout.n_e + 2 * len(cuts),) * 2
    if pol is Polarization.TMZ:
        np.testing.assert_allclose(matrix[cut.e_minus, hm], 1.0, atol=1e-12)
        np.testing.assert_allclose(matrix[cut.e_plus, hp], -1.0, atol=1e-12)
        expected_h = (alpha, alpha, -1.0, 1.0)
        expected_e = (-1.0, 1.0, beta, beta)
    else:
        np.testing.assert_allclose(matrix[cut.e_minus, hm], -1.0, atol=1e-12)
        np.testing.assert_allclose(matrix[cut.e_plus, hp], 1.0, atol=1e-12)
        np.testing.assert_allclose(matrix[cut.ey_here, hm], 0.5, atol=1e-12)
        np.testing.assert_allclose(matrix[cut.ey_here, hp], 0.5, atol=1e-12)
        np.testing.assert_allclose(matrix[cut.ey_right, hm], -0.5, atol=1e-12)
        np.testing.assert_allclose(matrix[cut.ey_right, hp], -0.5, atol=1e-12)
        expected_h = (-alpha, -alpha, -1.0, 1.0)
        expected_e = (-1.0, 1.0, -beta, -beta)
    columns = (cut.e_minus, cut.e_plus, hm, hp)
    for row, expected in ((hm, expected_h), (hp, expected_e)):
        got = [matrix[row, column] for column in columns]
        np.testing.assert_allclose(got, expected, atol=1e-12)
