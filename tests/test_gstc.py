"""Symmetric GSTC cell: closed-form R and T, the split magnetic face, and FDFD."""

import numpy as np

from maxwell_fd.analytics.sheets import (
    oblique_sheet_fields,
    sheet_coefficients,
    uniform_sheet_fields,
)
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.conductors import Circle, Conductors
from maxwell_fd.materials.sheets import SymmetricSheet
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.gstc import _cuts, incident_load, magnetic_face
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0, EPS0, ETA0, MU0

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


def test_normal_incidence_absorber_has_zero_reflection_and_transmission() -> None:
    k = 2.0 * np.pi
    chi = -2j / k
    for te in (False, True):
        reflected, transmitted = sheet_coefficients(chi, chi, k, te=te)
        np.testing.assert_allclose(reflected, 0.0, atol=1e-12)
        np.testing.assert_allclose(transmitted, 0.0, atol=1e-12)


def test_finite_sheet_selects_a_half_open_span() -> None:
    try:
        SymmetricSheet(4.5, 0.5, 0.25, x0=1.0)
    except ValueError as exc:
        assert "both" in str(exc)
    else:
        raise AssertionError("a one-sided extent was accepted")
    try:
        SymmetricSheet(4.5, 0.5, 0.25, x0=3.0, x1=1.0)
    except ValueError as exc:
        assert "x0" in str(exc)
    else:
        raise AssertionError("an inverted extent was accepted")
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(8, 16, 1.0, 1.0, pol, Boundary.PEC)
        bare = (
            UniformIsotropic().sample_tm(grid)
            if pol is Polarization.TMZ
            else UniformIsotropic().sample_te(grid)
        )
        full = FDFDOperator(grid, bare, sheet=SymmetricSheet(4.5, 0.5, 0.25))
        finite = FDFDOperator(
            grid, bare, sheet=SymmetricSheet(4.5, 0.5, 0.25, x0=2.0, x1=5.0)
        )
        face = magnetic_face(grid, 4.5)
        full_cuts, _full_index = _cuts(grid, full.layout, face)
        cuts, _index = _cuts(grid, finite.layout, face, finite.sheet)
        assert len(cuts) == 3, len(cuts)
        assert len(full_cuts) > len(cuts)
        omega = 2.0 * np.pi * C0 / 10.0
        matrix = finite.system_matrix(omega)
        assert matrix.shape == (finite.layout.n_e + 2 * len(cuts),) * 2
        assert full.system_matrix(omega).shape[0] > matrix.shape[0]
        if pol is Polarization.TEZ:
            cut = cuts[0]
            hm = finite.layout.n_e
            hp = hm + 1
            np.testing.assert_allclose(matrix[cut.ey_here, hm], 0.5, atol=1e-12)
            np.testing.assert_allclose(matrix[cut.ey_here, hp], 0.5, atol=1e-12)
            np.testing.assert_allclose(matrix[cut.ey_right, hm], -0.5, atol=1e-12)
            np.testing.assert_allclose(matrix[cut.ey_right, hp], -0.5, atol=1e-12)
            extra = np.asarray(
                matrix[cut.ey_here, finite.layout.n_e :].todense()
            ).ravel()
            assert np.count_nonzero(np.abs(extra) > 1e-12) == 2
    try:
        FDFDOperator(grid, bare, sheet=SymmetricSheet(4.5, 0.5, 0.25, x0=0.1, x1=0.2))
    except ValueError as exc:
        assert "span" in str(exc)
    else:
        raise AssertionError("a span with no magnetic sample was accepted")


def test_interior_sheet_keeps_a_pml_and_a_stretched_sheet_is_refused() -> None:
    grid = YeeGrid2D(16, 40, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    bare = UniformIsotropic().sample_tm(grid)
    interior = SymmetricSheet(20.5, 0.5, 0.25, x0=6.0, x1=10.0)
    operator = FDFDOperator(grid, bare, PMLSpec.box(4), sheet=interior)
    omega = 2.0 * np.pi * C0 / 10.0
    face = magnetic_face(grid, 20.5)
    cuts, _index = _cuts(grid, operator.layout, face, interior)
    assert operator.system_matrix(omega).shape[0] == operator.layout.n_e + 2 * len(cuts)
    try:
        FDFDOperator(
            grid,
            bare,
            PMLSpec.box(4),
            sheet=SymmetricSheet(20.5, 0.5, 0.25, x0=2.0, x1=8.0),
        )
    except ValueError as exc:
        assert "unstretched" in str(exc)
    else:
        raise AssertionError("a sheet entering the PML was accepted")
    try:
        FDFDOperator(
            grid,
            bare,
            PMLSpec.box(4),
            sheet=SymmetricSheet(2.5, 0.5, 0.25, x0=6.0, x1=10.0),
        )
    except ValueError as exc:
        assert "unstretched" in str(exc)
    else:
        raise AssertionError("a sheet on a stretched face was accepted")


def test_incident_load_matches_the_sheet_rows() -> None:
    for pol in (Polarization.TMZ, Polarization.TEZ):
        _check_incident_load(pol)


def test_finite_absorber_casts_a_shadow() -> None:
    # 4λ sheet, 10 cells per λ. The 10λ example grid stays out of the suite.
    for pol in (Polarization.TMZ, Polarization.TEZ):
        lit, shadow = _absorber_means(pol)
        assert shadow < 0.20, (pol, lit, shadow)
        assert lit > 0.95, (pol, lit, shadow)


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


def _check_incident_load(pol: Polarization) -> None:
    wavelength = 10.0
    omega = 2.0 * np.pi * C0 / wavelength
    k = omega / C0
    chi = -2j / k
    grid = YeeGrid2D(8, 16, 1.0, 1.0, pol, Boundary.PEC)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    y_sheet = 4.5
    sheet = SymmetricSheet(y_sheet, chi, chi, x0=2.0, x1=6.0)
    operator = FDFDOperator(grid, bare, sheet=sheet)
    if pol is Polarization.TMZ:
        x, y = grid.coordinates("ez")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        electric = np.exp(-1j * k * (yy - y_sheet))
        hx_x, hx_y = grid.coordinates("hx")
        hxx, hyy = np.meshgrid(hx_x, hx_y, indexing="ij")
        magnetic = np.exp(-1j * k * (hyy - y_sheet)) / ETA0
        fields = {"ez": electric}
        packed = operator.layout.pack_e({"ez": electric})
    else:
        x, y = grid.coordinates("ex")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        electric = np.exp(-1j * k * (yy - y_sheet))
        ey = np.zeros(grid.shapes()["ey"], dtype=np.complex128)
        hz_x, hz_y = grid.coordinates("hz")
        hxx, hyy = np.meshgrid(hz_x, hz_y, indexing="ij")
        magnetic = -np.exp(-1j * k * (hyy - y_sheet)) / ETA0
        fields = {"ex": electric, "ey": ey}
        packed = operator.layout.pack_e({"ex": electric, "ey": ey})
    load = incident_load(grid, operator.layout, sheet, omega, fields, magnetic)
    np.testing.assert_allclose(load[: operator.layout.n_e], 0.0, atol=0.0)
    matrix = operator.system_matrix(omega)
    face = magnetic_face(grid, y_sheet)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    state = np.zeros(matrix.shape[0], dtype=np.complex128)
    state[: operator.layout.n_e] = packed
    for index, cut in enumerate(cuts):
        state[operator.layout.n_e + 2 * index] = magnetic[cut.i, face]
        state[operator.layout.n_e + 2 * index + 1] = magnetic[cut.i, face]
    residual = matrix @ state
    np.testing.assert_allclose(
        residual[operator.layout.n_e :], -load[operator.layout.n_e :], atol=1e-8
    )
    alpha = 1j * omega * EPS0 * chi / 2.0
    beta = 1j * omega * MU0 * chi / 2.0
    below = np.exp(1j * k * 0.5)
    above = np.exp(-1j * k * 0.5)
    h_face = (1.0 if pol is Polarization.TMZ else -1.0) / ETA0
    if pol is Polarization.TMZ:
        expect_h = -alpha * (below + above)
        expect_e = -(above - below) - 2.0 * beta * h_face
    else:
        expect_h = alpha * (below + above)
        expect_e = -(above - below) + 2.0 * beta * h_face
    np.testing.assert_allclose(load[operator.layout.n_e], expect_h, atol=1e-9)
    np.testing.assert_allclose(load[operator.layout.n_e + 1], expect_e, atol=1e-9)


def _absorber_means(pol: Polarization) -> tuple[float, float]:
    wavelength = 1.0
    dx = 0.1
    pml = 6
    pad = 10
    sheet_cells = 40
    nx = pml + pad + sheet_cells + pad + pml
    ny = pml + pad + pad + pml
    x0 = (pml + pad) * dx
    x1 = x0 + sheet_cells * dx
    y_sheet = (0.5 * ny - 0.5) * dx
    omega = 2.0 * np.pi * C0 / wavelength
    k = omega / C0
    chi = -2j / k
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PEC)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    sheet = SymmetricSheet(y_sheet, chi, chi, x0, x1)
    operator = FDFDOperator(grid, bare, PMLSpec.box(pml), sheet=sheet)
    if pol is Polarization.TMZ:
        x, y = grid.coordinates("ez")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        electric = np.exp(-1j * k * (yy - y_sheet))
        hx_x, hx_y = grid.coordinates("hx")
        hxx, hyy = np.meshgrid(hx_x, hx_y, indexing="ij")
        magnetic = np.exp(-1j * k * (hyy - y_sheet)) / ETA0
        solved = operator.solve(
            omega,
            np.zeros(operator.layout.n_e),
            incident={"ez": electric, "hx": magnetic},
        )
        dof = operator.layout.e[0]
    else:
        x, y = grid.coordinates("ex")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        electric = np.exp(-1j * k * (yy - y_sheet))
        ey_x, ey_y = grid.coordinates("ey")
        eyx, eyy = np.meshgrid(ey_x, ey_y, indexing="ij")
        normal = np.zeros_like(eyy, dtype=np.complex128)
        hz_x, hz_y = grid.coordinates("hz")
        hxx, hyy = np.meshgrid(hz_x, hz_y, indexing="ij")
        magnetic = -np.exp(-1j * k * (hyy - y_sheet)) / ETA0
        solved = operator.solve(
            omega,
            np.zeros(operator.layout.n_e),
            incident={"ex": electric, "ey": normal, "hz": magnetic},
        )
        dof = operator.layout.e[0]
        solved = solved[: dof.i.size]
    total = dof.scatter(solved + dof.pack(electric), fill=0.0)
    margin = 0.2 * (x1 - x0)
    lit = _window_mean(
        total, x, y, x0 + margin, x1 - margin, y_sheet - 0.7, y_sheet - 0.25
    )
    shadow = _window_mean(
        total, x, y, x0 + margin, x1 - margin, y_sheet + 0.25, y_sheet + 0.7
    )
    return lit, shadow


# One transverse period at 30°, height 4λ. χ_ee = 0.5, χ_mm = 0.25,
# χ_mm^nn = 0.4. At 10 cells per wavelength the packed electric error is
# 1.498e-1 (TMz) and 1.212e-1 (TEz). Four times that resolution drops both
# by more than half. Pure χ_mm^nn = 0.4 at 20 cells per wavelength is
# 1.029e-1 in TMz.
NN_TMZ_BAR = 0.24
NN_TEZ_BAR = 0.20
NN_PURE_BAR = 0.17


def test_chi_mm_nn_derivative_is_on_the_tmz_jump() -> None:
    omega = 2.0 * np.pi * C0
    dx = 0.5
    chi_nn = 0.4
    gamma = chi_nn / (2.0 * 1j * omega * MU0 * dx * dx)
    grid = YeeGrid2D(4, 8, dx, dx, Polarization.TMZ, Boundary.PERIODIC_X)
    bare = UniformIsotropic().sample_tm(grid)
    sheet = SymmetricSheet(2.25, 0.0, 0.0, chi_mm_nn=chi_nn)
    operator = FDFDOperator(grid, bare, sheet=sheet)
    matrix = operator.system_matrix(omega).tocsr()
    face = magnetic_face(grid, 2.25)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    chosen = next(cut for cut in cuts if cut.i == 0)
    dof = operator.layout.e[0]
    ids = -np.ones(dof.shape, dtype=np.int64)
    ids[dof.i, dof.j] = np.arange(dof.i.size)
    row = operator.layout.n_e + 2 * cuts.index(chosen)
    j_plus = face + 1
    np.testing.assert_allclose(matrix[row, ids[0, face]], -2.0 * gamma, atol=1e-12)
    np.testing.assert_allclose(matrix[row, ids[1, face]], gamma, atol=1e-12)
    np.testing.assert_allclose(matrix[row, ids[3, face]], gamma, atol=1e-12)
    np.testing.assert_allclose(matrix[row, ids[0, j_plus]], -2.0 * gamma, atol=1e-12)
    np.testing.assert_allclose(matrix[row, row], -1.0, atol=1e-12)
    np.testing.assert_allclose(matrix[row, row + 1], 1.0, atol=1e-12)
    electric = matrix[row, : operator.layout.n_e].sum()
    np.testing.assert_allclose(electric, 0.0, atol=1e-12)
    tez = YeeGrid2D(6, 8, 1.0, 1.0, Polarization.TEZ, Boundary.PERIODIC_X)
    plain = FDFDOperator(
        tez, UniformIsotropic().sample_te(tez), sheet=SymmetricSheet(3.5, 0.5, 0.25)
    )
    with_nn = FDFDOperator(
        tez,
        UniformIsotropic().sample_te(tez),
        sheet=SymmetricSheet(3.5, 0.5, 0.25, chi_mm_nn=chi_nn),
    )
    frequency = 2.0 * np.pi * C0 / 46.0
    difference = plain.system_matrix(frequency) - with_nn.system_matrix(frequency)
    np.testing.assert_allclose(difference.toarray(), 0.0, atol=1e-12)
    closed = YeeGrid2D(8, 80, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    walled = FDFDOperator(
        closed,
        UniformIsotropic().sample_tm(closed),
        sheet=SymmetricSheet(40.5, 0.5, 0.25, chi_mm_nn=chi_nn),
    )
    try:
        walled.system_matrix(frequency)
    except ValueError as exc:
        assert "PEC" in str(exc)
    else:
        raise AssertionError("chi_mm_nn was accepted against the PEC wall")
    try:
        SymmetricSheet(40.5, 0.5, 0.25, chi_mm_nn=np.nan)
    except ValueError as exc:
        assert "chi" in str(exc)
    else:
        raise AssertionError("a non-finite chi_mm_nn was accepted")
    interior = FDFDOperator(
        closed,
        UniformIsotropic().sample_tm(closed),
        sheet=SymmetricSheet(40.5, 0.0, 0.0, x0=2.0, x1=6.0, chi_mm_nn=chi_nn),
    )
    interior.system_matrix(frequency)
    _check_normal_load()


def test_periodic_oblique_sheet_matches_analytic_coefficients() -> None:
    mixed = _periodic_errors(0.5, 0.25, 0.4, points=10)
    assert mixed[Polarization.TMZ] < NN_TMZ_BAR, mixed[Polarization.TMZ]
    assert mixed[Polarization.TEZ] < NN_TEZ_BAR, mixed[Polarization.TEZ]
    fine = _periodic_errors(0.5, 0.25, 0.4, points=40)
    assert fine[Polarization.TMZ] < mixed[Polarization.TMZ] / 2.0
    assert fine[Polarization.TEZ] < mixed[Polarization.TEZ] / 2.0
    pure = _periodic_error(Polarization.TMZ, 0.0, 0.0, 0.4, 20)
    assert pure < NN_PURE_BAR
    tangential = _periodic_errors(0.5, 0.25, 0.0, points=10)
    assert tangential[Polarization.TMZ] < NN_TMZ_BAR
    assert tangential[Polarization.TEZ] < NN_TEZ_BAR
    assert abs(mixed[Polarization.TEZ] - tangential[Polarization.TEZ]) < 1e-12


def _check_normal_load() -> None:
    wavelength = 10.0
    omega = 2.0 * np.pi * C0 / wavelength
    wavenumber = omega / C0
    grid = YeeGrid2D(6, 8, 1.0, 1.0, Polarization.TMZ, Boundary.PERIODIC_X)
    bare = UniformIsotropic().sample_tm(grid)
    y_sheet = 4.5
    chi_nn = 0.3
    sheet = SymmetricSheet(y_sheet, 0.5, 0.25, chi_mm_nn=chi_nn)
    operator = FDFDOperator(grid, bare, sheet=sheet)
    theta = np.deg2rad(30.0)
    x, y = grid.coordinates("ez")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    phase = np.exp(-1j * wavenumber * (np.sin(theta) * xx + np.cos(theta) * yy))
    hx_x, hx_y = grid.coordinates("hx")
    hxx, hyy = np.meshgrid(hx_x, hx_y, indexing="ij")
    magnetic = (
        np.cos(theta)
        * np.exp(-1j * wavenumber * (np.sin(theta) * hxx + np.cos(theta) * hyy))
        / ETA0
    )
    load = incident_load(grid, operator.layout, sheet, omega, {"ez": phase}, magnetic)
    np.testing.assert_allclose(load[: operator.layout.n_e], 0.0, atol=0.0)
    matrix = operator.system_matrix(omega)
    state = np.zeros(matrix.shape[0], dtype=np.complex128)
    state[: operator.layout.n_e] = operator.layout.pack_e({"ez": phase})
    face = magnetic_face(grid, y_sheet)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    for index, cut in enumerate(cuts):
        state[operator.layout.n_e + 2 * index] = magnetic[cut.i, face]
        state[operator.layout.n_e + 2 * index + 1] = magnetic[cut.i, face]
    residual = matrix @ state
    np.testing.assert_allclose(
        residual[operator.layout.n_e :], -load[operator.layout.n_e :], atol=1e-8
    )
    flat = np.exp(-1j * wavenumber * (yy - y_sheet))
    flat_h = np.full(grid.shapes()["hx"], 1.0 / ETA0, dtype=np.complex128)
    flat_load = incident_load(grid, operator.layout, sheet, omega, {"ez": flat}, flat_h)
    alpha = 1j * omega * EPS0 * 0.5 / 2.0
    below = np.exp(1j * wavenumber * 0.5)
    above = np.exp(-1j * wavenumber * 0.5)
    np.testing.assert_allclose(
        flat_load[operator.layout.n_e], -alpha * (below + above), atol=1e-9
    )


def _periodic_errors(
    chi_ee: complex, chi_mm: complex, chi_nn: complex, points: int
) -> dict[Polarization, float]:
    return {
        pol: _periodic_error(pol, chi_ee, chi_mm, chi_nn, points)
        for pol in (Polarization.TMZ, Polarization.TEZ)
    }


def _periodic_error(
    pol: Polarization,
    chi_ee: complex,
    chi_mm: complex,
    chi_nn: complex,
    points: int,
) -> float:
    wavelength = 1.0
    theta = np.deg2rad(30.0)
    length = wavelength / float(np.sin(theta))
    nx = int(round(points / float(np.sin(theta))))
    dx = length / nx
    ny = int(round(4.0 * wavelength / dx))
    y_sheet = (ny // 2 - 0.5) * dx
    omega = 2.0 * np.pi * C0 / wavelength
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PERIODIC_X)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    operator = FDFDOperator(
        grid,
        bare,
        sheet=SymmetricSheet(y_sheet, chi_ee, chi_mm, chi_mm_nn=chi_nn),
    )
    reflected, transmitted = sheet_coefficients(
        chi_ee,
        chi_mm,
        omega / C0,
        theta=theta,
        te=pol is Polarization.TEZ,
        chi_mm_nn=chi_nn,
    )
    if pol is Polarization.TMZ:
        x, y = grid.coordinates("ez")
        exact = oblique_sheet_fields(
            x,
            y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
            theta=theta,
        )["ez"]
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": exact}
        )
        reference = operator.layout.pack_e({"ez": exact})
    else:
        x, y = grid.coordinates("ex")
        exact_ex = oblique_sheet_fields(
            x,
            y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
            theta=theta,
        )["ex"]
        ey_x, ey_y = grid.coordinates("ey")
        exact_ey = oblique_sheet_fields(
            ey_x,
            ey_y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
            theta=theta,
        )["ey"]
        solved = operator.solve(
            omega,
            np.zeros(operator.layout.n_e),
            dirichlet={"ex": exact_ex, "ey": exact_ey},
        )
        reference = operator.layout.pack_e({"ex": exact_ex, "ey": exact_ey})
    return float(np.linalg.norm(solved - reference) / np.linalg.norm(reference))


def _window_mean(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> float:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    assert np.any(mask)
    return float(np.mean(np.abs(field[mask])))
