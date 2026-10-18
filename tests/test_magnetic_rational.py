"""A magnetic denominator in k_t is one auxiliary unknown per cut."""

import numpy as np

from maxwell_fd.analytics.sheets import (
    dispersive_susceptibility,
    oblique_sheet_fields,
    sheet_coefficients,
)
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.sheets import SymmetricSheet
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.gstc import _cuts, incident_load, magnetic_face
from maxwell_fd.utils.constants import C0, MU0

# One transverse period at 30°, height 4λ. χ_ee = 0.5, χ_mm0 = 0.25,
# χ_mm2 = 0.04, ξ_mm2 = 0.04. At 10 cells per wavelength the packed
# electric error is 0.1205 (TMz) and 0.1422 (TEz), both against
# (χ_mm0 + χ_mm2 (k sin θ)²) / (1 + ξ_mm2 (k sin θ)²). Four times that
# resolution drops both by more than half. A polynomial sheet, ξ_mm2 = 0,
# misses that target by more than the operator, in both polarizations.
_CHI_EE = 0.5
_CHI_MM0 = 0.25
_CHI_MM2 = 0.04
_XI = 0.04
_TMZ_BAR = 0.18
_TEZ_BAR = 0.21


def test_rational_susceptibility_moves_both_polarizations() -> None:
    k = 2.0 * np.pi
    theta = np.deg2rad(30.0)
    tangential = k * np.sin(theta)
    rational = dispersive_susceptibility(tangential, _CHI_MM0, _CHI_MM2, _XI)
    polynomial = dispersive_susceptibility(tangential, _CHI_MM0, _CHI_MM2)
    assert abs(rational - _CHI_MM0) > 1e-2
    assert abs(rational - polynomial) > 1e-2
    for te in (False, True):
        curved = sheet_coefficients(_CHI_EE, rational, k, theta=theta, te=te)
        local = sheet_coefficients(_CHI_EE, _CHI_MM0, k, theta=theta, te=te)
        straight = sheet_coefficients(_CHI_EE, polynomial, k, theta=theta, te=te)
        assert abs(curved[0] - local[0]) > 1e-3
        assert abs(curved[0] - straight[0]) > 1e-3


def test_rational_denominator_is_an_auxiliary() -> None:
    omega = 2.0 * np.pi * C0
    dx = 0.5
    chi_mm2 = 0.02
    xi = 0.01
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(4, 8, dx, dx, pol, Boundary.PERIODIC_X)
        sheet = _rational(2.25, chi_mm2, xi)
        operator = FDFDOperator(grid, _bare(grid), sheet=sheet)
        matrix = operator.system_matrix(omega).tocsr()
        polynomial = FDFDOperator(
            grid,
            _bare(grid),
            sheet=SymmetricSheet.magnetic_dispersion(
                2.25, _CHI_MM0, chi_mm2, chi_ee=_CHI_EE
            ),
        ).system_matrix(omega)
        face = magnetic_face(grid, sheet.y)
        cuts, _index = _cuts(grid, operator.layout, face, sheet)
        assert matrix.shape[0] == polynomial.shape[0] + len(cuts)
        _assert_stencil(grid, operator, matrix, sheet, omega, chi_mm2, xi)
        _assert_wave_symbol(grid, operator, matrix, sheet, omega, chi_mm2, xi)
        _assert_constant_cancels(grid, operator, matrix, sheet, omega)
        _assert_load_matches_matrix(grid, operator, sheet, omega, bloch=False)
        phased = YeeGrid2D(4, 8, dx, dx, pol, Boundary.PERIODIC_X, bloch_x=0.4)
        phased_sheet = _rational(2.25, chi_mm2, xi)
        phased_operator = FDFDOperator(phased, _bare(phased), sheet=phased_sheet)
        phased_matrix = phased_operator.system_matrix(omega).tocsr()
        _assert_bloch_weight(
            phased, phased_operator, phased_matrix, phased_sheet, xi, chi_mm2
        )
        _assert_load_matches_matrix(
            phased, phased_operator, phased_sheet, omega, bloch=True
        )
        zero = FDFDOperator(
            grid,
            _bare(grid),
            sheet=SymmetricSheet.magnetic_dispersion(
                2.25, _CHI_MM0, chi_mm2, chi_ee=_CHI_EE, xi_mm_2=0.0
            ),
        ).system_matrix(omega)
        np.testing.assert_allclose(
            polynomial.toarray(), zero.toarray(), atol=0.0, rtol=0.0
        )
    _assert_wall_and_gap(omega)
    try:
        SymmetricSheet.magnetic_dispersion(
            2.25, _CHI_MM0, _CHI_MM2, chi_ee=_CHI_EE, xi_mm_2=np.nan
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("a non-finite xi_mm_2 was accepted")


def test_periodic_rational_sheet_matches_the_effective_susceptibility() -> None:
    coarse = _errors(10)
    fine = _errors(40)
    assert coarse["tmz"] < _TMZ_BAR, coarse
    assert coarse["tez"] < _TEZ_BAR, coarse
    assert coarse["missed_tmz"] > coarse["tmz"]
    assert coarse["missed_tez"] > coarse["tez"]
    assert fine["tmz"] < coarse["tmz"] / 2.0
    assert fine["tez"] < coarse["tez"] / 2.0
    assert fine["missed_tmz"] > fine["tmz"]
    assert fine["missed_tez"] > fine["tez"]


def _rational(y: float, chi_mm2: complex, xi: complex, **kwargs) -> SymmetricSheet:
    return SymmetricSheet.magnetic_dispersion(
        y, _CHI_MM0, chi_mm2, chi_ee=_CHI_EE, xi_mm_2=xi, **kwargs
    )


def _assert_stencil(grid, operator, matrix, sheet, omega, chi_mm2, xi) -> None:
    side = 1.0 if grid.polarization is Polarization.TMZ else -1.0
    face = magnetic_face(grid, sheet.y)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    chosen = next(cut for cut in cuts if cut.i == 0)
    k = cuts.index(chosen)
    n_e = operator.layout.n_e
    n_cut = len(cuts)
    row = n_e + 2 * k + 1
    hm = n_e + 2 * k
    q = n_e + 2 * n_cut + k
    jump = side * 1j * omega * MU0 / 2.0
    spacing = grid.dx**2
    zeta_xi = xi / spacing
    zeta_chi = chi_mm2 / spacing
    left = next(cut for cut in cuts if cut.i == grid.nx - 1)
    right = next(cut for cut in cuts if cut.i == 1)
    left_k = cuts.index(left)
    right_k = cuts.index(right)
    np.testing.assert_allclose(matrix[row, hm], 0.0, atol=0.0)
    np.testing.assert_allclose(matrix[row, hm + 1], 0.0, atol=0.0)
    np.testing.assert_allclose(matrix[row, q], jump, atol=1e-9)
    np.testing.assert_allclose(matrix[row, n_e + 2 * left_k], 0.0, atol=0.0)
    aux = n_e + 2 * n_cut + k
    np.testing.assert_allclose(matrix[aux, q], 1.0 + 2.0 * zeta_xi, atol=1e-9)
    np.testing.assert_allclose(
        matrix[aux, n_e + 2 * n_cut + left_k], -zeta_xi, atol=1e-9
    )
    np.testing.assert_allclose(
        matrix[aux, n_e + 2 * n_cut + right_k], -zeta_xi, atol=1e-9
    )
    center_s = -(_CHI_MM0 + 2.0 * zeta_chi)
    np.testing.assert_allclose(matrix[aux, hm], center_s, atol=1e-9)
    np.testing.assert_allclose(matrix[aux, hm + 1], center_s, atol=1e-9)
    np.testing.assert_allclose(matrix[aux, n_e + 2 * left_k], zeta_chi, atol=1e-9)
    np.testing.assert_allclose(matrix[aux, n_e + 2 * left_k + 1], zeta_chi, atol=1e-9)
    np.testing.assert_allclose(matrix[hm, q], 0.0, atol=0.0)


def _assert_bloch_weight(grid, operator, matrix, sheet, xi, chi_mm2) -> None:
    forward, backward = grid.bloch_factors()
    face = magnetic_face(grid, sheet.y)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    chosen = next(cut for cut in cuts if cut.i == 0)
    end = next(cut for cut in cuts if cut.i == grid.nx - 1)
    n_e = operator.layout.n_e
    n_cut = len(cuts)
    spacing = grid.dx**2
    zeta_xi = xi / spacing
    zeta_chi = chi_mm2 / spacing
    aux = n_e + 2 * n_cut + cuts.index(chosen)
    left_q = n_e + 2 * n_cut + cuts.index(end)
    left_h = n_e + 2 * cuts.index(end)
    np.testing.assert_allclose(matrix[aux, left_q], -zeta_xi * backward, atol=1e-9)
    np.testing.assert_allclose(matrix[aux, left_h], zeta_chi * backward, atol=1e-9)
    end_aux = n_e + 2 * n_cut + cuts.index(end)
    right_q = n_e + 2 * n_cut + cuts.index(chosen)
    np.testing.assert_allclose(matrix[end_aux, right_q], -zeta_xi * forward, atol=1e-9)


def _assert_wave_symbol(grid, operator, matrix, sheet, omega, chi_mm2, xi) -> None:
    """The auxiliary row sees the discrete symbol, independent of the jump."""
    side = 1.0 if grid.polarization is Polarization.TMZ else -1.0
    sample = "hx" if grid.polarization is Polarization.TMZ else "hz"
    x = grid.coordinates(sample)[0]
    k_x = 2.0 * np.pi / (grid.nx * grid.dx)
    phi = k_x * grid.dx
    symbol = 4.0 * np.sin(phi / 2.0) ** 2 / grid.dx**2
    chi_eff = (_CHI_MM0 + chi_mm2 * symbol) / (1.0 + xi * symbol)
    face = magnetic_face(grid, sheet.y)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    n_e = operator.layout.n_e
    n_cut = len(cuts)
    denominator = np.zeros(matrix.shape[0], dtype=np.complex128)
    numerator = np.zeros(matrix.shape[0], dtype=np.complex128)
    for k, cut in enumerate(cuts):
        phase = np.exp(-1j * k_x * float(x[cut.i]))
        denominator[n_e + 2 * n_cut + k] = phase
        numerator[n_e + 2 * k] = phase
        numerator[n_e + 2 * k + 1] = phase
    den_residual = matrix @ denominator
    num_residual = matrix @ numerator
    for k, cut in enumerate(cuts):
        phase = np.exp(-1j * k_x * float(x[cut.i]))
        np.testing.assert_allclose(
            den_residual[n_e + 2 * n_cut + k], (1.0 + xi * symbol) * phase, atol=1e-8
        )
        np.testing.assert_allclose(
            num_residual[n_e + 2 * n_cut + k],
            -(_CHI_MM0 + chi_mm2 * symbol) * 2.0 * phase,
            atol=1e-8,
        )
    combined = numerator.copy()
    for k, cut in enumerate(cuts):
        phase = np.exp(-1j * k_x * float(x[cut.i]))
        combined[n_e + 2 * n_cut + k] = 2.0 * chi_eff * phase
    residual = matrix @ combined
    for k, cut in enumerate(cuts):
        phase = np.exp(-1j * k_x * float(x[cut.i]))
        np.testing.assert_allclose(residual[n_e + 2 * n_cut + k], 0.0, atol=1e-8)
        np.testing.assert_allclose(
            residual[n_e + 2 * k + 1],
            side * 1j * omega * MU0 * chi_eff * phase,
            atol=1e-8,
        )


def _assert_constant_cancels(grid, operator, matrix, sheet, omega) -> None:
    side = 1.0 if grid.polarization is Polarization.TMZ else -1.0
    face = magnetic_face(grid, sheet.y)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    n_e = operator.layout.n_e
    n_cut = len(cuts)
    state = np.zeros(matrix.shape[0], dtype=np.complex128)
    for k, _cut in enumerate(cuts):
        state[n_e + 2 * k] = 1.0
        state[n_e + 2 * k + 1] = 1.0
        state[n_e + 2 * n_cut + k] = 2.0 * _CHI_MM0
    residual = matrix @ state
    expected = side * 1j * omega * MU0 * _CHI_MM0
    for k, _cut in enumerate(cuts):
        np.testing.assert_allclose(residual[n_e + 2 * n_cut + k], 0.0, atol=1e-8)
        np.testing.assert_allclose(residual[n_e + 2 * k + 1], expected, atol=1e-8)


def _assert_load_matches_matrix(grid, operator, sheet, omega, bloch: bool) -> None:
    pol = grid.polarization
    face = magnetic_face(grid, sheet.y)
    sample = "hx" if pol is Polarization.TMZ else "hz"
    electric_name = "ez" if pol is Polarization.TMZ else "ex"
    x_e, y_e = grid.coordinates(electric_name)
    x_h, y_h = grid.coordinates(sample)
    if bloch:
        electric = np.ones(grid.shapes()[electric_name], dtype=np.complex128)
        magnetic = np.ones(grid.shapes()[sample], dtype=np.complex128)
    else:
        theta = np.deg2rad(30.0)
        k = omega / C0
        xx, yy = np.meshgrid(x_e, y_e, indexing="ij")
        electric = np.exp(-1j * k * (np.sin(theta) * xx + np.cos(theta) * yy))
        hxx, hyy = np.meshgrid(x_h, y_h, indexing="ij")
        magnetic = np.exp(-1j * k * (np.sin(theta) * hxx + np.cos(theta) * hyy))
    fields = {electric_name: electric}
    load = incident_load(grid, operator.layout, sheet, omega, fields, magnetic)
    np.testing.assert_allclose(load[: operator.layout.n_e], 0.0, atol=0.0)
    matrix = operator.system_matrix(omega)
    assert load.size == matrix.shape[0]
    state = np.zeros(matrix.shape[0], dtype=np.complex128)
    if pol is Polarization.TMZ:
        state[: operator.layout.n_e] = operator.layout.pack_e({"ez": electric})
    else:
        ey = np.zeros(grid.shapes()["ey"], dtype=np.complex128)
        state[: operator.layout.n_e] = operator.layout.pack_e(
            {"ex": electric, "ey": ey}
        )
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    for k, cut in enumerate(cuts):
        state[operator.layout.n_e + 2 * k] = magnetic[cut.i, face]
        state[operator.layout.n_e + 2 * k + 1] = magnetic[cut.i, face]
    residual = matrix @ state
    np.testing.assert_allclose(
        residual[operator.layout.n_e :], -load[operator.layout.n_e :], atol=1e-8
    )


def _assert_wall_and_gap(omega: float) -> None:
    for pol in (Polarization.TMZ, Polarization.TEZ):
        closed = YeeGrid2D(8, 80, 1.0, 1.0, pol, Boundary.PEC)
        walled = FDFDOperator(
            closed,
            _bare(closed),
            sheet=_rational(40.5, 0.0, 0.02),
        )
        try:
            walled.system_matrix(omega)
        except ValueError as exc:
            assert "xi_mm_2" in str(exc)
            assert "PEC" in str(exc)
        else:
            raise AssertionError("xi_mm_2 was accepted against the PEC wall")
        spanned = FDFDOperator(
            closed,
            _bare(closed),
            sheet=_rational(40.5, 0.0, 0.02, x0=2.0, x1=6.0),
        )
        try:
            spanned.system_matrix(omega)
        except ValueError as exc:
            assert "past" in str(exc)
        else:
            raise AssertionError("xi_mm_2 was accepted past the sheet")
    _assert_gapped_load(omega)


def _assert_gapped_load(omega: float) -> None:
    """A gap makes the cut index differ from the sample index."""
    y = 3.5
    grid = YeeGrid2D(8, 8, 1.0, 1.0, Polarization.TMZ, Boundary.PERIODIC_X)
    bare = SymmetricSheet(y, _CHI_EE, _CHI_MM0, x0=1.0, x1=4.0)
    hot = SymmetricSheet(
        y, _CHI_EE, _CHI_MM0, x0=4.0, x1=5.0, chi_mm_2=0.02, xi_mm_2=0.01
    )
    rest = SymmetricSheet(y, _CHI_EE, _CHI_MM0, x0=5.0, x1=8.0)
    sheet = SymmetricSheet.piecewise(y, (bare, hot, rest))
    operator = FDFDOperator(grid, _bare(grid), sheet=sheet)
    _assert_load_matches_matrix(grid, operator, sheet, omega, bloch=False)
    face = magnetic_face(grid, y)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    assert [cut.i for cut in cuts] == [1, 2, 3, 4, 5, 6, 7]
    matrix = operator.system_matrix(omega).tocsr()
    n_e = operator.layout.n_e
    n_cut = len(cuts)
    hot_k = cuts.index(next(cut for cut in cuts if cut.i == 4))
    left_k = cuts.index(next(cut for cut in cuts if cut.i == 3))
    aux = n_e + 2 * n_cut + hot_k
    np.testing.assert_allclose(matrix[aux, n_e + 2 * left_k], 0.02, atol=1e-9)
    np.testing.assert_allclose(matrix[aux, n_e + 2 * n_cut + left_k], -0.01, atol=1e-9)
    try:
        edge = SymmetricSheet(y, _CHI_EE, _CHI_MM0, x0=1.0, x1=2.0, xi_mm_2=0.02)
        cover = SymmetricSheet(y, _CHI_EE, _CHI_MM0, x0=2.0, x1=8.0)
        FDFDOperator(
            grid, _bare(grid), sheet=SymmetricSheet.piecewise(y, (edge, cover))
        ).system_matrix(omega)
    except ValueError as exc:
        assert "past" in str(exc)
    else:
        raise AssertionError("xi_mm_2 was accepted beside a gap")
    try:
        SymmetricSheet(y, 0.0, 0.0, xi_mm_2=0.02, pieces=(bare, rest))
    except ValueError as exc:
        assert "piece" in str(exc)
    else:
        raise AssertionError("a parent xi_mm_2 was accepted")


def _errors(points: int) -> dict[str, float]:
    return {
        "tmz": _periodic(Polarization.TMZ, points, "rational"),
        "tez": _periodic(Polarization.TEZ, points, "rational"),
        "missed_tmz": _periodic(Polarization.TMZ, points, "polynomial"),
        "missed_tez": _periodic(Polarization.TEZ, points, "polynomial"),
    }


def _periodic(pol: Polarization, points: int, sheet_kind: str) -> float:
    wavelength = 1.0
    theta = float(np.deg2rad(30.0))
    length = wavelength / float(np.sin(theta))
    nx = int(round(points / float(np.sin(theta))))
    dx = length / nx
    ny = int(round(4.0 * wavelength / dx))
    y_sheet = (ny // 2 - 0.5) * dx
    omega = 2.0 * np.pi * C0 / wavelength
    k = float(omega / C0)
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PERIODIC_X)
    if sheet_kind == "polynomial":
        sheet = SymmetricSheet.magnetic_dispersion(
            y_sheet, _CHI_MM0, _CHI_MM2, chi_ee=_CHI_EE
        )
    else:
        sheet = _rational(y_sheet, _CHI_MM2, _XI)
    chi_mm = dispersive_susceptibility(k * np.sin(theta), _CHI_MM0, _CHI_MM2, _XI)
    operator = FDFDOperator(grid, _bare(grid), sheet=sheet)
    reflected, transmitted = sheet_coefficients(
        _CHI_EE, chi_mm, k, theta=theta, te=pol is Polarization.TEZ
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


def _bare(grid: YeeGrid2D):
    if grid.polarization is Polarization.TMZ:
        return UniformIsotropic().sample_tm(grid)
    return UniformIsotropic().sample_te(grid)
