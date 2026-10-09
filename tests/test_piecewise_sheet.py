"""Piecewise GSTC sheet: one χ on each cut face, same jumps."""

import numpy as np

from maxwell_fd.analytics.sheets import sheet_coefficients, uniform_sheet_fields
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.sheets import SymmetricSheet
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.gstc import _cuts, _global_ids, _nn_columns, magnetic_face
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0, EPS0, ETA0, MU0

# 6λ sheet, 10 cells per λ, split into 3λ + 3λ. Left is the normal-incidence
# absorber. Right is χ_ee=0.5, χ_mm=0.25. Windows sit 1λ inside each piece.
# Measured relative L2 (shadow of the absorber is an rms, because T = 0):
# TMz 0.164, 0.131, 0.121, 0.368 and TEz 0.161, 0.084, 0.111, 0.372,
# in the order lit/shadow of the left piece, then lit/shadow of the right.
# A uniform sheet of the right-hand χ has the same shadow error: the ends
# diffract. The bars sit about halfway above those values.
_LEFT_LIT = 0.25
_LEFT_SHADOW = 0.20
_RIGHT_LIT = 0.18
_RIGHT_SHADOW = 0.55


def test_two_pieces_match_each_analytic_field() -> None:
    errors = two_piece_errors()
    for pol in ("tmz", "tez"):
        assert errors[(pol, "left", "lit")] < _LEFT_LIT
        assert errors[(pol, "left", "shadow")] < _LEFT_SHADOW
        assert errors[(pol, "right", "lit")] < _RIGHT_LIT
        assert errors[(pol, "right", "shadow")] < _RIGHT_SHADOW


def test_one_piece_matches_the_scalar_sheet() -> None:
    omega = 2.0 * np.pi * C0
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(8, 16, 1.0, 1.0, pol, Boundary.PEC)
        bare = _bare(grid)
        scalar = SymmetricSheet(
            8.5, 0.5, 0.25, x0=2.0, x1=6.0, chi_mm_nn=0.4, chi_em=0.2, chi_me=0.2
        )
        piece = SymmetricSheet.piecewise(
            8.5,
            (
                SymmetricSheet(
                    8.5,
                    0.5,
                    0.25,
                    x0=2.0,
                    x1=6.0,
                    chi_mm_nn=0.4,
                    chi_em=0.2,
                    chi_me=0.2,
                ),
            ),
        )
        left = FDFDOperator(grid, bare, sheet=scalar).system_matrix(omega)
        right = FDFDOperator(grid, bare, sheet=piece).system_matrix(omega)
        np.testing.assert_allclose(left.toarray(), right.toarray(), atol=0.0, rtol=0.0)


def test_each_face_uses_its_own_susceptibility() -> None:
    omega = 2.0 * np.pi * C0
    for pol in (Polarization.TMZ, Polarization.TEZ):
        _faces_differ(pol, omega)


def test_piecewise_sheet_rejects_a_bad_run() -> None:
    y = 8.5
    left = SymmetricSheet(y, 0.5, 0.25, x0=0.0, x1=4.0)
    right = SymmetricSheet(y, 0.2, 0.1, x0=4.0, x1=8.0)
    try:
        SymmetricSheet.piecewise(y, (left, SymmetricSheet(y, 0.2, 0.1, x0=3.0, x1=6.0)))
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping pieces were accepted")
    try:
        SymmetricSheet.piecewise(y, (SymmetricSheet(y, 0.5, 0.25),))
    except ValueError as exc:
        assert "x0" in str(exc)
    else:
        raise AssertionError("a piece without an extent was accepted")
    try:
        SymmetricSheet.piecewise(y, (SymmetricSheet.piecewise(y, (left,)),))
    except ValueError as exc:
        assert "contain" in str(exc)
    else:
        raise AssertionError("a nested piece was accepted")
    try:
        SymmetricSheet(y, 0.5, 0.25, pieces=(left, right))
    except ValueError as exc:
        assert "chi" in str(exc)
    else:
        raise AssertionError("a piecewise sheet carried its own chi")
    try:
        SymmetricSheet(y, 0.0, 0.0, x0=0.0, x1=8.0, pieces=(left, right))
    except ValueError as exc:
        assert "extent" in str(exc)
    else:
        raise AssertionError("a piecewise sheet carried its own extent")
    try:
        SymmetricSheet.piecewise(
            y, (SymmetricSheet(y + 1.0, 0.5, 0.25, x0=0.0, x1=4.0),)
        )
    except ValueError as exc:
        assert "coordinate" in str(exc)
    else:
        raise AssertionError("a piece on another face was accepted")
    gap = SymmetricSheet.piecewise(
        y,
        (
            SymmetricSheet(y, 0.5, 0.25, x0=0.0, x1=3.0),
            SymmetricSheet(y, 0.2, 0.1, x0=5.0, x1=8.0),
        ),
    )
    grid = YeeGrid2D(8, 16, 1.0, 1.0, Polarization.TMZ, Boundary.PEC)
    operator = FDFDOperator(grid, _bare(grid), sheet=gap)
    face = magnetic_face(grid, y)
    cuts, _index = _cuts(grid, operator.layout, face, gap)
    xs = grid.coordinates("hx")[0]
    covered = [float(xs[cut.i]) for cut in cuts]
    assert all(value < 3.0 or value >= 5.0 for value in covered)
    assert any(3.0 <= float(x) < 5.0 for x in xs)


def _faces_differ(pol: Polarization, omega: float) -> None:
    grid = YeeGrid2D(8, 16, 1.0, 1.0, pol, Boundary.PEC)
    y = 8.5
    sheet = SymmetricSheet.piecewise(
        y,
        (
            SymmetricSheet(y, 0.5, 0.25, x0=0.0, x1=4.0),
            SymmetricSheet(
                y,
                0.2,
                0.1,
                x0=4.0,
                x1=6.0,
                chi_mm_nn=0.4,
                chi_em=0.15,
                chi_me=-0.1,
            ),
        ),
    )
    operator = FDFDOperator(grid, _bare(grid), sheet=sheet)
    matrix = operator.system_matrix(omega).tocsr()
    face = magnetic_face(grid, y)
    cuts, _index = _cuts(grid, operator.layout, face, sheet)
    sample = "hz" if pol is Polarization.TEZ else "hx"
    xs = grid.coordinates(sample)[0]
    left = next(cut for k, cut in enumerate(cuts) if float(xs[cut.i]) < 4.0)
    right = next(cut for cut in cuts if float(xs[cut.i]) >= 4.0)
    left_k = next(k for k, cut in enumerate(cuts) if cut is left)
    right_k = next(k for k, cut in enumerate(cuts) if cut is right)
    _assert_jump(
        matrix, operator.layout.n_e, left_k, left, pol, omega, 0.5, 0.25, 0.0, 0.0, 0.0
    )
    _assert_jump(
        matrix,
        operator.layout.n_e,
        right_k,
        right,
        pol,
        omega,
        0.2,
        0.1,
        0.15,
        -0.1,
        0.4,
    )
    if pol is Polarization.TMZ:
        _assert_normal(grid, operator, matrix, face, cuts, xs, omega)


def _assert_jump(
    matrix,
    n_e: int,
    k: int,
    cut,
    pol: Polarization,
    omega: float,
    chi_ee: float,
    chi_mm: float,
    chi_em: float,
    chi_me: float,
    chi_nn: float,
) -> None:
    alpha = 1j * omega * EPS0 * chi_ee / 2.0
    beta = 1j * omega * MU0 * chi_mm / 2.0
    kappa_em = 0j if chi_em == 0 else 1j * omega * chi_em / (2.0 * C0)
    kappa_me = 0j if chi_me == 0 else 1j * omega * chi_me / (2.0 * C0)
    hm = n_e + 2 * k
    hp = hm + 1
    if pol is Polarization.TMZ:
        # The normal term puts -2γ on each of the two center Ez samples.
        shift = 0j
        if chi_nn != 0.0:
            gamma = chi_nn / (2.0 * 1j * omega * MU0)
            shift = -2.0 * gamma
        expected_h = (alpha + shift, alpha + shift, -1.0 - kappa_em, 1.0 - kappa_em)
        expected_e = (-1.0 + kappa_me, 1.0 + kappa_me, beta, beta)
    else:
        expected_h = (-alpha, -alpha, -1.0 - kappa_em, 1.0 - kappa_em)
        expected_e = (-1.0 + kappa_me, 1.0 + kappa_me, -beta, -beta)
    columns = (cut.e_minus, cut.e_plus, hm, hp)
    for row, expected in ((hm, expected_h), (hp, expected_e)):
        got = [matrix[row, column] for column in columns]
        np.testing.assert_allclose(got, expected, atol=1e-12)


def _assert_normal(grid, operator, matrix, face: int, cuts, xs, omega: float) -> None:
    """TMz χ_mm^nn is the local coefficient. TEz never reaches this helper."""
    gamma = 0.4 / (2.0 * 1j * omega * MU0 * grid.dx * grid.dx)
    ez_id = _global_ids(operator.layout.e)[0]
    j_plus = face + 1
    for k, cut in enumerate(cuts):
        # The end sample's second difference reaches the PEC wall. Its χ_nn is zero.
        if not 2.0 <= float(xs[cut.i]) <= 5.0:
            continue
        columns = _nn_columns(grid, ez_id, cut.i, face, j_plus)
        neighbor = next(
            column
            for column, weight in columns
            if column not in (cut.e_minus, cut.e_plus) and weight == 1
        )
        hm = operator.layout.n_e + 2 * k
        expected = gamma if float(xs[cut.i]) >= 4.0 else 0.0
        np.testing.assert_allclose(matrix[hm, neighbor], expected, atol=1e-12)


def _bare(grid: YeeGrid2D):
    if grid.polarization is Polarization.TMZ:
        return UniformIsotropic().sample_tm(grid)
    return UniformIsotropic().sample_te(grid)


def two_piece_errors() -> dict[tuple[str, str, str], float]:
    """Relative window error of each piece, away from the junction and the ends."""
    wavelength = 1.0
    dx = 0.1
    pml = 6
    pad = 20
    sheet_cells = 60
    half = sheet_cells // 2
    nx = pml + pad + sheet_cells + pad + pml
    ny = pml + pad + pad + pml
    x0 = (pml + pad) * dx
    x_split = x0 + half * dx
    x1 = x0 + sheet_cells * dx
    y_sheet = (0.5 * ny - 0.5) * dx
    omega = 2.0 * np.pi * C0 / wavelength
    k = omega / C0
    margin = wavelength
    windows = {
        "left": (x0 + margin, x_split - margin),
        "right": (x_split + margin, x1 - margin),
    }
    bands = {
        "lit": (y_sheet - wavelength, y_sheet - 0.25 * wavelength),
        "shadow": (y_sheet + 0.25 * wavelength, y_sheet + wavelength),
    }
    errors: dict[tuple[str, str, str], float] = {}
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PEC)
        absorber = -2j / k
        sheet = SymmetricSheet.piecewise(
            y_sheet,
            (
                SymmetricSheet(y_sheet, absorber, absorber, x0=x0, x1=x_split),
                SymmetricSheet(y_sheet, 0.5, 0.25, x0=x_split, x1=x1),
            ),
        )
        operator = FDFDOperator(grid, _bare(grid), PMLSpec.box(pml), sheet=sheet)
        total, analytic_left, analytic_right, x, y = _total_and_analytic(
            pol, grid, operator, omega, k, y_sheet
        )
        for piece, (xa, xb) in windows.items():
            reference = analytic_left if piece == "left" else analytic_right
            for band, (ya, yb) in bands.items():
                errors[(pol.value, piece, band)] = _window(
                    total, reference, x, y, xa, xb, ya, yb
                )
    return errors


def _total_and_analytic(pol, grid, operator, omega, k, y_sheet):
    reflected_l, transmitted_l = sheet_coefficients(
        -2j / k, -2j / k, k, te=pol is Polarization.TEZ
    )
    reflected_r, transmitted_r = sheet_coefficients(
        0.5, 0.25, k, te=pol is Polarization.TEZ
    )
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
        total = dof.scatter(solved + dof.pack(electric), fill=0.0)
        analytic_l = _broadcast(
            y, grid.shapes()["ez"], y_sheet, omega, reflected_l, transmitted_l
        )
        analytic_r = _broadcast(
            y, grid.shapes()["ez"], y_sheet, omega, reflected_r, transmitted_r
        )
        return total, analytic_l, analytic_r, x, y
    x, y = grid.coordinates("ex")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    electric = np.exp(-1j * k * (yy - y_sheet))
    ey = np.zeros(grid.shapes()["ey"], dtype=np.complex128)
    hz_x, hz_y = grid.coordinates("hz")
    hxx, hyy = np.meshgrid(hz_x, hz_y, indexing="ij")
    magnetic = -np.exp(-1j * k * (hyy - y_sheet)) / ETA0
    solved = operator.solve(
        omega,
        np.zeros(operator.layout.n_e),
        incident={"ex": electric, "ey": ey, "hz": magnetic},
    )
    dof = operator.layout.e[0]
    solved = solved[: dof.i.size]
    total = dof.scatter(solved + dof.pack(electric), fill=0.0)
    analytic_l = _broadcast(
        y, grid.shapes()["ex"], y_sheet, omega, reflected_l, transmitted_l
    )
    analytic_r = _broadcast(
        y, grid.shapes()["ex"], y_sheet, omega, reflected_r, transmitted_r
    )
    return total, analytic_l, analytic_r, x, y


def _broadcast(y, shape, y_sheet, omega, reflected, transmitted):
    electric, _hx, _hz, _ex = uniform_sheet_fields(
        y, y_sheet=y_sheet, omega=omega, reflected=reflected, transmitted=transmitted
    )
    return np.broadcast_to(electric, shape).copy()


def _window(total, analytic, x, y, x0, x1, y0, y1) -> float:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    if not np.any(mask):
        raise ValueError("window missed every sample")
    got = np.asarray(total)[mask]
    exact = np.asarray(analytic)[mask]
    scale = np.linalg.norm(exact)
    if scale < 1e-8:
        return float(np.linalg.norm(got) / np.sqrt(got.size))
    return float(np.linalg.norm(got - exact) / scale)
