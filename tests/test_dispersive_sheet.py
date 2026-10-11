"""Polynomial electric dispersion is the normal-susceptibility operator."""

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
from maxwell_fd.utils.constants import C0

# One transverse period at 30°, height 4λ. χ_ee0 = 0.5, χ_ee2 = 0.01,
# χ_mm = 0.25, so χ_mm^nn = χ_ee2 k² = 0.3948. At 10 cells per wavelength
# the packed electric error is 0.1497 (TMz, against χ̃_ee(k sin θ)) and
# 0.1212 (TEz, against χ_ee0). Four times that resolution drops both by
# more than half. A sheet that keeps χ_ee0 and drops χ_ee2 misses the
# TMz target by more than the operator does.
_TMZ_BAR = 0.22
_TEZ_BAR = 0.18
_CHI_EE0 = 0.5
_CHI_EE2 = 0.01
_CHI_MM = 0.25


def test_rational_susceptibility_follows_the_quotient() -> None:
    assert dispersive_susceptibility(0.0, 0.5, 0.2, 0.1) == 0.5
    straight = dispersive_susceptibility(2.0, 0.5, 0.01)
    np.testing.assert_allclose(straight, 0.5 + 0.01 * 4.0, atol=0.0)
    curved = dispersive_susceptibility(1.0, 0.5, 0.2, 0.1)
    np.testing.assert_allclose(curved, 0.7 / 1.1, atol=0.0)
    try:
        dispersive_susceptibility(1.0, 0.5, 0.0, -1.0)
    except ValueError as exc:
        assert "denominator" in str(exc)
    else:
        raise AssertionError("a vanishing dispersion denominator was accepted")
    try:
        dispersive_susceptibility(np.nan, 0.5)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("a non-finite k_t was accepted")


def test_polynomial_matches_the_normal_susceptibility() -> None:
    k = 2.0 * np.pi
    theta = np.deg2rad(30.0)
    tangential = k * np.sin(theta)
    effective = dispersive_susceptibility(tangential, _CHI_EE0, _CHI_EE2)
    chi_nn = _CHI_EE2 * k * k
    from_effective = sheet_coefficients(effective, _CHI_MM, k, theta=theta)
    from_normal = sheet_coefficients(
        _CHI_EE0, _CHI_MM, k, theta=theta, chi_mm_nn=chi_nn
    )
    np.testing.assert_allclose(from_effective, from_normal, atol=1e-12)
    tez_local = sheet_coefficients(_CHI_EE0, _CHI_MM, k, theta=theta, te=True)
    tez_effective = sheet_coefficients(effective, _CHI_MM, k, theta=theta, te=True)
    assert abs(tez_local[0] - tez_effective[0]) > 1e-3


def test_electric_dispersion_is_the_normal_operator() -> None:
    omega = 2.0 * np.pi * C0
    k = omega / C0
    chi_nn = _CHI_EE2 * k * k
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(6, 8, 1.0, 1.0, pol, Boundary.PERIODIC_X)
        bare = _bare(grid)
        dispersed = SymmetricSheet.electric_dispersion(
            3.5, _CHI_EE0, _CHI_EE2, k, chi_mm=_CHI_MM
        )
        scalar = SymmetricSheet(3.5, _CHI_EE0, _CHI_MM, chi_mm_nn=chi_nn)
        left = FDFDOperator(grid, bare, sheet=dispersed).system_matrix(omega)
        right = FDFDOperator(grid, bare, sheet=scalar).system_matrix(omega)
        np.testing.assert_allclose(left.toarray(), right.toarray(), atol=0.0, rtol=0.0)
    tez = YeeGrid2D(6, 8, 1.0, 1.0, Polarization.TEZ, Boundary.PERIODIC_X)
    local = FDFDOperator(
        tez,
        _bare(tez),
        sheet=SymmetricSheet(3.5, _CHI_EE0, _CHI_MM),
    ).system_matrix(omega)
    carried = FDFDOperator(
        tez,
        _bare(tez),
        sheet=SymmetricSheet.electric_dispersion(
            3.5, _CHI_EE0, _CHI_EE2, k, chi_mm=_CHI_MM
        ),
    ).system_matrix(omega)
    np.testing.assert_allclose((local - carried).toarray(), 0.0, atol=0.0, rtol=0.0)
    try:
        SymmetricSheet.electric_dispersion(3.5, _CHI_EE0, _CHI_EE2, 0.0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("a non-positive k was accepted")
    try:
        SymmetricSheet.electric_dispersion(3.5, _CHI_EE0, np.nan, k)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("a non-finite chi_ee_2 was accepted")


def test_periodic_dispersion_matches_the_effective_sheet() -> None:
    coarse = _errors(10)
    assert coarse["tmz"] < _TMZ_BAR, coarse["tmz"]
    assert coarse["tez"] < _TEZ_BAR, coarse["tez"]
    assert coarse["missed"] > coarse["tmz"]
    fine = _errors(40)
    assert fine["tmz"] < coarse["tmz"] / 2.0
    assert fine["tez"] < coarse["tez"] / 2.0
    assert fine["missed"] > fine["tmz"]


def _errors(points: int) -> dict[str, float]:
    return {
        "tmz": _periodic(Polarization.TMZ, points, "dispersion", "dispersive"),
        "tez": _periodic(Polarization.TEZ, points, "dispersion", "local"),
        "missed": _periodic(Polarization.TMZ, points, "local", "dispersive"),
    }


def _periodic(pol: Polarization, points: int, sheet_kind: str, target: str) -> float:
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
    if sheet_kind == "local":
        sheet = SymmetricSheet(y_sheet, _CHI_EE0, _CHI_MM)
    else:
        sheet = SymmetricSheet.electric_dispersion(
            y_sheet, _CHI_EE0, _CHI_EE2, k, chi_mm=_CHI_MM
        )
    chi_ee = (
        dispersive_susceptibility(k * np.sin(theta), _CHI_EE0, _CHI_EE2)
        if target == "dispersive"
        else _CHI_EE0
    )
    operator = FDFDOperator(grid, _bare(grid), sheet=sheet)
    reflected, transmitted = sheet_coefficients(
        chi_ee, _CHI_MM, k, theta=theta, te=pol is Polarization.TEZ
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
