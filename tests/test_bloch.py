"""Bloch phase on the periodic x wrap."""

import numpy as np

from maxwell_fd.analytics.sheets import oblique_sheet_fields, sheet_coefficients
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.drivers.fdtd2d import step_tmz, zeros_tmz
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.sheets import SymmetricSheet
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.curl2d import array_curl_e, array_curl_h, build_curls
from maxwell_fd.utils.constants import C0

# Width 0.8λ at 30°, so k_Bx a = 0.8π and the phase is not one.
# χ_ee=0.5, χ_mm=0.25, χ_mm^nn=0.4. Measured L2 at 10 cells per
# wavelength is 0.1498 (TMz) and 0.1212 (TEz). Four times that
# resolution is 0.0535 and 0.0559. Phase one on the same cell is about 1.3.
BLOCH_TMZ_BAR = 0.22
BLOCH_TEZ_BAR = 0.18
_WIDTH = 0.8
_WAVELENGTH = 1.0
_HEIGHT = 4.0


def test_bloch_wrap_matches_an_explicit_image() -> None:
    k_b = 1.7
    grid = YeeGrid2D(
        8, 6, 0.2, 0.25, Polarization.TMZ, Boundary.PERIODIC_X, bloch_x=k_b
    )
    phase = np.exp(-1j * k_b * grid.a)
    rng = np.random.default_rng(3)
    ez = rng.normal(size=grid.shapes()["ez"]) + 1j * rng.normal(
        size=grid.shapes()["ez"]
    )
    ez[:, 0] = 0.0
    ez[:, -1] = 0.0
    extended = np.concatenate([ez, ez[:1] * phase], axis=0)
    manual = -(extended[1:] - extended[:-1]) / grid.dx
    np.testing.assert_allclose(array_curl_e(grid, {"ez": ez})["hy"], manual, atol=1e-12)
    hy = rng.normal(size=grid.shapes()["hy"]) + 1j * rng.normal(
        size=grid.shapes()["hy"]
    )
    hx = np.zeros(grid.shapes()["hx"], dtype=np.complex128)
    left = np.concatenate([hy[-1:] * np.conjugate(phase), hy[:-1]], axis=0)
    manual_h = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    manual_h[:, 1:-1] = (hy[:, 1:-1] - left[:, 1:-1]) / grid.dx
    got_h = array_curl_h(grid, {"hx": hx, "hy": hy})["ez"]
    np.testing.assert_allclose(got_h[:, 1:-1], manual_h[:, 1:-1], atol=1e-12)
    tez = YeeGrid2D(8, 6, 0.2, 0.25, Polarization.TEZ, Boundary.PERIODIC_X, bloch_x=k_b)
    ey = rng.normal(size=tez.shapes()["ey"]) + 1j * rng.normal(size=tez.shapes()["ey"])
    ex = np.zeros(tez.shapes()["ex"], dtype=np.complex128)
    extended_ey = np.concatenate([ey, ey[:1] * phase], axis=0)
    manual_hz = (extended_ey[1:] - extended_ey[:-1]) / tez.dx
    np.testing.assert_allclose(
        array_curl_e(tez, {"ex": ex, "ey": ey})["hz"], manual_hz, atol=1e-12
    )
    for pol, boundary in (
        (Polarization.TMZ, Boundary.PERIODIC_X),
        (Polarization.TMZ, Boundary.PERIODIC),
        (Polarization.TEZ, Boundary.PERIODIC_X),
        (Polarization.TEZ, Boundary.PERIODIC),
    ):
        sample = YeeGrid2D(7, 5, 0.2, 0.25, pol, boundary, bloch_x=k_b)
        ops = build_curls(sample)
        np.testing.assert_allclose(
            ops.curl_h.toarray(),
            ops.curl_e.toarray().conj().T,
            atol=1e-12,
        )
        _fields_match_arrays(sample, ops)


def test_bloch_rejects_a_wall_and_a_real_step() -> None:
    try:
        YeeGrid2D(4, 4, 0.2, 0.2, Polarization.TMZ, Boundary.PEC, bloch_x=1.0)
    except ValueError as exc:
        assert "periodic" in str(exc)
    else:
        raise AssertionError("a PEC wall accepted a Bloch phase")
    try:
        YeeGrid2D(4, 4, 0.2, 0.2, Polarization.TMZ, Boundary.PERIODIC_X, bloch_x=True)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("a boolean wave number was accepted")
    grid = YeeGrid2D(4, 4, 0.2, 0.2, Polarization.TMZ, Boundary.PERIODIC_X, bloch_x=1.0)
    try:
        step_tmz(grid, UniformIsotropic().sample_tm(grid), zeros_tmz(grid), 1e-12)
    except ValueError as exc:
        assert "FDFD" in str(exc)
    else:
        raise AssertionError("the leapfrog accepted a Bloch phase")
    operator = FDFDOperator(grid, UniformIsotropic().sample_tm(grid))
    try:
        operator.spatial_operator()
    except ValueError as exc:
        assert "Bloch" in str(exc)
    else:
        raise AssertionError("the real eigenproblem accepted a Bloch phase")


def test_oblique_sheet_closes_with_the_bloch_phase() -> None:
    coarse = _sheet_errors(10)
    fine = _sheet_errors(40)
    assert coarse[Polarization.TMZ] < BLOCH_TMZ_BAR, coarse[Polarization.TMZ]
    assert coarse[Polarization.TEZ] < BLOCH_TEZ_BAR, coarse[Polarization.TEZ]
    assert fine[Polarization.TMZ] < coarse[Polarization.TMZ] / 2.0
    assert fine[Polarization.TEZ] < coarse[Polarization.TEZ] / 2.0
    plain = _sheet_errors(10, phased=False)
    assert plain[Polarization.TMZ] > 0.5, plain[Polarization.TMZ]
    assert plain[Polarization.TEZ] > 0.5, plain[Polarization.TEZ]
    _orders_are_specular(10)


def _fields_match_arrays(grid: YeeGrid2D, ops) -> None:
    rng = np.random.default_rng(4)
    electric = {
        dof.name: rng.normal(size=dof.shape) + 1j * rng.normal(size=dof.shape)
        for dof in ops.layout.e
    }
    magnetic = {
        dof.name: rng.normal(size=dof.shape) + 1j * rng.normal(size=dof.shape)
        for dof in ops.layout.h
    }
    if grid.boundary is Boundary.PERIODIC_X and grid.polarization is Polarization.TMZ:
        electric["ez"][:, 0] = 0.0
        electric["ez"][:, -1] = 0.0
    if grid.boundary is Boundary.PERIODIC_X and grid.polarization is Polarization.TEZ:
        electric["ex"][:, 0] = 0.0
        electric["ex"][:, -1] = 0.0
    np.testing.assert_allclose(
        ops.curl_e @ ops.layout.pack_e(electric),
        ops.layout.pack_h(array_curl_e(grid, electric)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ops.curl_h @ ops.layout.pack_h(magnetic),
        ops.layout.pack_e(array_curl_h(grid, magnetic)),
        rtol=1e-12,
        atol=1e-12,
    )


def _sheet_errors(points: int, *, phased: bool = True) -> dict[Polarization, float]:
    return {
        pol: _sheet_error(pol, points, phased=phased)
        for pol in (Polarization.TMZ, Polarization.TEZ)
    }


def _sheet_error(pol: Polarization, points: int, *, phased: bool) -> float:
    grid, operator, omega, reflected, transmitted, theta, y_sheet = _cell(
        pol, points, phased
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


def _orders_are_specular(points: int) -> None:
    for pol, name in ((Polarization.TMZ, "ez"), (Polarization.TEZ, "ex")):
        grid, operator, omega, reflected, transmitted, theta, y_sheet = _cell(
            pol, points, True
        )
        x, y = grid.coordinates(name)
        exact = oblique_sheet_fields(
            x,
            y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
            theta=theta,
        )[name]
        if pol is Polarization.TMZ:
            solved = operator.solve(
                omega, np.zeros(operator.layout.n_e), dirichlet={"ez": exact}
            )
            field = exact.copy()
            dof = operator.layout.e[0]
            field[dof.i, dof.j] = solved
        else:
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
                dirichlet={"ex": exact, "ey": exact_ey},
            )
            field = exact.copy()
            dof = operator.layout.e[0]
            field[dof.i, dof.j] = solved[: dof.i.size]
        line = field[:, grid.ny // 4]
        k_x = (omega / C0) * float(np.sin(theta))
        specular = abs(np.mean(line * np.exp(1j * k_x * x)))
        for order in (-1, 1):
            harmonic = k_x + 2.0 * np.pi * order / grid.a
            other = abs(np.mean(line * np.exp(1j * harmonic * x)))
            assert other < 1e-8 * specular, (pol.value, order, other, specular)


def _cell(pol: Polarization, points: int, phased: bool):
    theta = np.deg2rad(30.0)
    dx = _WAVELENGTH / points
    nx = int(round(_WIDTH / dx))
    ny = int(round(_HEIGHT / dx))
    y_sheet = (ny // 2 - 0.5) * dx
    omega = 2.0 * np.pi * C0 / _WAVELENGTH
    k_b = (omega / C0) * float(np.sin(theta)) if phased else 0.0
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PERIODIC_X, bloch_x=k_b)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    operator = FDFDOperator(
        grid,
        bare,
        sheet=SymmetricSheet(y_sheet, 0.5, 0.25, chi_mm_nn=0.4),
    )
    reflected, transmitted = sheet_coefficients(
        0.5,
        0.25,
        omega / C0,
        theta=theta,
        te=pol is Polarization.TEZ,
        chi_mm_nn=0.4,
    )
    return grid, operator, omega, reflected, transmitted, theta, y_sheet
