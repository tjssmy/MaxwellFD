"""TMz near-to-far integral against a line current."""

import numpy as np

from maxwell_fd.analytics.farfield2d import (
    line_current_far_ez,
    tmz_far_ez,
    tmz_magnetic,
    tmz_rectangle,
)
from maxwell_fd.analytics.green2d import line_current_ez, line_current_h
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0, MU0

# 3λ square, Δ = 0.05, PML of 8 cells, contour half-width 0.8λ.
# Complex Ez at ρ = 20λ on 72 angles. Measured L2 is 2.619e-2.
# The same physical contour at Δ = 0.025, with a PML of 16 cells, is 9.52e-3.
FAR_BAR = 0.05
FAR_FINE_BAR = 0.02
_WAVELENGTH = 1.0
_SIDE = 3.0
_HALF = 0.8
_RHO = 20.0
_ANGLES = 72


def test_circle_reproduces_the_hankel_asymptotic() -> None:
    omega = 2.0 * np.pi * C0
    count = 480
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    radius = 0.7
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    ez = line_current_ez(x, y, omega=omega)
    hx, hy = line_current_h(x, y, omega=omega)
    phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    got = tmz_far_ez(
        x,
        y,
        ez,
        hx,
        hy,
        np.cos(angle),
        np.sin(angle),
        np.full(count, radius * 2.0 * np.pi / count),
        phi=phi,
        rho=_RHO,
        omega=omega,
    )
    exact = line_current_far_ez(phi, _RHO, omega=omega)
    error = np.linalg.norm(got - exact) / np.linalg.norm(exact)
    assert error < 1e-8, error
    point = (0.35, -0.2)
    step = 1.0e-5
    plus = line_current_ez(point[0], point[1] + step, omega=omega)
    minus = line_current_ez(point[0], point[1] - step, omega=omega)
    d_dy = (plus - minus) / (2.0 * step)
    plus = line_current_ez(point[0] + step, point[1], omega=omega)
    minus = line_current_ez(point[0] - step, point[1], omega=omega)
    d_dx = (plus - minus) / (2.0 * step)
    hx_point, hy_point = line_current_h(point[0], point[1], omega=omega)
    np.testing.assert_allclose(hx_point, -d_dy / (1j * omega * MU0), rtol=1e-6)
    np.testing.assert_allclose(hy_point, d_dx / (1j * omega * MU0), rtol=1e-6)
    singular = line_current_h(0.0, 0.0, omega=omega)
    assert np.isnan(singular[0]) and np.isnan(singular[1])
    try:
        line_current_far_ez(0.0, _RHO, omega=0.0)
    except ValueError as exc:
        assert "omega" in str(exc)
    else:
        raise AssertionError("a nonpositive frequency was accepted")
    try:
        tmz_far_ez(
            x,
            y,
            ez,
            hx,
            hy,
            np.cos(angle),
            np.sin(angle),
            np.ones(count),
            phi=phi,
            rho=0.0,
            omega=omega,
        )
    except ValueError as exc:
        assert "rho" in str(exc)
    else:
        raise AssertionError("a nonpositive distance was accepted")


def test_rectangle_rejects_a_wall_and_a_short_contour() -> None:
    grid = YeeGrid2D(8, 6, 0.2, 0.2, Polarization.TMZ, Boundary.PEC)
    ez = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    hx = np.zeros(grid.shapes()["hx"], dtype=np.complex128)
    hy = np.zeros(grid.shapes()["hy"], dtype=np.complex128)
    try:
        tmz_rectangle(grid, ez, hx, hy, 0, 4, 1, 4)
    except ValueError as exc:
        assert "grid" in str(exc)
    else:
        raise AssertionError("a contour on the wall was accepted")
    try:
        tmz_far_ez(
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            phi=np.array([0.0]),
            rho=1.0,
            omega=1.0,
        )
    except ValueError as exc:
        assert "contour" in str(exc)
    else:
        raise AssertionError("an empty contour was accepted")
    tez = YeeGrid2D(8, 6, 0.2, 0.2, Polarization.TEZ, Boundary.PEC)
    try:
        tmz_magnetic(tez, np.zeros(tez.shapes()["ex"]), 1.0)
    except ValueError as exc:
        assert "TMz" in str(exc)
    else:
        raise AssertionError("a TEz magnetic reconstruction was accepted")


def test_line_current_far_field_matches_the_hankel_wave() -> None:
    coarse = _far_error(0.05, 8)
    fine = _far_error(0.025, 16)
    assert coarse < FAR_BAR, coarse
    assert fine < FAR_FINE_BAR, fine
    assert fine < coarse / 2.0, (fine, coarse)


def _far_error(dx: float, pml_cells: int) -> float:
    n = int(round(_SIDE / dx))
    grid = YeeGrid2D(n, n, dx, dx, Polarization.TMZ, Boundary.PEC)
    omega = 2.0 * np.pi * C0 / _WAVELENGTH
    operator = FDFDOperator(
        grid, UniformIsotropic().sample_tm(grid), PMLSpec.box(pml_cells)
    )
    i0 = n // 2
    jz = np.zeros(grid.shapes()["ez"])
    jz[i0, i0] = 1.0 / (dx * dx)
    solved = operator.solve(omega, operator.layout.pack_e({"ez": jz}))
    ez = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    dof = operator.layout.e[0]
    ez[dof.i, dof.j] = solved
    hx, hy = tmz_magnetic(grid, ez, omega)
    half = int(round(_HALF / dx))
    contour = tmz_rectangle(
        grid, ez, hx, hy, i0 - half, i0 + half, i0 - half, i0 + half
    )
    x_nodes, y_nodes = grid.coordinates("ez")
    phi = np.linspace(0.0, 2.0 * np.pi, _ANGLES, endpoint=False)
    got = tmz_far_ez(
        contour["x"] - x_nodes[i0],
        contour["y"] - y_nodes[i0],
        contour["ez"],
        contour["hx"],
        contour["hy"],
        contour["nx"],
        contour["ny"],
        contour["dl"],
        phi=phi,
        rho=_RHO * _WAVELENGTH,
        omega=omega,
    )
    exact = line_current_far_ez(phi, _RHO * _WAVELENGTH, omega=omega)
    return float(np.linalg.norm(got - exact) / np.linalg.norm(exact))
