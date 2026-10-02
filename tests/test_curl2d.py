"""Exact differences and Faraday/Ampere signs for the 2D curls."""

import numpy as np

from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.operators.curl2d import array_curl_e, array_curl_h, build_curls


def _grid(pol: Polarization, boundary: Boundary, nx: int = 8, ny: int = 6) -> YeeGrid2D:
    return YeeGrid2D(nx, ny, 0.2, 0.25, pol, boundary)


def test_linear_tm_curl_is_exact() -> None:
    grid = _grid(Polarization.TMZ, Boundary.PEC)
    x, y = grid.coordinates("ez")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    alpha, beta = 0.3, -0.7
    curls = array_curl_e(grid, {"ez": alpha * xx + beta * yy})
    np.testing.assert_allclose(curls["hx"], beta, atol=1e-12)
    np.testing.assert_allclose(curls["hy"], -alpha, atol=1e-12)


def test_linear_te_curl_is_exact() -> None:
    grid = _grid(Polarization.TEZ, Boundary.PEC)
    x_ex, y_ex = grid.coordinates("ex")
    x_ey, y_ey = grid.coordinates("ey")
    _, ex_y = np.meshgrid(x_ex, y_ex, indexing="ij")
    ey_x, _ = np.meshgrid(x_ey, y_ey, indexing="ij")
    alpha, beta = -1.1, 0.4
    curls = array_curl_e(grid, {"ex": beta * ex_y, "ey": alpha * ey_x})
    np.testing.assert_allclose(curls["hz"], alpha - beta, atol=1e-12)


def test_curl_h_is_transpose_of_curl_e() -> None:
    cases = (
        (Polarization.TMZ, Boundary.PEC),
        (Polarization.TMZ, Boundary.PERIODIC),
        (Polarization.TMZ, Boundary.PERIODIC_X),
        (Polarization.TEZ, Boundary.PEC),
        (Polarization.TEZ, Boundary.PERIODIC),
        (Polarization.TEZ, Boundary.PERIODIC_X),
    )
    for pol, boundary in cases:
        ops = build_curls(_grid(pol, boundary))
        np.testing.assert_allclose(
            ops.curl_h.toarray(),
            ops.curl_e.toarray().T,
            atol=1e-12,
        )


def test_matrix_matches_array_curl() -> None:
    rng = np.random.default_rng(2)
    cases = (
        (Polarization.TMZ, Boundary.PEC),
        (Polarization.TMZ, Boundary.PERIODIC),
        (Polarization.TMZ, Boundary.PERIODIC_X),
        (Polarization.TEZ, Boundary.PEC),
        (Polarization.TEZ, Boundary.PERIODIC),
        (Polarization.TEZ, Boundary.PERIODIC_X),
    )
    for pol, boundary in cases:
        grid = _grid(pol, boundary)
        ops = build_curls(grid)
        electric = {dof.name: rng.normal(size=dof.shape) for dof in ops.layout.e}
        magnetic = {dof.name: rng.normal(size=dof.shape) for dof in ops.layout.h}
        if boundary is Boundary.PEC and pol is Polarization.TMZ:
            electric["ez"][0, :] = 0.0
            electric["ez"][-1, :] = 0.0
            electric["ez"][:, 0] = 0.0
            electric["ez"][:, -1] = 0.0
        if boundary is Boundary.PEC and pol is Polarization.TEZ:
            electric["ex"][:, 0] = 0.0
            electric["ex"][:, -1] = 0.0
            electric["ey"][0, :] = 0.0
            electric["ey"][-1, :] = 0.0
        if boundary is Boundary.PERIODIC_X and pol is Polarization.TMZ:
            electric["ez"][:, 0] = 0.0
            electric["ez"][:, -1] = 0.0
        if boundary is Boundary.PERIODIC_X and pol is Polarization.TEZ:
            electric["ex"][:, 0] = 0.0
            electric["ex"][:, -1] = 0.0
        got_e = ops.curl_e @ ops.layout.pack_e(electric)
        ref_e = ops.layout.pack_h(array_curl_e(grid, electric))
        got_h = ops.curl_h @ ops.layout.pack_h(magnetic)
        ref_h = ops.layout.pack_e(array_curl_h(grid, magnetic))
        np.testing.assert_allclose(got_e, ref_e, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got_h, ref_h, rtol=1e-12, atol=1e-12)
