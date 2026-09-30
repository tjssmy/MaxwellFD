"""Exact differences, adjoint, and array agreement for the 3D curls."""

import numpy as np

from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.operators.curl3d import array_curl_e, array_curl_h, build_curls


def _grid(boundary: Boundary, nx: int = 6, ny: int = 5, nz: int = 4) -> YeeGrid3D:
    return YeeGrid3D(nx, ny, nz, 0.2, 0.25, 0.3, boundary)


def test_pec_shapes_follow_the_stagger() -> None:
    grid = YeeGrid3D(4, 5, 6, 0.2, 0.25, 0.3, Boundary.PEC)
    assert grid.shapes() == {
        "ex": (4, 6, 7),
        "ey": (5, 5, 7),
        "ez": (5, 6, 6),
        "hx": (5, 5, 6),
        "hy": (4, 6, 6),
        "hz": (4, 5, 7),
    }
    x, y, z = grid.coordinates("ex")
    assert x[0] == 0.5 * grid.dx
    assert y[0] == 0.0
    assert z[0] == 0.0
    periodic = YeeGrid3D(4, 5, 6, 0.2, 0.25, 0.3, Boundary.PERIODIC)
    assert set(periodic.shapes().values()) == {(4, 5, 6)}


def test_pec_grid_needs_two_cells() -> None:
    try:
        YeeGrid3D(1, 4, 4, 1.0, 1.0, 1.0, Boundary.PEC)
    except ValueError as exc:
        assert "2" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_linear_curl_is_exact() -> None:
    grid = _grid(Boundary.PEC)
    _x, y, _z = grid.coordinates("ez")
    _xx, yy, _zz = np.meshgrid(_x, y, _z, indexing="ij")
    beta = -0.7
    zeros = {
        "ex": np.zeros(grid.shapes()["ex"]),
        "ey": np.zeros(grid.shapes()["ey"]),
    }
    curls = array_curl_e(grid, {**zeros, "ez": beta * yy})
    np.testing.assert_allclose(curls["hx"], beta, atol=1e-12)
    np.testing.assert_allclose(curls["hy"], 0.0, atol=1e-12)
    np.testing.assert_allclose(curls["hz"], 0.0, atol=1e-12)

    x, _y, z = grid.coordinates("ey")
    _, _, zz = np.meshgrid(x, _y, z, indexing="ij")
    alpha = 0.4
    curls = array_curl_e(
        grid,
        {
            "ex": np.zeros(grid.shapes()["ex"]),
            "ey": alpha * zz,
            "ez": np.zeros(grid.shapes()["ez"]),
        },
    )
    # (∇×E)_x = −∂Ey/∂z, and Ey = α z is linear on the z nodes.
    np.testing.assert_allclose(curls["hx"], -alpha, atol=1e-12)
    np.testing.assert_allclose(curls["hy"], 0.0, atol=1e-12)
    np.testing.assert_allclose(curls["hz"], 0.0, atol=1e-12)


def test_curl_h_is_transpose_of_curl_e() -> None:
    for boundary in (Boundary.PEC, Boundary.PERIODIC):
        operators = build_curls(_grid(boundary))
        np.testing.assert_allclose(
            operators.curl_h.toarray(),
            operators.curl_e.toarray().T,
            atol=1e-12,
        )


def test_matrix_matches_array_curl() -> None:
    rng = np.random.default_rng(2)
    for boundary in (Boundary.PEC, Boundary.PERIODIC):
        grid = _grid(boundary)
        operators = build_curls(grid)
        electric = {dof.name: rng.normal(size=dof.shape) for dof in operators.layout.e}
        magnetic = {dof.name: rng.normal(size=dof.shape) for dof in operators.layout.h}
        if boundary is Boundary.PEC:
            _zero_tangential(electric)
        got_e = operators.curl_e @ operators.layout.pack_e(electric)
        ref_e = operators.layout.pack_h(array_curl_e(grid, electric))
        got_h = operators.curl_h @ operators.layout.pack_h(magnetic)
        ref_h = operators.layout.pack_e(array_curl_h(grid, magnetic))
        np.testing.assert_allclose(got_e, ref_e, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got_h, ref_h, rtol=1e-12, atol=1e-12)


def _zero_tangential(electric: dict[str, np.ndarray]) -> None:
    electric["ex"][:, 0, :] = 0.0
    electric["ex"][:, -1, :] = 0.0
    electric["ex"][:, :, 0] = 0.0
    electric["ex"][:, :, -1] = 0.0
    electric["ey"][0, :, :] = 0.0
    electric["ey"][-1, :, :] = 0.0
    electric["ey"][:, :, 0] = 0.0
    electric["ey"][:, :, -1] = 0.0
    electric["ez"][0, :, :] = 0.0
    electric["ez"][-1, :, :] = 0.0
    electric["ez"][:, 0, :] = 0.0
    electric["ez"][:, -1, :] = 0.0
