"""3D convolutional PML: grading, stretched curls, and the eigenproblem refusal."""

import numpy as np

from maxwell_fd.drivers.fdfd3d import FDFDOperator3D
from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.materials.volume3d import UniformIsotropic3D
from maxwell_fd.operators.curl3d import (
    CurlScale3,
    array_curl_e,
    array_curl_h,
    build_curls,
)
from maxwell_fd.operators.pml import PMLProfile3, PMLSpec3
from maxwell_fd.utils.constants import EPS0


def test_profile_matches_the_stretch_and_rejects_a_bad_grid() -> None:
    grid = YeeGrid3D(10, 8, 7, 0.2, 0.25, 0.3, Boundary.PEC)
    spec = PMLSpec3.box(2, m=3, r0=1e-6, kappa_max=1.4, alpha_max=0.02)
    profile = PMLProfile3(grid, spec)
    omega = 2.0 * np.pi * 1.0e9
    scale = profile.scale(omega)
    y = (np.arange(grid.ny) + 0.5) * grid.dy
    expected = 1.0 / (
        profile.y_mid.kappa
        + profile.y_mid.sigma / (profile.y_mid.alpha + 1j * omega * EPS0)
    )
    np.testing.assert_allclose(scale.y_on_hx[0, :, 0], expected)
    np.testing.assert_allclose(scale.y_on_hz[1, :, 2], expected)
    outside = (y > spec.y_lo * grid.dy) & (y < grid.b - spec.y_hi * grid.dy)
    np.testing.assert_allclose(expected[outside], 1.0)
    z_expected = 1.0 / (
        profile.z_node.kappa
        + profile.z_node.sigma / (profile.z_node.alpha + 1j * omega * EPS0)
    )
    np.testing.assert_allclose(scale.z_on_ex[0, 1, :], z_expected)
    periodic = YeeGrid3D(8, 8, 8, 0.2, 0.2, 0.2, Boundary.PERIODIC)
    try:
        PMLProfile3(periodic, PMLSpec3.box(2))
    except ValueError as exc:
        assert "PEC" in str(exc)
    else:
        raise AssertionError("periodic PML was accepted")
    try:
        PMLProfile3(grid, PMLSpec3(z_lo=4, z_hi=3))
    except ValueError as exc:
        assert "no interior" in str(exc)
    else:
        raise AssertionError("overlapping PML was accepted")


def test_stretched_matrix_matches_the_array_curl() -> None:
    rng = np.random.default_rng(7)
    omega = 1.7e8
    grid = YeeGrid3D(6, 5, 4, 0.3, 0.25, 0.2, Boundary.PEC)
    spec = PMLSpec3(
        x_lo=1,
        x_hi=2,
        y_lo=1,
        y_hi=1,
        z_lo=1,
        z_hi=1,
        m=3,
        r0=1e-5,
        kappa_max=2.0,
        alpha_max=0.04,
    )
    scale = PMLProfile3(grid, spec).scale(omega)
    operators = build_curls(grid, scale=scale)
    electric = {dof.name: rng.normal(size=dof.shape) for dof in operators.layout.e}
    magnetic = {dof.name: rng.normal(size=dof.shape) for dof in operators.layout.h}
    _zero_tangential(electric)
    got_e = operators.curl_e @ operators.layout.pack_e(electric)
    ref_e = operators.layout.pack_h(array_curl_e(grid, electric, scale=scale))
    got_h = operators.curl_h @ operators.layout.pack_h(magnetic)
    ref_h = operators.layout.pack_e(array_curl_h(grid, magnetic, scale=scale))
    np.testing.assert_allclose(got_e, ref_e, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_h, ref_h, rtol=1e-12, atol=1e-12)
    ones = _ones(grid)
    scaled = build_curls(grid, scale=ones)
    plain = build_curls(grid)
    np.testing.assert_allclose(
        scaled.curl_e.toarray(), plain.curl_e.toarray(), atol=1e-12
    )
    np.testing.assert_allclose(
        scaled.curl_h.toarray(), plain.curl_h.toarray(), atol=1e-12
    )


def test_spatial_operator_rejects_an_active_pml() -> None:
    grid = YeeGrid3D(6, 5, 4, 0.2, 0.2, 0.2, Boundary.PEC)
    operator = FDFDOperator3D(grid, UniformIsotropic3D().sample(grid), PMLSpec3.box(1))
    try:
        operator.spatial_operator()
    except ValueError as exc:
        assert "unstretched" in str(exc)
    else:
        raise AssertionError("eigenproblem accepted a PML")
    bare = FDFDOperator3D(grid, UniformIsotropic3D().sample(grid), PMLSpec3())
    stiffness, _mass = bare.spatial_operator()
    reference, _mass_ref = FDFDOperator3D(
        grid, UniformIsotropic3D().sample(grid)
    ).spatial_operator()
    np.testing.assert_allclose(stiffness.toarray(), reference.toarray(), atol=1e-12)


def _ones(grid: YeeGrid3D) -> CurlScale3:
    shapes = grid.shapes()
    return CurlScale3(
        y_on_hx=np.ones(shapes["hx"]),
        z_on_hx=np.ones(shapes["hx"]),
        z_on_hy=np.ones(shapes["hy"]),
        x_on_hy=np.ones(shapes["hy"]),
        x_on_hz=np.ones(shapes["hz"]),
        y_on_hz=np.ones(shapes["hz"]),
        y_on_ex=np.ones(shapes["ex"]),
        z_on_ex=np.ones(shapes["ex"]),
        z_on_ey=np.ones(shapes["ey"]),
        x_on_ey=np.ones(shapes["ey"]),
        x_on_ez=np.ones(shapes["ez"]),
        y_on_ez=np.ones(shapes["ez"]),
    )


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
