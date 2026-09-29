"""Fresnel slab and dielectric cylinder posed as a Dirichlet trace on a PEC box."""

import numpy as np

from maxwell_fd.analytics.fresnel import slab_fields
from maxwell_fd.analytics.mie2d import cylinder_ez, cylinder_te_electric
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import (
    Disk,
    SlabY,
    StaircaseIsotropic,
    UniformIsotropic,
)
from maxwell_fd.utils.constants import C0, EPS0


def test_fresnel_slab_converges_on_both_polarizations() -> None:
    # λ = 40 on the 8×80 box is a TEz eigenfrequency, so the trace problem is singular.
    # Once λ is detuned, Δ = 1 is still above the bar; this pair is the refinement.
    coarse = _fresnel_error(dx=0.5)
    fine = _fresnel_error(dx=0.25)
    for pol in coarse:
        assert coarse[pol] < 2.0e-2, f"{pol.value} coarse error {coarse[pol]}"
        assert (
            fine[pol] < coarse[pol] / 2.5
        ), f"{pol.value} fine {fine[pol]} vs {coarse[pol]}"


def test_dielectric_cylinder_matches_mie_off_the_staircase() -> None:
    for pol in (Polarization.TMZ, Polarization.TEZ):
        error = _cylinder_error(pol)
        assert error < 2.0e-2, f"{pol.value} relative L2 error {error}"


def _fresnel_error(dx: float) -> dict[Polarization, float]:
    length_x, length_y = 8.0, 80.0
    nx, ny = int(round(length_x / dx)), int(round(length_y / dx))
    # *.375 is off the electric nodes of both Δ = 1/2 and Δ = 1/4.
    y1, y2 = 30.375, 50.375
    omega = 2.0 * np.pi * C0 / 46.0
    eps_r = 4.0
    errors = {}
    for pol in (Polarization.TMZ, Polarization.TEZ):
        grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PEC)
        law = StaircaseIsotropic(UniformIsotropic(), (SlabY(y1, y2, eps=eps_r * EPS0),))
        components = (
            law.sample_tm(grid) if pol is Polarization.TMZ else law.sample_te(grid)
        )
        operator = FDFDOperator(grid, components)
        if pol is Polarization.TMZ:
            _x, y = grid.coordinates("ez")
            ez_line, _hx, _hz, _ex = slab_fields(
                y, y1=y1, y2=y2, omega=omega, eps_r=eps_r
            )
            ez = np.broadcast_to(ez_line, grid.shapes()["ez"]).copy()
            solved = operator.solve(
                omega, np.zeros(operator.layout.n_e), dirichlet={"ez": ez}
            )
            exact = operator.layout.pack_e({"ez": ez})
        else:
            _x, y = grid.coordinates("ex")
            _ez, _hx, _hz, ex_line = slab_fields(
                y, y1=y1, y2=y2, omega=omega, eps_r=eps_r
            )
            ex = np.broadcast_to(ex_line, grid.shapes()["ex"]).copy()
            ey = np.zeros(grid.shapes()["ey"], dtype=np.complex128)
            solved = operator.solve(
                omega, np.zeros(operator.layout.n_e), dirichlet={"ex": ex, "ey": ey}
            )
            exact = operator.layout.pack_e({"ex": ex, "ey": ey})
        errors[pol] = _relative(solved, exact)
    return errors


def _cylinder_error(pol: Polarization) -> float:
    nx = ny = 50
    dx = 1.0
    radius = 10.0
    center = (0.5 * nx * dx, 0.5 * ny * dx)
    eps_r = 2.25
    k = 1.0 / radius
    omega = k * C0
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PEC)
    law = StaircaseIsotropic(
        UniformIsotropic(),
        (Disk(center[0], center[1], radius, eps=eps_r * EPS0),),
    )
    if pol is Polarization.TMZ:
        components = law.sample_tm(grid)
        operator = FDFDOperator(grid, components)
        x, y = grid.coordinates("ez")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        exact = cylinder_ez(xx, yy, k=k, radius=radius, eps_r=eps_r, center=center)
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": exact}
        )
        reference = operator.layout.e[0].scatter(solved, fill=0.0)
        mask = _sample_mask(xx, yy, grid.a, grid.b, center, radius, dx)
        # scatter fills free samples and leaves the boundary at 0, which is outside the mask.
        return _relative(reference[mask], exact[mask])
    components = law.sample_te(grid)
    operator = FDFDOperator(grid, components)
    ex_x, ex_y = grid.coordinates("ex")
    ey_x, ey_y = grid.coordinates("ey")
    ex_xx, ex_yy = np.meshgrid(ex_x, ex_y, indexing="ij")
    ey_xx, ey_yy = np.meshgrid(ey_x, ey_y, indexing="ij")
    ex_exact, _ey_on_ex = cylinder_te_electric(
        ex_xx,
        ex_yy,
        k=k,
        radius=radius,
        eps_r=eps_r,
        omega=omega,
        eps0=EPS0,
        center=center,
    )
    _ex_on_ey, ey_exact = cylinder_te_electric(
        ey_xx,
        ey_yy,
        k=k,
        radius=radius,
        eps_r=eps_r,
        omega=omega,
        eps0=EPS0,
        center=center,
    )
    solved = operator.solve(
        omega,
        np.zeros(operator.layout.n_e),
        dirichlet={"ex": ex_exact, "ey": ey_exact},
    )
    ex_num = operator.layout.e[0].scatter(solved[: operator.layout.e[0].i.size])
    ey_num = operator.layout.e[1].scatter(solved[operator.layout.e[0].i.size :])
    ex_mask = _sample_mask(ex_xx, ex_yy, grid.a, grid.b, center, radius, dx)
    ey_mask = _sample_mask(ey_xx, ey_yy, grid.a, grid.b, center, radius, dx)
    return _relative(
        np.concatenate([ex_num[ex_mask], ey_num[ey_mask]]),
        np.concatenate([ex_exact[ex_mask], ey_exact[ey_mask]]),
    )


def _sample_mask(
    x: np.ndarray,
    y: np.ndarray,
    length_x: float,
    length_y: float,
    center: tuple[float, float],
    radius: float,
    dx: float,
) -> np.ndarray:
    rho = np.hypot(x - center[0], y - center[1])
    wall = np.minimum(np.minimum(x, length_x - x), np.minimum(y, length_y - y))
    return (np.abs(rho - radius) >= 2.0 * dx) & (wall >= 2.0 * dx)


def _relative(numerical: np.ndarray, exact: np.ndarray) -> float:
    scale = np.linalg.norm(exact)
    if scale == 0.0:
        raise AssertionError("exact field on the comparison set is zero")
    return float(np.linalg.norm(numerical - exact) / scale)
