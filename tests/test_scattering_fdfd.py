"""Fresnel slab and dielectric cylinders.

The closed box prescribes a Dirichlet trace on the PEC wall. The open
cylinder drives the scattered field with a contrast current and absorbs
that field in the convolutional PML.
"""

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
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0, EPS0, MU0

# 64 cells, Δ = 1, a = 8, eight PML cells, ka = 1, n = 1.5.
# Measured TMz 0.00449 and TEz 0.00709 on that grid.
OPEN_DIELECTRIC_TMZ_BAR = 8.0e-3
OPEN_DIELECTRIC_TEZ_BAR = 1.3e-2


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


def test_open_dielectric_cylinder_matches_mie() -> None:
    tmz = _open_dielectric_error(Polarization.TMZ)
    tez = _open_dielectric_error(Polarization.TEZ)
    assert tmz < OPEN_DIELECTRIC_TMZ_BAR, tmz
    assert tez < OPEN_DIELECTRIC_TEZ_BAR, tez


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


def _open_dielectric_error(pol: Polarization) -> float:
    """Scattered field on the PML, driven by ``J = jω(ε−ε0) E_inc``."""
    cells, dx, radius, pml_cells, ka, eps_r = 64, 1.0, 8.0, 8, 1.0, 2.25
    length = cells * dx
    center = (0.5 * length, 0.5 * length)
    omega = (ka / radius) * C0
    k = float(omega * np.sqrt(MU0 * EPS0))
    grid = YeeGrid2D(cells, cells, dx, dx, pol, Boundary.PEC)
    law = StaircaseIsotropic(
        UniformIsotropic(),
        (Disk(center[0], center[1], radius, eps=eps_r * EPS0),),
    )
    band = pml_cells * dx
    if pol is Polarization.TMZ:
        components = law.sample_tm(grid)
        operator = FDFDOperator(grid, components, PMLSpec.box(pml_cells))
        x, y = grid.coordinates("ez")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        incident = np.exp(-1j * k * (xx - center[0]))
        current = operator.layout.pack_e(
            {"ez": 1j * omega * (components.eps_z - EPS0) * incident}
        )
        solved = operator.solve(omega, current)
        total = incident.copy()
        dof = operator.layout.e[0]
        total[dof.i, dof.j] = solved + incident[dof.i, dof.j]
        exact = cylinder_ez(xx, yy, k=k, radius=radius, eps_r=eps_r, center=center)
        mask = _open_mask(xx, yy, length, band, center, radius, dx)
        assert int(np.count_nonzero(mask)) > 100
        return _relative(total[mask], exact[mask])
    components = law.sample_te(grid)
    operator = FDFDOperator(grid, components, PMLSpec.box(pml_cells))
    ex_x, ex_y = grid.coordinates("ex")
    ey_x, ey_y = grid.coordinates("ey")
    ex_xx, ex_yy = np.meshgrid(ex_x, ex_y, indexing="ij")
    ey_xx, ey_yy = np.meshgrid(ey_x, ey_y, indexing="ij")
    ex_inc = np.zeros(grid.shapes()["ex"], dtype=np.complex128)
    ey_inc = (k / (omega * EPS0)) * np.exp(-1j * k * (ey_xx - center[0]))
    current = operator.layout.pack_e(
        {
            "ex": 1j * omega * (components.eps_x - EPS0) * ex_inc,
            "ey": 1j * omega * (components.eps_y - EPS0) * ey_inc,
        }
    )
    solved = operator.solve(omega, current)
    n_ex = operator.layout.e[0].i.size
    ex = ex_inc.copy()
    ey = ey_inc.copy()
    ex[operator.layout.e[0].i, operator.layout.e[0].j] = (
        solved[:n_ex] + ex_inc[operator.layout.e[0].i, operator.layout.e[0].j]
    )
    ey[operator.layout.e[1].i, operator.layout.e[1].j] = (
        solved[n_ex:] + ey_inc[operator.layout.e[1].i, operator.layout.e[1].j]
    )
    exact_ex, _exact_ey = cylinder_te_electric(
        ex_xx,
        ex_yy,
        k=k,
        radius=radius,
        eps_r=eps_r,
        omega=omega,
        eps0=EPS0,
        center=center,
    )
    _exact_ex, exact_ey = cylinder_te_electric(
        ey_xx,
        ey_yy,
        k=k,
        radius=radius,
        eps_r=eps_r,
        omega=omega,
        eps0=EPS0,
        center=center,
    )
    mask_ex = _open_mask(ex_xx, ex_yy, length, band, center, radius, dx)
    mask_ey = _open_mask(ey_xx, ey_yy, length, band, center, radius, dx)
    assert int(np.count_nonzero(mask_ex)) > 100
    assert int(np.count_nonzero(mask_ey)) > 100
    return _relative(
        np.concatenate((ex[mask_ex], ey[mask_ey])),
        np.concatenate((exact_ex[mask_ex], exact_ey[mask_ey])),
    )


def _open_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    length: float,
    band: float,
    center: tuple[float, float],
    radius: float,
    dx: float,
) -> np.ndarray:
    rho = np.hypot(xx - center[0], yy - center[1])
    return (
        (xx >= band)
        & (xx <= length - band)
        & (yy >= band)
        & (yy <= length - band)
        & (np.abs(rho - radius) >= 2.0 * dx)
    )
