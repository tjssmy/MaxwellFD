"""Open dielectric sphere: contrast current, PML, and the 3D Mie series."""

import numpy as np

from maxwell_fd.analytics.mie3d import sphere_fields
from maxwell_fd.drivers.fdfd3d import FDFDOperator3D
from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.materials.volume3d import Ball, StaircaseIsotropic3D, UniformIsotropic3D
from maxwell_fd.operators.pml import PMLSpec3
from maxwell_fd.utils.constants import C0, EPS0, MU0

# 18 cells, Δ = 1, a = 4, three PML cells, ka = 1, n = 1.5.
# Measured electric L2 0.00977 on that grid.
OPEN_SPHERE_BAR = 1.8e-2
_COMPONENTS = ("ex", "ey", "ez")


def test_ball_is_closed_and_a_later_region_wins() -> None:
    ball = Ball(1.0, 2.0, 3.0, 0.5, eps=2.0 * EPS0)
    on = ball.contains(np.array([1.5]), np.array([2.0]), np.array([3.0]))
    off = ball.contains(np.array([1.5 + 1e-9]), np.array([2.0]), np.array([3.0]))
    assert bool(on[0])
    assert not bool(off[0])
    try:
        Ball(0.0, 0.0, 0.0, 0.0, eps=EPS0)
    except ValueError as exc:
        assert "radius" in str(exc)
    else:
        raise AssertionError("a zero radius was accepted")
    grid = YeeGrid3D(6, 6, 6, 1.0, 1.0, 1.0, Boundary.PEC)
    first = Ball(3.0, 3.0, 3.0, 1.2, eps=2.0 * EPS0)
    second = Ball(3.0, 3.0, 3.0, 1.2, eps=4.0 * EPS0, mu=2.0 * MU0, sigma=0.1)
    components = StaircaseIsotropic3D(UniformIsotropic3D(), (first, second)).sample(
        grid
    )
    x, y, z = grid.coordinates("ex")
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    inside = first.contains(xx, yy, zz)
    assert int(np.count_nonzero(inside)) > 0
    assert np.all(components.eps_x[inside] == 4.0 * EPS0)
    assert np.all(components.sigma_x[inside] == 0.1)
    assert np.all(components.eps_x[~inside] == EPS0)
    hx, hy, hz = grid.coordinates("hx")
    hh, yy, zz = np.meshgrid(hx, hy, hz, indexing="ij")
    inside_h = first.contains(hh, yy, zz)
    assert np.all(components.mu_x[inside_h] == 2.0 * MU0)
    assert np.all(components.mu_x[~inside_h] == MU0)


def test_transparent_contrast_current_vanishes() -> None:
    solved = _scattered(8, 1.5, 2, index=1.0)
    np.testing.assert_allclose(solved, 0.0, atol=1e-10)


def test_open_dielectric_sphere_matches_mie() -> None:
    error, counts = _sphere_error(18, 4.0, 3, index=1.5)
    assert min(counts) > 50, counts
    assert error < OPEN_SPHERE_BAR, error


def _scattered(
    cells: int, radius: float, pml_cells: int, *, index: float
) -> np.ndarray:
    dx = 1.0
    length = cells * dx
    center = (0.5 * length, 0.5 * length, 0.5 * length)
    omega = (1.0 / radius) * C0
    k = float(omega * np.sqrt(MU0 * EPS0))
    grid = YeeGrid3D(cells, cells, cells, dx, dx, dx, Boundary.PEC)
    components = StaircaseIsotropic3D(
        UniformIsotropic3D(),
        (Ball(*center, radius, eps=index**2 * EPS0),),
    ).sample(grid)
    operator = FDFDOperator3D(grid, components, PMLSpec3.box(pml_cells))
    eps = {
        "ex": components.eps_x,
        "ey": components.eps_y,
        "ez": components.eps_z,
    }
    current = operator.layout.pack_e(
        {
            name: 1j * omega * (eps[name] - EPS0) * _incident(grid, name, center, k)
            for name in _COMPONENTS
        }
    )
    return operator.solve(omega, current)


def _sphere_error(
    cells: int, radius: float, pml_cells: int, *, index: float
) -> tuple[float, tuple[int, int, int]]:
    dx = 1.0
    length = cells * dx
    center = (0.5 * length, 0.5 * length, 0.5 * length)
    omega = (1.0 / radius) * C0
    k = float(omega * np.sqrt(MU0 * EPS0))
    grid = YeeGrid3D(cells, cells, cells, dx, dx, dx, Boundary.PEC)
    components = StaircaseIsotropic3D(
        UniformIsotropic3D(),
        (Ball(*center, radius, eps=index**2 * EPS0),),
    ).sample(grid)
    operator = FDFDOperator3D(grid, components, PMLSpec3.box(pml_cells))
    eps = {
        "ex": components.eps_x,
        "ey": components.eps_y,
        "ez": components.eps_z,
    }
    incident = {name: _incident(grid, name, center, k) for name in _COMPONENTS}
    solved = operator.solve(
        omega,
        operator.layout.pack_e(
            {
                name: 1j * omega * (eps[name] - EPS0) * incident[name]
                for name in _COMPONENTS
            }
        ),
    )
    numerical = []
    exact = []
    counts = []
    offset = 0
    for dof, name in zip(operator.layout.e, _COMPONENTS, strict=True):
        total = incident[name].copy()
        count = dof.i.size
        total[dof.i, dof.j, dof.k] = (
            solved[offset : offset + count] + incident[name][dof.i, dof.j, dof.k]
        )
        offset += count
        x, y, z = grid.coordinates(name)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        reference = sphere_fields(
            xx, yy, zz, k=k, radius=radius, index=index, center=center
        )[0][{"ex": 0, "ey": 1, "ez": 2}[name]]
        observe = _mask(xx, yy, zz, grid, pml_cells * dx, center, radius, dx)
        numerical.append(total[observe])
        exact.append(reference[observe])
        counts.append(int(np.count_nonzero(observe)))
    got = np.concatenate(numerical)
    ref = np.concatenate(exact)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref)), (
        counts[0],
        counts[1],
        counts[2],
    )


def _incident(
    grid: YeeGrid3D, component: str, center: tuple[float, float, float], k: float
) -> np.ndarray:
    _x, _y, z = grid.coordinates(component)
    phase = np.exp(-1j * k * (z - center[2]))
    field = np.zeros(grid.shapes()[component], dtype=np.complex128)
    if component == "ex":
        field[...] = phase
    return field


def _mask(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    grid: YeeGrid3D,
    band: float,
    center: tuple[float, float, float],
    radius: float,
    spacing: float,
) -> np.ndarray:
    rho = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
    return (
        (xx >= band)
        & (xx <= grid.a - band)
        & (yy >= band)
        & (yy <= grid.b - band)
        & (zz >= band)
        & (zz <= grid.c - band)
        & (np.abs(rho - radius) >= 2.0 * spacing)
    )
