"""Hertzian dipole: the outgoing spherical wave, and one 3D FDFD check."""

import numpy as np

from maxwell_fd.analytics.dipole3d import hertzian_dipole_fields
from maxwell_fd.drivers.fdfd3d import FDFDOperator3D
from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.materials.volume3d import UniformIsotropic3D
from maxwell_fd.operators.pml import PMLSpec3
from maxwell_fd.utils.constants import C0, EPS0, MU0

# 16³ box, Δ = 0.1, λ = 1, three PML cells, moment Iℓ = 1.
# The electric comparison starts 0.25λ from the element and stays outside
# the PML. Measured L2 on that window is 9.652e-2.
DIPOLE_BAR = 0.16
_COMPONENTS = ("ex", "ey", "ez")


def test_near_field_is_the_static_dipole() -> None:
    # kr = 10^{-4}. The static dipole uses p = Iℓ / (jω).
    radius = 1.0
    omega = 1.0e-4 * C0
    moment = 1.0 + 0.25j
    axis = hertzian_dipole_fields(0.0, 0.0, radius, omega=omega, moment=moment)
    static_axis = moment / (2.0 * np.pi * 1j * omega * EPS0 * radius**3)
    np.testing.assert_allclose(axis["ez"], static_axis, rtol=1e-6)
    np.testing.assert_allclose(axis["ex"], 0.0, atol=1e-12)
    np.testing.assert_allclose(axis["hx"], 0.0, atol=1e-12)
    equator = hertzian_dipole_fields(radius, 0.0, 0.0, omega=omega, moment=moment)
    static_equator = 1j * moment / (4.0 * np.pi * omega * EPS0 * radius**3)
    np.testing.assert_allclose(equator["ez"], static_equator, rtol=1e-6)
    np.testing.assert_allclose(equator["ex"], 0.0, atol=1e-12)
    np.testing.assert_allclose(equator["ey"], 0.0, atol=1e-12)
    singular = hertzian_dipole_fields(0.0, 0.0, 0.0, omega=omega)
    assert np.all([np.isnan(singular[name]) for name in singular])


def test_dipole_is_outgoing_and_satisfies_maxwell() -> None:
    omega = 2.0 * np.pi * C0
    wavenumber = omega * np.sqrt(MU0 * EPS0)
    radii = np.array([2.0, 2.2])
    wave = hertzian_dipole_fields(radii, np.zeros(2), np.zeros(2), omega=omega)
    phase = np.angle(wave["hy"][1] / wave["hy"][0])
    advance = np.angle(np.exp(-1j * wavenumber * (radii[1] - radii[0])))
    np.testing.assert_allclose(phase, advance, atol=0.05)
    point = np.array([0.4, 0.1, 0.2])
    step = 1.0e-5
    gradient = np.zeros((3, 3), dtype=np.complex128)
    for axis in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[axis] += step
        minus[axis] -= step
        gradient[:, axis] = (_electric(plus, omega) - _electric(minus, omega)) / (
            2.0 * step
        )
    curl = np.array(
        [
            gradient[2, 1] - gradient[1, 2],
            gradient[0, 2] - gradient[2, 0],
            gradient[1, 0] - gradient[0, 1],
        ]
    )
    residual = curl + 1j * omega * MU0 * _magnetic(point, omega)
    assert np.linalg.norm(residual) < 1e-8 * np.linalg.norm(curl)
    try:
        hertzian_dipole_fields(1.0, 0.0, 0.0, omega=0.0)
    except ValueError as exc:
        assert "omega" in str(exc)
    else:
        raise AssertionError("a zero frequency was accepted")
    try:
        hertzian_dipole_fields(1.0, 0.0, 0.0, omega=omega, moment=np.nan)
    except ValueError as exc:
        assert "moment" in str(exc)
    else:
        raise AssertionError("a non-finite moment was accepted")


def test_hertzian_dipole_matches_the_spherical_wave() -> None:
    error, count = _dipole_error(16, 0.1, 3, wavelength=1.0, r_min=0.25)
    assert count > 1000, count
    assert error < DIPOLE_BAR, error


def _electric(point: np.ndarray, omega: float) -> np.ndarray:
    field = hertzian_dipole_fields(point[0], point[1], point[2], omega=omega)
    return np.array([field["ex"], field["ey"], field["ez"]], dtype=np.complex128)


def _magnetic(point: np.ndarray, omega: float) -> np.ndarray:
    field = hertzian_dipole_fields(point[0], point[1], point[2], omega=omega)
    return np.array([field["hx"], field["hy"], field["hz"]], dtype=np.complex128)


def _dipole_error(
    cells: int,
    dx: float,
    pml_cells: int,
    *,
    wavelength: float,
    r_min: float,
    moment: complex = 1.0,
) -> tuple[float, int]:
    omega = 2.0 * np.pi * C0 / wavelength
    grid = YeeGrid3D(cells, cells, cells, dx, dx, dx, Boundary.PEC)
    operator = FDFDOperator3D(
        grid, UniformIsotropic3D().sample(grid), PMLSpec3.box(pml_cells)
    )
    index = cells // 2
    current = {
        name: np.zeros(grid.shapes()[name], dtype=np.complex128) for name in _COMPONENTS
    }
    current["ez"][index, index, index] = moment / (dx * dx * dx)
    solved = operator.solve(omega, operator.layout.pack_e(current))
    x, y, z = grid.coordinates("ez")
    origin = (float(x[index]), float(y[index]), float(z[index]))
    numerical = []
    reference = []
    count = 0
    offset = 0
    band = pml_cells * dx
    length = cells * dx
    for dof, name in zip(operator.layout.e, _COMPONENTS, strict=True):
        width = dof.i.size
        field = dof.scatter(solved[offset : offset + width])
        offset += width
        cx, cy, cz = grid.coordinates(name)
        xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="ij")
        exact = hertzian_dipole_fields(
            xx, yy, zz, omega=omega, moment=moment, origin=origin
        )[name]
        radius = np.sqrt(
            (xx - origin[0]) ** 2 + (yy - origin[1]) ** 2 + (zz - origin[2]) ** 2
        )
        mask = (
            (xx >= band)
            & (xx <= length - band)
            & (yy >= band)
            & (yy <= length - band)
            & (zz >= band)
            & (zz <= length - band)
            & (radius >= r_min)
        )
        numerical.append(field[mask])
        reference.append(exact[mask])
        count += int(np.count_nonzero(mask))
    got = np.concatenate(numerical)
    exact = np.concatenate(reference)
    return float(np.linalg.norm(got - exact) / np.linalg.norm(exact)), count
