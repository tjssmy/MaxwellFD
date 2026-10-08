"""Paraxial Gaussian beam, and one TMz total-field/scattered-field check."""

import numpy as np

from maxwell_fd.analytics.beam2d import gaussian_beam_tmz
from maxwell_fd.drivers.fdfd2d import FDFDOperator, tfsf_y_current
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0, ETA0

# 8 by 6 box, Δ = 0.05, λ = 1, w0 = 1.5, PML of 8 cells. The total-field
# Ez row is y = 1.5 and the waist is the box center. Measured electric L2
# on the forward window is 5.571e-2. The scattered-side level is 7.54e-4.
BEAM_BAR = 0.10
QUIET_BAR = 2.0e-3
_WAVELENGTH = 1.0
_WIDTH = 8.0
_HEIGHT = 6.0
_DX = 0.05
_PML = 8
_W0 = 1.5
_SPLIT = 1.5


def test_waist_is_the_gaussian_profile() -> None:
    k = 2.0 * np.pi
    amplitude = 0.4 - 0.2j
    waist = gaussian_beam_tmz(0.0, 0.0, k=k, w0=2.0, amplitude=amplitude)
    np.testing.assert_allclose(waist["ez"], amplitude, atol=1e-12)
    np.testing.assert_allclose(waist["hx"], amplitude / ETA0, rtol=1e-12)
    np.testing.assert_allclose(waist["hy"], 0.0, atol=1e-12)
    off = gaussian_beam_tmz(2.0, 0.0, k=k, w0=2.0, amplitude=amplitude)
    np.testing.assert_allclose(np.abs(off["ez"]), np.abs(amplitude) / np.e, rtol=1e-12)
    z_r = 0.5 * k * 4.0
    z_coord = 1.25
    expected = amplitude * np.sqrt((1j * z_r) / (z_coord + 1j * z_r))
    expected = expected * np.exp(-1j * k * z_coord)
    forward = gaussian_beam_tmz(0.0, z_coord, k=k, w0=2.0, amplitude=amplitude)
    np.testing.assert_allclose(forward["ez"], expected, rtol=1e-12)
    rotated = gaussian_beam_tmz(
        z_coord, -0.5, k=k, w0=2.0, direction=(1.0, 0.0), amplitude=amplitude
    )
    upright = gaussian_beam_tmz(-0.5, z_coord, k=k, w0=2.0, amplitude=amplitude)
    np.testing.assert_allclose(rotated["ez"], upright["ez"], rtol=1e-12)
    np.testing.assert_allclose(rotated["hx"], 0.0, atol=1e-12)
    np.testing.assert_allclose(rotated["hy"], -rotated["ez"] / ETA0, rtol=1e-12)
    far = np.array([8.0, 8.2])
    wave = gaussian_beam_tmz(np.zeros(2), far, k=k, w0=4.0)
    phase = np.angle(wave["ez"][1] / wave["ez"][0])
    advance = np.angle(np.exp(-1j * k * (far[1] - far[0])))
    np.testing.assert_allclose(phase, advance, atol=0.02)
    try:
        gaussian_beam_tmz(0.0, 0.0, k=0.0, w0=1.0)
    except ValueError as exc:
        assert "k" in str(exc)
    else:
        raise AssertionError("a nonpositive wavenumber was accepted")
    try:
        gaussian_beam_tmz(0.0, 0.0, k=k, w0=-1.0)
    except ValueError as exc:
        assert "w0" in str(exc)
    else:
        raise AssertionError("a nonpositive waist was accepted")
    try:
        gaussian_beam_tmz(0.0, 0.0, k=k, w0=1.0, direction=(0.0, 0.0))
    except ValueError as exc:
        assert "direction" in str(exc)
    else:
        raise AssertionError("a zero direction was accepted")
    try:
        gaussian_beam_tmz(0.0, 0.0, k=k, w0=1.0, amplitude=np.nan)
    except ValueError as exc:
        assert "amplitude" in str(exc)
    else:
        raise AssertionError("a non-finite amplitude was accepted")


def test_tfsf_split_rejects_a_bad_row() -> None:
    grid = YeeGrid2D(8, 6, 0.2, 0.2, Polarization.TEZ, Boundary.PEC)
    operator = FDFDOperator(grid, UniformIsotropic().sample_te(grid))
    try:
        tfsf_y_current(operator, 1.0, np.zeros((2, 2)), 3)
    except ValueError as exc:
        assert "TMz" in str(exc)
    else:
        raise AssertionError("a TEz split was accepted")
    grid = YeeGrid2D(8, 6, 0.2, 0.2, Polarization.TMZ, Boundary.PEC)
    operator = FDFDOperator(grid, UniformIsotropic().sample_tm(grid))
    ez = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    try:
        tfsf_y_current(operator, 1.0, ez, 1)
    except ValueError as exc:
        assert "j_total" in str(exc)
    else:
        raise AssertionError("a split on the PEC wall was accepted")


def test_gaussian_beam_matches_on_the_total_field_side() -> None:
    error, quiet, total_count, quiet_count = _beam_error()
    assert total_count > 1000
    assert quiet_count > 100
    assert error < BEAM_BAR
    assert quiet < QUIET_BAR


def _beam_error() -> tuple[float, float, int, int]:
    dx = _DX
    nx = int(round(_WIDTH / dx))
    ny = int(round(_HEIGHT / dx))
    grid = YeeGrid2D(nx, ny, dx, dx, Polarization.TMZ, Boundary.PEC)
    omega = 2.0 * np.pi * C0 / _WAVELENGTH
    k = omega / C0
    operator = FDFDOperator(grid, UniformIsotropic().sample_tm(grid), PMLSpec.box(_PML))
    x, y = grid.coordinates("ez")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    incident = gaussian_beam_tmz(
        xx, yy, k=k, w0=_W0, origin=(0.5 * _WIDTH, 0.5 * _HEIGHT)
    )
    row = int(round(_SPLIT / dx))
    current = tfsf_y_current(operator, omega, incident["ez"], row)
    solved = operator.solve(omega, current)
    ez = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    dof = operator.layout.e[0]
    ez[dof.i, dof.j] = solved
    inset = _PML * dx
    edge = inset + 0.5 * _WAVELENGTH
    total = (
        (xx >= edge)
        & (xx <= x[-1] - edge)
        & (yy >= y[row] + 0.5 * _WAVELENGTH)
        & (yy <= y[-1] - edge)
    )
    scattered = (
        (xx >= edge)
        & (xx <= x[-1] - edge)
        & (yy >= inset + 0.25 * _WAVELENGTH)
        & (yy <= y[row] - 0.5 * _WAVELENGTH)
    )
    exact = incident["ez"]
    error = float(
        np.linalg.norm(ez[total] - exact[total]) / np.linalg.norm(exact[total])
    )
    quiet = float(np.linalg.norm(ez[scattered]) / np.linalg.norm(exact[total]))
    return error, quiet, int(np.sum(total)), int(np.sum(scattered))
