"""PEC cavity eigenvalues against the semi-discrete Yee formula."""

import numpy as np
from scipy import linalg

from maxwell_fd.analytics.cavity2d import semi_discrete_omega
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import UniformIsotropic


def _omegas(pol: Polarization) -> np.ndarray:
    grid = YeeGrid2D(16, 12, 1.0, 1.0, pol, Boundary.PEC)
    law = UniformIsotropic()
    components = law.sample_tm(grid) if pol is Polarization.TMZ else law.sample_te(grid)
    stiffness, mass = FDFDOperator(grid, components).spatial_operator()
    eigenvalues = linalg.eigh(stiffness.toarray(), mass.toarray(), eigvals_only=True)
    return np.sqrt(np.clip(eigenvalues, 0.0, None))


def _expected(pol: Polarization, m: int, n: int) -> float:
    grid = YeeGrid2D(16, 12, 1.0, 1.0, pol, Boundary.PEC)
    return semi_discrete_omega(m=m, n=n, a=grid.a, b=grid.b, dx=grid.dx, dy=grid.dy)


def test_tmz_fundamental_matches_sine_formula() -> None:
    omega = _omegas(Polarization.TMZ)
    expected = _expected(Polarization.TMZ, 1, 1)
    np.testing.assert_allclose(omega[0], expected, rtol=1e-10)


def test_tez_modes_match_sine_formula() -> None:
    # curl-curl on tangential E has a gradient kernel, so zero eigenvalues sit
    # under the physical cavity spectrum.
    omega = _omegas(Polarization.TEZ)
    positive = omega[omega > 1e-6 * omega[-1]]
    np.testing.assert_allclose(
        positive[0], _expected(Polarization.TEZ, 1, 0), rtol=1e-10
    )
    distance = np.min(np.abs(positive - _expected(Polarization.TEZ, 1, 1)))
    assert distance / _expected(Polarization.TEZ, 1, 1) < 1e-10
