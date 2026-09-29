"""Driven FDFD solve recovers a field that produced the impressed current."""

import numpy as np

from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.utils.constants import C0


def _recover(pol: Polarization) -> None:
    grid = YeeGrid2D(16, 12, 0.05, 0.04, pol, Boundary.PEC)
    law = UniformIsotropic()
    components = law.sample_tm(grid) if pol is Polarization.TMZ else law.sample_te(grid)
    operator = FDFDOperator(grid, components)
    # Below every cavity resonance, so the Helmholtz matrix stays well conditioned.
    omega = 0.25 * C0 * np.pi / max(grid.a, grid.b)
    rng = np.random.default_rng(4)
    electric = rng.normal(size=operator.layout.n_e) + 1j * rng.normal(
        size=operator.layout.n_e
    )
    current = -(operator.system_matrix(omega) @ electric)
    recovered = operator.solve(omega, current)
    np.testing.assert_allclose(recovered, electric, rtol=1e-10, atol=1e-10)


def test_manufactured_tmz() -> None:
    _recover(Polarization.TMZ)


def test_manufactured_tez() -> None:
    _recover(Polarization.TEZ)
