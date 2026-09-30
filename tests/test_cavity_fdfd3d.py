"""3D PEC cavity eigenvalues and a manufactured FDFD solve."""

import numpy as np

from maxwell_fd.analytics.cavity3d import cavity_mode, semi_discrete_omega
from maxwell_fd.drivers.fdfd3d import FDFDOperator3D
from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.materials.volume3d import UniformIsotropic3D

# 8×6×5, Δ = 1. The three pure modes are (1,1,0), (1,0,1), and (0,1,1).
_CELLS = (8, 6, 5)


def test_pure_cavity_modes_match_the_sine_formula() -> None:
    grid = YeeGrid3D(*_CELLS, 1.0, 1.0, 1.0, Boundary.PEC)
    operator = FDFDOperator3D(grid, UniformIsotropic3D().sample(grid))
    stiffness, mass = operator.spatial_operator()
    for indices in ((1, 1, 0), (1, 0, 1), (0, 1, 1)):
        expected = semi_discrete_omega(
            m=indices[0],
            n=indices[1],
            p=indices[2],
            a=grid.a,
            b=grid.b,
            c=grid.c,
            dx=grid.dx,
            dy=grid.dy,
            dz=grid.dz,
        )
        mode = operator.layout.pack_e(cavity_mode(grid, *indices))
        stiffness_mode = stiffness @ mode
        mass_mode = mass @ mode
        rayleigh = float(mode @ stiffness_mode / (mode @ mass_mode))
        residual = np.linalg.norm(stiffness_mode - rayleigh * mass_mode)
        scale = np.linalg.norm(stiffness_mode)
        assert residual / scale < 1e-10, (indices, residual / scale)
        np.testing.assert_allclose(np.sqrt(rayleigh), expected, rtol=1e-10)


def test_manufactured_fdfd_recovers_the_field() -> None:
    grid = YeeGrid3D(6, 5, 4, 0.5, 0.4, 0.3, Boundary.PEC)
    operator = FDFDOperator3D(grid, UniformIsotropic3D().sample(grid))
    rng = np.random.default_rng(3)
    field = rng.normal(size=operator.layout.n_e) + 1j * rng.normal(
        size=operator.layout.n_e
    )
    omega = 1.0e8
    matrix = operator.system_matrix(omega)
    recovered = operator.solve(omega, -(matrix @ field))
    np.testing.assert_allclose(recovered, field, rtol=1e-10, atol=1e-10)
