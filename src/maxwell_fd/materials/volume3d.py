"""Uniform ε, μ, and σ on the six 3D Yee locations.

A staircase ball is not in this module. The 2D staircase stays in
``materials.volume``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.utils.constants import EPS0, MU0

FloatArray = NDArray[np.float64]


class FieldComponents3D:
    """Constitutive samples shaped like the six field components."""

    def __init__(
        self,
        eps_x: NDArray,
        eps_y: NDArray,
        eps_z: NDArray,
        mu_x: NDArray,
        mu_y: NDArray,
        mu_z: NDArray,
        sigma_x: NDArray,
        sigma_y: NDArray,
        sigma_z: NDArray,
    ) -> None:
        self.eps_x = _real(eps_x, "eps_x")
        self.eps_y = _real(eps_y, "eps_y")
        self.eps_z = _real(eps_z, "eps_z")
        self.mu_x = _real(mu_x, "mu_x")
        self.mu_y = _real(mu_y, "mu_y")
        self.mu_z = _real(mu_z, "mu_z")
        self.sigma_x = _real(sigma_x, "sigma_x")
        self.sigma_y = _real(sigma_y, "sigma_y")
        self.sigma_z = _real(sigma_z, "sigma_z")


class UniformIsotropic3D:
    """One real ε, μ, and σ on every sample. σ is an electric conductivity."""

    def __init__(self, eps: float = EPS0, mu: float = MU0, sigma: float = 0.0) -> None:
        if eps <= 0.0 or mu <= 0.0:
            raise ValueError(f"eps and mu must be positive, got eps={eps}, mu={mu}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.eps = float(eps)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def sample(self, grid: YeeGrid3D) -> FieldComponents3D:
        shape = grid.shapes()
        return FieldComponents3D(
            eps_x=np.full(shape["ex"], self.eps),
            eps_y=np.full(shape["ey"], self.eps),
            eps_z=np.full(shape["ez"], self.eps),
            mu_x=np.full(shape["hx"], self.mu),
            mu_y=np.full(shape["hy"], self.mu),
            mu_z=np.full(shape["hz"], self.mu),
            sigma_x=np.full(shape["ex"], self.sigma),
            sigma_y=np.full(shape["ey"], self.sigma),
            sigma_z=np.full(shape["ez"], self.sigma),
        )


def _real(values: NDArray, name: str) -> FloatArray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    return np.asarray(array, dtype=np.float64)
