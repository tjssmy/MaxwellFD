"""Frequency-domain Yee system on a 3D grid.

The factored matrix is the same operator as eq. (4.1) of
``FD_LaTeX_Reference.tex``,

    curl_h @ diag(1/(j ω μ)) @ curl_e + j ω diag(ε),

applied to the free electric unknowns ``(Ex, Ey, Ez)``. It equals ``-J``.
Real ``ε`` and ``σ`` enter the diagonal as ``ε + σ/(jω)``.

``spatial_operator`` returns the real pair ``(K, M)`` of eq. (4.2) for
lossless media. A PML is not attached to this driver.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.materials.volume3d import FieldComponents3D
from maxwell_fd.operators.curl3d import CurlOperators3, Layout3, build_curls

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


class FDFDOperator3D:
    """Sparse 3D curls plus the constitutive samples on the free electric unknowns."""

    def __init__(self, grid: YeeGrid3D, components: FieldComponents3D) -> None:
        self.grid = grid
        self.components = components
        self.operators: CurlOperators3 = build_curls(grid)
        self.layout: Layout3 = self.operators.layout
        self.eps_e, self.sigma_e, self.mu_h = _sample_diagonals(
            grid, components, self.layout
        )

    def spatial_operator(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """``K`` and ``M`` of eq. (4.2), with real positive ``ε`` and ``μ``."""
        if np.any(self.sigma_e != 0.0):
            raise ValueError("spatial eigenproblem needs sigma = 0")
        if np.any(self.mu_h <= 0.0) or np.any(self.eps_e <= 0.0):
            raise ValueError("spatial eigenproblem needs positive real eps and mu")
        inv_mu = sparse.diags(1.0 / self.mu_h)
        stiffness = (self.operators.curl_h @ inv_mu @ self.operators.curl_e).tocsr()
        mass = sparse.diags(self.eps_e, format="csr")
        return stiffness, mass

    def system_matrix(self, omega: float) -> sparse.csr_matrix:
        """Driven matrix ``A`` such that ``A e = -J``."""
        if omega == 0.0:
            raise ValueError("omega must be nonzero")
        j_omega = 1j * float(omega)
        permittivity = self.eps_e + self.sigma_e / j_omega
        inv_j_omega_mu = sparse.diags(1.0 / (j_omega * self.mu_h))
        curl_curl = self.operators.curl_h @ inv_j_omega_mu @ self.operators.curl_e
        return (curl_curl + sparse.diags(j_omega * permittivity)).tocsr()

    def solve(self, omega: float, impressed_j: NDArray) -> ComplexArray:
        """Electric unknowns for ``A e = -J``."""
        current = np.asarray(impressed_j, dtype=np.complex128).reshape(-1)
        if current.size != self.layout.n_e:
            raise ValueError(
                f"impressed_j has length {current.size}, expected {self.layout.n_e}"
            )
        return np.asarray(
            spsolve(self.system_matrix(omega), -current), dtype=np.complex128
        )


def _sample_diagonals(
    grid: YeeGrid3D, components: FieldComponents3D, layout: Layout3
) -> tuple[RealArray, RealArray, RealArray]:
    shapes = grid.shapes()
    _match(components.eps_x, shapes["ex"], "eps_x")
    _match(components.eps_y, shapes["ey"], "eps_y")
    _match(components.eps_z, shapes["ez"], "eps_z")
    _match(components.mu_x, shapes["hx"], "mu_x")
    _match(components.mu_y, shapes["hy"], "mu_y")
    _match(components.mu_z, shapes["hz"], "mu_z")
    _match(components.sigma_x, shapes["ex"], "sigma_x")
    _match(components.sigma_y, shapes["ey"], "sigma_y")
    _match(components.sigma_z, shapes["ez"], "sigma_z")
    eps_e = np.concatenate(
        [
            layout.e[0].pack(components.eps_x),
            layout.e[1].pack(components.eps_y),
            layout.e[2].pack(components.eps_z),
        ]
    ).astype(np.float64, copy=False)
    sigma_e = np.concatenate(
        [
            layout.e[0].pack(components.sigma_x),
            layout.e[1].pack(components.sigma_y),
            layout.e[2].pack(components.sigma_z),
        ]
    ).astype(np.float64, copy=False)
    mu_h = np.concatenate(
        [
            layout.h[0].pack(components.mu_x),
            layout.h[1].pack(components.mu_y),
            layout.h[2].pack(components.mu_z),
        ]
    ).astype(np.float64, copy=False)
    return eps_e, sigma_e, mu_h


def _match(field: NDArray, shape: tuple[int, int, int], name: str) -> None:
    if field.shape != shape:
        raise ValueError(f"{name} shape {field.shape} != {shape}")
