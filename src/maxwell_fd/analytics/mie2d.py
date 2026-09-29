"""2D Mie series for a non-magnetic dielectric cylinder, ``e^{jωt}``.

Outgoing waves use ``H_n^{(2)}``. The incident expansion is eq:mie-inc, the
TMz coefficient is eq:mie-tm, and the TEz coefficient is eq:mie-te in
``FD_LaTeX_Reference.tex``. The PEC coefficient is a separate one-line
reduction, not the large-index limit of the dielectric formula.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import hankel2, jv

ComplexArray = NDArray[np.complex128]


def azimuthal_orders(ka: float) -> NDArray[np.int64]:
    """Truncation ``|n| <= ceil(|ka| + 4 |ka|^{1/3} + 8)``."""
    width = abs(float(ka))
    n_max = int(np.ceil(width + 4.0 * width ** (1.0 / 3.0) + 8.0))
    return np.arange(-n_max, n_max + 1, dtype=np.int64)


def bessel_derivative(
    order: NDArray, argument: NDArray | float, *, hankel: bool
) -> NDArray:
    """``f_n' = (f_{n-1} - f_{n+1}) / 2`` for ``J`` or ``H^{(2)}``."""
    function = hankel2 if hankel else jv
    return 0.5 * (function(order - 1, argument) - function(order + 1, argument))


def pec_tmz_coefficient(ka: float, order: NDArray) -> ComplexArray:
    """``a_n = -(-j)^n J_n(ka) / H_n^{(2)}(ka)``."""
    n = np.asarray(order, dtype=int)
    return np.asarray(-((-1j) ** n) * jv(n, ka) / hankel2(n, ka), dtype=np.complex128)


def dielectric_coefficients(
    ka: float, index: float, *, te: bool
) -> tuple[NDArray[np.int64], ComplexArray, ComplexArray]:
    """Return ``(n, a_n, b_n)`` for TEz when ``te`` is true, otherwise TMz."""
    if index <= 0.0:
        raise ValueError(f"index must be positive, got {index}")
    if ka <= 0.0:
        raise ValueError(f"ka must be positive, got {ka}")
    orders = azimuthal_orders(ka)
    beta = index * ka
    j_alpha = jv(orders, ka)
    j_beta = jv(orders, beta)
    h_alpha = hankel2(orders, ka)
    jp_alpha = bessel_derivative(orders, ka, hankel=False)
    jp_beta = bessel_derivative(orders, beta, hankel=False)
    hp_alpha = bessel_derivative(orders, ka, hankel=True)
    phase = (-1j) ** orders
    if te:
        numerator = index * j_beta * jp_alpha - jp_beta * j_alpha
        denominator = jp_beta * h_alpha - index * j_beta * hp_alpha
    else:
        numerator = index * jp_beta * j_alpha - j_beta * jp_alpha
        denominator = j_beta * hp_alpha - index * jp_beta * h_alpha
    scattered = phase * numerator / denominator
    interior = (phase * j_alpha + scattered * h_alpha) / j_beta
    return (
        orders,
        np.asarray(scattered, dtype=np.complex128),
        np.asarray(interior, dtype=np.complex128),
    )


def cylinder_ez(
    x: NDArray,
    y: NDArray,
    *,
    k: float,
    radius: float,
    eps_r: float,
    center: tuple[float, float] = (0.0, 0.0),
) -> ComplexArray:
    """Total TMz ``Ez`` of a unit wave ``e^{-jkx}`` on a dielectric cylinder."""
    return _scalar_field(x, y, k=k, radius=radius, eps_r=eps_r, center=center, te=False)


def cylinder_hz(
    x: NDArray,
    y: NDArray,
    *,
    k: float,
    radius: float,
    eps_r: float,
    center: tuple[float, float] = (0.0, 0.0),
) -> ComplexArray:
    """Total TEz ``Hz`` of a unit wave ``e^{-jkx}`` on a dielectric cylinder."""
    return _scalar_field(x, y, k=k, radius=radius, eps_r=eps_r, center=center, te=True)


def cylinder_te_electric(
    x: NDArray,
    y: NDArray,
    *,
    k: float,
    radius: float,
    eps_r: float,
    omega: float,
    eps0: float,
    center: tuple[float, float] = (0.0, 0.0),
) -> tuple[ComplexArray, ComplexArray]:
    """``(Ex, Ey)`` from ``Ex = ∂y Hz / (jωε)`` and ``Ey = -∂x Hz / (jωε)``."""
    if omega == 0.0:
        raise ValueError("omega must be nonzero")
    dx, dy = _cartesian_derivatives(
        x, y, k=k, radius=radius, eps_r=eps_r, center=center, te=True
    )
    rho, _phi = _polar(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), center
    )
    permittivity = np.where(rho <= radius, eps_r * eps0, eps0)
    scale = 1.0 / (1j * float(omega) * permittivity)
    return dy * scale, -dx * scale


def radial_limits(
    *,
    k: float,
    radius: float,
    eps_r: float,
    phi: float,
    te: bool,
) -> tuple[complex, complex, complex, complex]:
    """Field and ``∂/∂ρ`` just inside and outside ``ρ = a``, at one angle.

    Returns ``(value_in, value_out, deriv_in, deriv_out)``.
    """
    index = float(np.sqrt(eps_r))
    orders, scattered, interior = dielectric_coefficients(k * radius, index, te=te)
    phase = np.exp(1j * orders * phi)
    ka = k * radius
    beta = index * ka
    value_in = np.sum(interior * jv(orders, beta) * phase)
    value_out = np.sum(
        (((-1j) ** orders) * jv(orders, ka) + scattered * hankel2(orders, ka)) * phase
    )
    deriv_in = np.sum(
        interior * bessel_derivative(orders, beta, hankel=False) * (index * k) * phase
    )
    deriv_out = np.sum(
        (
            ((-1j) ** orders) * bessel_derivative(orders, ka, hankel=False)
            + scattered * bessel_derivative(orders, ka, hankel=True)
        )
        * k
        * phase
    )
    return complex(value_in), complex(value_out), complex(deriv_in), complex(deriv_out)


def _scalar_field(
    x: NDArray,
    y: NDArray,
    *,
    k: float,
    radius: float,
    eps_r: float,
    center: tuple[float, float],
    te: bool,
) -> ComplexArray:
    abscissa = np.asarray(x, dtype=np.float64)
    ordinate = np.asarray(y, dtype=np.float64)
    rho, phi = _polar(abscissa, ordinate, center)
    flat_rho = rho.ravel()
    flat_phi = phi.ravel()
    index = float(np.sqrt(eps_r))
    orders, scattered, interior = dielectric_coefficients(k * radius, index, te=te)
    acc = np.zeros(flat_rho.shape, dtype=np.complex128)
    inside = flat_rho <= radius
    outside = ~inside
    for n, a_n, b_n, phase0 in zip(
        orders, scattered, interior, (-1j) ** orders, strict=True
    ):
        angular = np.exp(1j * n * flat_phi)
        if np.any(inside):
            acc[inside] += b_n * jv(n, index * k * flat_rho[inside]) * angular[inside]
        if np.any(outside):
            radial = phase0 * jv(n, k * flat_rho[outside]) + a_n * hankel2(
                n, k * flat_rho[outside]
            )
            acc[outside] += radial * angular[outside]
    return acc.reshape(rho.shape)


def _cartesian_derivatives(
    x: NDArray,
    y: NDArray,
    *,
    k: float,
    radius: float,
    eps_r: float,
    center: tuple[float, float],
    te: bool,
) -> tuple[ComplexArray, ComplexArray]:
    abscissa = np.asarray(x, dtype=np.float64)
    ordinate = np.asarray(y, dtype=np.float64)
    rho, phi = _polar(abscissa, ordinate, center)
    flat_rho = np.maximum(rho.ravel(), 1e-14 * radius)
    flat_phi = phi.ravel()
    index = float(np.sqrt(eps_r))
    orders, scattered, interior = dielectric_coefficients(k * radius, index, te=te)
    dx = np.zeros(flat_rho.shape, dtype=np.complex128)
    dy = np.zeros(flat_rho.shape, dtype=np.complex128)
    inside = rho.ravel() <= radius
    cos_phi = np.cos(flat_phi)
    sin_phi = np.sin(flat_phi)
    for n, a_n, b_n, phase0 in zip(
        orders, scattered, interior, (-1j) ** orders, strict=True
    ):
        angular = np.exp(1j * n * flat_phi)
        radial = np.empty(flat_rho.shape, dtype=np.complex128)
        slope = np.empty(flat_rho.shape, dtype=np.complex128)
        if np.any(inside):
            argument = index * k * flat_rho[inside]
            radial[inside] = b_n * jv(n, argument)
            slope[inside] = (
                b_n * bessel_derivative(n, argument, hankel=False) * index * k
            )
        if np.any(~inside):
            argument = k * flat_rho[~inside]
            radial[~inside] = phase0 * jv(n, argument) + a_n * hankel2(n, argument)
            slope[~inside] = k * (
                phase0 * bessel_derivative(n, argument, hankel=False)
                + a_n * bessel_derivative(n, argument, hankel=True)
            )
        chain_x = slope * cos_phi + radial * (1j * n) * (-sin_phi) / flat_rho
        chain_y = slope * sin_phi + radial * (1j * n) * cos_phi / flat_rho
        dx += chain_x * angular
        dy += chain_y * angular
    return dx.reshape(rho.shape), dy.reshape(rho.shape)


def _polar(
    x: NDArray, y: NDArray, center: tuple[float, float]
) -> tuple[NDArray, NDArray]:
    dx = x - center[0]
    dy = y - center[1]
    return np.hypot(dx, dy), np.arctan2(dy, dx)
