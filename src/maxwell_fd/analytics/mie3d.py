"""3D Mie series for a sphere, ``e^{jωt}``.

Outgoing waves use ``h_n^{(2)}``. The incident wave propagates toward
``+z``, with ``E`` along ``+x`` and ``H`` along ``+y``:

    E_inc = x-hat exp(-j k (z - z_c)),
    H_inc = y-hat exp(-j k (z - z_c)) / η0.

``a_n`` and ``b_n`` are the scattered electric and magnetic coefficients.
``c_n`` and ``d_n`` are the interior ones. The sphere is non-magnetic.
Inside ``ρ < a`` the radial function is the spherical Bessel ``j_n``.
Outside, the total field is the incident wave plus the scattered series.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import spherical_jn, spherical_yn

from maxwell_fd.utils.constants import ETA0

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def multipole_count(size_parameter: float) -> int:
    """Wiscombe truncation plus four orders, so the surface series is tight."""
    width = abs(float(size_parameter))
    wiscombe = int(np.floor(width + 4.0 * width ** (1.0 / 3.0) + 2.0))
    return max(wiscombe + 4, 2)


def dielectric_coefficients(
    index: float, size_parameter: float
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Return ``(a, b, c, d)`` for orders ``n = 1 .. N``.

    The Riccati function outside is ``ξ = z h_n^{(2)}``. ``a`` multiplies
    the electric scattered wave and ``b`` the magnetic one.
    """
    m = _index(index)
    x = _size(size_parameter)
    count = multipole_count(x)
    a = np.empty(count, dtype=np.complex128)
    b = np.empty(count, dtype=np.complex128)
    c = np.empty(count, dtype=np.complex128)
    d = np.empty(count, dtype=np.complex128)
    mx = m * x
    for i, n in enumerate(range(1, count + 1)):
        psi_x, psip_x = _riccati(n, x, hankel=False)
        xi_x, xip_x = _riccati(n, x, hankel=True)
        psi_m, psip_m = _riccati(n, mx, hankel=False)
        den_a = m * psi_m * xip_x - xi_x * psip_m
        den_b = psi_m * xip_x - m * xi_x * psip_m
        wronskian = psi_x * xip_x - xi_x * psip_x
        a[i] = (m * psi_m * psip_x - psi_x * psip_m) / den_a
        b[i] = (psi_m * psip_x - m * psi_x * psip_m) / den_b
        c[i] = m * wronskian / den_b
        d[i] = m * wronskian / den_a
    return a, b, c, d


def pec_coefficients(size_parameter: float) -> tuple[ComplexArray, ComplexArray]:
    """PEC ``a_n = ψ'/ξ'`` and ``b_n = ψ/ξ``, with ``ξ = z h_n^{(2)}``."""
    x = _size(size_parameter)
    count = multipole_count(x)
    a = np.empty(count, dtype=np.complex128)
    b = np.empty(count, dtype=np.complex128)
    for i, n in enumerate(range(1, count + 1)):
        psi, psip = _riccati(n, x, hankel=False)
        xi, xip = _riccati(n, x, hankel=True)
        a[i] = psip / xip
        b[i] = psi / xi
    return a, b


def sphere_fields(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    *,
    k: float,
    radius: float,
    index: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    region: str | None = None,
) -> tuple[
    tuple[ComplexArray, ComplexArray, ComplexArray],
    tuple[ComplexArray, ComplexArray, ComplexArray],
]:
    """Total ``(Ex, Ey, Ez)`` and ``(Hx, Hy, Hz)``.

    ``region`` forces every sample through the interior series or the
    exterior series. The default splits on ``ρ < radius``.
    """
    if k <= 0.0:
        raise ValueError(f"k must be positive, got {k}")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    m = _index(index)
    xl, yl, zl = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64) - center[0],
        np.asarray(y, dtype=np.float64) - center[1],
        np.asarray(z, dtype=np.float64) - center[2],
    )
    rho = np.sqrt(xl * xl + yl * yl + zl * zl)
    inside = _region_mask(rho, radius, region)
    a, b, c, d = dielectric_coefficients(float(m.real), k * radius)
    electric, magnetic = _series(
        xl, yl, zl, rho, inside, k=k, radius=radius, index=m, a=a, b=b, c=c, d=d
    )
    return electric, magnetic


def _index(index: float) -> complex:
    value = complex(index)
    if value.imag != 0.0 or value.real <= 0.0:
        raise ValueError(f"index must be positive, got {index}")
    return value


def _size(size_parameter: float) -> float:
    value = float(size_parameter)
    if value <= 0.0:
        raise ValueError(f"size parameter must be positive, got {size_parameter}")
    return value


def _riccati(n: int, z: complex, *, hankel: bool) -> tuple[complex, complex]:
    sph, deriv = _spherical(n, z, hankel=hankel)
    return z * sph, sph + z * deriv


def _spherical(
    n: int, z: NDArray | complex, *, hankel: bool
) -> tuple[NDArray, NDArray]:
    jn = spherical_jn(n, z)
    jp = spherical_jn(n, z, derivative=True)
    if not hankel:
        return jn, jp
    return jn - 1j * spherical_yn(n, z), jp - 1j * spherical_yn(n, z, derivative=True)


def _region_mask(rho: NDArray, radius: float, region: str | None) -> NDArray:
    if region is None:
        return rho < radius
    if region == "interior":
        return np.ones(rho.shape, dtype=bool)
    if region == "exterior":
        return np.zeros(rho.shape, dtype=bool)
    raise ValueError(f"region must be 'interior' or 'exterior', got {region}")


def _series(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    rho: NDArray,
    inside: NDArray,
    *,
    k: float,
    radius: float,
    index: complex,
    a: ComplexArray,
    b: ComplexArray,
    c: ComplexArray,
    d: ComplexArray,
) -> tuple[
    tuple[ComplexArray, ComplexArray, ComplexArray],
    tuple[ComplexArray, ComplexArray, ComplexArray],
]:
    count = a.size
    orders = np.arange(1, count + 1)
    scale = ((-1j) ** orders) * (2 * orders + 1) / (orders * (orders + 1))
    safe = np.maximum(rho, 1e-30)
    mu = np.clip(z / safe, -1.0, 1.0)
    mu = np.where(mu >= 1.0, 0.999999, mu)
    mu = np.where(mu <= -1.0, -0.999999, mu)
    theta = np.arccos(np.clip(z / safe, -1.0, 1.0))
    phi = np.arctan2(y, x)
    pi, tau = _pi_tau(mu, count)
    e_sph = np.zeros((3,) + rho.shape, dtype=np.complex128)
    h_sph = np.zeros((3,) + rho.shape, dtype=np.complex128)
    origin = inside & (rho == 0.0)
    _add_multipoles(
        ~inside,
        hankel=True,
        medium=1.0 + 0.0j,
        electric=(a, b),
        magnetic=(b, a),
        rho=rho,
        theta=theta,
        phi=phi,
        pi=pi,
        tau=tau,
        scale=scale,
        orders=orders,
        k=k,
        e_sph=e_sph,
        h_sph=h_sph,
    )
    _add_multipoles(
        inside & ~origin,
        hankel=False,
        medium=index,
        electric=(-d, -c),
        magnetic=(-index * c, -index * d),
        rho=rho,
        theta=theta,
        phi=phi,
        pi=pi,
        tau=tau,
        scale=scale,
        orders=orders,
        k=k,
        e_sph=e_sph,
        h_sph=h_sph,
    )
    phase = np.exp(-1j * k * z)
    outside = ~inside
    e_sph[0][outside] += phase[outside] * np.sin(theta[outside]) * np.cos(phi[outside])
    e_sph[1][outside] += phase[outside] * np.cos(theta[outside]) * np.cos(phi[outside])
    e_sph[2][outside] += -phase[outside] * np.sin(phi[outside])
    h_sph[0][outside] += phase[outside] * np.sin(theta[outside]) * np.sin(phi[outside])
    h_sph[1][outside] += phase[outside] * np.cos(theta[outside]) * np.sin(phi[outside])
    h_sph[2][outside] += phase[outside] * np.cos(phi[outside])
    h_sph /= ETA0
    ex, ey, ez = _cartesian(e_sph, theta, phi)
    hx, hy, hz = _cartesian(h_sph, theta, phi)
    # The n = 1 interior term is finite at the center: Ex = d_1, Hy = m c_1 / η0.
    if np.any(origin):
        ex[origin] = d[0]
        ey[origin] = 0.0
        ez[origin] = 0.0
        hx[origin] = 0.0
        hy[origin] = index * c[0] / ETA0
        hz[origin] = 0.0
    return (ex, ey, ez), (hx, hy, hz)


def _pi_tau(mu: NDArray, count: int) -> tuple[NDArray, NDArray]:
    pi = np.zeros(mu.shape + (count,))
    tau = np.zeros(mu.shape + (count,))
    pi[..., 0] = 1.0
    previous = np.zeros(mu.shape, dtype=np.float64)
    for n in range(1, count):
        tau[..., n - 1] = n * mu * pi[..., n - 1] - (n + 1) * previous
        current = pi[..., n - 1].copy()
        pi[..., n] = ((2 * n + 1) * mu * current - (n + 1) * previous) / n
        previous = current
    tau[..., count - 1] = count * mu * pi[..., count - 1] - (count + 1) * previous
    return pi, tau


def _add_multipoles(
    mask: NDArray,
    *,
    hankel: bool,
    medium: complex,
    electric: tuple[ComplexArray, ComplexArray],
    magnetic: tuple[ComplexArray, ComplexArray],
    rho: NDArray,
    theta: NDArray,
    phi: NDArray,
    pi: NDArray,
    tau: NDArray,
    scale: ComplexArray,
    orders: NDArray,
    k: float,
    e_sph: NDArray,
    h_sph: NDArray,
) -> None:
    if not np.any(mask):
        return
    kr = k * medium * rho[mask]
    cos_p = np.cos(phi[mask])
    sin_p = np.sin(phi[mask])
    sin_t = np.sin(theta[mask])
    pi_m = pi[mask]
    tau_m = tau[mask]
    er = np.zeros(kr.shape, dtype=np.complex128)
    et = np.zeros_like(er)
    ep = np.zeros_like(er)
    hr = np.zeros_like(er)
    ht = np.zeros_like(er)
    hp = np.zeros_like(er)
    aa, bb = electric
    cc, dd = magnetic
    for i, n in enumerate(orders):
        sph, deriv = _spherical(int(n), kr, hankel=hankel)
        n1 = sph / kr
        n2 = deriv + sph / kr
        n_r = n * (n + 1) * sin_t * pi_m[..., i] * n1
        n_t = tau_m[..., i] * n2
        n_p = pi_m[..., i] * n2
        m_t = pi_m[..., i] * sph
        m_p = tau_m[..., i] * sph
        weight = scale[i]
        er += weight * ((-1j) * aa[i] * cos_p * n_r)
        et += weight * ((-1j) * aa[i] * cos_p * n_t - bb[i] * cos_p * m_t)
        ep += weight * ((-1j) * aa[i] * (-sin_p) * n_p - bb[i] * (-sin_p) * m_p)
        hr += weight * ((-1j) * cc[i] * sin_p * n_r)
        ht += weight * ((-1j) * cc[i] * sin_p * n_t + dd[i] * (-sin_p) * m_t)
        hp += weight * ((-1j) * cc[i] * cos_p * n_p + dd[i] * (-cos_p) * m_p)
    e_sph[0][mask] += er
    e_sph[1][mask] += et
    e_sph[2][mask] += ep
    h_sph[0][mask] += hr
    h_sph[1][mask] += ht
    h_sph[2][mask] += hp


def _cartesian(
    spherical: NDArray, theta: NDArray, phi: NDArray
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    fr, ft, fp = spherical
    fx = fr * st * cp + ft * ct * cp - fp * sp
    fy = fr * st * sp + ft * ct * sp + fp * cp
    fz = fr * ct - ft * st
    return fx, fy, fz
