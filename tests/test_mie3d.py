"""3D Mie series: a transparent sphere, surface continuity, and PEC ratios."""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from maxwell_fd.analytics.mie3d import (
    dielectric_coefficients,
    pec_coefficients,
    sphere_fields,
)
from maxwell_fd.utils.constants import ETA0


def test_transparent_sphere_is_the_incident_wave() -> None:
    k = 1.4
    radius = 0.8
    center = (0.2, -0.3, 0.15)
    z = np.linspace(-1.5, 1.6, 9)
    x = np.full_like(z, 0.35)
    y = np.full_like(z, -0.25)
    (ex, ey, ez), (hx, hy, hz) = sphere_fields(
        x, y, z, k=k, radius=radius, index=1.0, center=center
    )
    phase = np.exp(-1j * k * (z - center[2]))
    np.testing.assert_allclose(ex, phase, atol=1e-8)
    np.testing.assert_allclose(ey, 0.0, atol=1e-8)
    np.testing.assert_allclose(ez, 0.0, atol=1e-8)
    np.testing.assert_allclose(hx, 0.0, atol=1e-8)
    np.testing.assert_allclose(hy, phase / ETA0, atol=1e-8)
    np.testing.assert_allclose(hz, 0.0, atol=1e-8)
    (ex0, ey0, ez0), (hx0, hy0, hz0) = sphere_fields(
        np.array([center[0]]),
        np.array([center[1]]),
        np.array([center[2]]),
        k=k,
        radius=radius,
        index=1.0,
        center=center,
    )
    np.testing.assert_allclose(ex0, 1.0, atol=1e-12)
    np.testing.assert_allclose(ey0, 0.0, atol=1e-12)
    np.testing.assert_allclose(ez0, 0.0, atol=1e-12)
    np.testing.assert_allclose(hy0, 1.0 / ETA0, atol=1e-12)
    np.testing.assert_allclose(hx0, 0.0, atol=1e-12)
    np.testing.assert_allclose(hz0, 0.0, atol=1e-12)


def test_surface_is_continuous_and_coefficients_are_lossless() -> None:
    k, radius, index = 2.0, 0.5, 1.5
    theta = np.linspace(0.2, np.pi - 0.2, 8)
    phi = np.linspace(0.15, 2.0 * np.pi - 0.15, 10)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    xyz = (
        radius * np.sin(tt) * np.cos(pp),
        radius * np.sin(tt) * np.sin(pp),
        radius * np.cos(tt),
    )
    e_out, h_out = sphere_fields(
        *xyz, k=k, radius=radius, index=index, region="exterior"
    )
    e_in, h_in = sphere_fields(*xyz, k=k, radius=radius, index=index, region="interior")
    rhat = (
        np.sin(tt) * np.cos(pp),
        np.sin(tt) * np.sin(pp),
        np.cos(tt),
    )
    _assert_continuous(e_out, e_in, rhat, normal_scale=index**2)
    _assert_continuous(h_out, h_in, rhat, normal_scale=1.0)
    a, b, c, d = dielectric_coefficients(1.0, k * radius)
    np.testing.assert_allclose(a, 0.0, atol=1e-12)
    np.testing.assert_allclose(b, 0.0, atol=1e-12)
    np.testing.assert_allclose(c, 1.0, atol=1e-12)
    np.testing.assert_allclose(d, 1.0, atol=1e-12)
    a, b, _c, _d = dielectric_coefficients(index, k * radius)
    np.testing.assert_allclose(a.real, np.abs(a) ** 2, atol=1e-12)
    np.testing.assert_allclose(b.real, np.abs(b) ** 2, atol=1e-12)
    direct_a, direct_b, wronskian = _pec_from_scipy(1.3, 1)
    got_a, got_b = pec_coefficients(1.3)
    np.testing.assert_allclose(got_a[0], direct_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_b[0], direct_b, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wronskian, -1j, atol=1e-12)


def _assert_continuous(outer, inner, rhat, *, normal_scale: float) -> None:
    normal_out = outer[0] * rhat[0] + outer[1] * rhat[1] + outer[2] * rhat[2]
    normal_in = inner[0] * rhat[0] + inner[1] * rhat[1] + inner[2] * rhat[2]
    for component in range(3):
        tang_out = outer[component] - normal_out * rhat[component]
        tang_in = inner[component] - normal_in * rhat[component]
        np.testing.assert_allclose(tang_out, tang_in, atol=1e-8)
    np.testing.assert_allclose(normal_out, normal_scale * normal_in, atol=1e-8)


def _pec_from_scipy(argument: float, order: int) -> tuple[complex, complex, complex]:
    jn = spherical_jn(order, argument)
    jp = spherical_jn(order, argument, derivative=True)
    yn = spherical_yn(order, argument)
    yp = spherical_yn(order, argument, derivative=True)
    hankel = jn - 1j * yn
    hankel_p = jp - 1j * yp
    psi = argument * jn
    psi_p = jn + argument * jp
    xi = argument * hankel
    xi_p = hankel + argument * hankel_p
    return psi_p / xi_p, psi / xi, psi * xi_p - xi * psi_p
