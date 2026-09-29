"""Closed-form slab and cylinder fields, before any Yee solve."""

import numpy as np
from scipy.special import hankel2, jv

from maxwell_fd.analytics.fresnel import slab_coefficients, slab_fields
from maxwell_fd.analytics.mie2d import pec_tmz_coefficient, radial_limits
from maxwell_fd.utils.constants import C0, EPS0, MU0


def test_fresnel_faces_are_continuous_and_vacuum_is_the_incident_wave() -> None:
    omega = 2.0 * np.pi * C0 / 40.0
    y1, y2 = 30.5, 50.5
    eps_r = 4.0
    thickness = y2 - y1
    electric = slab_coefficients(eps_r, omega / C0, thickness)
    magnetic = slab_coefficients(eps_r, omega / C0, thickness, te=True)
    np.testing.assert_allclose(magnetic[0], -electric[0], atol=1e-12)
    np.testing.assert_allclose(magnetic[1], electric[1], atol=1e-12)
    for face, low, high in ((y1, "below", "slab"), (y2, "slab", "above")):
        left = slab_fields(
            np.array([face]), y1=y1, y2=y2, omega=omega, eps_r=eps_r, branch=low
        )
        right = slab_fields(
            np.array([face]), y1=y1, y2=y2, omega=omega, eps_r=eps_r, branch=high
        )
        for left_field, right_field in zip(left, right, strict=True):
            np.testing.assert_allclose(left_field, right_field, atol=1e-12)
    reflected, transmitted, _n = slab_coefficients(1.0, omega / C0, thickness)
    assert abs(reflected) < 1e-12
    # τ keeps the transit phase e^{-jkd}, so the field of a vacuum slab is the incident wave.
    y = np.linspace(0.0, 80.0, 9)
    ez, hx, hz, ex = slab_fields(y, y1=y1, y2=y2, omega=omega, eps_r=1.0)
    incident = np.exp(-1j * (omega / C0) * (y - y1))
    eta = float(np.sqrt(MU0 / EPS0))
    np.testing.assert_allclose(ez, incident, atol=1e-12)
    np.testing.assert_allclose(hz, incident, atol=1e-12)
    np.testing.assert_allclose(hx, incident / eta, atol=1e-12)
    np.testing.assert_allclose(ex, -eta * incident, atol=1e-12)
    np.testing.assert_allclose(
        transmitted, np.exp(-1j * (omega / C0) * thickness), atol=1e-12
    )


def test_fresnel_hx_and_ex_match_the_maxwell_derivatives() -> None:
    omega = 2.0 * np.pi * C0 / 40.0
    y1, y2 = 30.5, 50.5
    y = np.array([10.0, 40.0, 70.0])
    step = 1e-6
    ez, hx, hz, ex = slab_fields(y, y1=y1, y2=y2, omega=omega, eps_r=4.0)
    ez_above, _hx_above, hz_above, _ex_above = slab_fields(
        y + step, y1=y1, y2=y2, omega=omega, eps_r=4.0
    )
    ez_below, _hx_below, hz_below, _ex_below = slab_fields(
        y - step, y1=y1, y2=y2, omega=omega, eps_r=4.0
    )
    d_ez = (ez_above - ez_below) / (2.0 * step)
    d_hz = (hz_above - hz_below) / (2.0 * step)
    np.testing.assert_allclose(hx, -d_ez / (1j * omega * MU0), rtol=1e-8, atol=1e-8)
    eps = np.array([EPS0, 4.0 * EPS0, EPS0])
    np.testing.assert_allclose(ex, d_hz / (1j * omega * eps), rtol=1e-8, atol=1e-8)
    del ez, hz


def test_mie_tangential_field_is_continuous_and_pec_coefficient_matches() -> None:
    k, radius, eps_r, phi = 0.1, 10.0, 2.25, 0.7
    for te in (False, True):
        value_in, value_out, deriv_in, deriv_out = radial_limits(
            k=k, radius=radius, eps_r=eps_r, phi=phi, te=te
        )
        np.testing.assert_allclose(value_in, value_out, atol=1e-8)
        if te:
            # Tangential E is continuous when (1/ε) ∂H/∂ρ is.
            np.testing.assert_allclose(deriv_in / eps_r, deriv_out, atol=1e-8)
        else:
            np.testing.assert_allclose(deriv_in, deriv_out, atol=1e-8)
    ka = 1.3
    orders = np.arange(-6, 7)
    direct = -((-1j) ** orders) * jv(orders, ka) / hankel2(orders, ka)
    np.testing.assert_allclose(
        pec_tmz_coefficient(ka, orders), direct, rtol=1e-12, atol=1e-12
    )
