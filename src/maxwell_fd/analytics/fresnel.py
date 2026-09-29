"""Normal-incidence Fresnel field of a dielectric slab in vacuum.

The slab coefficients are eq:fresnel-R. The TMz field is eq:fresnel-field
and the TEz field is eq:fresnel-te in ``FD_LaTeX_Reference.tex``. Phase is
zero for the incident wave at the front face. The two scalars agree only
when the index is one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.utils.constants import C0, EPS0, MU0

ComplexArray = NDArray[np.complex128]


def slab_coefficients(
    eps_r: float,
    k: float,
    thickness: float,
    *,
    te: bool = False,
) -> tuple[complex, complex, float]:
    """Return ``(R, τ, n)`` for a slab of thickness ``d`` and index ``n``.

    ``te=False`` uses the TMz interface coefficient ``(1-n)/(1+n)``.
    ``te=True`` uses ``(n-1)/(n+1)`` for ``Hz``. Transmission depends on
    ``r**2``, so the two values of ``τ`` match and the reflections differ
    by a sign.
    """
    if eps_r <= 0.0:
        raise ValueError(f"eps_r must be positive, got {eps_r}")
    if thickness <= 0.0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    n = float(np.sqrt(eps_r))
    reflection = (n - 1.0) / (n + 1.0) if te else (1.0 - n) / (1.0 + n)
    delta = n * k * thickness
    phase = np.exp(-2j * delta)
    denom = 1.0 - reflection**2 * phase
    reflected = reflection * (1.0 - phase) / denom
    transmitted = (1.0 - reflection**2) * np.exp(-1j * delta) / denom
    return complex(reflected), complex(transmitted), n


def slab_fields(
    y: NDArray,
    *,
    y1: float,
    y2: float,
    omega: float,
    eps_r: float,
    c: float = C0,
    branch: str = "auto",
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Return ``(ez, hx, hz, ex)`` on ordinates ``y``.

    ``ey`` is identically zero and is not returned. ``hx`` follows
    ``Hx = -∂y Ez / (jωμ)``. ``ex`` follows ``Ex = ∂y Hz / (jωε)``.
    ``branch`` is ``"auto"``, ``"below"``, ``"slab"``, or ``"above"``.
    The forced branches evaluate one piece of the piecewise field at a
    face, where two pieces must agree.
    """
    if y2 <= y1:
        raise ValueError(f"slab needs y2 > y1, got {y1}, {y2}")
    if omega == 0.0:
        raise ValueError("omega must be nonzero")
    ordinate = np.asarray(y, dtype=np.float64)
    k = float(omega) / c
    thickness = y2 - y1
    reflected, transmitted, n = slab_coefficients(eps_r, k, thickness)
    reflected_h, transmitted_h, _n_h = slab_coefficients(eps_r, k, thickness, te=True)
    eta = float(np.sqrt(MU0 / EPS0))
    incident = np.exp(-1j * k * (ordinate - y1))
    mirror = np.exp(1j * k * (ordinate - y1))
    into = np.exp(-1j * n * k * (ordinate - y1))
    back = np.exp(1j * n * k * (ordinate - y1))
    leaving = np.exp(-1j * k * (ordinate - y2))
    below, above = _region_masks(ordinate, y1, y2, branch)
    forward = 0.5 * (1.0 + reflected + (1.0 - reflected) / n)
    backward = 0.5 * (1.0 + reflected - (1.0 - reflected) / n)
    forward_h = 0.5 * (1.0 + reflected_h + n * (1.0 - reflected_h))
    backward_h = 0.5 * (1.0 + reflected_h - n * (1.0 - reflected_h))
    ez = _scalar_piece(
        below,
        above,
        incident,
        mirror,
        into,
        back,
        leaving,
        reflected,
        transmitted,
        forward,
        backward,
    )
    hz = _scalar_piece(
        below,
        above,
        incident,
        mirror,
        into,
        back,
        leaving,
        reflected_h,
        transmitted_h,
        forward_h,
        backward_h,
    )
    hx = (
        _difference_piece(
            below,
            above,
            incident,
            mirror,
            into,
            back,
            leaving,
            reflected,
            transmitted,
            n * forward,
            n * backward,
        )
        / eta
    )
    ex = -eta * _difference_piece(
        below,
        above,
        incident,
        mirror,
        into,
        back,
        leaving,
        reflected_h,
        transmitted_h,
        forward_h / n,
        backward_h / n,
    )
    return (
        np.asarray(ez, dtype=np.complex128),
        np.asarray(hx, dtype=np.complex128),
        np.asarray(hz, dtype=np.complex128),
        np.asarray(ex, dtype=np.complex128),
    )


def _region_masks(
    ordinate: NDArray, y1: float, y2: float, branch: str
) -> tuple[NDArray, NDArray]:
    if branch == "auto":
        return ordinate <= y1, ordinate >= y2
    zeros = np.zeros(ordinate.shape, dtype=bool)
    ones = np.ones(ordinate.shape, dtype=bool)
    if branch == "below":
        return ones, zeros
    if branch == "above":
        return zeros, ones
    if branch == "slab":
        return zeros, zeros
    raise ValueError(f"branch must be auto, below, slab, or above, got {branch!r}")


def _scalar_piece(
    below: NDArray,
    above: NDArray,
    incident: NDArray,
    mirror: NDArray,
    into: NDArray,
    back: NDArray,
    leaving: NDArray,
    reflected: complex,
    transmitted: complex,
    forward: complex,
    backward: complex,
) -> NDArray:
    return np.where(
        below,
        incident + reflected * mirror,
        np.where(above, transmitted * leaving, forward * into + backward * back),
    )


def _difference_piece(
    below: NDArray,
    above: NDArray,
    incident: NDArray,
    mirror: NDArray,
    into: NDArray,
    back: NDArray,
    leaving: NDArray,
    reflected: complex,
    transmitted: complex,
    forward: complex,
    backward: complex,
) -> NDArray:
    """Forward amplitude minus backward amplitude, region by region."""
    return np.where(
        below,
        incident - reflected * mirror,
        np.where(above, transmitted * leaving, forward * into - backward * back),
    )
