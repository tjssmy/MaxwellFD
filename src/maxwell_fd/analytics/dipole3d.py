"""Hertzian dipole in a uniform medium, ``e^{+jωt}``.

A ``z``-directed current element of moment ``I ℓ`` at the origin radiates
the outgoing spherical wave. Off the source,

    H = (j k Iℓ / 4π) (sinθ) (1 + 1/(j k r)) (e^{-j k r} / r) φ-hat,

    E_r = η (Iℓ cosθ) / (2π r²) (1 + 1/(j k r)) e^{-j k r},

    E_θ = j η (k Iℓ sinθ) / (4π r)
          (1 + 1/(j k r) - 1/(k r)²) e^{-j k r}.

``k = ω √(μ ε)`` and ``η = √(μ / ε)``. On a Yee cell the same moment is the
sample ``J_z = Iℓ / (Δx Δy Δz)``. The field diverges at the element.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.utils.constants import EPS0, MU0

ComplexArray = NDArray[np.complex128]


def hertzian_dipole_fields(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    *,
    omega: float,
    moment: complex = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mu: float = MU0,
    eps: float = EPS0,
) -> dict[str, ComplexArray]:
    """Return ``ex, ey, ez, hx, hy, hz`` of a ``z``-directed current element.

    ``moment`` is ``I ℓ``. Coordinates may be broadcast. Samples at the
    element are NaN.
    """
    if not np.isfinite(omega) or omega == 0.0:
        raise ValueError(f"omega must be a nonzero finite number, got {omega}")
    if not np.isfinite(mu) or not np.isfinite(eps) or mu <= 0.0 or eps <= 0.0:
        raise ValueError(f"mu and eps must be positive, got {mu}, {eps}")
    amplitude = complex(moment)
    if not np.isfinite(amplitude):
        raise ValueError(f"moment must be finite, got {moment}")
    if len(origin) != 3 or not all(np.isfinite(float(value)) for value in origin):
        raise ValueError(f"origin must be three finite coordinates, got {origin}")
    xx, yy, zz = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64) - float(origin[0]),
        np.asarray(y, dtype=np.float64) - float(origin[1]),
        np.asarray(z, dtype=np.float64) - float(origin[2]),
    )
    radius = np.sqrt(xx * xx + yy * yy + zz * zz)
    shape = radius.shape
    k = float(omega) * float(np.sqrt(mu * eps))
    eta = float(np.sqrt(mu / eps))
    fields = {
        name: np.full(shape, np.nan, dtype=np.complex128)
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    }
    mask = radius > 0.0
    if not np.any(mask):
        return fields
    radial = radius[mask]
    rx = xx[mask]
    ry = yy[mask]
    rz = zz[mask]
    phase = np.exp(-1j * k * radial) / radial
    near = 1.0 + 1.0 / (1j * k * radial)
    inv_r2 = 1.0 / (radial * radial)
    electric_r = (
        eta * amplitude * near * np.exp(-1j * k * radial) * inv_r2 / (2.0 * np.pi)
    )
    electric_theta = (
        1j * eta * k * amplitude * phase * (near - inv_r2 / (k * k)) / (4.0 * np.pi)
    )
    magnetic = 1j * k * amplitude * phase * near / (4.0 * np.pi)
    couple = rz * inv_r2
    fields["ex"][mask] = (electric_r + electric_theta) * couple * rx
    fields["ey"][mask] = (electric_r + electric_theta) * couple * ry
    fields["ez"][mask] = (
        electric_r * rz * rz * inv_r2 - electric_theta * (rx * rx + ry * ry) * inv_r2
    )
    fields["hx"][mask] = magnetic * (-ry * inv_r2 * radial)
    fields["hy"][mask] = magnetic * (rx * inv_r2 * radial)
    fields["hz"][mask] = 0.0
    return fields
