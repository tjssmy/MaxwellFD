"""Two-dimensional paraxial Gaussian beam, ``e^{+jωt}``.

A beam of amplitude ``A`` and waist radius ``w0`` propagates in a unit
direction ``k-hat``. With ``z`` the distance along that direction from the
waist, ``ρ`` the transverse distance, ``z_R = k w0² / 2``, and
``q = z + j z_R``,

    E_z = A √(j z_R / q) exp(-j k z - j k ρ² / (2 q)).

The Gouy factor is the cylindrical one, half of the three-dimensional
phase. ``H`` is the local plane-wave field ``(k-hat × z-hat) E_z / η``.
On a Yee grid the beam is the incident field of a horizontal
total-field/scattered-field split: the impressed current sits on the two
``Ez`` rows that touch the split.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.utils.constants import ETA0

ComplexArray = NDArray[np.complex128]


def gaussian_beam_tmz(
    x: NDArray,
    y: NDArray,
    *,
    k: float,
    w0: float,
    origin: tuple[float, float] = (0.0, 0.0),
    direction: tuple[float, float] = (0.0, 1.0),
    amplitude: complex = 1.0,
    eta: float = ETA0,
) -> dict[str, ComplexArray]:
    """Return ``ez``, ``hx``, and ``hy`` of a paraxial TMz Gaussian beam.

    ``w0`` is the ``1/e`` amplitude radius at the waist. ``origin`` is the
    waist center. Coordinates may be broadcast.
    """
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"k must be positive, got {k}")
    if not np.isfinite(w0) or w0 <= 0.0:
        raise ValueError(f"w0 must be positive, got {w0}")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError(f"eta must be positive, got {eta}")
    scale = complex(amplitude)
    if not np.isfinite(scale):
        raise ValueError(f"amplitude must be finite, got {amplitude}")
    if len(origin) != 2 or not all(np.isfinite(float(value)) for value in origin):
        raise ValueError(f"origin must be two finite coordinates, got {origin}")
    axis = np.asarray(direction, dtype=np.float64).reshape(-1)
    if axis.size != 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"direction must be two finite components, got {direction}")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-15:
        raise ValueError(f"direction must be nonzero, got {direction}")
    axis = axis / norm
    xx = np.asarray(x, dtype=np.float64) - float(origin[0])
    yy = np.asarray(y, dtype=np.float64) - float(origin[1])
    z_coord = xx * axis[0] + yy * axis[1]
    transverse_x = xx - z_coord * axis[0]
    transverse_y = yy - z_coord * axis[1]
    rho2 = transverse_x * transverse_x + transverse_y * transverse_y
    rayleigh = 0.5 * float(k) * float(w0) * float(w0)
    q_factor = z_coord + 1j * rayleigh
    field = scale * np.sqrt((1j * rayleigh) / q_factor)
    field = field * np.exp(
        -1j * float(k) * z_coord - 1j * float(k) * rho2 / (2.0 * q_factor)
    )
    ez = np.asarray(field, dtype=np.complex128)
    return {
        "ez": ez,
        "hx": np.asarray(axis[1] * ez / float(eta), dtype=np.complex128),
        "hy": np.asarray(-axis[0] * ez / float(eta), dtype=np.complex128),
    }
