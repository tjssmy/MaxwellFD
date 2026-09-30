"""Two-dimensional line-current field in a uniform medium.

The outgoing cylindrical wave for ``e^{+jωt}`` is ``H_0^{(2)}``. A
z-directed current whose integral is ``current`` amperes produces

    (∇² + k²) E_z = j ω μ J_z,
    E_z = -(ω μ current / 4) H_0^{(2)}(k ρ),

with ``k = ω √(μ ε)`` and ``ρ`` the distance from the source. On the Yee
grid that current is the sample ``J_z = current / (Δx Δy)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import hankel2

from maxwell_fd.utils.constants import EPS0, MU0

ComplexArray = NDArray[np.complex128]


def line_current_ez(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    omega: float,
    current: float = 1.0,
    mu: float = MU0,
    eps: float = EPS0,
) -> ComplexArray:
    """``E_z`` of a line current at the origin of ``(x, y)``.

    The sample at ``ρ = 0`` is left as NaN. The field diverges there.
    """
    if omega == 0.0:
        raise ValueError("omega must be nonzero")
    if mu <= 0.0 or eps <= 0.0:
        raise ValueError("mu and eps must be positive")
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    rho = np.hypot(xx, yy)
    field = np.full(rho.shape, np.nan, dtype=np.complex128)
    mask = rho > 0.0
    k = float(omega) * float(np.sqrt(mu * eps))
    field[mask] = -0.25 * float(omega) * mu * current * hankel2(0, k * rho[mask])
    return field
