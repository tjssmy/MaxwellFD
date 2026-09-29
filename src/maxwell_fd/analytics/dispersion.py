"""Fully discrete Yee dispersion and the leapfrog recurrence.

On a uniform grid the time frequency of a spatial mode ``(k_x, k_y)`` satisfies

    (sin(ω Δt / 2) / (c Δt))²
        = (sin(k_x Δx / 2) / Δx)² + (sin(k_y Δy / 2) / Δy)².

A pure spatial eigenmode sampled at successive integer or half-integer steps
obeys the scalar leapfrog recurrence, so three samples determine ``ω``.
The dispersion relation is eq. (3.3) of ``FD_LaTeX_Reference.tex``.
"""

from __future__ import annotations

import math

import numpy as np

from maxwell_fd.utils.constants import C0


def fully_discrete_omega(
    *,
    kx: float,
    ky: float,
    dx: float,
    dy: float,
    dt: float,
    c: float = C0,
) -> float:
    """Leapfrog frequency of a plane-wave mode, eq. (3.3). Raises past the mode's limit."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    spatial = (math.sin(kx * dx / 2.0) / dx) ** 2 + (math.sin(ky * dy / 2.0) / dy) ** 2
    argument = c * dt * math.sqrt(spatial)
    if argument > 1.0:
        raise ValueError(f"mode is past the leapfrog limit: sin argument {argument}")
    return (2.0 / dt) * math.asin(argument)


def leapfrog_omega_from_samples(a0: float, a1: float, a2: float, dt: float) -> float:
    """Frequency from three samples of one leapfrog eigenmode, Sec. 3.2 of the note."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if a1 == 0.0:
        raise ValueError("middle sample is zero; the recurrence does not determine ω")
    omega0_sq = -((a2 - 2.0 * a1 + a0) / (dt**2 * a1))
    if omega0_sq < 0.0:
        raise ValueError(f"samples are not a stable oscillation, ω0²={omega0_sq}")
    argument = 0.5 * dt * math.sqrt(omega0_sq)
    if argument > 1.0:
        raise ValueError(f"samples imply an unstable step, sin argument {argument}")
    return (2.0 / dt) * math.asin(argument)


def cfl_number(dx: float, dy: float, dt: float, c: float = C0) -> float:
    """Courant number ``c Δt √(Δx⁻² + Δy⁻²)``, eq. (3.1). Stable leapfrog needs this ≤ 1."""
    return float(c * dt * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
