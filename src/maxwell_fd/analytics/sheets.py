"""Reflection and transmission of a uniform sheet.

The coefficients are eq:R and eq:T of ``FD_LaTeX_Reference.tex``. TMz is
the s polarization, which MaxwellMOM calls TE. TEz is the p polarization,
which that code calls TM. ``θ`` is measured from the sheet normal. A
resistive sheet is the pure-electric reduction ``χ_mm = 0`` with
``χ_ee = η0 / (j k Z_s)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from maxwell_fd.utils.constants import C0, ETA0

ComplexArray = NDArray[np.complex128]


def chi_ee_from_zs(zs: complex, k: float) -> complex:
    """``χ_ee = η0 / (j k Z_s)``, eq:chi-from-z."""
    impedance = complex(zs)
    if impedance == 0:
        raise ValueError("zs must be nonzero")
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"k must be positive, got {k}")
    return ETA0 / (1j * float(k) * impedance)


def sheet_coefficients(
    chi_ee: complex,
    chi_mm: complex,
    k: float,
    *,
    theta: float = 0.0,
    te: bool = False,
    chi_mm_nn: complex = 0.0,
) -> tuple[complex, complex]:
    """Return ``(R, T)`` from eq:R and eq:T.

    ``te=False`` is TMz and ``te=True`` is TEz. ``theta`` is in radians.
    ``chi_mm_nn`` is added to the TMz electric coefficient, eq:chi-nn.
    """
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"k must be positive, got {k}")
    if not np.isfinite(theta) or abs(theta) >= 0.5 * np.pi:
        raise ValueError(f"|theta| must be below π/2, got {theta}")
    cosine = float(np.cos(theta))
    if abs(cosine) < 1e-14:
        raise ValueError("cos(theta) is zero")
    alpha = 1j * float(k) * complex(chi_ee) / 2.0
    beta = 1j * float(k) * complex(chi_mm) / 2.0
    if te:
        alpha, beta = alpha * cosine, beta / cosine
    else:
        alpha, beta = alpha / cosine, beta * cosine
        normal = complex(chi_mm_nn)
        if normal != 0:
            sine = float(np.sin(theta))
            alpha = alpha + 1j * float(k) * sine**2 * normal / (2.0 * cosine)
    electric = (1.0 - alpha) / (1.0 + alpha)
    magnetic = (1.0 - beta) / (1.0 + beta)
    reflected = 0.5 * (electric - magnetic)
    transmitted = 0.5 * (electric + magnetic)
    return complex(reflected), complex(transmitted)


def resistive_coefficients(
    zs: float, k: float, *, theta: float = 0.0, te: bool = False
) -> tuple[complex, complex]:
    """``(R, T)`` of a resistive sheet, ``χ_mm = 0``."""
    if not np.isfinite(zs) or zs <= 0.0:
        raise ValueError(f"zs must be positive, got {zs}")
    return sheet_coefficients(chi_ee_from_zs(zs, k), 0.0, k, theta=theta, te=te)


def uniform_sheet_fields(
    y: NDArray,
    *,
    y_sheet: float,
    omega: float,
    reflected: complex,
    transmitted: complex,
    branch: str = "auto",
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Return ``(ez, hx, hz, ex)`` for a normal-incidence uniform sheet.

    The incident electric amplitude is 1 and its phase is zero on the sheet.
    The wave arrives from ``y < y_sheet``. ``branch`` is ``"auto"``,
    ``"below"``, or ``"above"``. The automatic split uses the transmitted
    side for ``y >= y_sheet``. ``hx = (forward - backward)/η0`` and
    ``hz = -hx``, the electric-amplitude convention shared by both
    polarizations.
    """
    if omega == 0.0:
        raise ValueError("omega must be nonzero")
    if branch not in {"auto", "below", "above"}:
        raise ValueError(f"branch must be auto, below, or above, got {branch}")
    ordinate = np.asarray(y, dtype=np.float64)
    k = float(omega) / C0
    delta = ordinate - float(y_sheet)
    incident = np.exp(-1j * k * delta)
    mirror = np.exp(1j * k * delta)
    if branch == "below":
        illuminated = np.ones(ordinate.shape, dtype=bool)
    elif branch == "above":
        illuminated = np.zeros(ordinate.shape, dtype=bool)
    else:
        illuminated = ordinate < float(y_sheet)
    coefficient_r = complex(reflected)
    coefficient_t = complex(transmitted)
    electric = np.where(
        illuminated, incident + coefficient_r * mirror, coefficient_t * incident
    )
    forward = np.where(illuminated, incident, coefficient_t * incident)
    backward = np.where(illuminated, coefficient_r * mirror, 0.0)
    magnetic = (forward - backward) / ETA0
    return electric, magnetic, -magnetic, electric


def oblique_sheet_fields(
    x: NDArray,
    y: NDArray,
    *,
    y_sheet: float,
    omega: float,
    reflected: complex,
    transmitted: complex,
    theta: float = 0.0,
    x_ref: float = 0.0,
) -> dict[str, ComplexArray]:
    """Uniform-sheet fields at an angle ``theta`` from the normal.

    ``x`` and ``y`` are one component's sample coordinates. The returned
    arrays are that mesh, so a staggered component is evaluated with its own
    coordinates. The incident electric amplitude is 1 and its phase is zero
    at ``(x_ref, y_sheet)``. The wave arrives from ``y < y_sheet``. ``R`` and
    ``T`` multiply that amplitude. Below the sheet the field is the incident
    wave plus the reflected wave; on and above the sheet it is the
    transmitted wave. At ``theta = 0`` the tangential fields match
    ``uniform_sheet_fields``.
    """
    if omega == 0.0:
        raise ValueError("omega must be nonzero")
    if not np.isfinite(theta) or abs(theta) >= 0.5 * np.pi:
        raise ValueError(f"|theta| must be below π/2, got {theta}")
    abscissa = np.asarray(x, dtype=np.float64)
    ordinate = np.asarray(y, dtype=np.float64)
    xx, yy = np.meshgrid(abscissa, ordinate, indexing="ij")
    k = float(omega) / C0
    sine = float(np.sin(theta))
    cosine = float(np.cos(theta))
    dx = xx - float(x_ref)
    dy = yy - float(y_sheet)
    forward = np.exp(-1j * k * (sine * dx + cosine * dy))
    backward = np.exp(-1j * k * (sine * dx - cosine * dy))
    below = yy < float(y_sheet)
    coefficient_r = complex(reflected)
    coefficient_t = complex(transmitted)
    going = np.where(below, forward, coefficient_t * forward)
    coming = np.where(below, coefficient_r * backward, 0.0)
    amplitude = going + coming
    magnetic = (going - coming) / ETA0
    return {
        "ez": amplitude,
        "hx": cosine * magnetic,
        "hy": -sine * amplitude / ETA0,
        "ex": cosine * amplitude,
        "ey": sine * (coming - going),
        "hz": -magnetic,
    }


def resistive_sheet_fields(
    y: NDArray,
    *,
    y_sheet: float,
    omega: float,
    zs: float,
    branch: str = "auto",
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Return ``(ez, hx, hz, ex)`` for a normal-incidence resistive sheet.

    The incident electric amplitude is 1 and its phase is zero on the sheet.
    The wave arrives from ``y < y_sheet``. ``branch`` is ``"auto"``,
    ``"below"``, or ``"above"``. The automatic split uses the transmitted
    side on the sheet itself, where ``E`` is continuous and ``H`` jumps.
    """
    if omega == 0.0:
        raise ValueError("omega must be nonzero")
    reflected, transmitted = resistive_coefficients(zs, float(omega) / C0)
    return uniform_sheet_fields(
        y,
        y_sheet=y_sheet,
        omega=omega,
        reflected=reflected,
        transmitted=transmitted,
        branch=branch,
    )
