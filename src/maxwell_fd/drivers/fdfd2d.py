"""Frequency-domain Yee system on a 2D grid.

The factored matrix is

    curl_h @ diag(1/(j ω μ)) @ curl_e + j ω diag(ε)

applied to the free electric unknowns, and it equals ``-J`` for an impressed
current ``J`` living on those same unknowns. That is eq. (4.1) of
``FD_LaTeX_Reference.tex``. PEC samples fixed at zero are removed from the
unknown vector rather than penalized. A non-zero trace is moved to the
right-hand side, eq:dirichlet.

Real ``ε`` and ``σ`` enter the diagonal as ``ε + σ/(jω)``, eq:eps-sigma.
A complex ``ε(ω)`` replaces that sum and requires ``σ = 0``.

``spatial_operator`` returns the real pair ``(K, M)`` of eq. (4.2),
``K e = ω² M e``, for lossless media. Conductivity, a complex permittivity,
and a PML stay in the driven system. A PML multiplies each derivative by
``1/s_w(ω)`` from eq:stretch. The curls are rebuilt at the solve frequency
because that factor depends on ``ω``.

Embedded PEC and PMC objects remove samples from that same unknown vector.
PEC drops electric samples and PMC drops magnetic samples. The eigenproblem
accepts the conductors. It still refuses a PML.

A non-magnetic open dielectric keeps every sample. The caller passes the
contrast current ``J = j ω (ε − ε0) E_inc`` and the unknown is the
scattered field. The diagonal still holds the full staircase permittivity.

A resistive sheet is the same diagonal. The caller adds ``1/(Z_s Δy)`` to
``σ`` on the tangential electric samples of the Ampere cell that contains
the sheet, and the matrix uses ``ε + σ/(jω)`` with that conductivity.

A symmetric GSTC sheet is a different model. ``sheet`` splits the magnetic
sample on one face into ``H^-`` and ``H^+`` and appends that pair after the
electric unknowns. An optional ``x`` interval keeps a finite run of those
samples. ``solve`` returns the electric prefix. The jump keeps the
unstretched curl, so every cut sample must sit where the PML stretch is 1.
``chi_mm_nn`` adds ``χ ∂_x H_y`` to the TMz jump and leaves TEz unchanged.
The sheet is not combined with an embedded conductor, and the spatial
eigenproblem refuses it. Passing ``incident`` makes the unknown the
scattered field: the continuous incident wave enters through the sheet rows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from maxwell_fd.grid.yee2d import Polarization, YeeGrid2D
from maxwell_fd.materials.conductors import Conductors
from maxwell_fd.materials.sheets import SymmetricSheet
from maxwell_fd.materials.volume import TEComponents, TMComponents
from maxwell_fd.operators.curl2d import (
    CurlOperators,
    CurlScale,
    Layout,
    array_curl_e,
    array_curl_h,
    build_curls,
)
from maxwell_fd.operators.gstc import (
    _cuts,
    gstc_system,
    incident_load,
    magnetic_face,
    sheet_face,
)
from maxwell_fd.operators.pml import Grade, PMLProfile, PMLSpec

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


class FDFDOperator:
    """Sparse curls plus the constitutive samples on the free degrees of freedom."""

    def __init__(
        self,
        grid: YeeGrid2D,
        components: TMComponents | TEComponents,
        pml: PMLSpec | None = None,
        *,
        conductors: Conductors | None = None,
        sheet: SymmetricSheet | None = None,
    ) -> None:
        self.grid = grid
        self.components = components
        self.pml = pml
        self.sheet = sheet
        self.conductors = (
            conductors if conductors is not None and conductors.active else None
        )
        self.profile: PMLProfile | None = None
        if pml is not None and pml.active:
            self.profile = PMLProfile(grid, pml)
        if self.sheet is not None and self.conductors is not None:
            raise ValueError(
                "symmetric sheet is not combined with an embedded conductor"
            )
        self._fixed_e: dict[str, NDArray[np.bool_]] | None = None
        self._fixed_h: dict[str, NDArray[np.bool_]] | None = None
        if self.conductors is not None:
            masks = self.conductors.masks(grid)
            self._fixed_e = masks.pec
            self._fixed_h = masks.pmc
        self.operators: CurlOperators = build_curls(
            grid, fixed_e=self._fixed_e, fixed_h=self._fixed_h
        )
        self.layout: Layout = self.operators.layout
        self.eps_e, self.sigma_e, self.mu_h, self.eps_is_complex = _sample_diagonals(
            grid, components, self.layout
        )
        if self.sheet is not None:
            sheet_face(self.grid, self.layout, self.sheet)
            if self.profile is not None:
                _require_unstretched_sheet(
                    self.grid, self.layout, self.sheet, self.profile
                )

    def spatial_operator(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """``K`` and ``M`` of eq. (4.2), with real positive ``ε`` and ``μ``."""
        if self.sheet is not None:
            raise ValueError("spatial eigenproblem has no GSTC sheet")
        if self.profile is not None:
            raise ValueError("spatial eigenproblem uses the unstretched real curl")
        if self.eps_is_complex or np.any(self.sigma_e != 0.0):
            raise ValueError("spatial eigenproblem needs real eps and sigma = 0")
        if np.any(self.mu_h <= 0.0) or np.any(self.eps_e <= 0.0):
            raise ValueError("spatial eigenproblem needs positive real eps and mu")
        inv_mu = sparse.diags(1.0 / self.mu_h)
        stiffness = (self.operators.curl_h @ inv_mu @ self.operators.curl_e).tocsr()
        mass = sparse.diags(np.real(self.eps_e), format="csr")
        return stiffness, mass

    def system_matrix(self, omega: float) -> sparse.csr_matrix:
        """Driven matrix ``A`` of eq. (4.1), such that ``A e = -J``."""
        if omega == 0.0:
            raise ValueError("omega must be nonzero")
        permittivity = self._permittivity(omega)
        j_omega = 1j * float(omega)
        inv_j_omega_mu = sparse.diags(1.0 / (j_omega * self.mu_h))
        operators = self._operators(omega)
        curl_curl = operators.curl_h @ inv_j_omega_mu @ operators.curl_e
        base = (curl_curl + sparse.diags(j_omega * permittivity)).tocsr()
        if self.sheet is None:
            return base
        return gstc_system(
            self.grid,
            self.layout,
            operators.curl_e,
            operators.curl_h,
            self.mu_h,
            self.sheet,
            omega,
            base,
        )

    def solve(
        self,
        omega: float,
        impressed_j: NDArray,
        dirichlet: dict[str, NDArray] | None = None,
        *,
        incident: dict[str, NDArray] | None = None,
    ) -> ComplexArray:
        """Electric unknowns for ``A e = -J``, with an optional boundary trace.

        ``incident`` is the continuous incident wave on the full component
        arrays, including the tangential magnetic field. The unknown is then
        the scattered electric field and the wave enters through the sheet
        rows. TMz needs ``ez`` and ``hx``. TEz needs ``ex``, ``ey``, and ``hz``.
        """
        current = np.asarray(impressed_j, dtype=np.complex128).reshape(-1)
        if current.size != self.layout.n_e:
            raise ValueError(
                f"impressed_j has length {current.size}, expected {self.layout.n_e}"
            )
        sheet = self.sheet
        if incident is not None and sheet is None:
            raise ValueError("incident field requires a symmetric sheet")
        if incident is not None and dirichlet is not None:
            raise ValueError("incident field is not combined with a Dirichlet trace")
        rhs = -current
        if dirichlet is not None:
            rhs = rhs - self._dirichlet_load(omega, dirichlet)
        matrix = self.system_matrix(omega)
        if incident is not None and sheet is not None:
            electric, magnetic = _incident_parts(self.grid, incident)
            load = incident_load(
                self.grid, self.layout, sheet, omega, electric, magnetic
            )
            if load.size != matrix.shape[0]:
                raise RuntimeError("incident load does not match the sheet system")
            rhs = np.concatenate(
                [rhs, np.zeros(load.size - rhs.size, dtype=np.complex128)]
            )
            rhs = rhs + load
        elif matrix.shape[0] != rhs.size:
            extra = matrix.shape[0] - rhs.size
            rhs = np.concatenate([rhs, np.zeros(extra, dtype=np.complex128)])
        solved = np.asarray(spsolve(matrix, rhs), dtype=np.complex128)
        return solved[: self.layout.n_e]

    def _permittivity(self, omega: float) -> ComplexArray:
        if self.eps_is_complex:
            if np.any(self.sigma_e != 0.0):
                raise ValueError("complex eps already includes loss; sigma must be 0")
            return np.asarray(self.eps_e, dtype=np.complex128)
        return np.asarray(self.eps_e, dtype=np.complex128) + self.sigma_e / (
            1j * float(omega)
        )

    def _operators(self, omega: float) -> CurlOperators:
        scale = self._scale(omega)
        if scale is None:
            return self.operators
        return build_curls(
            self.grid, scale=scale, fixed_e=self._fixed_e, fixed_h=self._fixed_h
        )

    def _scale(self, omega: float) -> CurlScale | None:
        if self.profile is None:
            return None
        return self.profile.scale(omega)

    def _dirichlet_load(
        self, omega: float, dirichlet: dict[str, NDArray]
    ) -> ComplexArray:
        boundary = {
            dof.name: _boundary_only(dof.i, dof.j, dirichlet[dof.name], dof.shape)
            for dof in self.layout.e
        }
        return self.layout.pack_e(self._array_lhs(omega, boundary))

    def _array_lhs(
        self, omega: float, electric: dict[str, NDArray]
    ) -> dict[str, NDArray]:
        """Left-hand side of eq. (4.1) on the full electric arrays."""
        scale = self._scale(omega)
        scaled = _scale_magnetic(
            self.grid,
            array_curl_e(self.grid, electric, scale=scale),
            self.components,
            omega,
        )
        scaled = _zero_fixed(scaled, self._fixed_h)
        ampere = array_curl_h(self.grid, scaled, scale=scale)
        permittivity = _permittivity_fields(self.components, omega, self.eps_is_complex)
        return {
            name: ampere[name] + 1j * float(omega) * permittivity[name] * electric[name]
            for name in ampere
        }


def _require_unstretched_sheet(
    grid: YeeGrid2D,
    layout: Layout,
    sheet: SymmetricSheet,
    profile: PMLProfile,
) -> None:
    """Refuse a cut whose own sample sits inside a PML band."""
    face = magnetic_face(grid, sheet.y)
    cuts, _index = _cuts(grid, layout, face, sheet)
    x_grade = profile.x_node if grid.polarization is Polarization.TMZ else profile.x_mid
    y_grade = profile.y_mid
    stretched = _graded(y_grade, face) or any(_graded(x_grade, cut.i) for cut in cuts)
    if stretched:
        raise ValueError("sheet must stay in the unstretched region")


def _graded(grade: Grade, index: int) -> bool:
    return float(grade.sigma[index]) != 0.0 or float(grade.kappa[index]) != 1.0


def _incident_parts(
    grid: YeeGrid2D, incident: dict[str, NDArray]
) -> tuple[dict[str, NDArray], NDArray]:
    shapes = grid.shapes()
    if grid.polarization is Polarization.TMZ:
        _present(incident, "ez", shapes["ez"])
        _present(incident, "hx", shapes["hx"])
        return {"ez": incident["ez"]}, incident["hx"]
    _present(incident, "ex", shapes["ex"])
    _present(incident, "ey", shapes["ey"])
    _present(incident, "hz", shapes["hz"])
    return {"ex": incident["ex"], "ey": incident["ey"]}, incident["hz"]


def _present(incident: dict[str, NDArray], name: str, shape: tuple[int, int]) -> None:
    if name not in incident:
        raise ValueError(f"incident field needs {name}")
    array = np.asarray(incident[name])
    if array.shape != shape:
        raise ValueError(f"incident {name} shape {array.shape} != {shape}")


def _zero_fixed(
    fields: dict[str, NDArray], fixed: dict[str, NDArray[np.bool_]] | None
) -> dict[str, NDArray]:
    """Set PMC magnetic samples to zero so the array Ampere matches the sparse curl."""
    if not fixed:
        return fields
    out: dict[str, NDArray] = {}
    for name, values in fields.items():
        mask = fixed.get(name)
        if mask is None or not np.any(mask):
            out[name] = values
            continue
        copied = np.array(values, copy=True)
        copied[mask] = 0.0
        out[name] = copied
    return out


def _boundary_only(
    free_i: NDArray, free_j: NDArray, values: NDArray, shape: tuple[int, int]
) -> ComplexArray:
    field = np.asarray(values, dtype=np.complex128)
    if field.shape != shape:
        raise ValueError(f"dirichlet shape {field.shape} != {shape}")
    kept = field.copy()
    kept[free_i, free_j] = 0.0
    return kept


def _scale_magnetic(
    grid: YeeGrid2D,
    curls: dict[str, NDArray],
    components: TMComponents | TEComponents,
    omega: float,
) -> dict[str, NDArray]:
    scale = 1.0 / (1j * float(omega))
    if grid.polarization is Polarization.TMZ:
        assert isinstance(components, TMComponents)
        return {
            "hx": curls["hx"] * scale / components.mu_x,
            "hy": curls["hy"] * scale / components.mu_y,
        }
    assert isinstance(components, TEComponents)
    return {"hz": curls["hz"] * scale / components.mu_z}


def _permittivity_fields(
    components: TMComponents | TEComponents, omega: float, eps_is_complex: bool
) -> dict[str, NDArray]:
    def combine(eps: NDArray, sigma: NDArray) -> ComplexArray:
        if eps_is_complex:
            return np.asarray(eps, dtype=np.complex128)
        return np.asarray(eps, dtype=np.complex128) + np.asarray(sigma) / (
            1j * float(omega)
        )

    if isinstance(components, TMComponents):
        return {"ez": combine(components.eps_z, components.sigma_z)}
    return {
        "ex": combine(components.eps_x, components.sigma_x),
        "ey": combine(components.eps_y, components.sigma_y),
    }


def _sample_diagonals(
    grid: YeeGrid2D,
    components: TMComponents | TEComponents,
    layout: Layout,
) -> tuple[NDArray, RealArray, RealArray, bool]:
    if grid.polarization is Polarization.TMZ:
        if not isinstance(components, TMComponents):
            raise TypeError("TMz grid requires TMComponents")
        _match(components.eps_z, grid.shapes()["ez"], "eps_z")
        _match(components.mu_x, grid.shapes()["hx"], "mu_x")
        _match(components.mu_y, grid.shapes()["hy"], "mu_y")
        _match(components.sigma_z, grid.shapes()["ez"], "sigma_z")
        eps_e = layout.e[0].pack(components.eps_z)
        sigma_e = layout.e[0].pack(components.sigma_z).astype(np.float64, copy=False)
        mu_h = np.concatenate(
            [layout.h[0].pack(components.mu_x), layout.h[1].pack(components.mu_y)]
        ).astype(np.float64, copy=False)
        return eps_e, sigma_e, mu_h, bool(np.iscomplexobj(eps_e))
    if not isinstance(components, TEComponents):
        raise TypeError("TEz grid requires TEComponents")
    _match(components.eps_x, grid.shapes()["ex"], "eps_x")
    _match(components.eps_y, grid.shapes()["ey"], "eps_y")
    _match(components.mu_z, grid.shapes()["hz"], "mu_z")
    _match(components.sigma_x, grid.shapes()["ex"], "sigma_x")
    _match(components.sigma_y, grid.shapes()["ey"], "sigma_y")
    eps_e = np.concatenate(
        [layout.e[0].pack(components.eps_x), layout.e[1].pack(components.eps_y)]
    )
    sigma_e = np.concatenate(
        [layout.e[0].pack(components.sigma_x), layout.e[1].pack(components.sigma_y)]
    ).astype(np.float64, copy=False)
    mu_h = layout.h[0].pack(components.mu_z).astype(np.float64, copy=False)
    complex_eps = bool(
        np.iscomplexobj(components.eps_x) or np.iscomplexobj(components.eps_y)
    )
    if complex_eps:
        eps_e = np.asarray(eps_e, dtype=np.complex128)
    return eps_e, sigma_e, mu_h, complex_eps


def _match(field: NDArray, shape: tuple[int, int], name: str) -> None:
    if field.shape != shape:
        raise ValueError(f"{name} shape {field.shape} != {shape}")
