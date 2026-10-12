"""Symmetric GSTC cell on one magnetic face.

The sheet lies halfway between tangential electric rows ``j`` and ``j+1``.
The Yee magnetic sample on that face is removed and replaced by the pair
``H^-``, ``H^+``. Equations ``eq:sc-h`` and ``eq:sc-e`` close the pair for
TMz; TEz uses the same susceptibilities with the cross-product signs of a
``y``-normal sheet. The bulk curl-curl contribution of the removed sample
is subtracted, and the two side values re-enter the tangential Ampere rows.
TEz also returns their average to the normal electric samples on the cut.
The unknowns appended after the electric vector are ``(H^-, H^+)`` at each
free magnetic location on the face. A half-open ``x`` interval keeps a
finite run of those locations. The default is the whole face. The jump
uses the unstretched stencil, so a cut sample inside a PML is refused.
``chi_mm_nn`` adds ``χ ∂_x H_y`` to the TMz magnetic jump. ``H_y`` is not
split: the derivative is the average of the Faraday samples on the two
electric rows, written as a second difference of ``E_z``. TEz ignores it.
``chi_em`` and ``chi_me`` add ``(jω/c) χ (n̂ × field)`` to both
polarizations. ``chi_em`` multiplies the averaged tangential ``H`` in the
magnetic jump and ``chi_me`` multiplies the averaged tangential ``E`` in
the electric jump. A piecewise sheet evaluates each susceptibility on
the piece that contains the cut sample. The tangential, normal, and
magneto-electric stencils are unchanged. ``chi_mm_2`` adds a centered
second difference of the appended tangential ``H`` to the electric jump,
in both polarizations. On a plane wave ``k_t²`` is ``−∂_xx``. A neighbor
that is not a cut is refused when that coefficient is nonzero, and a
wrap uses the same Bloch weight as the normal derivative.
``incident_load`` moves a continuous incident wave onto the sheet rows.
The electric rows of that load stay zero: the incident field already
satisfies the bulk stencil up to Yee dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from maxwell_fd.grid.yee2d import Polarization, YeeGrid2D
from maxwell_fd.materials.sheets import SymmetricSheet
from maxwell_fd.operators.curl2d import Layout
from maxwell_fd.utils.constants import C0, EPS0, MU0

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class _Cut:
    e_minus: int
    e_plus: int
    ey_here: int
    ey_right: int
    i: int


def sheet_face(grid: YeeGrid2D, layout: Layout, sheet: SymmetricSheet) -> int:
    """Magnetic-row index of ``sheet``, or an error when that face is unusable."""
    face = magnetic_face(grid, sheet.y)
    _cuts(grid, layout, face, sheet)
    return face


def gstc_system(
    grid: YeeGrid2D,
    layout: Layout,
    curl_e: sparse.csr_matrix,
    curl_h: sparse.csr_matrix,
    mu_h: NDArray,
    sheet: SymmetricSheet,
    omega: float,
    base: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Electric system plus two magnetic unknowns on every cut sample."""
    if omega == 0.0:
        raise ValueError("omega must be nonzero")
    face = magnetic_face(grid, sheet.y)
    cuts, h_index = _cuts(grid, layout, face, sheet)
    reduced = _without_cut(curl_e, curl_h, mu_h, h_index, omega, base)
    return _augment(grid, layout, sheet, omega, reduced, cuts)


def incident_load(
    grid: YeeGrid2D,
    layout: Layout,
    sheet: SymmetricSheet,
    omega: float,
    electric: dict[str, NDArray],
    magnetic: NDArray,
) -> NDArray:
    """Augmented right-hand side for a scattered field driven by one incident wave.

    ``electric`` is the incident tangential electric field on the full
    arrays. ``magnetic`` is the incident tangential ``H`` on the full ``Hx``
    or ``Hz`` array. The wave is continuous, so ``H^+ = H^-``. The sheet
    rows receive minus that wave's GSTC residual. The electric rows stay
    zero.
    """
    if omega == 0.0:
        raise ValueError("omega must be nonzero")
    face = magnetic_face(grid, sheet.y)
    cuts, _h_index = _cuts(grid, layout, face, sheet)
    cut_at = {cut.i: k for k, cut in enumerate(cuts)}
    free = _free_magnetic(grid, layout, face)
    periodic_y = grid.periodic_axes()[1]
    j_plus = (face + 1) % grid.ny if periodic_y else face + 1
    sample = "hz" if grid.polarization is Polarization.TEZ else "hx"
    x_samples = grid.coordinates(sample)[0]
    rhs = np.zeros(layout.n_e + 2 * len(cuts), dtype=np.complex128)
    if grid.polarization is Polarization.TEZ:
        tangential = _complex_field(electric, "ex", grid.shapes()["ex"])
        normal_h = _complex_field({"hz": magnetic}, "hz", grid.shapes()["hz"])
        for k, cut in enumerate(cuts):
            alpha, beta, kappa_em, kappa_me, _gamma, eta = _jump_coefficients(
                _chi_at(sheet, float(x_samples[cut.i])), omega, grid.dx
            )
            e_minus = tangential[cut.i, face]
            e_plus = tangential[cut.i, j_plus]
            h_face = normal_h[cut.i, face]
            rhs[layout.n_e + 2 * k] = (
                alpha * (e_minus + e_plus) + 2.0 * kappa_em * h_face
            )
            rhs[layout.n_e + 2 * k + 1] = (
                -(e_plus - e_minus)
                + _dispersive_h_load(
                    grid, cut_at, free, normal_h, face, cut.i, beta, eta, -1.0
                )
                - kappa_me * (e_minus + e_plus)
            )
        return rhs
    tangential = _complex_field(electric, "ez", grid.shapes()["ez"])
    normal_h = _complex_field({"hx": magnetic}, "hx", grid.shapes()["hx"])
    ez_id = _global_ids(layout.e)[0] if sheet.carries_normal() else None
    for k, cut in enumerate(cuts):
        alpha, beta, kappa_em, kappa_me, gamma, eta = _jump_coefficients(
            _chi_at(sheet, float(x_samples[cut.i])), omega, grid.dx
        )
        e_minus = tangential[cut.i, face]
        e_plus = tangential[cut.i, j_plus]
        h_face = normal_h[cut.i, face]
        normal = 0.0
        if gamma != 0 and ez_id is not None:
            normal = _nn_on_field(grid, ez_id, tangential, cut.i, face, j_plus, gamma)
        rhs[layout.n_e + 2 * k] = (
            -alpha * (e_minus + e_plus) - normal + 2.0 * kappa_em * h_face
        )
        rhs[layout.n_e + 2 * k + 1] = (
            -(e_plus - e_minus)
            + _dispersive_h_load(
                grid, cut_at, free, normal_h, face, cut.i, beta, eta, 1.0
            )
            - kappa_me * (e_minus + e_plus)
        )
    return rhs


def magnetic_face(grid: YeeGrid2D, y_sheet: float) -> int:
    """Index of the magnetic row whose coordinate is ``y_sheet``."""
    component = "hx" if grid.polarization is Polarization.TMZ else "hz"
    y = grid.coordinates(component)[1]
    tolerance = 1e-8 * float(grid.dy)
    hit = np.flatnonzero(np.abs(y - float(y_sheet)) <= tolerance)
    if hit.size != 1:
        raise ValueError("sheet must lie on one magnetic face")
    return int(hit[0])


def _without_cut(
    curl_e: sparse.csr_matrix,
    curl_h: sparse.csr_matrix,
    mu_h: NDArray,
    h_index: IntArray,
    omega: float,
    base: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Remove ``curl_h diag(1/(jωμ)) curl_e`` on the cut magnetic samples."""
    update = None
    for index in h_index:
        factor = 1.0 / (1j * float(omega) * float(mu_h[int(index)]))
        column = curl_h.getcol(int(index))
        row = curl_e.getrow(int(index))
        term = (column @ row).astype(np.complex128) * factor
        update = term if update is None else update + term
    if update is None:
        raise ValueError("sheet face has no magnetic sample")
    return (base.astype(np.complex128) - update).tocsr()


def _augment(
    grid: YeeGrid2D,
    layout: Layout,
    sheet: SymmetricSheet,
    omega: float,
    reduced: sparse.csr_matrix,
    cuts: list[_Cut],
) -> sparse.csr_matrix:
    n_e = reduced.shape[0]
    n_cut = len(cuts)
    coo = reduced.tocoo()
    rows = [coo.row.astype(np.int64, copy=False)]
    cols = [coo.col.astype(np.int64, copy=False)]
    data: list[NDArray] = [coo.data.astype(np.complex128, copy=False)]
    dy = float(grid.dy)
    dx = float(grid.dx)
    tez = grid.polarization is Polarization.TEZ
    ez_id = _global_ids(layout.e)[0] if sheet.carries_normal() and not tez else None
    periodic_x, periodic_y = grid.periodic_axes()
    _forward, backward = grid.bloch_factors()
    face = magnetic_face(grid, sheet.y)
    j_plus = (face + 1) % grid.ny if periodic_y else face + 1
    sample = "hz" if tez else "hx"
    x_samples = grid.coordinates(sample)[0]
    cut_at = {cut.i: index for index, cut in enumerate(cuts)}
    free = _free_magnetic(grid, layout, face)
    for k, cut in enumerate(cuts):
        alpha, beta, kappa_em, kappa_me, gamma, eta = _jump_coefficients(
            _chi_at(sheet, float(x_samples[cut.i])), omega, dx
        )
        hm = n_e + 2 * k
        hp = hm + 1
        side = -1.0 if tez else 1.0
        center = side * (beta + 2.0 * eta)
        if tez:
            _add(rows, cols, data, cut.e_minus, hm, -1.0 / dy)
            _add(rows, cols, data, cut.e_plus, hp, 1.0 / dy)
            _average(rows, cols, data, cut.ey_here, hm, hp, 0.5 / dx)
            right_phase = backward if periodic_x and cut.i == grid.nx - 1 else 1.0
            _average(rows, cols, data, cut.ey_right, hm, hp, -0.5 / dx * right_phase)
            _equation(
                rows,
                cols,
                data,
                n_e + 2 * k,
                (cut.e_minus, cut.e_plus, hm, hp),
                (-alpha, -alpha, -1.0 - kappa_em, 1.0 - kappa_em),
            )
        else:
            _add(rows, cols, data, cut.e_minus, hm, 1.0 / dy)
            _add(rows, cols, data, cut.e_plus, hp, -1.0 / dy)
            _equation(
                rows,
                cols,
                data,
                n_e + 2 * k,
                (cut.e_minus, cut.e_plus, hm, hp),
                (alpha, alpha, -1.0 - kappa_em, 1.0 - kappa_em),
            )
            if gamma != 0 and ez_id is not None:
                for column, weight in _nn_columns(grid, ez_id, cut.i, face, j_plus):
                    _add(rows, cols, data, hm, column, gamma * weight)
        _equation(
            rows,
            cols,
            data,
            n_e + 2 * k + 1,
            (cut.e_minus, cut.e_plus, hm, hp),
            (-1.0 + kappa_me, 1.0 + kappa_me, center, center),
        )
        if eta != 0:
            _add_mm2_neighbors(
                rows,
                cols,
                data,
                grid,
                cut_at,
                free,
                n_e,
                cut.i,
                n_e + 2 * k + 1,
                eta,
                side,
            )
    size = n_e + 2 * n_cut
    return sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
        dtype=np.complex128,
    ).tocsr()


def _cross_kappa(chi: complex, omega: float) -> complex:
    """``j ω χ / (2 c)`` for one magneto-electric susceptibility."""
    value = complex(chi)
    if value == 0:
        return 0j
    return 1j * float(omega) * value / (2.0 * C0)


def _chi_at(
    sheet: SymmetricSheet, x: float
) -> tuple[complex, complex, complex, complex, complex, complex]:
    found = sheet.susceptibility(x)
    if found is None:
        raise RuntimeError("a cut sample has no susceptibility")
    return found


def _jump_coefficients(
    chi: tuple[complex, complex, complex, complex, complex, complex],
    omega: float,
    dx: float,
) -> tuple[complex, complex, complex, complex, complex, complex]:
    """``(α, β, κ_em, κ_me, γ, η)`` of one cut face."""
    chi_ee, chi_mm, chi_nn, chi_em, chi_me, chi_mm_2 = chi
    alpha = 1j * float(omega) * EPS0 * complex(chi_ee) / 2.0
    beta = 1j * float(omega) * MU0 * complex(chi_mm) / 2.0
    return (
        alpha,
        beta,
        _cross_kappa(chi_em, omega),
        _cross_kappa(chi_me, omega),
        _nn_gamma(chi_nn, omega, dx),
        _mm2_eta(chi_mm_2, omega, dx),
    )


def _mm2_eta(chi_mm_2: complex, omega: float, dx: float) -> complex:
    """``j ω μ0 χ_mm2 / (2 Δx²)`` on the electric jump.

    ``χ_mm(k_t) = χ_mm0 + χ_mm2 k_t²`` replaces ``χ_mm H`` by
    ``χ_mm0 H − χ_mm2 ∂_xx H``. The centered second difference puts
    ``+2η`` on the cut and ``−η`` on each neighbor, with the sign of
    ``β``: positive for TMz and negative for TEz.
    """
    strength = complex(chi_mm_2)
    if strength == 0:
        return 0j
    spacing = float(dx)
    return 1j * float(omega) * MU0 * strength / (2.0 * spacing * spacing)


def _mm2_neighbors(
    grid: YeeGrid2D, i: int
) -> tuple[tuple[int, complex], tuple[int, complex]]:
    """Left and right magnetic samples, with the Bloch weight on a wrap."""
    periodic_x = grid.periodic_axes()[0]
    forward, backward = grid.bloch_factors()
    if periodic_x:
        left = (i - 1) % grid.nx
        right = (i + 1) % grid.nx
        left_weight = backward if i == 0 else 1.0
        right_weight = forward if i == grid.nx - 1 else 1.0
        return (left, left_weight), (right, right_weight)
    left = i - 1 if i > 0 else -1
    right = i + 1 if i + 1 < grid.nx else -1
    return (left, 1.0), (right, 1.0)


def _free_magnetic(grid: YeeGrid2D, layout: Layout, face: int) -> set[int]:
    """Sample indices whose tangential ``H`` is a degree of freedom."""
    magnetic = _global_ids(layout.h)[0]
    return {i for i in range(magnetic.shape[0]) if int(magnetic[i, face]) >= 0}


def _require_neighbor(index: int, cut_at: dict[int, int], free: set[int]) -> int:
    if index < 0 or index not in free:
        raise ValueError("chi_mm_2 reaches the PEC wall")
    if index not in cut_at:
        raise ValueError("chi_mm_2 reaches past the sheet")
    return cut_at[index]


def _add_mm2_neighbors(
    rows: list[NDArray],
    cols: list[NDArray],
    data: list[NDArray],
    grid: YeeGrid2D,
    cut_at: dict[int, int],
    free: set[int],
    n_e: int,
    i: int,
    row: int,
    eta: complex,
    side: float,
) -> None:
    for index, weight in _mm2_neighbors(grid, i):
        neighbor = _require_neighbor(index, cut_at, free)
        coeff = side * (-eta) * weight
        hm = n_e + 2 * neighbor
        _add(rows, cols, data, row, hm, coeff)
        _add(rows, cols, data, row, hm + 1, coeff)


def _dispersive_h_load(
    grid: YeeGrid2D,
    cut_at: dict[int, int],
    free: set[int],
    magnetic: NDArray,
    face: int,
    i: int,
    beta: complex,
    eta: complex,
    side: float,
) -> complex:
    """Minus the electric jump's tangential-H part, with ``H^+ = H^-``."""
    h_face = complex(magnetic[i, face])
    if eta == 0:
        return -2.0 * side * beta * h_face
    (left, left_weight), (right, right_weight) = _mm2_neighbors(grid, i)
    _require_neighbor(left, cut_at, free)
    _require_neighbor(right, cut_at, free)
    h_left = complex(magnetic[left, face])
    h_right = complex(magnetic[right, face])
    return (
        -2.0 * side * (beta + 2.0 * eta) * h_face
        + 2.0 * side * eta * left_weight * h_left
        + 2.0 * side * eta * right_weight * h_right
    )


def _nn_gamma(chi_nn: complex, omega: float, dx: float) -> complex:
    """Coefficient of one ``E_z`` second difference in the TMz jump.

    ``H_y`` from Faraday is ``-∂_x E_z / (j ω μ_0)``. Averaging the two
    electric rows and differentiating in ``x`` puts
    ``χ_mm^nn / (2 j ω μ_0 Δx²)`` on ``E_z[i+1] - 2 E_z[i] + E_z[i-1]``.
    """
    normal = complex(chi_nn)
    if normal == 0:
        return 0j
    return normal / (2.0 * 1j * float(omega) * MU0 * float(dx) * float(dx))


def _nn_points(
    grid: YeeGrid2D, i: int, face: int, j_plus: int
) -> list[tuple[int, int, complex]]:
    """``(i, j, weight)`` of the two-row second difference, before ``γ``."""
    periodic_x = grid.periodic_axes()[0]
    forward, backward = grid.bloch_factors()
    if periodic_x:
        left, right = (i - 1) % grid.nx, (i + 1) % grid.nx
        left_weight = backward if i == 0 else 1.0
        right_weight = forward if i == grid.nx - 1 else 1.0
    else:
        left, right = i - 1, i + 1
        left_weight, right_weight = 1.0, 1.0
    points: list[tuple[int, int, complex]] = []
    for j in (face, j_plus):
        points.append((left, j, left_weight))
        points.append((i, j, -2.0))
        points.append((right, j, right_weight))
    return points


def _nn_column(grid: YeeGrid2D, ez_id: IntArray, index: int, j: int) -> int:
    periodic_x = grid.periodic_axes()[0]
    if periodic_x:
        return int(ez_id[index, j])
    if index < 0 or index >= ez_id.shape[0] or j < 0 or j >= ez_id.shape[1]:
        return -1
    return int(ez_id[index, j])


def _nn_columns(
    grid: YeeGrid2D, ez_id: IntArray, i: int, face: int, j_plus: int
) -> list[tuple[int, complex]]:
    columns: list[tuple[int, complex]] = []
    for index, j, weight in _nn_points(grid, i, face, j_plus):
        column = _nn_column(grid, ez_id, index, j)
        if column < 0:
            raise ValueError("chi_mm_nn reaches the PEC wall")
        columns.append((column, weight))
    return columns


def _nn_on_field(
    grid: YeeGrid2D,
    ez_id: IntArray,
    ez: NDArray,
    i: int,
    face: int,
    j_plus: int,
    gamma: complex,
) -> complex:
    """``χ ∂_x H_y`` of one incident ``E_z``, on the free samples only."""
    total = 0j
    for index, j, weight in _nn_points(grid, i, face, j_plus):
        column = _nn_column(grid, ez_id, index, j)
        if column < 0:
            raise ValueError("chi_mm_nn reaches the PEC wall")
        total += weight * complex(ez[index, j])
    return gamma * total


def _cuts(
    grid: YeeGrid2D,
    layout: Layout,
    face: int,
    sheet: SymmetricSheet | None = None,
) -> tuple[list[_Cut], IntArray]:
    periodic_x, periodic_y = grid.periodic_axes()
    electric = _global_ids(layout.e)
    magnetic = _global_ids(layout.h)
    if grid.polarization is Polarization.TMZ:
        sample = "hx"
        ez_id = electric[0]
        hx_id = magnetic[0]
        x = grid.coordinates(sample)[0]
        j_plus = (face + 1) % grid.ny if periodic_y else face + 1
        cuts: list[_Cut] = []
        indices: list[int] = []
        for i in range(hx_id.shape[0]):
            if not _on_sheet(float(x[i]), sheet):
                continue
            h = int(hx_id[i, face])
            if h < 0:
                continue
            e_minus = int(ez_id[i, face])
            e_plus = int(ez_id[i, j_plus]) if j_plus < ez_id.shape[1] else -1
            if e_minus < 0 or e_plus < 0:
                raise ValueError("sheet lies on the PEC wall")
            cuts.append(_Cut(e_minus, e_plus, -1, -1, i))
            indices.append(h)
    else:
        sample = "hz"
        ex_id = electric[0]
        ey_id = electric[1]
        hz_id = magnetic[0]
        x = grid.coordinates(sample)[0]
        j_plus = (face + 1) % grid.ny if periodic_y else face + 1
        cuts = []
        indices = []
        for i in range(hz_id.shape[0]):
            if not _on_sheet(float(x[i]), sheet):
                continue
            h = int(hz_id[i, face])
            if h < 0:
                continue
            e_minus = int(ex_id[i, face])
            e_plus = int(ex_id[i, j_plus]) if j_plus < ex_id.shape[1] else -1
            if e_minus < 0 or e_plus < 0:
                raise ValueError("sheet lies on the PEC wall")
            ey_here = int(ey_id[i, face]) if i < ey_id.shape[0] else -1
            right = (i + 1) % grid.nx if periodic_x else i + 1
            ey_right = int(ey_id[right, face]) if right < ey_id.shape[0] else -1
            cuts.append(_Cut(e_minus, e_plus, ey_here, ey_right, i))
            indices.append(h)
    if not cuts:
        if sheet is not None and sheet.x0 is not None:
            raise ValueError("sheet span misses every magnetic sample")
        raise ValueError("sheet face has no magnetic sample")
    return cuts, np.asarray(indices, dtype=np.int64)


def _on_sheet(x: float, sheet: SymmetricSheet | None) -> bool:
    if sheet is None:
        return True
    return sheet.susceptibility(x) is not None


def _complex_field(
    fields: dict[str, NDArray], name: str, shape: tuple[int, int]
) -> NDArray:
    if name not in fields:
        raise ValueError(f"incident field needs {name}")
    array = np.asarray(fields[name], dtype=np.complex128)
    if array.shape != shape:
        raise ValueError(f"incident {name} shape {array.shape} != {shape}")
    return array


def _global_ids(dofs: tuple) -> list[IntArray]:
    maps: list[IntArray] = []
    offset = 0
    for dof in dofs:
        ids = -np.ones(dof.shape, dtype=np.int64)
        ids[dof.i, dof.j] = np.arange(dof.i.size, dtype=np.int64) + offset
        maps.append(ids)
        offset += int(dof.i.size)
    return maps


def _add(
    rows: list[NDArray],
    cols: list[NDArray],
    data: list[NDArray],
    row: int,
    col: int,
    value: complex,
) -> None:
    if row < 0:
        return
    rows.append(np.asarray([row], dtype=np.int64))
    cols.append(np.asarray([col], dtype=np.int64))
    data.append(np.asarray([value], dtype=np.complex128))


def _average(
    rows: list[NDArray],
    cols: list[NDArray],
    data: list[NDArray],
    row: int,
    hm: int,
    hp: int,
    value: complex,
) -> None:
    _add(rows, cols, data, row, hm, value)
    _add(rows, cols, data, row, hp, value)


def _equation(
    rows: list[NDArray],
    cols: list[NDArray],
    data: list[NDArray],
    row: int,
    columns: tuple[int, int, int, int],
    values: tuple[complex, complex, complex, complex],
) -> None:
    rows.append(np.full(4, row, dtype=np.int64))
    cols.append(np.asarray(columns, dtype=np.int64))
    data.append(np.asarray(values, dtype=np.complex128))
