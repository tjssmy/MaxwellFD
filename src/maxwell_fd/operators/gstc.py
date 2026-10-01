"""Symmetric GSTC cell on one magnetic face.

The sheet lies halfway between tangential electric rows ``j`` and ``j+1``.
The Yee magnetic sample on that face is removed and replaced by the pair
``H^-``, ``H^+``. Equations ``eq:sc-h`` and ``eq:sc-e`` close the pair for
TMz; TEz uses the same susceptibilities with the cross-product signs of a
``y``-normal sheet. The bulk curl-curl contribution of the removed sample
is subtracted, and the two side values re-enter the tangential Ampere rows.
TEz also returns their average to the normal electric samples on the cut.
The unknowns appended after the electric vector are ``(H^-, H^+)`` at each
free magnetic location on the face.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.sheets import SymmetricSheet
from maxwell_fd.operators.curl2d import Layout
from maxwell_fd.utils.constants import EPS0, MU0

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class _Cut:
    e_minus: int
    e_plus: int
    ey_here: int
    ey_right: int


def sheet_face(grid: YeeGrid2D, layout: Layout, sheet: SymmetricSheet) -> int:
    """Magnetic-row index of ``sheet``, or an error when that face is unusable."""
    face = magnetic_face(grid, sheet.y)
    _cuts(grid, layout, face)
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
    cuts, h_index = _cuts(grid, layout, face)
    reduced = _without_cut(curl_e, curl_h, mu_h, h_index, omega, base)
    return _augment(grid, sheet, omega, reduced, cuts)


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
    alpha = 1j * float(omega) * EPS0 * complex(sheet.chi_ee) / 2.0
    beta = 1j * float(omega) * MU0 * complex(sheet.chi_mm) / 2.0
    dy = float(grid.dy)
    dx = float(grid.dx)
    tez = grid.polarization is Polarization.TEZ
    for k, cut in enumerate(cuts):
        hm = n_e + 2 * k
        hp = hm + 1
        if tez:
            _add(rows, cols, data, cut.e_minus, hm, -1.0 / dy)
            _add(rows, cols, data, cut.e_plus, hp, 1.0 / dy)
            _average(rows, cols, data, cut.ey_here, hm, hp, 0.5 / dx)
            _average(rows, cols, data, cut.ey_right, hm, hp, -0.5 / dx)
            _equation(
                rows,
                cols,
                data,
                n_e + 2 * k,
                (cut.e_minus, cut.e_plus, hm, hp),
                (-alpha, -alpha, -1.0, 1.0),
            )
            _equation(
                rows,
                cols,
                data,
                n_e + 2 * k + 1,
                (cut.e_minus, cut.e_plus, hm, hp),
                (-1.0, 1.0, -beta, -beta),
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
                (alpha, alpha, -1.0, 1.0),
            )
            _equation(
                rows,
                cols,
                data,
                n_e + 2 * k + 1,
                (cut.e_minus, cut.e_plus, hm, hp),
                (-1.0, 1.0, beta, beta),
            )
    size = n_e + 2 * n_cut
    return sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
        dtype=np.complex128,
    ).tocsr()


def _cuts(grid: YeeGrid2D, layout: Layout, face: int) -> tuple[list[_Cut], IntArray]:
    periodic = grid.boundary is Boundary.PERIODIC
    electric = _global_ids(layout.e)
    magnetic = _global_ids(layout.h)
    if grid.polarization is Polarization.TMZ:
        ez_id = electric[0]
        hx_id = magnetic[0]
        j_plus = (face + 1) % grid.ny if periodic else face + 1
        cuts: list[_Cut] = []
        indices: list[int] = []
        for i in range(hx_id.shape[0]):
            h = int(hx_id[i, face])
            if h < 0:
                continue
            e_minus = int(ez_id[i, face])
            e_plus = int(ez_id[i, j_plus]) if j_plus < ez_id.shape[1] else -1
            if e_minus < 0 or e_plus < 0:
                raise ValueError("sheet lies on the PEC wall")
            cuts.append(_Cut(e_minus, e_plus, -1, -1))
            indices.append(h)
    else:
        ex_id = electric[0]
        ey_id = electric[1]
        hz_id = magnetic[0]
        j_plus = (face + 1) % grid.ny if periodic else face + 1
        cuts = []
        indices = []
        for i in range(hz_id.shape[0]):
            h = int(hz_id[i, face])
            if h < 0:
                continue
            e_minus = int(ex_id[i, face])
            e_plus = int(ex_id[i, j_plus]) if j_plus < ex_id.shape[1] else -1
            if e_minus < 0 or e_plus < 0:
                raise ValueError("sheet lies on the PEC wall")
            ey_here = int(ey_id[i, face]) if i < ey_id.shape[0] else -1
            right = i + 1
            ey_right = int(ey_id[right, face]) if right < ey_id.shape[0] else -1
            cuts.append(_Cut(e_minus, e_plus, ey_here, ey_right))
            indices.append(h)
    if not cuts:
        raise ValueError("sheet face has no magnetic sample")
    return cuts, np.asarray(indices, dtype=np.int64)


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
    value: float,
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
