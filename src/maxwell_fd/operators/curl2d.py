"""Sparse and array curls for the 2D Yee cell.

``curl_e @ e`` stores ∇×E at the H degrees of freedom. ``curl_h @ h`` stores
∇×H at the E degrees of freedom. Signs match ``FD_LaTeX_Reference.tex``
(2.1)--(2.3): ``curl_h`` is the transpose of ``curl_e``, and
``curl_h diag(1/μ) curl_e`` is the positive semi-discrete curl-curl matrix.

PEC degrees of freedom omit electric samples fixed at zero. The array curls
still accept the full stored arrays; on a PEC grid they agree with the
matrices only when those boundary samples are zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DofSet:
    """One stacked component inside the E or H unknown vector."""

    name: str
    i: NDArray[np.int64]
    j: NDArray[np.int64]
    shape: tuple[int, int]

    def pack(self, field: NDArray[np.float64] | NDArray[np.complex128]) -> NDArray:
        return np.asarray(field)[self.i, self.j]

    def scatter(self, values: NDArray, fill: float = 0.0) -> NDArray:
        out = np.full(self.shape, fill, dtype=values.dtype)
        out[self.i, self.j] = values
        return out


@dataclass(frozen=True)
class Layout:
    """Degree-of-freedom order used by both curl matrices."""

    e: tuple[DofSet, ...]
    h: tuple[DofSet, ...]

    @property
    def n_e(self) -> int:
        return sum(set_.i.size for set_ in self.e)

    @property
    def n_h(self) -> int:
        return sum(set_.i.size for set_ in self.h)

    def pack_e(self, fields: dict[str, NDArray]) -> NDArray:
        return _pack(self.e, fields)

    def pack_h(self, fields: dict[str, NDArray]) -> NDArray:
        return _pack(self.h, fields)


@dataclass(frozen=True)
class CurlOperators:
    """Real sparse curls and the layout that indexes them."""

    layout: Layout
    curl_e: sparse.csr_matrix
    curl_h: sparse.csr_matrix


def build_curls(grid: YeeGrid2D) -> CurlOperators:
    """Assemble ``curl_e`` (H rows, E columns) and ``curl_h`` (E rows, H columns)."""
    layout = _layout(grid)
    if grid.polarization is Polarization.TMZ:
        curl_e, curl_h = _curls_tm(grid, layout)
    else:
        curl_e, curl_h = _curls_te(grid, layout)
    return CurlOperators(layout=layout, curl_e=curl_e, curl_h=curl_h)


def array_curl_e(
    grid: YeeGrid2D, fields: dict[str, FloatArray]
) -> dict[str, FloatArray]:
    """∇×E on the full H arrays. TMz returns ``hx`` = ∂Ez/∂y and ``hy`` = −∂Ez/∂x."""
    _check_fields(grid, fields, electric=True)
    if grid.polarization is Polarization.TMZ:
        return _array_curl_e_tm(grid, fields["ez"])
    return _array_curl_e_te(grid, fields["ex"], fields["ey"])


def array_curl_h(
    grid: YeeGrid2D, fields: dict[str, FloatArray]
) -> dict[str, FloatArray]:
    """∇×H on the full E arrays."""
    _check_fields(grid, fields, electric=False)
    if grid.polarization is Polarization.TMZ:
        return {"ez": _array_curl_h_tm(grid, fields["hx"], fields["hy"])}
    return _array_curl_h_te(grid, fields["hz"])


def _pack(sets: tuple[DofSet, ...], fields: dict[str, NDArray]) -> NDArray:
    parts = [dof.pack(fields[dof.name]) for dof in sets]
    if not parts:
        raise ValueError("no degrees of freedom")
    return np.concatenate(parts)


def _layout(grid: YeeGrid2D) -> Layout:
    if grid.polarization is Polarization.TMZ:
        return _layout_tm(grid)
    return _layout_te(grid)


def _layout_tm(grid: YeeGrid2D) -> Layout:
    shapes = grid.shapes()
    if grid.boundary is Boundary.PERIODIC:
        ez = _dof("ez", shapes["ez"], 0, grid.nx, 0, grid.ny)
        hx = _dof("hx", shapes["hx"], 0, grid.nx, 0, grid.ny)
        hy = _dof("hy", shapes["hy"], 0, grid.nx, 0, grid.ny)
    else:
        ez = _dof("ez", shapes["ez"], 1, grid.nx, 1, grid.ny)
        hx = _dof("hx", shapes["hx"], 1, grid.nx, 0, grid.ny)
        hy = _dof("hy", shapes["hy"], 0, grid.nx, 1, grid.ny)
    return Layout(e=(ez,), h=(hx, hy))


def _layout_te(grid: YeeGrid2D) -> Layout:
    shapes = grid.shapes()
    if grid.boundary is Boundary.PERIODIC:
        hz = _dof("hz", shapes["hz"], 0, grid.nx, 0, grid.ny)
        ex = _dof("ex", shapes["ex"], 0, grid.nx, 0, grid.ny)
        ey = _dof("ey", shapes["ey"], 0, grid.nx, 0, grid.ny)
    else:
        hz = _dof("hz", shapes["hz"], 0, grid.nx, 0, grid.ny)
        ex = _dof("ex", shapes["ex"], 0, grid.nx, 1, grid.ny)
        ey = _dof("ey", shapes["ey"], 1, grid.nx, 0, grid.ny)
    return Layout(e=(ex, ey), h=(hz,))


def _dof(
    name: str, shape: tuple[int, int], i0: int, i1: int, j0: int, j1: int
) -> DofSet:
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    return DofSet(name=name, i=ii.ravel(), j=jj.ravel(), shape=shape)


def _id_map(dof: DofSet) -> NDArray[np.int64]:
    ids = -np.ones(dof.shape, dtype=np.int64)
    ids[dof.i, dof.j] = np.arange(dof.i.size, dtype=np.int64)
    return ids


def _coo(
    rows: list[NDArray[np.int64]],
    cols: list[NDArray[np.int64]],
    data: list[NDArray[np.float64]],
    shape: tuple[int, int],
) -> sparse.csr_matrix:
    if not rows:
        return sparse.csr_matrix(shape, dtype=np.float64)
    return sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=shape,
        dtype=np.float64,
    ).tocsr()


def _masked(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    value: float,
    offset: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]] | None:
    """Keep stencil entries whose column id is a real degree of freedom.

    ``offset`` is added after the mask. Applying it first would turn a missing
    id of -1 into a legitimate column.
    """
    keep = cols >= 0
    if not np.any(keep):
        return None
    count = int(np.count_nonzero(keep))
    return rows[keep], cols[keep] + offset, np.full(count, value, dtype=np.float64)


def _curls_tm(
    grid: YeeGrid2D, layout: Layout
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    ez, hx, hy = layout.e[0], layout.h[0], layout.h[1]
    ez_id = _id_map(ez)
    hx_id = _id_map(hx)
    hy_id = _id_map(hy)
    n_hx = hx.i.size
    periodic = grid.boundary is Boundary.PERIODIC
    dx, dy = grid.dx, grid.dy

    e_rows: list[NDArray[np.int64]] = []
    e_cols: list[NDArray[np.int64]] = []
    e_data: list[NDArray[np.float64]] = []

    def add_e(
        block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]] | None,
    ) -> None:
        if block is not None:
            e_rows.append(block[0])
            e_cols.append(block[1])
            e_data.append(block[2])

    hx_rows = np.arange(n_hx, dtype=np.int64)
    j_hi = (hx.j + 1) % grid.ny if periodic else hx.j + 1
    add_e(_masked(hx_rows, ez_id[hx.i, j_hi], 1.0 / dy))
    add_e(_masked(hx_rows, ez_id[hx.i, hx.j], -1.0 / dy))

    hy_rows = n_hx + np.arange(hy.i.size, dtype=np.int64)
    i_hi = (hy.i + 1) % grid.nx if periodic else hy.i + 1
    add_e(_masked(hy_rows, ez_id[i_hi, hy.j], -1.0 / dx))
    add_e(_masked(hy_rows, ez_id[hy.i, hy.j], 1.0 / dx))

    h_rows: list[NDArray[np.int64]] = []
    h_cols: list[NDArray[np.int64]] = []
    h_data: list[NDArray[np.float64]] = []

    def add_h(
        block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]] | None,
    ) -> None:
        if block is not None:
            h_rows.append(block[0])
            h_cols.append(block[1])
            h_data.append(block[2])

    ez_rows = np.arange(ez.i.size, dtype=np.int64)
    i_lo = (ez.i - 1) % grid.nx if periodic else ez.i - 1
    j_lo = (ez.j - 1) % grid.ny if periodic else ez.j - 1
    add_h(_masked(ez_rows, hy_id[ez.i, ez.j], 1.0 / dx, offset=n_hx))
    add_h(_masked(ez_rows, hy_id[i_lo, ez.j], -1.0 / dx, offset=n_hx))
    add_h(_masked(ez_rows, hx_id[ez.i, ez.j], -1.0 / dy))
    add_h(_masked(ez_rows, hx_id[ez.i, j_lo], 1.0 / dy))

    curl_e = _coo(e_rows, e_cols, e_data, (layout.n_h, layout.n_e))
    curl_h = _coo(h_rows, h_cols, h_data, (layout.n_e, layout.n_h))
    return curl_e, curl_h


def _curls_te(
    grid: YeeGrid2D, layout: Layout
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    ex, ey, hz = layout.e[0], layout.e[1], layout.h[0]
    ex_id = _id_map(ex)
    ey_id = _id_map(ey)
    hz_id = _id_map(hz)
    n_ex = ex.i.size
    periodic = grid.boundary is Boundary.PERIODIC
    dx, dy = grid.dx, grid.dy

    e_rows: list[NDArray[np.int64]] = []
    e_cols: list[NDArray[np.int64]] = []
    e_data: list[NDArray[np.float64]] = []

    def add_e(
        block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]] | None,
    ) -> None:
        if block is not None:
            e_rows.append(block[0])
            e_cols.append(block[1])
            e_data.append(block[2])

    hz_rows = np.arange(hz.i.size, dtype=np.int64)
    i_hi = (hz.i + 1) % grid.nx if periodic else hz.i + 1
    j_hi = (hz.j + 1) % grid.ny if periodic else hz.j + 1
    add_e(_masked(hz_rows, ey_id[i_hi, hz.j], 1.0 / dx, offset=n_ex))
    add_e(_masked(hz_rows, ey_id[hz.i, hz.j], -1.0 / dx, offset=n_ex))
    add_e(_masked(hz_rows, ex_id[hz.i, j_hi], -1.0 / dy))
    add_e(_masked(hz_rows, ex_id[hz.i, hz.j], 1.0 / dy))

    h_rows: list[NDArray[np.int64]] = []
    h_cols: list[NDArray[np.int64]] = []
    h_data: list[NDArray[np.float64]] = []

    def add_h(
        block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]] | None,
    ) -> None:
        if block is not None:
            h_rows.append(block[0])
            h_cols.append(block[1])
            h_data.append(block[2])

    ex_rows = np.arange(n_ex, dtype=np.int64)
    j_lo = (ex.j - 1) % grid.ny if periodic else ex.j - 1
    add_h(_masked(ex_rows, hz_id[ex.i, ex.j], 1.0 / dy))
    add_h(_masked(ex_rows, hz_id[ex.i, j_lo], -1.0 / dy))

    ey_rows = n_ex + np.arange(ey.i.size, dtype=np.int64)
    i_lo = (ey.i - 1) % grid.nx if periodic else ey.i - 1
    add_h(_masked(ey_rows, hz_id[ey.i, ey.j], -1.0 / dx))
    add_h(_masked(ey_rows, hz_id[i_lo, ey.j], 1.0 / dx))

    curl_e = _coo(e_rows, e_cols, e_data, (layout.n_h, layout.n_e))
    curl_h = _coo(h_rows, h_cols, h_data, (layout.n_e, layout.n_h))
    return curl_e, curl_h


def _array_curl_e_tm(grid: YeeGrid2D, ez: FloatArray) -> dict[str, FloatArray]:
    if grid.boundary is Boundary.PERIODIC:
        d_ez_dy = (np.roll(ez, -1, axis=1) - ez) / grid.dy
        minus_d_ez_dx = -(np.roll(ez, -1, axis=0) - ez) / grid.dx
    else:
        d_ez_dy = (ez[:, 1:] - ez[:, :-1]) / grid.dy
        minus_d_ez_dx = -(ez[1:, :] - ez[:-1, :]) / grid.dx
    return {"hx": d_ez_dy, "hy": minus_d_ez_dx}


def _array_curl_h_tm(grid: YeeGrid2D, hx: FloatArray, hy: FloatArray) -> FloatArray:
    if grid.boundary is Boundary.PERIODIC:
        return (hy - np.roll(hy, 1, axis=0)) / grid.dx - (
            hx - np.roll(hx, 1, axis=1)
        ) / grid.dy
    ez = np.zeros(grid.shapes()["ez"], dtype=np.result_type(hx, hy, np.float64))
    ez[1:-1, 1:-1] = (hy[1:, 1:-1] - hy[:-1, 1:-1]) / grid.dx - (
        hx[1:-1, 1:] - hx[1:-1, :-1]
    ) / grid.dy
    return ez


def _array_curl_e_te(
    grid: YeeGrid2D, ex: FloatArray, ey: FloatArray
) -> dict[str, FloatArray]:
    if grid.boundary is Boundary.PERIODIC:
        d_ey_dx = (np.roll(ey, -1, axis=0) - ey) / grid.dx
        d_ex_dy = (np.roll(ex, -1, axis=1) - ex) / grid.dy
    else:
        d_ey_dx = (ey[1:, :] - ey[:-1, :]) / grid.dx
        d_ex_dy = (ex[:, 1:] - ex[:, :-1]) / grid.dy
    return {"hz": d_ey_dx - d_ex_dy}


def _array_curl_h_te(grid: YeeGrid2D, hz: FloatArray) -> dict[str, FloatArray]:
    if grid.boundary is Boundary.PERIODIC:
        d_hz_dy = (hz - np.roll(hz, 1, axis=1)) / grid.dy
        minus_d_hz_dx = -(hz - np.roll(hz, 1, axis=0)) / grid.dx
        return {"ex": d_hz_dy, "ey": minus_d_hz_dx}
    dtype = np.result_type(hz, np.float64)
    ex = np.zeros(grid.shapes()["ex"], dtype=dtype)
    ey = np.zeros(grid.shapes()["ey"], dtype=dtype)
    ex[:, 1:-1] = (hz[:, 1:] - hz[:, :-1]) / grid.dy
    ey[1:-1, :] = -(hz[1:, :] - hz[:-1, :]) / grid.dx
    return {"ex": ex, "ey": ey}


def _check_fields(
    grid: YeeGrid2D, fields: dict[str, FloatArray], *, electric: bool
) -> None:
    shapes = grid.shapes()
    if grid.polarization is Polarization.TMZ:
        needed = ("ez",) if electric else ("hx", "hy")
    else:
        needed = ("ex", "ey") if electric else ("hz",)
    for name in needed:
        if name not in fields:
            raise KeyError(name)
        if fields[name].shape != shapes[name]:
            raise ValueError(f"{name} shape {fields[name].shape} != {shapes[name]}")
