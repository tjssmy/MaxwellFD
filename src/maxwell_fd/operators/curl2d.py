"""Sparse and array curls for the 2D Yee cell.

``curl_e @ e`` stores ∇×E at the H degrees of freedom. ``curl_h @ h`` stores
∇×H at the E degrees of freedom. Signs match ``FD_LaTeX_Reference.tex``
(2.1)--(2.3): with no scale, ``curl_h`` is the transpose of ``curl_e``, and
``curl_h diag(1/μ) curl_e`` is the positive semi-discrete curl-curl matrix.

PEC degrees of freedom omit electric samples fixed at zero, including an
embedded conductor passed as ``fixed_e``. PMC magnetic samples passed as
``fixed_h`` are omitted the same way. The array curls still accept the full
stored arrays; they agree with the matrices only when the omitted samples
are zero.

An optional :class:`CurlScale` multiplies each derivative by the CFS factor
``1/s_w`` of eq:stretch, evaluated where that derivative is centered. A
varying scale is complex and the two curls are then not transposes, because
``s_w`` at an E sample and at the neighboring H sample differ by half a cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from maxwell_fd.grid.yee2d import Polarization, YeeGrid2D

FloatArray = NDArray[np.float64]
FieldArray = NDArray[np.float64] | NDArray[np.complex128]


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
    """Sparse curls and the layout that indexes them."""

    layout: Layout
    curl_e: sparse.csr_matrix
    curl_h: sparse.csr_matrix


@dataclass(frozen=True)
class CurlScale:
    """``1/s_w`` on each derivative, sampled where that derivative is centered.

    TMz stores ``∂Ez/∂y`` at ``Hx`` and ``∂Ez/∂x`` at ``Hy``. The Ampere
    derivatives ``∂Hy/∂x`` and ``∂Hx/∂y`` are centered at ``Ez``. TEz stores
    ``∂Ey/∂x`` and ``∂Ex/∂y`` at ``Hz``, ``∂Hz/∂y`` at ``Ex``, and ``∂Hz/∂x``
    at ``Ey``. Missing arrays mean a factor of one. A polarization needs its
    whole set.
    """

    tm_y_on_hx: FieldArray | None = None
    tm_x_on_hy: FieldArray | None = None
    tm_x_on_ez: FieldArray | None = None
    tm_y_on_ez: FieldArray | None = None
    te_x_on_hz: FieldArray | None = None
    te_y_on_hz: FieldArray | None = None
    te_y_on_ex: FieldArray | None = None
    te_x_on_ey: FieldArray | None = None


def build_curls(
    grid: YeeGrid2D,
    *,
    scale: CurlScale | None = None,
    fixed_e: dict[str, NDArray[np.bool_]] | None = None,
    fixed_h: dict[str, NDArray[np.bool_]] | None = None,
) -> CurlOperators:
    """Assemble ``curl_e`` (H rows, E columns) and ``curl_h`` (E rows, H columns).

    ``fixed_e`` and ``fixed_h`` are boolean masks, true on samples that stay
    out of the unknown vector. The outer PEC wall is already out. An embedded
    PEC adds electric samples and an embedded PMC adds magnetic samples.
    """
    layout = _layout(grid, fixed_e, fixed_h)
    if layout.n_e == 0:
        raise ValueError("conductor removes every electric unknown")
    if grid.polarization is Polarization.TMZ:
        curl_e, curl_h = _curls_tm(grid, layout, scale)
    else:
        curl_e, curl_h = _curls_te(grid, layout, scale)
    return CurlOperators(layout=layout, curl_e=curl_e, curl_h=curl_h)


def array_curl_e(
    grid: YeeGrid2D,
    fields: dict[str, FieldArray],
    *,
    scale: CurlScale | None = None,
) -> dict[str, FieldArray]:
    """∇×E on the full H arrays. TMz returns ``hx`` = ∂Ez/∂y and ``hy`` = −∂Ez/∂x."""
    _check_fields(grid, fields, electric=True)
    if grid.polarization is Polarization.TMZ:
        return _array_curl_e_tm(grid, fields["ez"], scale)
    return _array_curl_e_te(grid, fields["ex"], fields["ey"], scale)


def array_curl_h(
    grid: YeeGrid2D,
    fields: dict[str, FieldArray],
    *,
    scale: CurlScale | None = None,
) -> dict[str, FieldArray]:
    """∇×H on the full E arrays."""
    _check_fields(grid, fields, electric=False)
    if grid.polarization is Polarization.TMZ:
        return {"ez": _array_curl_h_tm(grid, fields["hx"], fields["hy"], scale)}
    return _array_curl_h_te(grid, fields["hz"], scale)


def _pack(sets: tuple[DofSet, ...], fields: dict[str, NDArray]) -> NDArray:
    parts = [dof.pack(fields[dof.name]) for dof in sets]
    if not parts:
        raise ValueError("no degrees of freedom")
    return np.concatenate(parts)


def _layout(
    grid: YeeGrid2D,
    fixed_e: dict[str, NDArray[np.bool_]] | None,
    fixed_h: dict[str, NDArray[np.bool_]] | None,
) -> Layout:
    if grid.polarization is Polarization.TMZ:
        return _layout_tm(grid, fixed_e, fixed_h)
    return _layout_te(grid, fixed_e, fixed_h)


def _layout_tm(
    grid: YeeGrid2D,
    fixed_e: dict[str, NDArray[np.bool_]] | None,
    fixed_h: dict[str, NDArray[np.bool_]] | None,
) -> Layout:
    shapes = grid.shapes()
    px, py = grid.periodic_axes()
    ez = _dof("ez", shapes["ez"], 0 if px else 1, grid.nx, 0 if py else 1, grid.ny)
    hx = _dof("hx", shapes["hx"], 0 if px else 1, grid.nx, 0, grid.ny)
    hy = _dof("hy", shapes["hy"], 0, grid.nx, 0 if py else 1, grid.ny)
    return Layout(
        e=(_drop(ez, fixed_e),),
        h=(_drop(hx, fixed_h), _drop(hy, fixed_h)),
    )


def _layout_te(
    grid: YeeGrid2D,
    fixed_e: dict[str, NDArray[np.bool_]] | None,
    fixed_h: dict[str, NDArray[np.bool_]] | None,
) -> Layout:
    shapes = grid.shapes()
    px, py = grid.periodic_axes()
    hz = _dof("hz", shapes["hz"], 0, grid.nx, 0, grid.ny)
    ex = _dof("ex", shapes["ex"], 0, grid.nx, 0 if py else 1, grid.ny)
    ey = _dof("ey", shapes["ey"], 0 if px else 1, grid.nx, 0, grid.ny)
    return Layout(
        e=(_drop(ex, fixed_e), _drop(ey, fixed_e)),
        h=(_drop(hz, fixed_h),),
    )


def _drop(dof: DofSet, fixed: dict[str, NDArray[np.bool_]] | None) -> DofSet:
    """Remove samples marked true in ``fixed`` from one component."""
    if fixed is None or dof.name not in fixed:
        return dof
    mask = np.asarray(fixed[dof.name], dtype=bool)
    if mask.shape != dof.shape:
        raise ValueError(f"{dof.name} fixed-sample shape {mask.shape} != {dof.shape}")
    keep = ~mask[dof.i, dof.j]
    return DofSet(name=dof.name, i=dof.i[keep], j=dof.j[keep], shape=dof.shape)


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
    data: list[NDArray],
    shape: tuple[int, int],
) -> sparse.csr_matrix:
    if not rows:
        return sparse.csr_matrix(shape, dtype=np.float64)
    values = np.concatenate(data)
    dtype = np.complex128 if np.iscomplexobj(values) else np.float64
    return sparse.coo_matrix(
        (
            values.astype(dtype, copy=False),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=shape,
        dtype=dtype,
    ).tocsr()


def _masked(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    value: float | FieldArray,
    offset: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray] | None:
    """Keep stencil entries whose column id is a real degree of freedom.

    ``offset`` is added after the mask. Applying it first would turn a missing
    id of -1 into a legitimate column. ``value`` is either one coefficient or
    one coefficient per row, aligned with ``rows`` before the mask.
    """
    keep = cols >= 0
    if not np.any(keep):
        return None
    raw = np.asarray(value)
    if raw.ndim == 0:
        data = np.full(
            int(np.count_nonzero(keep)), raw, dtype=np.result_type(raw, np.float64)
        )
    else:
        if raw.shape != rows.shape:
            raise ValueError(f"stencil weight shape {raw.shape} != {rows.shape}")
        data = raw[keep]
    return rows[keep], cols[keep] + offset, data


def _weights(
    factor: FieldArray | None, coeff: float, i: NDArray, j: NDArray
) -> float | FieldArray:
    """Per-row stencil weight. A missing factor leaves the bare coefficient."""
    if factor is None:
        return coeff
    return coeff * factor[i, j]


def _curls_tm(
    grid: YeeGrid2D, layout: Layout, scale: CurlScale | None
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    ez, hx, hy = layout.e[0], layout.h[0], layout.h[1]
    sy_hx, sx_hy, sx_ez, sy_ez = _tm_factors(scale, grid)
    ez_id = _id_map(ez)
    hx_id = _id_map(hx)
    hy_id = _id_map(hy)
    n_hx = hx.i.size
    px, py = grid.periodic_axes()
    dx, dy = grid.dx, grid.dy

    e_rows: list[NDArray[np.int64]] = []
    e_cols: list[NDArray[np.int64]] = []
    e_data: list[NDArray] = []

    def add_e(
        block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray] | None,
    ) -> None:
        if block is not None:
            e_rows.append(block[0])
            e_cols.append(block[1])
            e_data.append(block[2])

    hx_rows = np.arange(n_hx, dtype=np.int64)
    j_hi = (hx.j + 1) % grid.ny if py else hx.j + 1
    why = _weights(sy_hx, 1.0 / dy, hx.i, hx.j)
    add_e(_masked(hx_rows, ez_id[hx.i, j_hi], why))
    add_e(_masked(hx_rows, ez_id[hx.i, hx.j], -why))

    hy_rows = n_hx + np.arange(hy.i.size, dtype=np.int64)
    i_hi = (hy.i + 1) % grid.nx if px else hy.i + 1
    whx = _weights(sx_hy, 1.0 / dx, hy.i, hy.j)
    add_e(_masked(hy_rows, ez_id[i_hi, hy.j], -whx))
    add_e(_masked(hy_rows, ez_id[hy.i, hy.j], whx))

    h_rows: list[NDArray[np.int64]] = []
    h_cols: list[NDArray[np.int64]] = []
    h_data: list[NDArray] = []

    def add_h(
        block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray] | None,
    ) -> None:
        if block is not None:
            h_rows.append(block[0])
            h_cols.append(block[1])
            h_data.append(block[2])

    ez_rows = np.arange(ez.i.size, dtype=np.int64)
    i_lo = (ez.i - 1) % grid.nx if px else ez.i - 1
    j_lo = (ez.j - 1) % grid.ny if py else ez.j - 1
    wx = _weights(sx_ez, 1.0 / dx, ez.i, ez.j)
    wy = _weights(sy_ez, 1.0 / dy, ez.i, ez.j)
    add_h(_masked(ez_rows, hy_id[ez.i, ez.j], wx, offset=n_hx))
    add_h(_masked(ez_rows, hy_id[i_lo, ez.j], -wx, offset=n_hx))
    add_h(_masked(ez_rows, hx_id[ez.i, ez.j], -wy))
    add_h(_masked(ez_rows, hx_id[ez.i, j_lo], wy))

    curl_e = _coo(e_rows, e_cols, e_data, (layout.n_h, layout.n_e))
    curl_h = _coo(h_rows, h_cols, h_data, (layout.n_e, layout.n_h))
    return curl_e, curl_h


def _curls_te(
    grid: YeeGrid2D, layout: Layout, scale: CurlScale | None
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    ex, ey, hz = layout.e[0], layout.e[1], layout.h[0]
    sx_hz, sy_hz, sy_ex, sx_ey = _te_factors(scale, grid)
    ex_id = _id_map(ex)
    ey_id = _id_map(ey)
    hz_id = _id_map(hz)
    n_ex = ex.i.size
    px, py = grid.periodic_axes()
    dx, dy = grid.dx, grid.dy

    e_rows: list[NDArray[np.int64]] = []
    e_cols: list[NDArray[np.int64]] = []
    e_data: list[NDArray] = []

    def add_e(
        block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray] | None,
    ) -> None:
        if block is not None:
            e_rows.append(block[0])
            e_cols.append(block[1])
            e_data.append(block[2])

    hz_rows = np.arange(hz.i.size, dtype=np.int64)
    i_hi = (hz.i + 1) % grid.nx if px else hz.i + 1
    j_hi = (hz.j + 1) % grid.ny if py else hz.j + 1
    wx = _weights(sx_hz, 1.0 / dx, hz.i, hz.j)
    wy = _weights(sy_hz, 1.0 / dy, hz.i, hz.j)
    add_e(_masked(hz_rows, ey_id[i_hi, hz.j], wx, offset=n_ex))
    add_e(_masked(hz_rows, ey_id[hz.i, hz.j], -wx, offset=n_ex))
    add_e(_masked(hz_rows, ex_id[hz.i, j_hi], -wy))
    add_e(_masked(hz_rows, ex_id[hz.i, hz.j], wy))

    h_rows: list[NDArray[np.int64]] = []
    h_cols: list[NDArray[np.int64]] = []
    h_data: list[NDArray] = []

    def add_h(
        block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray] | None,
    ) -> None:
        if block is not None:
            h_rows.append(block[0])
            h_cols.append(block[1])
            h_data.append(block[2])

    ex_rows = np.arange(n_ex, dtype=np.int64)
    j_lo = (ex.j - 1) % grid.ny if py else ex.j - 1
    wy_ex = _weights(sy_ex, 1.0 / dy, ex.i, ex.j)
    add_h(_masked(ex_rows, hz_id[ex.i, ex.j], wy_ex))
    add_h(_masked(ex_rows, hz_id[ex.i, j_lo], -wy_ex))

    ey_rows = n_ex + np.arange(ey.i.size, dtype=np.int64)
    i_lo = (ey.i - 1) % grid.nx if px else ey.i - 1
    wx_ey = _weights(sx_ey, 1.0 / dx, ey.i, ey.j)
    add_h(_masked(ey_rows, hz_id[ey.i, ey.j], -wx_ey))
    add_h(_masked(ey_rows, hz_id[i_lo, ey.j], wx_ey))

    curl_e = _coo(e_rows, e_cols, e_data, (layout.n_h, layout.n_e))
    curl_h = _coo(h_rows, h_cols, h_data, (layout.n_e, layout.n_h))
    return curl_e, curl_h


def _array_curl_e_tm(
    grid: YeeGrid2D, ez: FieldArray, scale: CurlScale | None
) -> dict[str, FieldArray]:
    px, py = grid.periodic_axes()
    d_ez_dy = (
        (np.roll(ez, -1, axis=1) - ez) / grid.dy
        if py
        else (ez[:, 1:] - ez[:, :-1]) / grid.dy
    )
    minus_d_ez_dx = (
        -(np.roll(ez, -1, axis=0) - ez) / grid.dx
        if px
        else -(ez[1:, :] - ez[:-1, :]) / grid.dx
    )
    sy_hx, sx_hy, _, _ = _tm_factors(scale, grid)
    if sy_hx is not None:
        d_ez_dy = d_ez_dy * sy_hx
    if sx_hy is not None:
        minus_d_ez_dx = minus_d_ez_dx * sx_hy
    return {"hx": d_ez_dy, "hy": minus_d_ez_dx}


def _array_curl_h_tm(
    grid: YeeGrid2D, hx: FieldArray, hy: FieldArray, scale: CurlScale | None
) -> FieldArray:
    _, _, sx_ez, sy_ez = _tm_factors(scale, grid)
    px, py = grid.periodic_axes()
    if px and py:
        d_hy_dx = (hy - np.roll(hy, 1, axis=0)) / grid.dx
        d_hx_dy = (hx - np.roll(hx, 1, axis=1)) / grid.dy
    elif px:
        dtype = np.result_type(hy, hx, np.float64)
        d_hy_dx = np.zeros(grid.shapes()["ez"], dtype=dtype)
        d_hx_dy = np.zeros(grid.shapes()["ez"], dtype=dtype)
        d_hy_dx[:, 1:-1] = (hy[:, 1:-1] - np.roll(hy[:, 1:-1], 1, axis=0)) / grid.dx
        d_hx_dy[:, 1:-1] = (hx[:, 1:] - hx[:, :-1]) / grid.dy
    else:
        d_hy_dx = np.zeros(grid.shapes()["ez"], dtype=np.result_type(hy, np.float64))
        d_hx_dy = np.zeros(grid.shapes()["ez"], dtype=np.result_type(hx, np.float64))
        d_hy_dx[1:-1, 1:-1] = (hy[1:, 1:-1] - hy[:-1, 1:-1]) / grid.dx
        d_hx_dy[1:-1, 1:-1] = (hx[1:-1, 1:] - hx[1:-1, :-1]) / grid.dy
    if sx_ez is not None:
        d_hy_dx = d_hy_dx * sx_ez
    if sy_ez is not None:
        d_hx_dy = d_hx_dy * sy_ez
    return d_hy_dx - d_hx_dy


def _array_curl_e_te(
    grid: YeeGrid2D, ex: FieldArray, ey: FieldArray, scale: CurlScale | None
) -> dict[str, FieldArray]:
    px, py = grid.periodic_axes()
    d_ey_dx = (
        (np.roll(ey, -1, axis=0) - ey) / grid.dx
        if px
        else (ey[1:, :] - ey[:-1, :]) / grid.dx
    )
    d_ex_dy = (
        (np.roll(ex, -1, axis=1) - ex) / grid.dy
        if py
        else (ex[:, 1:] - ex[:, :-1]) / grid.dy
    )
    sx_hz, sy_hz, _, _ = _te_factors(scale, grid)
    if sx_hz is not None:
        d_ey_dx = d_ey_dx * sx_hz
    if sy_hz is not None:
        d_ex_dy = d_ex_dy * sy_hz
    return {"hz": d_ey_dx - d_ex_dy}


def _array_curl_h_te(
    grid: YeeGrid2D, hz: FieldArray, scale: CurlScale | None
) -> dict[str, FieldArray]:
    _, _, sy_ex, sx_ey = _te_factors(scale, grid)
    px, py = grid.periodic_axes()
    if px and py:
        d_hz_dy = (hz - np.roll(hz, 1, axis=1)) / grid.dy
        minus_d_hz_dx = -(hz - np.roll(hz, 1, axis=0)) / grid.dx
    elif px:
        dtype = np.result_type(hz, np.float64)
        d_hz_dy = np.zeros(grid.shapes()["ex"], dtype=dtype)
        d_hz_dy[:, 1:-1] = (hz[:, 1:] - hz[:, :-1]) / grid.dy
        minus_d_hz_dx = -(hz - np.roll(hz, 1, axis=0)) / grid.dx
    else:
        dtype = np.result_type(hz, np.float64)
        d_hz_dy = np.zeros(grid.shapes()["ex"], dtype=dtype)
        minus_d_hz_dx = np.zeros(grid.shapes()["ey"], dtype=dtype)
        d_hz_dy[:, 1:-1] = (hz[:, 1:] - hz[:, :-1]) / grid.dy
        minus_d_hz_dx[1:-1, :] = -(hz[1:, :] - hz[:-1, :]) / grid.dx
    if sy_ex is not None:
        d_hz_dy = d_hz_dy * sy_ex
    if sx_ey is not None:
        minus_d_hz_dx = minus_d_hz_dx * sx_ey
    return {"ex": d_hz_dy, "ey": minus_d_hz_dx}


def _tm_factors(
    scale: CurlScale | None, grid: YeeGrid2D
) -> tuple[FieldArray | None, FieldArray | None, FieldArray | None, FieldArray | None]:
    if scale is None:
        return None, None, None, None
    shapes = grid.shapes()
    sy_hx = _factor(scale.tm_y_on_hx, shapes["hx"], "tm_y_on_hx")
    sx_hy = _factor(scale.tm_x_on_hy, shapes["hy"], "tm_x_on_hy")
    sx_ez = _factor(scale.tm_x_on_ez, shapes["ez"], "tm_x_on_ez")
    sy_ez = _factor(scale.tm_y_on_ez, shapes["ez"], "tm_y_on_ez")
    present = (
        sy_hx is not None,
        sx_hy is not None,
        sx_ez is not None,
        sy_ez is not None,
    )
    if not any(present):
        raise ValueError("TMz curl scale is empty")
    if not all(present):
        raise ValueError("TMz curl scale needs all four derivative multipliers")
    return sy_hx, sx_hy, sx_ez, sy_ez


def _te_factors(
    scale: CurlScale | None, grid: YeeGrid2D
) -> tuple[FieldArray | None, FieldArray | None, FieldArray | None, FieldArray | None]:
    if scale is None:
        return None, None, None, None
    shapes = grid.shapes()
    sx_hz = _factor(scale.te_x_on_hz, shapes["hz"], "te_x_on_hz")
    sy_hz = _factor(scale.te_y_on_hz, shapes["hz"], "te_y_on_hz")
    sy_ex = _factor(scale.te_y_on_ex, shapes["ex"], "te_y_on_ex")
    sx_ey = _factor(scale.te_x_on_ey, shapes["ey"], "te_x_on_ey")
    present = (
        sx_hz is not None,
        sy_hz is not None,
        sy_ex is not None,
        sx_ey is not None,
    )
    if not any(present):
        raise ValueError("TEz curl scale is empty")
    if not all(present):
        raise ValueError("TEz curl scale needs all four derivative multipliers")
    return sx_hz, sy_hz, sy_ex, sx_ey


def _factor(
    values: FieldArray | None, shape: tuple[int, int], name: str
) -> FieldArray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} != {shape}")
    return array


def _check_fields(
    grid: YeeGrid2D, fields: dict[str, FieldArray], *, electric: bool
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
