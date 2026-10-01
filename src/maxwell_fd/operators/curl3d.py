"""Sparse and array curls for the 3D Yee cell.

``curl_e @ e`` stores ∇×E at the H degrees of freedom. ``curl_h @ h`` stores
∇×H at the E degrees of freedom. The partial derivatives are the same ones
as ``FD_LaTeX_Reference.tex`` (2.1)--(2.3), taken on every axis:

    (∇×E)_x = ∂Ez/∂y − ∂Ey/∂z,   (∇×H)_x = ∂Hz/∂y − ∂Hy/∂z,
    (∇×E)_y = ∂Ex/∂z − ∂Ez/∂x,   (∇×H)_y = ∂Hx/∂z − ∂Hz/∂x,
    (∇×E)_z = ∂Ey/∂x − ∂Ex/∂y,   (∇×H)_z = ∂Hy/∂x − ∂Hx/∂y.

With no stretch, ``curl_h`` is the transpose of ``curl_e``. A scale
multiplies each partial derivative by ``1/s_w`` at the sample where that
derivative is centered. A varying scale is complex, and the two curls are
then not transposes. PEC tangential electric samples are omitted.
Magnetic samples that do not enter ``curl_h`` at a free electric sample
are omitted too. The array curls accept the full stored arrays and agree
with the matrices when those omitted electric samples are zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D

FloatArray = NDArray[np.float64]
FieldArray = NDArray[np.float64] | NDArray[np.complex128]
_ELECTRIC = ("ex", "ey", "ez")
_MAGNETIC = ("hx", "hy", "hz")


@dataclass(frozen=True)
class DofSet3:
    """One stacked component inside the 3D E or H unknown vector."""

    name: str
    i: NDArray[np.int64]
    j: NDArray[np.int64]
    k: NDArray[np.int64]
    shape: tuple[int, int, int]

    def pack(self, field: NDArray[np.float64] | NDArray[np.complex128]) -> NDArray:
        return np.asarray(field)[self.i, self.j, self.k]

    def scatter(self, values: NDArray, fill: float = 0.0) -> NDArray:
        out = np.full(self.shape, fill, dtype=values.dtype)
        out[self.i, self.j, self.k] = values
        return out


@dataclass(frozen=True)
class Layout3:
    """Degree-of-freedom order: ``(Ex, Ey, Ez)`` and ``(Hx, Hy, Hz)``."""

    e: tuple[DofSet3, DofSet3, DofSet3]
    h: tuple[DofSet3, DofSet3, DofSet3]

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
class CurlOperators3:
    """Sparse curls and the layout that indexes them."""

    layout: Layout3
    curl_e: sparse.csr_matrix
    curl_h: sparse.csr_matrix


@dataclass(frozen=True)
class CurlScale3:
    """``1/s_w`` on ``∂/∂w``, shaped like the curl component that difference enters."""

    y_on_hx: NDArray
    z_on_hx: NDArray
    z_on_hy: NDArray
    x_on_hy: NDArray
    x_on_hz: NDArray
    y_on_hz: NDArray
    y_on_ex: NDArray
    z_on_ex: NDArray
    z_on_ey: NDArray
    x_on_ey: NDArray
    x_on_ez: NDArray
    y_on_ez: NDArray


_SCALE_COMPONENT = {
    "y_on_hx": "hx",
    "z_on_hx": "hx",
    "z_on_hy": "hy",
    "x_on_hy": "hy",
    "x_on_hz": "hz",
    "y_on_hz": "hz",
    "y_on_ex": "ex",
    "z_on_ex": "ex",
    "z_on_ey": "ey",
    "x_on_ey": "ey",
    "x_on_ez": "ez",
    "y_on_ez": "ez",
}


def build_curls(grid: YeeGrid3D, *, scale: CurlScale3 | None = None) -> CurlOperators3:
    """Assemble ``curl_e`` (H rows, E columns) and ``curl_h`` (E rows, H columns)."""
    layout = _layout(grid)
    if layout.n_e == 0:
        raise ValueError("grid removes every electric unknown")
    factors = _factors(scale, grid)
    curl_e, curl_h = _curls(grid, layout, factors)
    return CurlOperators3(layout=layout, curl_e=curl_e, curl_h=curl_h)


def array_curl_e(
    grid: YeeGrid3D,
    fields: dict[str, FieldArray],
    scale: CurlScale3 | None = None,
) -> dict[str, FieldArray]:
    """∇×E on the full H arrays."""
    _check_fields(grid, fields, electric=True)
    return _array_curl_e(
        grid, fields["ex"], fields["ey"], fields["ez"], _factors(scale, grid)
    )


def array_curl_h(
    grid: YeeGrid3D,
    fields: dict[str, FieldArray],
    scale: CurlScale3 | None = None,
) -> dict[str, FieldArray]:
    """∇×H on the full E arrays. PEC tangential samples stay zero."""
    _check_fields(grid, fields, electric=False)
    return _array_curl_h(
        grid, fields["hx"], fields["hy"], fields["hz"], _factors(scale, grid)
    )


def _pack(sets: tuple[DofSet3, ...], fields: dict[str, NDArray]) -> NDArray:
    return np.concatenate([dof.pack(fields[dof.name]) for dof in sets])


def _layout(grid: YeeGrid3D) -> Layout3:
    shapes = grid.shapes()
    ranges = _ranges(grid)
    ex, ey, ez = (_dof(name, shapes[name], ranges[name]) for name in _ELECTRIC)
    hx, hy, hz = (_dof(name, shapes[name], ranges[name]) for name in _MAGNETIC)
    return Layout3(e=(ex, ey, ez), h=(hx, hy, hz))


def _ranges(
    grid: YeeGrid3D,
) -> dict[str, tuple[int, int, int, int, int, int]]:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if grid.boundary is Boundary.PERIODIC:
        full = (0, nx, 0, ny, 0, nz)
        return {name: full for name in (*_ELECTRIC, *_MAGNETIC)}
    return {
        "ex": (0, nx, 1, ny, 1, nz),
        "ey": (1, nx, 0, ny, 1, nz),
        "ez": (1, nx, 1, ny, 0, nz),
        "hx": (1, nx, 0, ny, 0, nz),
        "hy": (0, nx, 1, ny, 0, nz),
        "hz": (0, nx, 0, ny, 1, nz),
    }


def _dof(
    name: str, shape: tuple[int, int, int], bounds: tuple[int, int, int, int, int, int]
) -> DofSet3:
    i0, i1, j0, j1, k0, k1 = bounds
    ii, jj, kk = np.meshgrid(
        np.arange(i0, i1), np.arange(j0, j1), np.arange(k0, k1), indexing="ij"
    )
    return DofSet3(name=name, i=ii.ravel(), j=jj.ravel(), k=kk.ravel(), shape=shape)


def _id_map(dof: DofSet3) -> NDArray[np.int64]:
    ids = -np.ones(dof.shape, dtype=np.int64)
    ids[dof.i, dof.j, dof.k] = np.arange(dof.i.size, dtype=np.int64)
    return ids


def _curls(
    grid: YeeGrid3D, layout: Layout3, factors: dict[str, NDArray] | None
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    ex, ey, ez = layout.e
    hx, hy, hz = layout.h
    ex_id, ey_id, ez_id = _id_map(ex), _id_map(ey), _id_map(ez)
    hx_id, hy_id, hz_id = _id_map(hx), _id_map(hy), _id_map(hz)
    n_ex, n_ey = ex.i.size, ey.i.size
    n_hx, n_hy = hx.i.size, hy.i.size
    periodic = grid.boundary is Boundary.PERIODIC
    dims = (grid.nx, grid.ny, grid.nz)
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    e_rows: list[NDArray[np.int64]] = []
    e_cols: list[NDArray[np.int64]] = []
    e_data: list[NDArray] = []
    h_rows: list[NDArray[np.int64]] = []
    h_cols: list[NDArray[np.int64]] = []
    h_data: list[NDArray] = []

    def add_e(blocks: list[tuple[NDArray, NDArray, NDArray] | None]) -> None:
        for block in blocks:
            if block is not None:
                e_rows.append(block[0])
                e_cols.append(block[1])
                e_data.append(block[2])

    def add_h(blocks: list[tuple[NDArray, NDArray, NDArray] | None]) -> None:
        for block in blocks:
            if block is not None:
                h_rows.append(block[0])
                h_cols.append(block[1])
                h_data.append(block[2])

    hx_rows = np.arange(hx.i.size, dtype=np.int64)
    add_e(
        _difference(
            hx_rows,
            ez_id,
            hx,
            axis=1,
            high=True,
            weight=1.0 / dy,
            offset=n_ex + n_ey,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "y_on_hx", hx),
        )
    )
    add_e(
        _difference(
            hx_rows,
            ey_id,
            hx,
            axis=2,
            high=True,
            weight=-1.0 / dz,
            offset=n_ex,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "z_on_hx", hx),
        )
    )

    hy_rows = n_hx + np.arange(hy.i.size, dtype=np.int64)
    add_e(
        _difference(
            hy_rows,
            ex_id,
            hy,
            axis=2,
            high=True,
            weight=1.0 / dz,
            offset=0,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "z_on_hy", hy),
        )
    )
    add_e(
        _difference(
            hy_rows,
            ez_id,
            hy,
            axis=0,
            high=True,
            weight=-1.0 / dx,
            offset=n_ex + n_ey,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "x_on_hy", hy),
        )
    )

    hz_rows = n_hx + n_hy + np.arange(hz.i.size, dtype=np.int64)
    add_e(
        _difference(
            hz_rows,
            ey_id,
            hz,
            axis=0,
            high=True,
            weight=1.0 / dx,
            offset=n_ex,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "x_on_hz", hz),
        )
    )
    add_e(
        _difference(
            hz_rows,
            ex_id,
            hz,
            axis=1,
            high=True,
            weight=-1.0 / dy,
            offset=0,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "y_on_hz", hz),
        )
    )

    ex_rows = np.arange(ex.i.size, dtype=np.int64)
    add_h(
        _difference(
            ex_rows,
            hz_id,
            ex,
            axis=1,
            high=False,
            weight=1.0 / dy,
            offset=n_hx + n_hy,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "y_on_ex", ex),
        )
    )
    add_h(
        _difference(
            ex_rows,
            hy_id,
            ex,
            axis=2,
            high=False,
            weight=-1.0 / dz,
            offset=n_hx,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "z_on_ex", ex),
        )
    )

    ey_rows = n_ex + np.arange(ey.i.size, dtype=np.int64)
    add_h(
        _difference(
            ey_rows,
            hx_id,
            ey,
            axis=2,
            high=False,
            weight=1.0 / dz,
            offset=0,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "z_on_ey", ey),
        )
    )
    add_h(
        _difference(
            ey_rows,
            hz_id,
            ey,
            axis=0,
            high=False,
            weight=-1.0 / dx,
            offset=n_hx + n_hy,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "x_on_ey", ey),
        )
    )

    ez_rows = n_ex + n_ey + np.arange(ez.i.size, dtype=np.int64)
    add_h(
        _difference(
            ez_rows,
            hy_id,
            ez,
            axis=0,
            high=False,
            weight=1.0 / dx,
            offset=n_hx,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "x_on_ez", ez),
        )
    )
    add_h(
        _difference(
            ez_rows,
            hx_id,
            ez,
            axis=1,
            high=False,
            weight=-1.0 / dy,
            offset=0,
            periodic=periodic,
            dims=dims,
            factors=_row_scale(factors, "y_on_ez", ez),
        )
    )

    return (
        _coo(e_rows, e_cols, e_data, (layout.n_h, layout.n_e)),
        _coo(h_rows, h_cols, h_data, (layout.n_e, layout.n_h)),
    )


def _difference(
    rows: NDArray[np.int64],
    col_ids: NDArray[np.int64],
    dof: DofSet3,
    *,
    axis: int,
    high: bool,
    weight: float,
    offset: int,
    periodic: bool,
    dims: tuple[int, int, int],
    factors: NDArray | None,
) -> list[tuple[NDArray, NDArray, NDArray] | None]:
    """Centered difference of the column field, sampled at ``dof``.

    ``high`` is ``(f[p+e] - f[p])``. The low form is ``(f[p] - f[p-e])``,
    which is the transpose stencil. ``offset`` is added after the mask so
    a missing id of -1 is not shifted into a real column. ``factors``
    multiplies both stencil entries and is aligned with ``rows``.
    """
    here = _masked(
        rows,
        col_ids[dof.i, dof.j, dof.k],
        -weight if high else weight,
        offset,
        factors,
    )
    step = 1 if high else -1
    neighbor = _shift(dof.i, dof.j, dof.k, axis, step, periodic, dims)
    other = _masked(
        rows,
        col_ids[neighbor[0], neighbor[1], neighbor[2]],
        weight if high else -weight,
        offset,
        factors,
    )
    return [other, here]


def _shift(
    i: NDArray[np.int64],
    j: NDArray[np.int64],
    k: NDArray[np.int64],
    axis: int,
    step: int,
    periodic: bool,
    dims: tuple[int, int, int],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    coords = [i, j, k]
    moved = coords[axis] + step
    if periodic:
        moved = np.mod(moved, dims[axis])
    coords[axis] = moved
    return coords[0], coords[1], coords[2]


def _masked(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    value: float,
    offset: int,
    factors: NDArray | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray] | None:
    keep = cols >= 0
    if not np.any(keep):
        return None
    if factors is None:
        data: NDArray = np.full(int(np.count_nonzero(keep)), value, dtype=np.float64)
    else:
        data = np.asarray(factors[keep]) * value
        dtype = np.complex128 if np.iscomplexobj(data) else np.float64
        data = np.asarray(data, dtype=dtype)
    return rows[keep], cols[keep] + offset, data


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
        (np.asarray(values, dtype=dtype), (np.concatenate(rows), np.concatenate(cols))),
        shape=shape,
        dtype=dtype,
    ).tocsr()


def _array_curl_e(
    grid: YeeGrid3D,
    ex: FieldArray,
    ey: FieldArray,
    ez: FieldArray,
    factors: dict[str, NDArray] | None,
) -> dict[str, FieldArray]:
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    dtype = _dtype(
        ex,
        ey,
        ez,
        factors,
        ("y_on_hx", "z_on_hx", "z_on_hy", "x_on_hy", "x_on_hz", "y_on_hz"),
    )
    if grid.boundary is Boundary.PERIODIC:
        d_ez_dy = (np.roll(ez, -1, axis=1) - ez) / dy
        d_ey_dz = (np.roll(ey, -1, axis=2) - ey) / dz
        d_ex_dz = (np.roll(ex, -1, axis=2) - ex) / dz
        d_ez_dx = (np.roll(ez, -1, axis=0) - ez) / dx
        d_ey_dx = (np.roll(ey, -1, axis=0) - ey) / dx
        d_ex_dy = (np.roll(ex, -1, axis=1) - ex) / dy
    else:
        d_ez_dy = (ez[:, 1:, :] - ez[:, :-1, :]) / dy
        d_ey_dz = (ey[:, :, 1:] - ey[:, :, :-1]) / dz
        d_ex_dz = (ex[:, :, 1:] - ex[:, :, :-1]) / dz
        d_ez_dx = (ez[1:, :, :] - ez[:-1, :, :]) / dx
        d_ey_dx = (ey[1:, :, :] - ey[:-1, :, :]) / dx
        d_ex_dy = (ex[:, 1:, :] - ex[:, :-1, :]) / dy
    if factors is not None:
        d_ez_dy = d_ez_dy * factors["y_on_hx"]
        d_ey_dz = d_ey_dz * factors["z_on_hx"]
        d_ex_dz = d_ex_dz * factors["z_on_hy"]
        d_ez_dx = d_ez_dx * factors["x_on_hy"]
        d_ey_dx = d_ey_dx * factors["x_on_hz"]
        d_ex_dy = d_ex_dy * factors["y_on_hz"]
    return {
        "hx": np.asarray(d_ez_dy - d_ey_dz, dtype=dtype),
        "hy": np.asarray(d_ex_dz - d_ez_dx, dtype=dtype),
        "hz": np.asarray(d_ey_dx - d_ex_dy, dtype=dtype),
    }


def _array_curl_h(
    grid: YeeGrid3D,
    hx: FieldArray,
    hy: FieldArray,
    hz: FieldArray,
    factors: dict[str, NDArray] | None,
) -> dict[str, FieldArray]:
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    dtype = _dtype(
        hx,
        hy,
        hz,
        factors,
        ("y_on_ex", "z_on_ex", "z_on_ey", "x_on_ey", "x_on_ez", "y_on_ez"),
    )
    if grid.boundary is Boundary.PERIODIC:
        d_hz_dy = (hz - np.roll(hz, 1, axis=1)) / dy
        d_hy_dz = (hy - np.roll(hy, 1, axis=2)) / dz
        d_hx_dz = (hx - np.roll(hx, 1, axis=2)) / dz
        d_hz_dx = (hz - np.roll(hz, 1, axis=0)) / dx
        d_hy_dx = (hy - np.roll(hy, 1, axis=0)) / dx
        d_hx_dy = (hx - np.roll(hx, 1, axis=1)) / dy
        if factors is not None:
            d_hz_dy = d_hz_dy * factors["y_on_ex"]
            d_hy_dz = d_hy_dz * factors["z_on_ex"]
            d_hx_dz = d_hx_dz * factors["z_on_ey"]
            d_hz_dx = d_hz_dx * factors["x_on_ey"]
            d_hy_dx = d_hy_dx * factors["x_on_ez"]
            d_hx_dy = d_hx_dy * factors["y_on_ez"]
        return {
            "ex": np.asarray(d_hz_dy - d_hy_dz, dtype=dtype),
            "ey": np.asarray(d_hx_dz - d_hz_dx, dtype=dtype),
            "ez": np.asarray(d_hy_dx - d_hx_dy, dtype=dtype),
        }
    shapes = grid.shapes()
    ex = np.zeros(shapes["ex"], dtype=dtype)
    ey = np.zeros(shapes["ey"], dtype=dtype)
    ez = np.zeros(shapes["ez"], dtype=dtype)
    d_hz_dy = (hz[:, 1:, 1:-1] - hz[:, :-1, 1:-1]) / dy
    d_hy_dz = (hy[:, 1:-1, 1:] - hy[:, 1:-1, :-1]) / dz
    d_hx_dz = (hx[1:-1, :, 1:] - hx[1:-1, :, :-1]) / dz
    d_hz_dx = (hz[1:, :, 1:-1] - hz[:-1, :, 1:-1]) / dx
    d_hy_dx = (hy[1:, 1:-1, :] - hy[:-1, 1:-1, :]) / dx
    d_hx_dy = (hx[1:-1, 1:, :] - hx[1:-1, :-1, :]) / dy
    if factors is not None:
        d_hz_dy = factors["y_on_ex"][:, 1:-1, 1:-1] * d_hz_dy
        d_hy_dz = factors["z_on_ex"][:, 1:-1, 1:-1] * d_hy_dz
        d_hx_dz = factors["z_on_ey"][1:-1, :, 1:-1] * d_hx_dz
        d_hz_dx = factors["x_on_ey"][1:-1, :, 1:-1] * d_hz_dx
        d_hy_dx = factors["x_on_ez"][1:-1, 1:-1, :] * d_hy_dx
        d_hx_dy = factors["y_on_ez"][1:-1, 1:-1, :] * d_hx_dy
    ex[:, 1:-1, 1:-1] = d_hz_dy - d_hy_dz
    ey[1:-1, :, 1:-1] = d_hx_dz - d_hz_dx
    ez[1:-1, 1:-1, :] = d_hy_dx - d_hx_dy
    return {"ex": ex, "ey": ey, "ez": ez}


def _factors(scale: CurlScale3 | None, grid: YeeGrid3D) -> dict[str, NDArray] | None:
    if scale is None:
        return None
    shapes = grid.shapes()
    out: dict[str, NDArray] = {}
    for name, component in _SCALE_COMPONENT.items():
        array = np.asarray(getattr(scale, name))
        if array.shape != shapes[component]:
            raise ValueError(f"{name} shape {array.shape} != {shapes[component]}")
        out[name] = array
    return out


def _row_scale(
    factors: dict[str, NDArray] | None, name: str, dof: DofSet3
) -> NDArray | None:
    if factors is None:
        return None
    return factors[name][dof.i, dof.j, dof.k]


def _dtype(
    first: NDArray,
    second: NDArray,
    third: NDArray,
    factors: dict[str, NDArray] | None,
    names: tuple[str, ...],
) -> np.dtype:
    pieces: list[NDArray] = [first, second, third]
    if factors is not None:
        pieces.extend(factors[name] for name in names)
    return np.result_type(*pieces, np.float64)


def _check_fields(
    grid: YeeGrid3D, fields: dict[str, FieldArray], *, electric: bool
) -> None:
    needed = _ELECTRIC if electric else _MAGNETIC
    shapes = grid.shapes()
    for name in needed:
        if name not in fields:
            raise KeyError(name)
        if fields[name].shape != shapes[name]:
            raise ValueError(f"{name} shape {fields[name].shape} != {shapes[name]}")
