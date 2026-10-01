"""Open dielectric sphere in 3D FDFD, compared with the Mie series.

The unknown is the scattered field. The staircase ball stays in the
operator, and the incident wave enters as the contrast current
``J = j ω (ε − ε0) E_inc``. The incident field is ``x-hat exp(-j k (z-z_c))``.
The outer wall is a homogeneous PEC, backed by a convolutional PML.
``--radius`` is the sphere radius and ``--index`` is the refractive index,
with ``ε_r = n²``. ``--size`` is the cube side and ``--dx`` is the cell
size. ``--pml`` is the cell count on each face. ``ka`` stays 1, so the
wavelength scales with the radius. A small direct factor is left on one
BLAS thread when that count is unset. Results go in
``dielectric_sphere_<parameters>/`` next to this script.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from maxwell_fd.analytics.mie3d import sphere_fields
from maxwell_fd.drivers.fdfd3d import FDFDOperator3D
from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.materials.volume3d import Ball, StaircaseIsotropic3D, UniformIsotropic3D
from maxwell_fd.operators.pml import PMLSpec3
from maxwell_fd.utils.constants import C0, EPS0, MU0

KA = 1.0
INDEX = 1.5
RADIUS = 4.0
SIZE = 18.0
DX = 1.0
PML = 3
_COMPONENTS = ("ex", "ey", "ez")


def _relative(numerical: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def _cells(length: float, dx: float) -> int:
    if not np.isfinite(length) or not np.isfinite(dx) or dx <= 0.0 or length <= 0.0:
        raise ValueError(f"length and dx must be positive, got {length}, {dx}")
    count = int(round(length / dx))
    if count < 1 or abs(count * dx - length) > 1e-8 * max(length, dx):
        raise ValueError(f"{length:g} is not a multiple of dx={dx:g}")
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an open 3D dielectric sphere against the Mie series."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=RADIUS,
        help=f"Sphere radius a. ka stays {KA:g} (default: {RADIUS:g}).",
    )
    parser.add_argument(
        "--index",
        type=float,
        default=INDEX,
        help=f"Refractive index n, with ε_r = n² (default: {INDEX:g}).",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=SIZE,
        help=f"Side length of the cubic computational region (default: {SIZE:g}).",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=DX,
        help=f"Cell size Δx = Δy = Δz (default: {DX:g}).",
    )
    parser.add_argument(
        "--pml",
        type=int,
        default=PML,
        help=f"PML cells on each face (default: {PML}).",
    )
    args = parser.parse_args(argv)
    try:
        n = _cells(args.size, args.dx)
    except ValueError as exc:
        parser.error(str(exc))
    if n < 2:
        parser.error("size must contain at least 2 cells")
    if isinstance(args.pml, bool) or args.pml < 1 or 2 * args.pml >= n:
        parser.error("pml must be a positive cell count that leaves an interior")
    interior = 0.5 * args.size - args.pml * args.dx
    if not np.isfinite(args.radius) or args.radius <= 0.0 or args.radius >= interior:
        parser.error(
            f"radius must be positive and lie inside the interior (half-width {interior:g})"
        )
    if not np.isfinite(args.index) or args.index <= 0.0:
        parser.error("index must be positive")
    args.cells = n
    return args


def _wave(radius: float) -> tuple[float, float]:
    omega = (KA / radius) * C0
    k = float(omega * np.sqrt(MU0 * EPS0))
    return omega, k


def _incident(
    grid: YeeGrid3D, component: str, center: tuple[float, float, float], k: float
) -> np.ndarray:
    _x, _y, z = grid.coordinates(component)
    phase = np.exp(-1j * k * (z - center[2]))
    field = np.zeros(grid.shapes()[component], dtype=np.complex128)
    if component == "ex":
        field[...] = phase
    return field


def _mask(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    size: float,
    band: float,
    center: tuple[float, float, float],
    radius: float,
    dx: float,
) -> np.ndarray:
    rho = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
    return (
        (xx >= band)
        & (xx <= size - band)
        & (yy >= band)
        & (yy <= size - band)
        & (zz >= band)
        & (zz <= size - band)
        & (np.abs(rho - radius) >= 2.0 * dx)
    )


def _solve(
    radius: float, index: float, size: float, dx: float, pml_cells: int
) -> dict[str, np.ndarray | float]:
    n = _cells(size, dx)
    center = (0.5 * size, 0.5 * size, 0.5 * size)
    omega, k = _wave(radius)
    grid = YeeGrid3D(n, n, n, dx, dx, dx, Boundary.PEC)
    components = StaircaseIsotropic3D(
        UniformIsotropic3D(),
        (Ball(*center, radius, eps=index**2 * EPS0),),
    ).sample(grid)
    operator = FDFDOperator3D(grid, components, PMLSpec3.box(pml_cells))
    eps = {
        "ex": components.eps_x,
        "ey": components.eps_y,
        "ez": components.eps_z,
    }
    incident = {name: _incident(grid, name, center, k) for name in _COMPONENTS}
    solved = operator.solve(
        omega,
        operator.layout.pack_e(
            {
                name: 1j * omega * (eps[name] - EPS0) * incident[name]
                for name in _COMPONENTS
            }
        ),
    )
    numerical = []
    reference = []
    offset = 0
    band = pml_cells * dx
    totals: dict[str, np.ndarray] = {}
    exacts: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    coords: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dof, name in zip(operator.layout.e, _COMPONENTS, strict=True):
        total = incident[name].copy()
        count = dof.i.size
        total[dof.i, dof.j, dof.k] = (
            solved[offset : offset + count] + incident[name][dof.i, dof.j, dof.k]
        )
        offset += count
        x, y, z = grid.coordinates(name)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        exact = sphere_fields(
            xx, yy, zz, k=k, radius=radius, index=index, center=center
        )[0][{"ex": 0, "ey": 1, "ez": 2}[name]]
        observe = _mask(xx, yy, zz, size, band, center, radius, dx)
        numerical.append(total[observe])
        reference.append(exact[observe])
        totals[name] = total
        exacts[name] = exact
        masks[name] = observe
        coords[name] = (x, y, z)
    return {
        "error": _relative(np.concatenate(numerical), np.concatenate(reference)),
        "center": np.array(center),
        "totals": totals,
        "exacts": exacts,
        "masks": masks,
        "coords": coords,
    }


def _panel(
    ax: plt.Axes,
    x: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    limit: float,
    center: tuple[float, float],
    radius: float,
    band: float,
    size: float,
):
    xx, zz = np.meshgrid(x, z, indexing="ij")
    mesh = ax.pcolormesh(
        xx, zz, values, shading="gouraud", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    width = size - 2.0 * band
    ax.add_patch(Circle(center, radius, fill=False, ec="0.1", lw=0.9, zorder=3))
    ax.add_patch(
        Rectangle(
            (band, band),
            width,
            width,
            fill=False,
            edgecolor="#d94801",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(z)), float(np.max(z)))
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    return mesh


def plot_fields(
    path: Path,
    radius: float,
    index: float,
    size: float,
    dx: float,
    pml_cells: int,
) -> float:
    result = _solve(radius, index, size, dx, pml_cells)
    totals = result["totals"]
    exacts = result["exacts"]
    masks = result["masks"]
    x, y, z = result["coords"]["ex"]
    center = result["center"]
    j = int(np.argmin(np.abs(y - center[1])))
    shown = np.array(totals["ex"], copy=True)
    shown[:, 0, :] = 0.0
    shown[:, -1, :] = 0.0
    shown[:, :, 0] = 0.0
    shown[:, :, -1] = 0.0
    field = np.real(shown[:, j, :])
    exact = np.real(np.asarray(exacts["ex"])[:, j, :])
    mask = np.asarray(masks["ex"])[:, j, :]
    error = float(result["error"])
    limit = max(
        float(np.max(np.abs(field[mask]))), float(np.max(np.abs(exact[mask]))), 1e-6
    )
    diff = field - exact
    diff_limit = max(float(np.max(np.abs(diff[mask]))), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    band = pml_cells * dx
    panels = (
        (axes[0], field, limit, "FD"),
        (axes[1], exact, limit, "Mie"),
        (axes[2], diff, diff_limit, "FD − Mie"),
    )
    for ax, values, scale, label in panels:
        image = _panel(
            ax,
            x,
            z,
            values,
            scale,
            (float(center[0]), float(center[2])),
            radius,
            band,
            size,
        )
        ax.set_title(label)
        if label == "FD":
            ax.text(
                0.02,
                0.98,
                f"$E_x$, L2 {error:.2%}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="0.15",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        rf"Open dielectric sphere, $n={index:g}$, $a={radius:g}$, $ka={KA:g}$, "
        rf"size ${size:g}$, $\Delta={dx:g}$, PML {pml_cells} cells, "
        rf"incident $e^{{-jk(z-z_c)}}$",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return error


def plot_domain(
    path: Path,
    radius: float,
    index: float,
    size: float,
    dx: float,
    pml_cells: int,
) -> None:
    n = _cells(size, dx)
    origin = pml_cells * dx
    interior = size - 2.0 * origin
    center = (0.5 * size, 0.5 * size)
    edges = np.linspace(0.0, size, n + 1)
    segments = [[(float(x), 0.0), (float(x), size)] for x in edges]
    segments += [[(0.0, float(z)), (size, float(z))] for z in edges]
    fig, ax = plt.subplots(figsize=(6.8, 7.2))
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), size, size, facecolor="#fdd0a2", edgecolor="none", zorder=0
        )
    )
    ax.add_patch(
        Rectangle(
            (origin, origin),
            interior,
            interior,
            facecolor="#ffffff",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_collection(
        LineCollection(segments, colors="#bdbdbd", linewidths=0.45, zorder=2)
    )
    ax.add_patch(
        Circle(
            center,
            radius,
            facecolor="#6baed6",
            edgecolor="#08519c",
            linewidth=1.2,
            alpha=0.55,
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle(
            (origin, origin),
            interior,
            interior,
            fill=False,
            edgecolor="#e6550d",
            linewidth=1.3,
            zorder=4,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            size,
            size,
            fill=False,
            edgecolor="#252525",
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(0.0, size)
    ax.set_ylim(0.0, size)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(
        "Mesh, PML, and dielectric sphere\n"
        rf"$n={index:g}$, $a={radius:g}$, size ${size:g}$, $\Delta={dx:g}$, "
        rf"PML {pml_cells} cells"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", linewidth=1.0, label="mesh"),
            Patch(
                facecolor="#fdd0a2",
                edgecolor="#e6550d",
                label=f"PML ({pml_cells} cells)",
            ),
            Patch(
                facecolor="#6baed6",
                edgecolor="#08519c",
                alpha=0.55,
                label=rf"dielectric, $n={index:g}$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import sys

    examples = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(examples))
    from run_dir import output_dir

    args = _parse_args()
    folder = output_dir(
        __file__,
        index=args.index,
        ka=KA,
        a=args.radius,
        size=args.size,
        dx=args.dx,
        pml=args.pml,
    )
    error = plot_fields(
        folder / "dielectric_sphere.png",
        args.radius,
        args.index,
        args.size,
        args.dx,
        args.pml,
    )
    plot_domain(
        folder / "domain.png",
        args.radius,
        args.index,
        args.size,
        args.dx,
        args.pml,
    )
    print(f"{folder.name}  L2 {error:.4%}")


if __name__ == "__main__":
    main()
