"""Z-directed Hertzian dipole in 3D FDFD, compared with the spherical wave.

The element sits on one ``Ez`` sample. Its moment is ``Iℓ = 1``, so that
sample carries ``J_z = 1/(Δx Δy Δz)``. A convolutional PML backs the PEC
wall on every face. The comparison is the electric field outside the PML
and at least a quarter wavelength from the element. ``--wavelength``,
``--size``, ``--dx``, and ``--pml`` set the free-space wavelength, the
cube side, the cell size, and the PML depth. A small direct factor is
left on one BLAS thread when that count is unset. Results go in
``hertzian_dipole_<parameters>/`` next to this script.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

from maxwell_fd.analytics.dipole3d import hertzian_dipole_fields  # noqa: E402
from maxwell_fd.drivers.fdfd3d import FDFDOperator3D  # noqa: E402
from maxwell_fd.grid.yee2d import Boundary  # noqa: E402
from maxwell_fd.grid.yee3d import YeeGrid3D  # noqa: E402
from maxwell_fd.materials.volume3d import UniformIsotropic3D  # noqa: E402
from maxwell_fd.operators.pml import PMLSpec3  # noqa: E402
from maxwell_fd.utils.constants import C0  # noqa: E402

WAVELENGTH = 1.0
SIZE = 1.6
DX = 0.1
PML = 3
MOMENT = 1.0
R_MIN = 0.25
_COMPONENTS = ("ex", "ey", "ez")


def _cells(length: float, dx: float) -> int:
    if not np.isfinite(length) or not np.isfinite(dx) or dx <= 0.0 or length <= 0.0:
        raise ValueError(f"length and dx must be positive, got {length}, {dx}")
    count = int(round(length / dx))
    if count < 1 or abs(count * dx - length) > 1e-8 * max(length, dx):
        raise ValueError(f"{length:g} is not a multiple of dx={dx:g}")
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a z-directed Hertzian dipole against the spherical wave."
    )
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH)
    parser.add_argument("--size", type=float, default=SIZE)
    parser.add_argument("--dx", type=float, default=DX)
    parser.add_argument("--pml", type=int, default=PML)
    args = parser.parse_args(argv)
    try:
        cells = _cells(args.size, args.dx)
    except ValueError as exc:
        parser.error(str(exc))
    if cells < 4:
        parser.error("size must contain at least 4 cells")
    if isinstance(args.pml, bool) or args.pml < 1 or 2 * args.pml >= cells:
        parser.error("pml must be a positive cell count that leaves an interior")
    if not np.isfinite(args.wavelength) or args.wavelength <= 0.0:
        parser.error("wavelength must be positive")
    index = cells // 2
    source_z = (index + 0.5) * args.dx
    band = args.pml * args.dx
    clearance = min(index * args.dx - band, args.size - band - source_z)
    if clearance < R_MIN * args.wavelength:
        parser.error("leave 0.25 wavelengths between the element and the PML")
    args.cells = cells
    return args


def _solve(wavelength: float, size: float, dx: float, pml_cells: int) -> dict:
    cells = _cells(size, dx)
    index = cells // 2
    omega = 2.0 * np.pi * C0 / wavelength
    grid = YeeGrid3D(cells, cells, cells, dx, dx, dx, Boundary.PEC)
    operator = FDFDOperator3D(
        grid, UniformIsotropic3D().sample(grid), PMLSpec3.box(pml_cells)
    )
    current = {
        name: np.zeros(grid.shapes()[name], dtype=np.complex128) for name in _COMPONENTS
    }
    current["ez"][index, index, index] = MOMENT / (dx * dx * dx)
    solved = operator.solve(omega, operator.layout.pack_e(current))
    x, y, z = grid.coordinates("ez")
    origin = (float(x[index]), float(y[index]), float(z[index]))
    numerical = []
    reference = []
    offset = 0
    band = pml_cells * dx
    r_min = R_MIN * wavelength
    totals = {}
    exacts = {}
    masks = {}
    coords = {}
    for dof, name in zip(operator.layout.e, _COMPONENTS, strict=True):
        width = dof.i.size
        field = dof.scatter(solved[offset : offset + width])
        offset += width
        cx, cy, cz = grid.coordinates(name)
        xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="ij")
        exact = hertzian_dipole_fields(
            xx, yy, zz, omega=omega, moment=MOMENT, origin=origin
        )[name]
        radius = np.sqrt(
            (xx - origin[0]) ** 2 + (yy - origin[1]) ** 2 + (zz - origin[2]) ** 2
        )
        observe = (
            (xx >= band)
            & (xx <= size - band)
            & (yy >= band)
            & (yy <= size - band)
            & (zz >= band)
            & (zz <= size - band)
            & (radius >= r_min)
        )
        numerical.append(field[observe])
        reference.append(exact[observe])
        totals[name] = field
        exacts[name] = exact
        masks[name] = observe
        coords[name] = (cx, cy, cz)
    got = np.concatenate(numerical)
    exact = np.concatenate(reference)
    return {
        "error": float(np.linalg.norm(got - exact) / np.linalg.norm(exact)),
        "origin": origin,
        "totals": totals,
        "exacts": exacts,
        "masks": masks,
        "coords": coords,
        "r_min": r_min,
        "band": band,
    }


def _panel(
    ax: plt.Axes,
    x,
    z,
    values,
    limit: float,
    origin,
    r_min: float,
    band: float,
    size: float,
):
    xx, zz = np.meshgrid(x, z, indexing="ij")
    mesh = ax.pcolormesh(
        xx, zz, values, shading="gouraud", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    width = size - 2.0 * band
    ax.add_patch(
        Circle(origin, r_min, fill=False, ec="0.35", lw=0.8, ls="--", zorder=3)
    )
    ax.plot(origin[0], origin[1], marker="o", color="#08519c", ms=4.5, zorder=4)
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
    path: Path, wavelength: float, size: float, dx: float, pml_cells: int
) -> float:
    result = _solve(wavelength, size, dx, pml_cells)
    x, y, z = result["coords"]["ez"]
    origin = result["origin"]
    j = int(np.argmin(np.abs(y - origin[1])))
    field = np.real(np.asarray(result["totals"]["ez"])[:, j, :])
    exact = np.real(np.asarray(result["exacts"]["ez"])[:, j, :])
    exact = np.where(np.isfinite(exact), exact, 0.0)
    mask = np.asarray(result["masks"]["ez"])[:, j, :]
    error = float(result["error"])
    limit = max(
        float(np.max(np.abs(field[mask]))), float(np.max(np.abs(exact[mask]))), 1e-6
    )
    diff = field - exact
    diff_limit = max(float(np.max(np.abs(diff[mask]))), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    panels = (
        (axes[0], field, limit, "FD"),
        (axes[1], exact, limit, "Analytic"),
        (axes[2], diff, diff_limit, "FD − Analytic"),
    )
    center = (origin[0], origin[2])
    for ax, values, scale, label in panels:
        image = _panel(
            ax,
            x,
            z,
            values,
            scale,
            center,
            float(result["r_min"]),
            float(result["band"]),
            size,
        )
        ax.set_title(label)
        if label == "FD":
            ax.text(
                0.02,
                0.98,
                f"$E_z$, L2 {error:.2%}",
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
        rf"Hertzian dipole, $I\ell=1$, $\lambda={wavelength:g}$, "
        rf"size ${size:g}$, $\Delta={dx:g}$, PML {pml_cells} cells",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return error


def plot_domain(
    path: Path, wavelength: float, size: float, dx: float, pml_cells: int
) -> None:
    cells = _cells(size, dx)
    index = cells // 2
    origin_x = index * dx
    origin_z = (index + 0.5) * dx
    band = pml_cells * dx
    interior = size - 2.0 * band
    edges = np.linspace(0.0, size, cells + 1)
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
            (band, band),
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
    ax.plot(
        [origin_x, origin_x],
        [origin_z - 0.5 * dx, origin_z + 0.5 * dx],
        color="#08519c",
        lw=2.4,
        solid_capstyle="round",
        zorder=4,
    )
    ax.add_patch(
        Circle(
            (origin_x, origin_z),
            R_MIN * wavelength,
            fill=False,
            edgecolor="0.35",
            linewidth=0.9,
            linestyle="--",
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle(
            (band, band),
            interior,
            interior,
            fill=False,
            edgecolor="#e6550d",
            linewidth=1.3,
            zorder=5,
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
        "Mesh, PML, and Hertzian dipole\n"
        rf"$\lambda={wavelength:g}$, size ${size:g}$, $\Delta={dx:g}$, PML {pml_cells} cells"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", lw=1.0, label="mesh"),
            Patch(
                facecolor="#fdd0a2",
                edgecolor="#e6550d",
                label=f"PML ({pml_cells} cells)",
            ),
            Line2D([0], [0], color="#08519c", lw=2.4, label=r"dipole, $I\ell=1$"),
            Line2D(
                [0],
                [0],
                color="0.35",
                lw=0.9,
                ls="--",
                label=r"comparison radius $0.25\lambda$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import sys

    examples = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(examples))
    from run_dir import output_dir

    args = _parse_args()
    folder = output_dir(
        __file__, lam=args.wavelength, size=args.size, dx=args.dx, pml=args.pml
    )
    error = plot_fields(
        folder / "hertzian_dipole.png", args.wavelength, args.size, args.dx, args.pml
    )
    plot_domain(folder / "domain.png", args.wavelength, args.size, args.dx, args.pml)
    print(f"{folder.name}  L2 {error:.4%}")


if __name__ == "__main__":
    main()
