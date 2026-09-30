"""Open dielectric cylinder in 2D FDFD, compared with the Mie series.

The unknown is the scattered field. The staircase disk stays in the
operator, and the incident wave enters as the contrast current
``J = j ω (ε − ε0) E_inc``. The outer wall is a homogeneous PEC, backed
by a convolutional PML. ``--radius`` is the cylinder radius and
``--index`` is the refractive index, with ``ε_r = n²``. ``--size`` is the
side length and ``--dx`` is the cell size. ``--pml`` is the cell count on
each side. ``ka`` stays 1, so the wavelength scales with the radius.
Results go in ``dielectric_cylinder_open_<parameters>/`` next to this
script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from maxwell_fd.analytics.mie2d import cylinder_ez, cylinder_hz, cylinder_te_electric
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import Disk, StaircaseIsotropic, UniformIsotropic
from maxwell_fd.operators.curl2d import array_curl_e
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0, EPS0, MU0

KA = 1.0
INDEX = 1.5
RADIUS = 8.0
SIZE = 64.0
DX = 1.0
PML = 8


def _relative(numerical: np.ndarray, exact: np.ndarray) -> float:
    scale = np.linalg.norm(exact)
    return float(np.linalg.norm(numerical - exact) / scale)


def _cells(length: float, dx: float) -> int:
    if not np.isfinite(length) or not np.isfinite(dx) or dx <= 0.0 or length <= 0.0:
        raise ValueError(f"length and dx must be positive, got {length}, {dx}")
    count = int(round(length / dx))
    if count < 1 or abs(count * dx - length) > 1e-8 * max(length, dx):
        raise ValueError(f"{length:g} is not a multiple of dx={dx:g}")
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an open 2D dielectric cylinder against the Mie series."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=RADIUS,
        help=f"Cylinder radius a. ka stays {KA:g} (default: {RADIUS:g}).",
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
        help=f"Side length of the square computational region (default: {SIZE:g}).",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=DX,
        help=f"Cell size Δx = Δy (default: {DX:g}).",
    )
    parser.add_argument(
        "--pml",
        type=int,
        default=PML,
        help=f"PML cells on each side (default: {PML}).",
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


def _observe(
    xx: np.ndarray,
    yy: np.ndarray,
    length: float,
    band: float,
    center: tuple[float, float],
    radius: float,
    dx: float,
) -> np.ndarray:
    rho = np.hypot(xx - center[0], yy - center[1])
    return (
        (xx >= band)
        & (xx <= length - band)
        & (yy >= band)
        & (yy <= length - band)
        & (np.abs(rho - radius) >= 2.0 * dx)
    )


def _wave(radius: float) -> tuple[float, float]:
    omega = (KA / radius) * C0
    k = float(omega * np.sqrt(MU0 * EPS0))
    return omega, k


def _law(
    center: tuple[float, float], radius: float, eps_r: float
) -> StaircaseIsotropic:
    return StaircaseIsotropic(
        UniformIsotropic(),
        (Disk(center[0], center[1], radius, eps=eps_r * EPS0),),
    )


def _field(
    polarization: Polarization,
    radius: float,
    index: float,
    size: float,
    dx: float,
    pml_cells: int,
) -> dict[str, np.ndarray | float | str]:
    n = _cells(size, dx)
    center = (0.5 * size, 0.5 * size)
    omega, k = _wave(radius)
    eps_r = index**2
    grid = YeeGrid2D(n, n, dx, dx, polarization, Boundary.PEC)
    band = pml_cells * dx
    if polarization is Polarization.TMZ:
        components = _law(center, radius, eps_r).sample_tm(grid)
        operator = FDFDOperator(grid, components, PMLSpec.box(pml_cells))
        x, y = grid.coordinates("ez")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        incident = np.exp(-1j * k * (xx - center[0]))
        current = operator.layout.pack_e(
            {"ez": 1j * omega * (components.eps_z - EPS0) * incident}
        )
        solved = operator.solve(omega, current)
        total = incident.copy()
        dof = operator.layout.e[0]
        total[dof.i, dof.j] = solved + incident[dof.i, dof.j]
        total[0, :] = total[-1, :] = total[:, 0] = total[:, -1] = 0.0
        exact = cylinder_ez(xx, yy, k=k, radius=radius, eps_r=eps_r, center=center)
        mask = _observe(xx, yy, size, band, center, radius, dx)
        error = _relative(total[mask], exact[mask])
        return {
            "x": x,
            "y": y,
            "field": total,
            "exact": exact,
            "mask": mask,
            "title": f"TMz $E_z$, L2 {error:.2%}",
            "error": error,
        }
    components = _law(center, radius, eps_r).sample_te(grid)
    operator = FDFDOperator(grid, components, PMLSpec.box(pml_cells))
    ex_x, ex_y = grid.coordinates("ex")
    ey_x, ey_y = grid.coordinates("ey")
    ex_xx, ex_yy = np.meshgrid(ex_x, ex_y, indexing="ij")
    ey_xx, ey_yy = np.meshgrid(ey_x, ey_y, indexing="ij")
    ex_inc = np.zeros(grid.shapes()["ex"], dtype=np.complex128)
    ey_inc = (k / (omega * EPS0)) * np.exp(-1j * k * (ey_xx - center[0]))
    current = operator.layout.pack_e(
        {
            "ex": 1j * omega * (components.eps_x - EPS0) * ex_inc,
            "ey": 1j * omega * (components.eps_y - EPS0) * ey_inc,
        }
    )
    solved = operator.solve(omega, current)
    n_ex = operator.layout.e[0].i.size
    ex = ex_inc.copy()
    ey = ey_inc.copy()
    ex[operator.layout.e[0].i, operator.layout.e[0].j] = (
        solved[:n_ex] + ex_inc[operator.layout.e[0].i, operator.layout.e[0].j]
    )
    ey[operator.layout.e[1].i, operator.layout.e[1].j] = (
        solved[n_ex:] + ey_inc[operator.layout.e[1].i, operator.layout.e[1].j]
    )
    exact_ex, _ey_unused = cylinder_te_electric(
        ex_xx,
        ex_yy,
        k=k,
        radius=radius,
        eps_r=eps_r,
        omega=omega,
        eps0=EPS0,
        center=center,
    )
    _ex_unused, exact_ey = cylinder_te_electric(
        ey_xx,
        ey_yy,
        k=k,
        radius=radius,
        eps_r=eps_r,
        omega=omega,
        eps0=EPS0,
        center=center,
    )
    mask_ex = _observe(ex_xx, ex_yy, size, band, center, radius, dx)
    mask_ey = _observe(ey_xx, ey_yy, size, band, center, radius, dx)
    error = _relative(
        np.concatenate((ex[mask_ex], ey[mask_ey])),
        np.concatenate((exact_ex[mask_ex], exact_ey[mask_ey])),
    )
    ex[:, 0] = ex[:, -1] = 0.0
    ey[0, :] = ey[-1, :] = 0.0
    x, y = grid.coordinates("hz")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    curls = array_curl_e(grid, {"ex": ex, "ey": ey})
    field = -curls["hz"] / (1j * omega * components.mu_z)
    exact = cylinder_hz(xx, yy, k=k, radius=radius, eps_r=eps_r, center=center)
    return {
        "x": x,
        "y": y,
        "field": field,
        "exact": exact,
        "mask": _observe(xx, yy, size, band, center, radius, dx),
        "title": rf"TEz $H_z$, L2 {error:.2%}",
        "error": error,
    }


def _panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    limit: float,
    center: tuple[float, float],
    radius: float,
    band: float,
    size: float,
):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mesh = ax.pcolormesh(
        xx,
        yy,
        values,
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
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
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def plot_fields(
    path: Path,
    radius: float,
    index: float,
    size: float,
    dx: float,
    pml_cells: int,
) -> tuple[float, float]:
    rows = (
        _field(Polarization.TMZ, radius, index, size, dx, pml_cells),
        _field(Polarization.TEZ, radius, index, size, dx, pml_cells),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6), constrained_layout=True)
    center = (0.5 * size, 0.5 * size)
    band = pml_cells * dx
    for result, row in zip(rows, axes, strict=True):
        field = np.real(np.asarray(result["field"]))
        exact = np.real(np.asarray(result["exact"]))
        mask = np.asarray(result["mask"], dtype=bool)
        limit = max(
            float(np.max(np.abs(field[mask]))), float(np.max(np.abs(exact[mask]))), 1e-6
        )
        diff = field - exact
        diff_limit = max(float(np.max(np.abs(diff[mask]))), 1e-6)
        panels = (
            (row[0], field, limit, "FD"),
            (row[1], exact, limit, "Mie"),
            (row[2], diff, diff_limit, "FD − Mie"),
        )
        for ax, values, scale, label in panels:
            image = _panel(
                ax,
                result["x"],
                result["y"],
                values,
                scale,
                center,
                radius,
                band,
                size,
            )
            ax.set_title(label)
            if label == "FD":
                ax.text(
                    0.02,
                    0.98,
                    str(result["title"]),
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
        rf"Open dielectric cylinder, $n={index:g}$, $a={radius:g}$, $ka={KA:g}$, "
        rf"size ${size:g}$, $\Delta={dx:g}$, PML {pml_cells} cells, "
        rf"incident $e^{{-jk(x-x_c)}}$",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return float(rows[0]["error"]), float(rows[1]["error"])


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
    segments += [[(0.0, float(y)), (size, float(y))] for y in edges]

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
    ax.set_ylabel("y")
    ax.set_title(
        "Mesh, PML, and dielectric cylinder\n"
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
    tmz, tez = plot_fields(
        folder / "dielectric_cylinder_open.png",
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
    print(f"{folder.name}  TMz L2 {tmz:.4%}  TEz L2 {tez:.4%}")


if __name__ == "__main__":
    main()
