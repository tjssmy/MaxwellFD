"""Dielectric cylinder in 2D FDFD, compared with the Mie series.

The PEC wall carries the analytic trace and the impressed current is zero.
``--radius`` is the cylinder radius and ``--index`` is the refractive index
``n``, with ``ε_r = n²``. ``--size`` is the side length of the square region
and ``--dx`` is the cell size. ``--pml`` is the PML thickness marked on each
side of the domain figure. ``ka`` stays 1, so the wavelength scales with the
radius. Results go in ``dielectric_cylinder_<parameters>/`` next to this script.
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
from maxwell_fd.utils.constants import C0, EPS0

INDEX = 1.5
KA = 1.0
RADIUS = 10.0
SIZE = 50.0
DX = 1.0
PML = 8.0


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
        description="Plot a 2D dielectric cylinder against the Mie series."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=RADIUS,
        help=(
            f"Cylinder radius a. ka stays {KA:g}, so the wavelength scales with a "
            f"(default: {RADIUS:g})."
        ),
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
        type=float,
        default=PML,
        help=f"PML thickness on each side of the domain figure (default: {PML:g}).",
    )
    args = parser.parse_args(argv)
    try:
        n = _cells(args.size, args.dx)
        n_pml = _cells(args.pml, args.dx)
    except ValueError as exc:
        parser.error(str(exc))
    if n < 2:
        parser.error("size must contain at least 2 cells")
    if 2 * n_pml >= n:
        parser.error("pml must leave an interior on a grid of at least 2 cells")
    interior = 0.5 * args.size - args.pml
    if not np.isfinite(args.radius) or args.radius <= 0.0 or args.radius >= interior:
        parser.error(
            f"radius must be positive and lie inside the interior (half-width {interior:g})"
        )
    if not np.isfinite(args.index) or args.index <= 0.0:
        parser.error("index must be positive")
    return args


def _field(
    polarization: Polarization,
    radius: float,
    eps_r: float,
    size: float,
    dx: float,
) -> dict[str, np.ndarray | float | str]:
    n = _cells(size, dx)
    center = (0.5 * size, 0.5 * size)
    k = KA / radius
    omega = k * C0
    grid = YeeGrid2D(n, n, dx, dx, polarization, Boundary.PEC)
    law = StaircaseIsotropic(
        UniformIsotropic(),
        (Disk(center[0], center[1], radius, eps=eps_r * EPS0),),
    )
    if polarization is Polarization.TMZ:
        components = law.sample_tm(grid)
        operator = FDFDOperator(grid, components)
        x, y = grid.coordinates("ez")
        xx, yy = np.meshgrid(x, y, indexing="ij")
        exact = cylinder_ez(xx, yy, k=k, radius=radius, eps_r=eps_r, center=center)
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": exact}
        )
        field = exact.copy()
        dof = operator.layout.e[0]
        field[dof.i, dof.j] = solved
        error = _relative(dof.pack(field), dof.pack(exact))
        return {
            "x": x,
            "y": y,
            "field": field,
            "exact": exact,
            "title": f"TMz $E_z$, L2 {error:.2%}",
        }
    components = law.sample_te(grid)
    operator = FDFDOperator(grid, components)
    x, y = grid.coordinates("hz")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    exact = cylinder_hz(xx, yy, k=k, radius=radius, eps_r=eps_r, center=center)
    ex_x, ex_y = grid.coordinates("ex")
    ey_x, ey_y = grid.coordinates("ey")
    ex_xx, ex_yy = np.meshgrid(ex_x, ex_y, indexing="ij")
    ey_xx, ey_yy = np.meshgrid(ey_x, ey_y, indexing="ij")
    ex_exact, _ey = cylinder_te_electric(
        ex_xx,
        ex_yy,
        k=k,
        radius=radius,
        eps_r=eps_r,
        omega=omega,
        eps0=EPS0,
        center=center,
    )
    _ex, ey_exact = cylinder_te_electric(
        ey_xx,
        ey_yy,
        k=k,
        radius=radius,
        eps_r=eps_r,
        omega=omega,
        eps0=EPS0,
        center=center,
    )
    solved = operator.solve(
        omega,
        np.zeros(operator.layout.n_e),
        dirichlet={"ex": ex_exact, "ey": ey_exact},
    )
    n_ex = operator.layout.e[0].i.size
    ex = ex_exact.copy()
    ey = ey_exact.copy()
    ex[operator.layout.e[0].i, operator.layout.e[0].j] = solved[:n_ex]
    ey[operator.layout.e[1].i, operator.layout.e[1].j] = solved[n_ex:]
    curls = array_curl_e(grid, {"ex": ex, "ey": ey})
    field = -curls["hz"] / (1j * omega * components.mu_z)
    error = _relative(
        operator.layout.pack_e({"ex": ex, "ey": ey}),
        operator.layout.pack_e({"ex": ex_exact, "ey": ey_exact}),
    )
    return {
        "x": x,
        "y": y,
        "field": field,
        "exact": exact,
        "title": rf"TEz $H_z$, L2 {error:.2%}",
    }


def _image(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    limit: float,
    center: tuple[float, float],
    radius: float,
):
    xx, yy = np.meshgrid(np.asarray(x), np.asarray(y), indexing="ij")
    mesh = ax.pcolormesh(
        xx,
        yy,
        np.asarray(values),
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.add_patch(Circle(center, radius, fill=False, ec="0.1", lw=0.9, zorder=3))
    ax.set_aspect("equal")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def plot_fields(
    path: Path, radius: float, index: float, size: float, dx: float
) -> None:
    eps_r = index**2
    rows = (
        _field(Polarization.TMZ, radius, eps_r, size, dx),
        _field(Polarization.TEZ, radius, eps_r, size, dx),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0), constrained_layout=True)
    center = (0.5 * size, 0.5 * size)
    for result, row in zip(rows, axes, strict=True):
        field = np.real(np.asarray(result["field"]))
        exact = np.real(np.asarray(result["exact"]))
        limit = max(float(np.max(np.abs(field))), float(np.max(np.abs(exact))), 1e-6)
        diff = field - exact
        diff_limit = max(float(np.max(np.abs(diff))), 1e-6)
        panels = (
            (row[0], field, limit, "FD"),
            (row[1], exact, limit, "Mie"),
            (row[2], diff, diff_limit, "FD − Mie"),
        )
        for ax, values, scale, label in panels:
            image = _image(ax, result["x"], result["y"], values, scale, center, radius)
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
        rf"Dielectric cylinder, $n={index:g}$, $a={radius:g}$, $ka={KA:g}$, "
        rf"size ${size:g}$, $\Delta={dx:g}$, incident $e^{{-jkx}}$",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_domain(
    path: Path,
    radius: float,
    index: float,
    size: float,
    dx: float,
    pml: float,
) -> None:
    n = _cells(size, dx)
    n_pml = _cells(pml, dx)
    origin = n_pml * dx
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
        rf"Mesh, PML, and cylinder"
        "\n"
        rf"$n={index:g}$, $a={radius:g}$, size ${size:g}$, $\Delta={dx:g}$, PML ${pml:g}$"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", linewidth=1.0, label="mesh"),
            Patch(facecolor="#fdd0a2", edgecolor="#e6550d", label=f"PML ({pml:g})"),
            Patch(
                facecolor="#6baed6",
                edgecolor="#08519c",
                alpha=0.55,
                label=rf"cylinder, $n={index:g}$",
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
    plot_fields(
        folder / "dielectric_cylinder.png",
        args.radius,
        args.index,
        args.size,
        args.dx,
    )
    plot_domain(
        folder / "domain.png",
        args.radius,
        args.index,
        args.size,
        args.dx,
        args.pml,
    )
    print(folder)


if __name__ == "__main__":
    main()
