"""Normal-incidence Fresnel slab, solved by 2D FDFD.

The PEC wall carries the analytic trace and the impressed current is zero.
``--index`` is the refractive index ``n``, with ``ε_r = n²``. ``--thickness``
is the slab thickness and ``--y1`` is the front face. ``--length`` and
``--height`` are the sides of the rectangular region, ``--dx`` is the cell
size, and ``--pml`` is the PML thickness marked at each end in ``y``.
The walls ``x = 0`` and ``x =`` length carry the analytic trace.
``--wavelength`` is the free-space wavelength. Results go in
``fresnel_slab_<parameters>/`` next to this script.
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
from matplotlib.patches import Patch, Rectangle

from maxwell_fd.analytics.fresnel import slab_fields
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import SlabY, StaircaseIsotropic, UniformIsotropic
from maxwell_fd.operators.curl2d import array_curl_e
from maxwell_fd.utils.constants import C0, EPS0

INDEX = 2.0
THICKNESS = 20.0
WAVELENGTH = 46.0
Y1 = 30.375
LENGTH = 8.0
HEIGHT = 80.0
DX = 0.5
PML = 1.0


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
        description="Plot a 2D Fresnel slab against the analytic field."
    )
    parser.add_argument(
        "--index",
        type=float,
        default=INDEX,
        help=f"Refractive index n, with ε_r = n² (default: {INDEX:g}).",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=THICKNESS,
        help=f"Slab thickness (default: {THICKNESS:g}).",
    )
    parser.add_argument(
        "--y1",
        type=float,
        default=Y1,
        help=f"Front face. The default sits off the electric nodes (default: {Y1:g}).",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=WAVELENGTH,
        help=f"Free-space wavelength (default: {WAVELENGTH:g}).",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=LENGTH,
        help=f"Region size along x (default: {LENGTH:g}).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=HEIGHT,
        help=f"Region size along y, the direction of propagation (default: {HEIGHT:g}).",
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
        help=(
            "PML thickness at each end in y. "
            f"The x walls are the analytic trace (default: {PML:g})."
        ),
    )
    args = parser.parse_args(argv)
    try:
        nx = _cells(args.length, args.dx)
        ny = _cells(args.height, args.dx)
        n_pml = _cells(args.pml, args.dx)
    except ValueError as exc:
        parser.error(str(exc))
    if nx < 2 or ny < 2:
        parser.error("length and height must each contain at least 2 cells")
    if 2 * n_pml >= ny:
        parser.error("pml must leave an interior along y")
    y2 = args.y1 + args.thickness
    interior_top = args.height - args.pml
    if (
        not np.isfinite(args.y1)
        or not np.isfinite(args.thickness)
        or args.thickness <= 0.0
        or args.y1 <= args.pml
        or y2 >= interior_top
    ):
        parser.error(
            "thickness must be positive and the slab must lie inside the interior "
            f"({args.pml:g} < y < {interior_top:g})"
        )
    if not np.isfinite(args.index) or args.index <= 0.0:
        parser.error("index must be positive")
    if not np.isfinite(args.wavelength) or args.wavelength <= 0.0:
        parser.error("wavelength must be positive")
    return args


def _solve(
    polarization: Polarization,
    *,
    index: float,
    thickness: float,
    wavelength: float,
    y1: float,
    length: float,
    height: float,
    dx: float,
) -> dict[str, np.ndarray | float | str]:
    """Solve one polarization and keep the 2D map plus the center-column cut."""
    eps_r = index**2
    y2 = y1 + thickness
    nx = _cells(length, dx)
    ny = _cells(height, dx)
    omega = 2.0 * np.pi * C0 / wavelength
    grid = YeeGrid2D(nx, ny, dx, dx, polarization, Boundary.PEC)
    law = StaircaseIsotropic(UniformIsotropic(), (SlabY(y1, y2, eps=eps_r * EPS0),))
    if polarization is Polarization.TMZ:
        components = law.sample_tm(grid)
        operator = FDFDOperator(grid, components)
        x, y = grid.coordinates("ez")
        ez_line, _hx, _hz, _ex = slab_fields(y, y1=y1, y2=y2, omega=omega, eps_r=eps_r)
        exact = np.broadcast_to(ez_line, grid.shapes()["ez"]).copy()
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": exact}
        )
        electric = exact.copy()
        dof = operator.layout.e[0]
        electric[dof.i, dof.j] = solved
        curls = array_curl_e(grid, {"ez": electric})
        magnetic = -curls["hx"] / (1j * omega * components.mu_x)
        _xm, y_h = grid.coordinates("hx")
        error = _relative(solved, dof.pack(exact))
        return {
            "x": x,
            "y": y,
            "field": electric,
            "exact": exact,
            "y_e": y,
            "electric": electric[nx // 2],
            "y_h": y_h,
            "magnetic": magnetic[nx // 2],
            "error": error,
            "title": f"TMz $E_z$, L2 {error:.2%}",
            "electric_name": r"$\mathrm{Re}\,E_z$",
            "magnetic_name": r"$\mathrm{Re}\,H_x$",
        }
    components = law.sample_te(grid)
    operator = FDFDOperator(grid, components)
    _ex_x, ex_y = grid.coordinates("ex")
    _ez, _hx, _hz, ex_line = slab_fields(ex_y, y1=y1, y2=y2, omega=omega, eps_r=eps_r)
    ex = np.broadcast_to(ex_line, grid.shapes()["ex"]).copy()
    ey = np.zeros(grid.shapes()["ey"], dtype=np.complex128)
    solved = operator.solve(
        omega, np.zeros(operator.layout.n_e), dirichlet={"ex": ex, "ey": ey}
    )
    n_ex = operator.layout.e[0].i.size
    ex_num = ex.copy()
    ey_num = ey.copy()
    ex_num[operator.layout.e[0].i, operator.layout.e[0].j] = solved[:n_ex]
    ey_num[operator.layout.e[1].i, operator.layout.e[1].j] = solved[n_ex:]
    curls = array_curl_e(grid, {"ex": ex_num, "ey": ey_num})
    hz = -curls["hz"] / (1j * omega * components.mu_z)
    hz_x, hz_y = grid.coordinates("hz")
    _ez_h, _hx_h, hz_line, _ex_h = slab_fields(
        hz_y, y1=y1, y2=y2, omega=omega, eps_r=eps_r
    )
    exact = np.broadcast_to(hz_line, grid.shapes()["hz"]).copy()
    error = _relative(solved, operator.layout.pack_e({"ex": ex, "ey": ey}))
    return {
        "x": hz_x,
        "y": hz_y,
        "field": hz,
        "exact": exact,
        "y_e": ex_y,
        "electric": ex_num[nx // 2],
        "y_h": hz_y,
        "magnetic": hz[nx // 2],
        "error": error,
        "title": f"TEz $H_z$, electric L2 {error:.2%}",
        "electric_name": r"$\mathrm{Re}\,E_x$",
        "magnetic_name": r"$\mathrm{Re}\,H_z$",
    }


def _image(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    limit: float,
    y1: float,
    y2: float,
):
    """Draw ``values[i, j]`` with propagation ``y`` horizontal."""
    ordinate, abscissa = np.meshgrid(np.asarray(y), np.asarray(x), indexing="ij")
    mesh = ax.pcolormesh(
        ordinate,
        abscissa,
        np.asarray(values).T,
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.axvline(y1, color="0.15", lw=0.8, zorder=3)
    ax.axvline(y2, color="0.15", lw=0.8, zorder=3)
    ax.set_xlim(float(np.min(y)), float(np.max(y)))
    ax.set_ylim(float(np.min(x)), float(np.max(x)))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    return mesh


def plot_fields(
    path: Path,
    tm: dict[str, np.ndarray | float | str],
    te: dict[str, np.ndarray | float | str],
    *,
    index: float,
    thickness: float,
    wavelength: float,
    y1: float,
    dx: float,
) -> None:
    y2 = y1 + thickness
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 5.4), constrained_layout=True)
    for result, row in zip((tm, te), axes, strict=True):
        field = np.real(np.asarray(result["field"]))
        exact = np.real(np.asarray(result["exact"]))
        limit = max(float(np.max(np.abs(field))), float(np.max(np.abs(exact))), 1e-6)
        diff = field - exact
        diff_limit = max(float(np.max(np.abs(diff))), 1e-6)
        panels = (
            (row[0], field, limit, "FD"),
            (row[1], exact, limit, "Fresnel"),
            (row[2], diff, diff_limit, "FD − Fresnel"),
        )
        for ax, values, scale, label in panels:
            image = _image(
                ax,
                np.asarray(result["x"]),
                np.asarray(result["y"]),
                values,
                scale,
                y1,
                y2,
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
                    fontsize=8,
                    color="0.15",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.75,
                        "edgecolor": "none",
                        "pad": 1.5,
                    },
                )
            fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(
        rf"Fresnel slab, $n={index:g}$, $d={thickness:g}$, $\lambda={wavelength:g}$, "
        rf"$\Delta={dx:g}$, incident $e^{{-jky}}$",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _draw_cut(
    ax: plt.Axes,
    result: dict[str, np.ndarray | float | str],
    *,
    magnetic: bool,
    dense: dict[str, np.ndarray],
    y1: float,
    y2: float,
    height: float,
    eps_r: float,
) -> None:
    ax.axvspan(y1, y2, color="0.85", zorder=0, label=rf"slab $\varepsilon_r={eps_r:g}$")
    if magnetic:
        y_key, num_key, name = "y_h", "magnetic", "magnetic_name"
        y_dense, exact_dense = dense["y"], dense["magnetic"]
    else:
        y_key, num_key, name = "y_e", "electric", "electric_name"
        y_dense, exact_dense = dense["y"], dense["electric"]
    ax.plot(
        y_dense, np.real(exact_dense), color="0.15", lw=1.4, label="Fresnel", zorder=2
    )
    y = np.asarray(result[y_key])
    step = max(1, y.size // 40)
    ax.plot(
        y[::step],
        np.real(np.asarray(result[num_key])[::step]),
        linestyle="none",
        marker="o",
        ms=3.5,
        color="C1",
        label="FD",
        zorder=3,
    )
    ax.set_xlim(0.0, height)
    ax.set_xlabel("y")
    ax.set_ylabel(str(result[name]))
    ax.grid(True, alpha=0.3)


def plot_cuts(
    path: Path,
    tm: dict[str, np.ndarray | float | str],
    te: dict[str, np.ndarray | float | str],
    *,
    index: float,
    thickness: float,
    wavelength: float,
    y1: float,
    height: float,
    dx: float,
) -> None:
    eps_r = index**2
    y2 = y1 + thickness
    omega = 2.0 * np.pi * C0 / wavelength
    y_dense = np.linspace(0.0, height, 900)
    ez, hx, hz, ex = slab_fields(y_dense, y1=y1, y2=y2, omega=omega, eps_r=eps_r)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), sharex=True)
    panels = (
        (axes[0, 0], tm, False, {"y": y_dense, "electric": ez}),
        (axes[0, 1], tm, True, {"y": y_dense, "magnetic": hx}),
        (axes[1, 0], te, False, {"y": y_dense, "electric": ex}),
        (axes[1, 1], te, True, {"y": y_dense, "magnetic": hz}),
    )
    for ax, result, magnetic, dense in panels:
        _draw_cut(
            ax,
            result,
            magnetic=magnetic,
            dense=dense,
            y1=y1,
            y2=y2,
            height=height,
            eps_r=eps_r,
        )
    axes[0, 0].set_title(f"TMz electric, free-sample L2 {tm['error']:.2%}")
    axes[0, 1].set_title(r"TMz magnetic, $H=-\nabla\times E/(j\omega\mu)$")
    axes[1, 0].set_title(f"TEz electric, free-sample L2 {te['error']:.2%}")
    axes[1, 1].set_title(r"TEz magnetic, $H=-\nabla\times E/(j\omega\mu)$")
    axes[0, 0].legend(loc="upper left", frameon=True)
    fig.suptitle(
        rf"Fresnel slab, $n={index:g}$, $d={thickness:g}$, $\lambda={wavelength:g}$, $\Delta={dx:g}$",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_domain(
    path: Path,
    *,
    index: float,
    thickness: float,
    y1: float,
    length: float,
    height: float,
    dx: float,
    pml: float,
) -> None:
    """Mesh, y-end PML bands, and the slab. Propagation ``y`` is horizontal.

    The walls ``x = 0`` and ``x =`` length are drawn as the PEC trace.
    """
    nx = _cells(length, dx)
    ny = _cells(height, dx)
    n_pml = _cells(pml, dx)
    origin = n_pml * dx
    y2 = y1 + thickness
    y_edges = np.linspace(0.0, height, ny + 1)
    x_edges = np.linspace(0.0, length, nx + 1)
    segments = [[(float(y), 0.0), (float(y), length)] for y in y_edges]
    segments += [[(0.0, float(x)), (height, float(x))] for x in x_edges]
    trace = "#006d2c"

    fig, ax = plt.subplots(figsize=(12.2, 3.8))
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), origin, length, facecolor="#fdd0a2", edgecolor="none", zorder=0
        )
    )
    ax.add_patch(
        Rectangle(
            (height - origin, 0.0),
            origin,
            length,
            facecolor="#fdd0a2",
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (y1, 0.0),
            thickness,
            length,
            facecolor="#6baed6",
            edgecolor="none",
            alpha=0.55,
            zorder=2,
        )
    )
    ax.add_collection(
        LineCollection(segments, colors="#bdbdbd", linewidths=0.35, zorder=3)
    )
    ax.plot([y1, y1], [0.0, length], color="#08519c", lw=1.1, zorder=4)
    ax.plot([y2, y2], [0.0, length], color="#08519c", lw=1.1, zorder=4)
    ax.plot([origin, origin], [0.0, length], color="#e6550d", lw=1.3, zorder=4)
    ax.plot(
        [height - origin, height - origin],
        [0.0, length],
        color="#e6550d",
        lw=1.3,
        zorder=4,
    )
    ax.plot([0.0, 0.0], [0.0, length], color="#252525", lw=1.0, zorder=5)
    ax.plot([height, height], [0.0, length], color="#252525", lw=1.0, zorder=5)
    ax.plot(
        [0.0, height],
        [0.0, 0.0],
        color=trace,
        lw=2.2,
        zorder=6,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.plot(
        [0.0, height],
        [length, length],
        color=trace,
        lw=2.2,
        zorder=6,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.set_aspect("equal")
    ax.set_xlim(0.0, height)
    ax.set_ylim(0.0, length)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_title(
        "Mesh, y-end PML, and slab\n"
        rf"$n={index:g}$, $d={thickness:g}$, $y_1={y1:g}$, "
        rf"region ${length:g}\times{height:g}$, $\Delta={dx:g}$, PML ${pml:g}$"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", linewidth=1.0, label="mesh"),
            Patch(facecolor="#fdd0a2", edgecolor="#e6550d", label=f"PML ({pml:g})"),
            Patch(
                facecolor="#6baed6",
                edgecolor="#08519c",
                alpha=0.55,
                label=rf"slab, $n={index:g}$",
            ),
            Line2D([0], [0], color=trace, linewidth=2.2, label="PEC trace"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
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
        d=args.thickness,
        lam=args.wavelength,
        y1=args.y1,
        length=args.length,
        height=args.height,
        dx=args.dx,
        pml=args.pml,
    )
    shared = {
        "index": args.index,
        "thickness": args.thickness,
        "wavelength": args.wavelength,
        "y1": args.y1,
        "length": args.length,
        "height": args.height,
        "dx": args.dx,
    }
    tm = _solve(Polarization.TMZ, **shared)
    te = _solve(Polarization.TEZ, **shared)
    plot_fields(
        folder / "fresnel_slab.png",
        tm,
        te,
        index=args.index,
        thickness=args.thickness,
        wavelength=args.wavelength,
        y1=args.y1,
        dx=args.dx,
    )
    plot_cuts(
        folder / "cuts.png",
        tm,
        te,
        index=args.index,
        thickness=args.thickness,
        wavelength=args.wavelength,
        y1=args.y1,
        height=args.height,
        dx=args.dx,
    )
    plot_domain(
        folder / "domain.png",
        index=args.index,
        thickness=args.thickness,
        y1=args.y1,
        length=args.length,
        height=args.height,
        dx=args.dx,
        pml=args.pml,
    )
    print(folder)


if __name__ == "__main__":
    main()
