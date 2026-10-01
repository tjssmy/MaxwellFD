"""Normal-incidence resistive sheet, solved by 2D FDFD.

The sheet is zero thickness. Its conductance is added to the Ampere cell
that contains ``--y``, and the PEC wall carries the analytic trace. ``--zs``
is the surface impedance in ohms per square. ``--wavelength`` is the
free-space wavelength. ``--length`` and ``--height`` are the sides of the
rectangle and ``--dx`` is the cell size. Results go in
``resistive_sheet_<parameters>/`` next to this script.
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

from maxwell_fd.analytics.sheets import resistive_sheet_fields
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.sheets import ResistiveSheet
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.utils.constants import C0, ETA0

ZS = float(ETA0)
WAVELENGTH = 46.0
Y_SHEET = 40.0
LENGTH = 8.0
HEIGHT = 80.0
DX = 1.0


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
        description="Plot a resistive sheet against its analytic field."
    )
    parser.add_argument(
        "--zs", type=float, default=ZS, help=f"Surface impedance (default: {ZS:.6g})."
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=WAVELENGTH,
        help=f"Free-space wavelength (default: {WAVELENGTH:g}).",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=Y_SHEET,
        help=f"Sheet position (default: {Y_SHEET:g}).",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=LENGTH,
        help=f"Size along x (default: {LENGTH:g}).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=HEIGHT,
        help=f"Size along y (default: {HEIGHT:g}).",
    )
    parser.add_argument(
        "--dx", type=float, default=DX, help=f"Cell size (default: {DX:g})."
    )
    return parser.parse_args(argv)


def _solve(
    pol: Polarization,
    zs: float,
    wavelength: float,
    y_sheet: float,
    length: float,
    height: float,
    dx: float,
):
    if not 0.0 < y_sheet < height:
        raise ValueError(f"sheet must lie inside the domain, got y={y_sheet}")
    nx, ny = _cells(length, dx), _cells(height, dx)
    omega = 2.0 * np.pi * C0 / wavelength
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PEC)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    components = ResistiveSheet(y_sheet, zs).apply(grid, bare)
    operator = FDFDOperator(grid, components)
    if pol is Polarization.TMZ:
        x, y = grid.coordinates("ez")
        ez_line, _hx, _hz, _ex = resistive_sheet_fields(
            y, y_sheet=y_sheet, omega=omega, zs=zs
        )
        exact = np.broadcast_to(ez_line, grid.shapes()["ez"]).copy()
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": exact}
        )
        field = exact.copy()
        dof = operator.layout.e[0]
        field[dof.i, dof.j] = solved
        error = _relative(solved, operator.layout.pack_e({"ez": exact}))
        return {
            "x": x,
            "y": y,
            "field": field,
            "exact": exact,
            "error": error,
            "title": f"TMz $E_z$, L2 {error:.2%}",
        }
    x, y = grid.coordinates("ex")
    _ez, _hx, _hz, ex_line = resistive_sheet_fields(
        y, y_sheet=y_sheet, omega=omega, zs=zs
    )
    exact = np.broadcast_to(ex_line, grid.shapes()["ex"]).copy()
    ey = np.zeros(grid.shapes()["ey"], dtype=np.complex128)
    solved = operator.solve(
        omega, np.zeros(operator.layout.n_e), dirichlet={"ex": exact, "ey": ey}
    )
    field = exact.copy()
    dof = operator.layout.e[0]
    field[dof.i, dof.j] = solved[: dof.i.size]
    error = _relative(solved, operator.layout.pack_e({"ex": exact, "ey": ey}))
    return {
        "x": x,
        "y": y,
        "field": field,
        "exact": exact,
        "error": error,
        "title": f"TEz $E_x$, L2 {error:.2%}",
    }


def _image(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    limit: float,
    y_sheet: float,
):
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
    ax.axvline(y_sheet, color="#08519c", lw=0.9, zorder=3)
    ax.set_xlim(float(np.min(y)), float(np.max(y)))
    ax.set_ylim(float(np.min(x)), float(np.max(x)))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    return mesh


def plot_fields(
    path: Path,
    tm: dict,
    te: dict,
    *,
    zs: float,
    wavelength: float,
    y_sheet: float,
    dx: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 5.4), constrained_layout=True)
    for result, row in zip((tm, te), axes, strict=True):
        field = np.real(np.asarray(result["field"]))
        exact = np.real(np.asarray(result["exact"]))
        limit = max(float(np.max(np.abs(field))), float(np.max(np.abs(exact))), 1e-6)
        diff = field - exact
        diff_limit = max(float(np.max(np.abs(diff))), 1e-6)
        panels = (
            (row[0], field, limit, "FD"),
            (row[1], exact, limit, "Sheet"),
            (row[2], diff, diff_limit, "FD − Sheet"),
        )
        for ax, values, scale, label in panels:
            image = _image(
                ax,
                np.asarray(result["x"]),
                np.asarray(result["y"]),
                values,
                scale,
                y_sheet,
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
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        rf"Resistive sheet, $Z_s={zs:.6g}\,\Omega$, $\lambda={wavelength:g}$, "
        rf"$y={y_sheet:g}$, $\Delta={dx:g}$, incident $e^{{-jk(y-y_s)}}$",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_domain(
    path: Path, *, zs: float, y_sheet: float, length: float, height: float, dx: float
) -> None:
    nx, ny = _cells(length, dx), _cells(height, dx)
    y_edges = np.linspace(0.0, height, ny + 1)
    x_edges = np.linspace(0.0, length, nx + 1)
    segments = [[(float(y), 0.0), (float(y), length)] for y in y_edges]
    segments += [[(0.0, float(x)), (height, float(x))] for x in x_edges]
    trace = "#006d2c"
    fig, ax = plt.subplots(figsize=(12.2, 3.8))
    ax.add_collection(
        LineCollection(segments, colors="#bdbdbd", linewidths=0.35, zorder=3)
    )
    ax.plot([y_sheet, y_sheet], [0.0, length], color="#08519c", lw=1.6, zorder=4)
    ax.plot([0.0, 0.0], [0.0, length], color=trace, lw=1.4, zorder=5)
    ax.plot([height, height], [0.0, length], color=trace, lw=1.4, zorder=5)
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
        "Mesh and resistive sheet\n"
        rf"$Z_s={zs:.6g}\,\Omega$, $y={y_sheet:g}$, region ${length:g}\times{height:g}$, $\Delta={dx:g}$"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", linewidth=1.0, label="mesh"),
            Line2D(
                [0],
                [0],
                color="#08519c",
                linewidth=1.6,
                label="resistive sheet",
            ),
            Line2D([0], [0], color=trace, linewidth=2.2, label="analytic trace"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
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
        zs=args.zs,
        lam=args.wavelength,
        y=args.y,
        length=args.length,
        height=args.height,
        dx=args.dx,
    )
    tm = _solve(
        Polarization.TMZ,
        args.zs,
        args.wavelength,
        args.y,
        args.length,
        args.height,
        args.dx,
    )
    te = _solve(
        Polarization.TEZ,
        args.zs,
        args.wavelength,
        args.y,
        args.length,
        args.height,
        args.dx,
    )
    plot_fields(
        folder / "resistive_sheet.png",
        tm,
        te,
        zs=args.zs,
        wavelength=args.wavelength,
        y_sheet=args.y,
        dx=args.dx,
    )
    plot_domain(
        folder / "domain.png",
        zs=args.zs,
        y_sheet=args.y,
        length=args.length,
        height=args.height,
        dx=args.dx,
    )
    print(f"{folder.name}  TMz L2 {tm['error']:.4%}  TEz L2 {te['error']:.4%}")


if __name__ == "__main__":
    main()
