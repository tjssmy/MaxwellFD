"""Oblique GSTC sheet on one periodic cell, compared with analytic R and T.

The sheet spans the cell. ``x`` is periodic with phase one and the width is
one transverse period, so an oblique wave closes without a Bloch phase.
The ``y`` ends carry the analytic total field. ``--chi-nn`` is the normal
magnetic susceptibility. It changes TMz and leaves TEz on the tangential
formula. ``--theta`` is degrees from the sheet normal. Results go in
``periodic_sheet_<parameters>/`` next to this script.
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

from maxwell_fd.analytics.sheets import (  # noqa: E402
    oblique_sheet_fields,
    sheet_coefficients,
)
from maxwell_fd.drivers.fdfd2d import FDFDOperator  # noqa: E402
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D  # noqa: E402
from maxwell_fd.materials.sheets import SymmetricSheet  # noqa: E402
from maxwell_fd.materials.volume import UniformIsotropic  # noqa: E402
from maxwell_fd.utils.constants import C0  # noqa: E402

CHI_EE = 0.5
CHI_MM = 0.25
CHI_NN = 0.4
WAVELENGTH = 1.0
POINTS = 20
THETA = 30.0
HEIGHT = 4.0


def _relative(numerical: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def _fmt(value: complex) -> str:
    return f"{value.real:.4g}{value.imag:+.4g}j"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a periodic oblique GSTC sheet with analytic R and T."
    )
    parser.add_argument("--chi-ee", type=float, default=CHI_EE)
    parser.add_argument("--chi-mm", type=float, default=CHI_MM)
    parser.add_argument("--chi-nn", type=float, default=CHI_NN)
    parser.add_argument(
        "--wavelength",
        type=float,
        default=WAVELENGTH,
        help=f"Free-space wavelength (default: {WAVELENGTH:g}).",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=POINTS,
        help=f"Cells per wavelength (default: {POINTS}).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=THETA,
        help=f"Incidence angle in degrees (default: {THETA:g}).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=HEIGHT,
        help=f"Domain height in wavelengths (default: {HEIGHT:g}).",
    )
    return parser.parse_args(argv)


def _grid_shape(wavelength: float, points: int, theta_deg: float, height: float):
    if int(points) != points or points < 4:
        raise ValueError(f"points must be an integer of at least 4, got {points}")
    if not np.isfinite(theta_deg) or not 0.0 < abs(theta_deg) < 90.0:
        raise ValueError(f"theta must lie between 0 and 90 degrees, got {theta_deg}")
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError(f"height must be positive, got {height}")
    theta = float(np.deg2rad(theta_deg))
    length = wavelength / float(np.sin(theta))
    nx = int(round(points / float(np.sin(theta))))
    if nx < 4:
        raise ValueError("the transverse period needs at least 4 cells")
    dx = length / nx
    ny = int(round(height * wavelength / dx))
    if ny < 4:
        raise ValueError("the height needs at least 4 cells")
    y_sheet = (ny // 2 - 0.5) * dx
    return nx, ny, dx, length, ny * dx, y_sheet, theta


def _solve(pol: Polarization, args: argparse.Namespace) -> dict:
    nx, ny, dx, length, height, y_sheet, theta = _grid_shape(
        args.wavelength, args.points, args.theta, args.height
    )
    omega = 2.0 * np.pi * C0 / args.wavelength
    grid = YeeGrid2D(nx, ny, dx, dx, pol, Boundary.PERIODIC_X)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    operator = FDFDOperator(
        grid,
        bare,
        sheet=SymmetricSheet(y_sheet, args.chi_ee, args.chi_mm, chi_mm_nn=args.chi_nn),
    )
    reflected, transmitted = sheet_coefficients(
        args.chi_ee,
        args.chi_mm,
        omega / C0,
        theta=theta,
        te=pol is Polarization.TEZ,
        chi_mm_nn=args.chi_nn,
    )
    if pol is Polarization.TMZ:
        x, y = grid.coordinates("ez")
        exact = oblique_sheet_fields(
            x,
            y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
            theta=theta,
        )["ez"]
        solved = operator.solve(
            omega, np.zeros(operator.layout.n_e), dirichlet={"ez": exact}
        )
        field = exact.copy()
        dof = operator.layout.e[0]
        field[dof.i, dof.j] = solved
        error = _relative(solved, operator.layout.pack_e({"ez": exact}))
        name = "TMz $E_z$"
    else:
        x, y = grid.coordinates("ex")
        exact = oblique_sheet_fields(
            x,
            y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
            theta=theta,
        )["ex"]
        ey_x, ey_y = grid.coordinates("ey")
        ey = oblique_sheet_fields(
            ey_x,
            ey_y,
            y_sheet=y_sheet,
            omega=omega,
            reflected=reflected,
            transmitted=transmitted,
            theta=theta,
        )["ey"]
        solved = operator.solve(
            omega,
            np.zeros(operator.layout.n_e),
            dirichlet={"ex": exact, "ey": ey},
        )
        field = exact.copy()
        dof = operator.layout.e[0]
        field[dof.i, dof.j] = solved[: dof.i.size]
        error = _relative(solved, operator.layout.pack_e({"ex": exact, "ey": ey}))
        name = "TEz $E_x$"
    return {
        "x": x,
        "y": y,
        "field": field,
        "exact": exact,
        "error": error,
        "reflected": reflected,
        "transmitted": transmitted,
        "length": length,
        "height": height,
        "y_sheet": y_sheet,
        "dx": dx,
        "nx": nx,
        "ny": ny,
        "title": f"{name}\nL2 {error:.2%}\n$R={_fmt(reflected)}$\n$T={_fmt(transmitted)}$",
    }


def _image(ax: plt.Axes, x, y, values, limit: float, y_sheet: float):
    xx, yy = np.meshgrid(np.asarray(x), np.asarray(y), indexing="ij")
    mesh = ax.pcolormesh(
        xx,
        yy,
        np.real(np.asarray(values)),
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.axhline(y_sheet, color="#08519c", lw=0.9, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def plot_fields(path: Path, tm: dict, te: dict, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), constrained_layout=True)
    for result, row in zip((tm, te), axes, strict=True):
        field = np.real(np.asarray(result["field"]))
        exact = np.real(np.asarray(result["exact"]))
        limit = max(float(np.max(np.abs(field))), float(np.max(np.abs(exact))), 1e-6)
        diff = field - exact
        diff_limit = max(float(np.max(np.abs(diff))), 1e-6)
        panels = (
            (row[0], field, limit, "FD"),
            (row[1], exact, limit, "Analytic"),
            (row[2], diff, diff_limit, "FD − Analytic"),
        )
        for ax, values, scale, label in panels:
            image = _image(
                ax,
                result["x"],
                result["y"],
                values,
                scale,
                float(result["y_sheet"]),
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
                        "alpha": 0.8,
                        "edgecolor": "none",
                        "pad": 1.5,
                    },
                )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        rf"Periodic sheet, $\theta={args.theta:g}^\circ$, "
        rf"$\chi_{{ee}}={args.chi_ee:g}$, $\chi_{{mm}}={args.chi_mm:g}$, "
        rf"$\chi_{{mm}}^{{nn}}={args.chi_nn:g}$, $\lambda={args.wavelength:g}$",
        fontsize=12,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_domain(path: Path, result: dict, args: argparse.Namespace) -> None:
    length = float(result["length"])
    height = float(result["height"])
    y_sheet = float(result["y_sheet"])
    nx, ny = int(result["nx"]), int(result["ny"])
    x_edges = np.linspace(0.0, length, nx + 1)
    y_edges = np.linspace(0.0, height, ny + 1)
    segments = [[(float(x), 0.0), (float(x), height)] for x in x_edges]
    segments += [[(0.0, float(y)), (length, float(y))] for y in y_edges]
    trace = "#006d2c"
    fig, ax = plt.subplots(figsize=(4.8, 8.2))
    ax.add_collection(
        LineCollection(segments, colors="#bdbdbd", linewidths=0.25, zorder=2)
    )
    ax.plot([0.0, length], [y_sheet, y_sheet], color="#08519c", lw=1.6, zorder=4)
    ax.plot([0.0, length], [0.0, 0.0], color=trace, lw=1.8, zorder=5)
    ax.plot([0.0, length], [height, height], color=trace, lw=1.8, zorder=5)
    ax.plot([0.0, 0.0], [0.0, height], color="#6a3d9a", lw=1.4, ls="--", zorder=5)
    ax.plot([length, length], [0.0, height], color="#6a3d9a", lw=1.4, ls="--", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "One transverse period\n"
        rf"$\theta={args.theta:g}^\circ$, $\chi_{{mm}}^{{nn}}={args.chi_nn:g}$, "
        rf"$\Delta={float(result['dx']):.4g}$"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", lw=1.0, label="mesh"),
            Line2D([0], [0], color="#08519c", lw=1.6, label="sheet"),
            Line2D([0], [0], color=trace, lw=1.8, label="analytic trace"),
            Line2D([0], [0], color="#6a3d9a", lw=1.4, ls="--", label="periodic"),
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
        __file__,
        lam=args.wavelength,
        theta=args.theta,
        points=args.points,
        height=args.height,
        chiee=args.chi_ee,
        chimm=args.chi_mm,
        chinn=args.chi_nn,
    )
    tm = _solve(Polarization.TMZ, args)
    te = _solve(Polarization.TEZ, args)
    plot_fields(folder / "periodic_sheet.png", tm, te, args)
    plot_domain(folder / "domain.png", tm, args)
    print(
        f"{folder.name}  "
        f"TMz R {_fmt(tm['reflected'])} T {_fmt(tm['transmitted'])} "
        f"L2 {tm['error']:.4%}  "
        f"TEz R {_fmt(te['reflected'])} T {_fmt(te['transmitted'])} "
        f"L2 {te['error']:.4%}"
    )


if __name__ == "__main__":
    main()
