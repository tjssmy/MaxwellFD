"""Periodic sheet with a rational magnetic susceptibility.

``chi_mm(k_t) = (chi_mm0 + chi_mm2 k_t²) / (1 + xi_mm2 k_t²)`` appends
one auxiliary unknown per cut. The cell is one transverse period wide,
``x`` is periodic with phase one, and the ``y`` ends carry the analytic
total field. Both polarizations are compared with ``chi_mm`` evaluated
at ``k sin θ``. ``--theta`` is degrees from the sheet normal. Results
go in ``magnetic_rational_sheet_<parameters>/`` next to this script.
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
    dispersive_susceptibility,
    oblique_sheet_fields,
    sheet_coefficients,
)
from maxwell_fd.drivers.fdfd2d import FDFDOperator  # noqa: E402
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D  # noqa: E402
from maxwell_fd.materials.sheets import SymmetricSheet  # noqa: E402
from maxwell_fd.materials.volume import UniformIsotropic  # noqa: E402
from maxwell_fd.utils.constants import C0  # noqa: E402

CHI_EE = 0.5
CHI_MM0 = 0.25
CHI_MM2 = 0.04
XI_MM2 = 0.04
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
        description="Compare a rational magnetic periodic sheet with analytic R and T."
    )
    parser.add_argument("--chi-ee", type=float, default=CHI_EE)
    parser.add_argument("--chi-mm0", type=float, default=CHI_MM0)
    parser.add_argument("--chi-mm2", type=float, default=CHI_MM2)
    parser.add_argument("--xi-mm2", type=float, default=XI_MM2)
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
        help=f"Incidence angle in degrees, from the sheet normal (default: {THETA:g}).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=HEIGHT,
        help=f"Cell height in wavelengths (default: {HEIGHT:g}).",
    )
    args = parser.parse_args(argv)
    for name in ("chi_ee", "chi_mm0", "chi_mm2", "xi_mm2"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"{name.replace('_', '-')} must be finite")
    if not np.isfinite(args.wavelength) or args.wavelength <= 0.0:
        parser.error("wavelength must be positive")
    if isinstance(args.points, bool) or args.points < 4:
        parser.error("points must be an integer of at least 4")
    if not np.isfinite(args.theta) or not 0.0 < abs(args.theta) < 90.0:
        parser.error("theta must lie in (-90, 0) or (0, 90) degrees")
    if not np.isfinite(args.height) or args.height <= 0.0:
        parser.error("height must be positive")
    args.theta_rad = float(np.deg2rad(args.theta))
    length = args.wavelength / abs(np.sin(args.theta_rad))
    args.nx = int(round(args.points / abs(np.sin(args.theta_rad))))
    args.dx = length / args.nx
    args.ny = int(round(args.height * args.wavelength / args.dx))
    if args.nx < 4 or args.ny < 4:
        parser.error("the cell needs at least 4 cells on each side")
    return args


def _solve(pol: Polarization, args: argparse.Namespace) -> dict:
    dx = float(args.dx)
    length = args.nx * dx
    height = args.ny * dx
    y_sheet = (args.ny // 2 - 0.5) * dx
    theta = float(args.theta_rad)
    omega = 2.0 * np.pi * C0 / args.wavelength
    k = float(omega / C0)
    grid = YeeGrid2D(args.nx, args.ny, dx, dx, pol, Boundary.PERIODIC_X)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    operator = FDFDOperator(
        grid,
        bare,
        sheet=SymmetricSheet.magnetic_dispersion(
            y_sheet,
            args.chi_mm0,
            args.chi_mm2,
            chi_ee=args.chi_ee,
            xi_mm_2=args.xi_mm2,
        ),
    )
    chi_mm = dispersive_susceptibility(
        k * np.sin(theta), args.chi_mm0, args.chi_mm2, args.xi_mm2
    )
    reflected, transmitted = sheet_coefficients(
        args.chi_ee, chi_mm, k, theta=theta, te=pol is Polarization.TEZ
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
        reference = operator.layout.pack_e({"ez": exact})
        name = r"TMz $E_z$"
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
        reference = operator.layout.pack_e({"ex": exact, "ey": ey})
        name = r"TEz $E_x$"
    error = _relative(solved, reference)
    return {
        "x": x,
        "y": y,
        "field": field,
        "exact": exact,
        "error": error,
        "reflected": reflected,
        "transmitted": transmitted,
        "chi_mm": chi_mm,
        "length": length,
        "height": height,
        "y_sheet": y_sheet,
        "dx": dx,
        "nx": args.nx,
        "ny": args.ny,
        "title": (
            f"{name}\nL2 {error:.2%}\n$R={_fmt(reflected)}$\n$T={_fmt(transmitted)}$"
        ),
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
                ax, result["x"], result["y"], values, scale, float(result["y_sheet"])
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
        rf"Rational magnetic sheet, $\theta={args.theta:g}^\circ$, "
        rf"$\chi_{{ee}}={args.chi_ee:g}$, $\chi_{{mm0}}={args.chi_mm0:g}$, "
        rf"$\chi_{{mm2}}={args.chi_mm2:g}$, $\xi_{{mm2}}={args.xi_mm2:g}$, "
        rf"$\lambda={args.wavelength:g}$",
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
        rf"$\theta={args.theta:g}^\circ$, $\xi_{{mm2}}={args.xi_mm2:g}$, "
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
        chimm=args.chi_mm0,
        chimm2=args.chi_mm2,
        ximm2=args.xi_mm2,
    )
    tm = _solve(Polarization.TMZ, args)
    te = _solve(Polarization.TEZ, args)
    plot_fields(folder / "magnetic_rational_sheet.png", tm, te, args)
    plot_domain(folder / "domain.png", tm, args)
    print(
        f"{folder.name}  "
        f"chi_mm {tm['chi_mm'].real:.6g}  "
        f"TMz R {_fmt(tm['reflected'])} T {_fmt(tm['transmitted'])} "
        f"L2 {tm['error']:.4%}  "
        f"TEz R {_fmt(te['reflected'])} T {_fmt(te['transmitted'])} "
        f"L2 {te['error']:.4%}"
    )


if __name__ == "__main__":
    main()
