"""Oblique GSTC sheet on a Bloch cell, compared with analytic R and T.

The sheet spans the cell. ``x`` is periodic with ``k_{B,x} = k sin θ``, and
the width is not one transverse period, so the forward wrap carries
``e^{-j k_{B,x} a}``. The ``y`` ends carry the analytic total field.
``--width`` is the cell width in wavelengths. ``--chi-nn`` is the normal
magnetic susceptibility. It changes TMz and leaves TEz on the tangential
formula. ``--chi-em`` and ``--chi-me`` are the magneto-electric strengths.
A reciprocal sheet uses the same value for both. They default to zero and
stay out of the folder name at that default. ``--theta`` is degrees from
the sheet normal. Results go in ``bloch_sheet_<parameters>/`` next to this
script.
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
WIDTH = 0.8


def _relative(numerical: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def _fmt(value: complex) -> str:
    return f"{value.real:.4g}{value.imag:+.4g}j"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a Bloch-periodic oblique GSTC sheet with analytic R and T."
    )
    parser.add_argument("--chi-ee", type=float, default=CHI_EE)
    parser.add_argument("--chi-mm", type=float, default=CHI_MM)
    parser.add_argument("--chi-nn", type=float, default=CHI_NN)
    parser.add_argument("--chi-em", type=float, default=0.0)
    parser.add_argument("--chi-me", type=float, default=0.0)
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH)
    parser.add_argument("--points", type=int, default=POINTS)
    parser.add_argument("--theta", type=float, default=THETA)
    parser.add_argument("--height", type=float, default=HEIGHT)
    parser.add_argument("--width", type=float, default=WIDTH)
    args = parser.parse_args(argv)
    for name in ("chi_ee", "chi_mm", "chi_nn", "chi_em", "chi_me"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"{name.replace('_', '-')} must be finite")
    _validate(parser, args)
    return args


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if int(args.points) != args.points or args.points < 4:
        parser.error("points must be an integer of at least 4")
    if not np.isfinite(args.theta) or not 0.0 < abs(args.theta) < 90.0:
        parser.error("theta must lie between 0 and 90 degrees")
    if not np.isfinite(args.wavelength) or args.wavelength <= 0.0:
        parser.error("wavelength must be positive")
    if not np.isfinite(args.height) or args.height <= 0.0:
        parser.error("height must be positive")
    if not np.isfinite(args.width) or args.width <= 0.0:
        parser.error("width must be positive")
    nx = int(round(args.width * args.points))
    ny = int(round(args.height * args.points))
    if nx < 4 or abs(nx / args.points - args.width) > 1e-8:
        parser.error("width must be an integer number of cells, at least 4")
    if ny < 4 or abs(ny / args.points - args.height) > 1e-8:
        parser.error("height must be an integer number of cells, at least 4")
    args.nx = nx
    args.ny = ny


def _solve(pol: Polarization, args: argparse.Namespace) -> dict:
    dx = args.wavelength / args.points
    theta = float(np.deg2rad(args.theta))
    y_sheet = (args.ny // 2 - 0.5) * dx
    omega = 2.0 * np.pi * C0 / args.wavelength
    bloch_x = omega / C0 * float(np.sin(theta))
    grid = YeeGrid2D(
        args.nx, args.ny, dx, dx, pol, Boundary.PERIODIC_X, bloch_x=bloch_x
    )
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    operator = FDFDOperator(
        grid,
        bare,
        sheet=SymmetricSheet(
            y_sheet,
            args.chi_ee,
            args.chi_mm,
            chi_mm_nn=args.chi_nn,
            chi_em=args.chi_em,
            chi_me=args.chi_me,
        ),
    )
    reflected, transmitted = sheet_coefficients(
        args.chi_ee,
        args.chi_mm,
        omega / C0,
        theta=theta,
        te=pol is Polarization.TEZ,
        chi_mm_nn=args.chi_nn,
        chi_em=args.chi_em,
        chi_me=args.chi_me,
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
        "length": args.nx * dx,
        "height": args.ny * dx,
        "y_sheet": y_sheet,
        "dx": dx,
        "nx": args.nx,
        "ny": args.ny,
        "phase": complex(np.exp(-1j * bloch_x * args.nx * dx)),
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
    # Two end ticks: the column is a fraction of a wavelength wide.
    ax.set_xticks([float(np.min(x)), float(np.max(x))])
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
            if label == "FD":
                ax.set_title("FD\n" + str(result["title"]), fontsize=8)
            else:
                ax.set_title(label)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cross = ""
    if args.chi_em != 0.0 or args.chi_me != 0.0:
        cross = rf", $\chi_{{em}}={args.chi_em:g}$, $\chi_{{me}}={args.chi_me:g}$"
    fig.suptitle(
        rf"Bloch cell, $\theta={args.theta:g}^\circ$, width ${args.width:g}\lambda$, "
        rf"$\chi_{{ee}}={args.chi_ee:g}$, $\chi_{{mm}}={args.chi_mm:g}$, "
        rf"$\chi_{{mm}}^{{nn}}={args.chi_nn:g}${cross}",
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
    fig, ax = plt.subplots(figsize=(3.4, 8.4))
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
    phase = complex(result["phase"])
    ax.set_title(
        "Bloch cell\n"
        rf"$\theta={args.theta:g}^\circ$, width ${args.width:g}\lambda$, "
        rf"$e^{{-j\phi}}={phase.real:.3f}{phase.imag:+.3f}j$"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", lw=1.0, label="mesh"),
            Line2D([0], [0], color="#08519c", lw=1.6, label="sheet"),
            Line2D([0], [0], color=trace, lw=1.8, label="analytic trace"),
            Line2D([0], [0], color="#6a3d9a", lw=1.4, ls="--", label="Bloch"),
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
    parameters: dict[str, float | int] = {
        "lam": args.wavelength,
        "theta": args.theta,
        "width": args.width,
        "points": args.points,
        "height": args.height,
        "chiee": args.chi_ee,
        "chimm": args.chi_mm,
        "chinn": args.chi_nn,
    }
    if args.chi_em != 0.0:
        parameters["chiem"] = args.chi_em
    if args.chi_me != 0.0:
        parameters["chime"] = args.chi_me
    folder = output_dir(__file__, **parameters)
    tm = _solve(Polarization.TMZ, args)
    te = _solve(Polarization.TEZ, args)
    plot_fields(folder / "bloch_sheet.png", tm, te, args)
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
