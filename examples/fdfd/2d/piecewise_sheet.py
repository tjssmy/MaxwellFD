"""Two-piece GSTC sheet, solved as a scattered-field 2D FDFD problem.

The sheet is ``--sheet`` wavelengths long and lies on a magnetic face.
``--split`` is the length of the left piece, in wavelengths. That piece
is the normal-incidence absorber ``chi_ee = chi_mm = -2j/k``. The right
piece is ``chi_ee = 0.5`` and ``chi_mm = 0.25``. A convolutional PML
backs the PEC wall on all four sides. ``--theta`` is the incidence angle
in degrees, measured from the sheet normal, and the default is 0. The
unknown is the scattered field. Each piece is compared with the uniform
sheet of its own susceptibilities, on a window one wavelength inside
that piece. The absorber shadow is a mean ``|E|``, because that
transmission is zero. Results go in ``piecewise_sheet_<parameters>/``
next to this script.
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from maxwell_fd.analytics.sheets import (  # noqa: E402
    oblique_sheet_fields,
    sheet_coefficients,
)
from maxwell_fd.drivers.fdfd2d import FDFDOperator  # noqa: E402
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D  # noqa: E402
from maxwell_fd.materials.sheets import SymmetricSheet  # noqa: E402
from maxwell_fd.materials.volume import UniformIsotropic  # noqa: E402
from maxwell_fd.operators.pml import PMLSpec  # noqa: E402
from maxwell_fd.utils.constants import C0, ETA0  # noqa: E402

WAVELENGTH = 1.0
POINTS = 20
THETA = 0.0
PML = 12
SHEET = 10.0
SPLIT = 5.0
X_PAD = 2.0
Y_PAD = 2.0
RIGHT_EE = 0.5
RIGHT_MM = 0.25
LEFT_COLOR = "#08519c"
RIGHT_COLOR = "#006d2c"


def _cells(n_wavelengths: float, points: int, name: str) -> int:
    length = float(n_wavelengths) * int(points)
    count = int(round(length))
    if count < 1 or abs(count - length) > 1e-8:
        raise ValueError(f"{name} is not a positive integer number of cells")
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a two-piece GSTC sheet in an open FDFD box."
    )
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
        "--pml",
        type=int,
        default=PML,
        help=f"PML cells on each side (default: {PML}).",
    )
    parser.add_argument(
        "--sheet",
        type=float,
        default=SHEET,
        help=f"Sheet length in wavelengths (default: {SHEET:g}).",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=SPLIT,
        help=f"Left-piece length in wavelengths (default: {SPLIT:g}).",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.wavelength) or args.wavelength <= 0.0:
        parser.error("wavelength must be positive")
    if isinstance(args.points, bool) or args.points < 4:
        parser.error("points must be an integer of at least 4")
    if not np.isfinite(args.theta) or abs(args.theta) >= 90.0:
        parser.error("theta must lie in (-90, 90) degrees")
    if isinstance(args.pml, bool) or args.pml < 1:
        parser.error("pml must be a positive cell count")
    if not np.isfinite(args.sheet) or args.sheet <= 0.0:
        parser.error("sheet length must be positive")
    if not np.isfinite(args.split) or args.split <= 0.0 or args.split >= args.sheet:
        parser.error("split must lie strictly inside the sheet")
    try:
        args.sheet_cells = _cells(args.sheet, args.points, "sheet")
        args.split_cells = _cells(args.split, args.points, "split")
        args.x_pad = _cells(X_PAD, args.points, "x pad")
        args.y_pad = _cells(Y_PAD, args.points, "y pad")
    except ValueError as exc:
        parser.error(str(exc))
    if args.split_cells >= args.sheet_cells:
        parser.error("split must leave a right-hand piece")
    if min(args.split, args.sheet - args.split) <= 2.0:
        parser.error("each piece must be longer than two wavelengths")
    args.nx = args.pml + args.x_pad + args.sheet_cells + args.x_pad + args.pml
    args.ny = args.pml + args.y_pad + args.y_pad + args.pml
    args.dx = args.wavelength / args.points
    return args


def _phase(
    x: np.ndarray,
    y: np.ndarray,
    k: float,
    theta: float,
    x_ref: float,
    y_sheet: float,
) -> np.ndarray:
    return np.exp(
        -1j * k * (np.sin(theta) * (x - x_ref) + np.cos(theta) * (y - y_sheet))
    )


def _fmt(value: complex) -> str:
    return f"{value.real:.6g}{value.imag:+.6g}j"


def _window(
    total: np.ndarray,
    analytic: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> float:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    if not np.any(mask):
        raise ValueError("field window missed every sample")
    got = np.asarray(total)[mask]
    exact = np.asarray(analytic)[mask]
    scale = np.linalg.norm(exact)
    # The absorber transmits nothing, so that shadow window is an rms of |E|.
    if scale < 1e-8:
        return float(np.linalg.norm(got) / np.sqrt(got.size))
    return float(np.linalg.norm(got - exact) / scale)


def _solve(pol: Polarization, args: argparse.Namespace) -> dict:
    dx = float(args.dx)
    length = args.nx * dx
    height = args.ny * dx
    x0 = (args.pml + args.x_pad) * dx
    x_split = x0 + args.split_cells * dx
    x1 = x0 + args.sheet_cells * dx
    y_sheet = (0.5 * args.ny - 0.5) * dx
    theta = float(np.deg2rad(args.theta))
    omega = 2.0 * np.pi * C0 / args.wavelength
    k = float(omega / C0)
    absorber = -2j / k
    x_ref = 0.5 * (x0 + x1)
    grid = YeeGrid2D(args.nx, args.ny, dx, dx, pol, Boundary.PEC)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    sheet = SymmetricSheet.piecewise(
        y_sheet,
        (
            SymmetricSheet(y_sheet, absorber, absorber, x0=x0, x1=x_split),
            SymmetricSheet(y_sheet, RIGHT_EE, RIGHT_MM, x0=x_split, x1=x1),
        ),
    )
    operator = FDFDOperator(grid, bare, PMLSpec.box(args.pml), sheet=sheet)
    reflected_l, transmitted_l = sheet_coefficients(
        absorber, absorber, k, theta=theta, te=pol is Polarization.TEZ
    )
    reflected_r, transmitted_r = sheet_coefficients(
        RIGHT_EE, RIGHT_MM, k, theta=theta, te=pol is Polarization.TEZ
    )
    if pol is Polarization.TMZ:
        ex, ey = grid.coordinates("ez")
        xx, yy = np.meshgrid(ex, ey, indexing="ij")
        electric = _phase(xx, yy, k, theta, x_ref, y_sheet)
        hx_x, hx_y = grid.coordinates("hx")
        hxx, hyy = np.meshgrid(hx_x, hx_y, indexing="ij")
        magnetic = np.cos(theta) / ETA0 * _phase(hxx, hyy, k, theta, x_ref, y_sheet)
        solved = operator.solve(
            omega,
            np.zeros(operator.layout.n_e),
            incident={"ez": electric, "hx": magnetic},
        )
        dof = operator.layout.e[0]
        component = "ez"
        name = "ez"
        title = r"TMz $E_z$"
    else:
        ex_x, ex_y = grid.coordinates("ex")
        ey_x, ey_y = grid.coordinates("ey")
        exx, exy = np.meshgrid(ex_x, ex_y, indexing="ij")
        eyx, eyy = np.meshgrid(ey_x, ey_y, indexing="ij")
        electric = np.cos(theta) * _phase(exx, exy, k, theta, x_ref, y_sheet)
        normal = -np.sin(theta) * _phase(eyx, eyy, k, theta, x_ref, y_sheet)
        hz_x, hz_y = grid.coordinates("hz")
        hxx, hyy = np.meshgrid(hz_x, hz_y, indexing="ij")
        magnetic = -_phase(hxx, hyy, k, theta, x_ref, y_sheet) / ETA0
        solved = operator.solve(
            omega,
            np.zeros(operator.layout.n_e),
            incident={"ex": electric, "ey": normal, "hz": magnetic},
        )
        dof = operator.layout.e[0]
        solved = solved[: dof.i.size]
        ex, ey = ex_x, ex_y
        component = "ex"
        name = "ex"
        title = r"TEz $E_x$"
    scattered = dof.scatter(solved, fill=0.0)
    total = dof.scatter(solved + dof.pack(electric), fill=0.0)
    analytic_l = oblique_sheet_fields(
        ex,
        ey,
        y_sheet=y_sheet,
        omega=omega,
        reflected=reflected_l,
        transmitted=transmitted_l,
        theta=theta,
        x_ref=x_ref,
    )[component]
    analytic_r = oblique_sheet_fields(
        ex,
        ey,
        y_sheet=y_sheet,
        omega=omega,
        reflected=reflected_r,
        transmitted=transmitted_r,
        theta=theta,
        x_ref=x_ref,
    )[component]
    margin = args.wavelength
    lit = (y_sheet - args.wavelength, y_sheet - 0.25 * args.wavelength)
    shadow = (y_sheet + 0.25 * args.wavelength, y_sheet + args.wavelength)
    spans = {
        "left": (x0 + margin, x_split - margin),
        "right": (x_split + margin, x1 - margin),
    }
    errors = {}
    for piece, (xa, xb) in spans.items():
        reference = analytic_l if piece == "left" else analytic_r
        errors[f"{piece}_lit"] = _window(total, reference, ex, ey, xa, xb, *lit)
        errors[f"{piece}_shadow"] = _window(total, reference, ex, ey, xa, xb, *shadow)
    return {
        "x": ex,
        "y": ey,
        "incident": electric,
        "scattered": scattered,
        "total": total,
        "title": title,
        "length": length,
        "height": height,
        "x0": x0,
        "x_split": x_split,
        "x1": x1,
        "y_sheet": y_sheet,
        "k": k,
        "theta": theta,
        "name": name,
        "absorber": absorber,
        "reflected_l": reflected_l,
        "transmitted_l": transmitted_l,
        "reflected_r": reflected_r,
        "transmitted_r": transmitted_r,
        **errors,
    }


def _draw_sheet(ax: plt.Axes, result: dict) -> None:
    y_sheet = float(result["y_sheet"])
    ax.plot(
        [float(result["x0"]), float(result["x_split"])],
        [y_sheet, y_sheet],
        color=LEFT_COLOR,
        lw=1.6,
        zorder=5,
        solid_capstyle="butt",
    )
    ax.plot(
        [float(result["x_split"]), float(result["x1"])],
        [y_sheet, y_sheet],
        color=RIGHT_COLOR,
        lw=1.6,
        zorder=5,
        solid_capstyle="butt",
    )


def _draw_pml(ax: plt.Axes, length: float, height: float, band: float) -> None:
    ax.add_patch(
        Rectangle(
            (band, band),
            length - 2.0 * band,
            height - 2.0 * band,
            fill=False,
            edgecolor="#d94801",
            linewidth=1.1,
            zorder=4,
        )
    )


def _image(
    ax: plt.Axes,
    result: dict,
    values: np.ndarray,
    limit: float,
    band: float,
) -> plt.cm.ScalarMappable:
    xx, yy = np.meshgrid(
        np.asarray(result["x"]), np.asarray(result["y"]), indexing="ij"
    )
    mesh = ax.pcolormesh(
        xx,
        yy,
        values,
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    _draw_sheet(ax, result)
    _draw_pml(ax, float(result["length"]), float(result["height"]), band)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(result["length"]))
    ax.set_ylim(0.0, float(result["height"]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def _annotation(result: dict) -> str:
    return (
        f"{result['title']}\n"
        f"left lit {result['left_lit']:.2f}\n"
        f"left shadow |E| {result['left_shadow']:.2f}\n"
        f"right lit {result['right_lit']:.2f}\n"
        f"right shadow {result['right_shadow']:.2f}"
    )


def plot_fields(
    path: Path,
    tm: dict,
    te: dict,
    *,
    theta: float,
    wavelength: float,
    band: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 7.2), constrained_layout=True)
    for result, row in zip((tm, te), axes, strict=True):
        incident = np.real(np.asarray(result["incident"]))
        scattered = np.real(np.asarray(result["scattered"]))
        total = np.real(np.asarray(result["total"]))
        limit = max(
            float(np.max(np.abs(incident))),
            float(np.max(np.abs(scattered))),
            float(np.max(np.abs(total))),
            1e-6,
        )
        panels = (
            (row[0], incident, "Incident"),
            (row[1], scattered, "Scattered"),
            (row[2], total, "Total"),
        )
        for ax, values, label in panels:
            image = _image(ax, result, values, limit, band)
            ax.set_title(label)
            if label == "Total":
                ax.text(
                    0.02,
                    0.02,
                    _annotation(result),
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=8,
                    color="0.15",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.78,
                        "edgecolor": "none",
                        "pad": 1.5,
                    },
                )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    length = float(tm["x1"]) - float(tm["x0"])
    fig.suptitle(
        rf"Two-piece sheet, left absorber, $\chi_{{ee}}={RIGHT_EE:g}$, "
        rf"$\chi_{{mm}}={RIGHT_MM:g}$, $\lambda={wavelength:g}$, "
        rf"length ${length:.4g}$, split ${float(tm['x_split']) - float(tm['x0']):.4g}$, "
        rf"$\theta={theta:g}^\circ$",
        fontsize=12,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_domain(path: Path, args: argparse.Namespace, result: dict) -> None:
    dx = float(args.dx)
    length = args.nx * dx
    height = args.ny * dx
    band = args.pml * dx
    y_edges = np.linspace(0.0, height, args.ny + 1)
    x_edges = np.linspace(0.0, length, args.nx + 1)
    segments = [[(float(x), 0.0), (float(x), height)] for x in x_edges]
    segments += [[(0.0, float(y)), (length, float(y))] for y in y_edges]
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), length, height, facecolor="#fdd0a2", edgecolor="none", zorder=0
        )
    )
    ax.add_patch(
        Rectangle(
            (band, band),
            length - 2.0 * band,
            height - 2.0 * band,
            facecolor="#ffffff",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_collection(
        LineCollection(segments, colors="#bdbdbd", linewidths=0.25, zorder=2)
    )
    _draw_pml(ax, length, height, band)
    _draw_sheet(ax, result)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Mesh, PML, and two-piece sheet\n"
        rf"$\lambda={args.wavelength:g}$, sheet ${args.sheet:g}\lambda$, "
        rf"split ${args.split:g}\lambda$, "
        rf"$\Delta={dx:g}$, PML {args.pml} cells, $\theta={args.theta:g}^\circ$"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", linewidth=1.0, label="mesh"),
            Patch(
                facecolor="#fdd0a2",
                edgecolor="#e6550d",
                label=f"PML ({args.pml} cells)",
            ),
            Line2D([0], [0], color=LEFT_COLOR, linewidth=1.6, label="absorber"),
            Line2D(
                [0],
                [0],
                color=RIGHT_COLOR,
                linewidth=1.6,
                label=rf"$\chi_{{ee}}={RIGHT_EE:g}$, $\chi_{{mm}}={RIGHT_MM:g}$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
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
        points=args.points,
        theta=args.theta,
        pml=args.pml,
        sheet=args.sheet,
        split=args.split,
    )
    tm = _solve(Polarization.TMZ, args)
    te = _solve(Polarization.TEZ, args)
    plot_fields(
        folder / "piecewise_sheet.png",
        tm,
        te,
        theta=args.theta,
        wavelength=args.wavelength,
        band=args.pml * args.dx,
    )
    plot_domain(folder / "domain.png", args, tm)
    chi = tm["absorber"]
    print(
        f"{folder.name}  "
        f"left chi_ee = chi_mm = {chi.real:.6g}{chi.imag:+.6g}j  "
        f"TMz R {_fmt(tm['reflected_l'])} T {_fmt(tm['transmitted_l'])}  "
        f"TEz R {_fmt(te['reflected_l'])} T {_fmt(te['transmitted_l'])}  "
        f"right chi_ee={RIGHT_EE:g} chi_mm={RIGHT_MM:g}  "
        f"TMz R {_fmt(tm['reflected_r'])} T {_fmt(tm['transmitted_r'])}  "
        f"TEz R {_fmt(te['reflected_r'])} T {_fmt(te['transmitted_r'])}  "
        f"TMz left lit {tm['left_lit']:.4f} shadow {tm['left_shadow']:.4f} "
        f"right lit {tm['right_lit']:.4f} shadow {tm['right_shadow']:.4f}  "
        f"TEz left lit {te['left_lit']:.4f} shadow {te['left_shadow']:.4f} "
        f"right lit {te['right_lit']:.4f} shadow {te['right_shadow']:.4f}"
    )


if __name__ == "__main__":
    main()
