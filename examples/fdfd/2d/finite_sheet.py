"""Finite GSTC sheet, solved as a scattered-field 2D FDFD problem.

The sheet is ``--sheet`` wavelengths long and lies on a magnetic face.
With no susceptibilities on the command line, ``chi_ee = chi_mm = -2j/k``
is the normal-incidence perfect absorber and ``chi_mm_nn`` is zero.
``--chi-ee``, ``--chi-mm``, and ``--chi-nn`` replace that absorber with
three real values. ``chi_nn`` enters the TMz jump and leaves TEz unchanged.
A convolutional PML backs the PEC wall on all four sides. ``--theta`` is
the incidence angle in degrees, measured from the sheet normal, and the
default is 0. The unknown is the scattered field. Results go in
``finite_sheet_<parameters>/`` next to this script.
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

from maxwell_fd.analytics.sheets import sheet_coefficients  # noqa: E402
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
X_PAD = 2.0
Y_PAD = 2.0


def _cells(n_wavelengths: float, points: int, name: str) -> int:
    length = float(n_wavelengths) * int(points)
    count = int(round(length))
    if count < 1 or abs(count - length) > 1e-8:
        raise ValueError(f"{name} is not a positive integer number of cells")
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a finite absorbing GSTC sheet in an open FDFD box."
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
    parser.add_argument("--chi-ee", type=float, default=None)
    parser.add_argument("--chi-mm", type=float, default=None)
    parser.add_argument("--chi-nn", type=float, default=None)
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
    supplied = (args.chi_ee, args.chi_mm, args.chi_nn)
    if all(value is None for value in supplied):
        args.absorber = True
    elif any(value is None for value in supplied):
        parser.error("pass --chi-ee, --chi-mm, and --chi-nn together")
    else:
        args.absorber = False
        for name, value in zip(("chi-ee", "chi-mm", "chi-nn"), supplied, strict=True):
            if not np.isfinite(value):
                parser.error(f"{name} must be finite")
    try:
        args.sheet_cells = _cells(args.sheet, args.points, "sheet")
        args.x_pad = _cells(X_PAD, args.points, "x pad")
        args.y_pad = _cells(Y_PAD, args.points, "y pad")
    except ValueError as exc:
        parser.error(str(exc))
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
    return f"{value.real:.4g}{value.imag:+.4g}j"


def _mean_abs(
    field: np.ndarray,
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
    return float(np.mean(np.abs(field[mask])))


def _solve(pol: Polarization, args: argparse.Namespace) -> dict:
    dx = float(args.dx)
    length = args.nx * dx
    height = args.ny * dx
    x0 = (args.pml + args.x_pad) * dx
    x1 = x0 + args.sheet_cells * dx
    y_sheet = (0.5 * args.ny - 0.5) * dx
    theta = float(np.deg2rad(args.theta))
    omega = 2.0 * np.pi * C0 / args.wavelength
    k = float(omega / C0)
    if args.absorber:
        chi_ee = chi_mm = -2j / k
        chi_nn = 0.0
    else:
        chi_ee = complex(args.chi_ee)
        chi_mm = complex(args.chi_mm)
        chi_nn = complex(args.chi_nn)
    x_ref = 0.5 * (x0 + x1)
    grid = YeeGrid2D(args.nx, args.ny, dx, dx, pol, Boundary.PEC)
    bare = (
        UniformIsotropic().sample_tm(grid)
        if pol is Polarization.TMZ
        else UniformIsotropic().sample_te(grid)
    )
    operator = FDFDOperator(
        grid,
        bare,
        PMLSpec.box(args.pml),
        sheet=SymmetricSheet(y_sheet, chi_ee, chi_mm, x0, x1, chi_mm_nn=chi_nn),
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
        name = "ex"
        title = r"TEz $E_x$"
    scattered = dof.scatter(solved, fill=0.0)
    total = dof.scatter(solved + dof.pack(electric), fill=0.0)
    margin = 0.2 * (x1 - x0)
    probe_x0 = x0 + margin
    probe_x1 = x1 - margin
    lit = _mean_abs(
        total,
        ex,
        ey,
        probe_x0,
        probe_x1,
        y_sheet - args.wavelength,
        y_sheet - 0.25 * args.wavelength,
    )
    shadow = _mean_abs(
        total,
        ex,
        ey,
        probe_x0,
        probe_x1,
        y_sheet + 0.25 * args.wavelength,
        y_sheet + args.wavelength,
    )
    return {
        "x": ex,
        "y": ey,
        "incident": electric,
        "scattered": scattered,
        "total": total,
        "lit": lit,
        "shadow": shadow,
        "title": title,
        "length": length,
        "height": height,
        "x0": x0,
        "x1": x1,
        "y_sheet": y_sheet,
        "chi_ee": chi_ee,
        "chi_mm": chi_mm,
        "chi_nn": chi_nn,
        "k": k,
        "theta": theta,
        "name": name,
    }


def _draw_sheet(ax: plt.Axes, x0: float, x1: float, y_sheet: float) -> None:
    ax.plot(
        [x0, x1],
        [y_sheet, y_sheet],
        color="#08519c",
        lw=1.5,
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
    _draw_sheet(ax, float(result["x0"]), float(result["x1"]), float(result["y_sheet"]))
    _draw_pml(ax, float(result["length"]), float(result["height"]), band)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(result["length"]))
    ax.set_ylim(0.0, float(result["height"]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def plot_fields(
    path: Path,
    tm: dict,
    te: dict,
    *,
    theta: float,
    wavelength: float,
    band: float,
    heading: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 6.4), constrained_layout=True)
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
                    0.98,
                    f"{result['title']}\nshadow |E| {result['shadow']:.2f}\nlit |E| {result['lit']:.2f}",
                    transform=ax.transAxes,
                    va="top",
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
        heading
        + rf", $\lambda={wavelength:g}$, length ${length:.4g}$, $\theta={theta:g}^\circ$",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_domain(
    path: Path, args: argparse.Namespace, y_sheet: float, x0: float, x1: float
) -> None:
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
    _draw_sheet(ax, x0, x1, y_sheet)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Mesh, PML, and finite sheet\n"
        rf"$\lambda={args.wavelength:g}$, sheet ${args.sheet:g}\lambda$, "
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
            Line2D([0], [0], color="#08519c", linewidth=1.6, label="sheet"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
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
        "points": args.points,
        "theta": args.theta,
        "pml": args.pml,
        "sheet": args.sheet,
    }
    if args.absorber:
        heading = r"Finite absorbing sheet, $\chi_{ee}=\chi_{mm}=-2j/k$"
    else:
        parameters["chiee"] = args.chi_ee
        parameters["chimm"] = args.chi_mm
        parameters["chinn"] = args.chi_nn
        heading = (
            rf"Finite sheet, $\chi_{{ee}}={args.chi_ee:g}$, "
            rf"$\chi_{{mm}}={args.chi_mm:g}$, $\chi_{{mm}}^{{nn}}={args.chi_nn:g}$"
        )
    folder = output_dir(__file__, **parameters)
    tm = _solve(Polarization.TMZ, args)
    te = _solve(Polarization.TEZ, args)
    plot_fields(
        folder / "finite_sheet.png",
        tm,
        te,
        theta=args.theta,
        wavelength=args.wavelength,
        band=args.pml * args.dx,
        heading=heading,
    )
    plot_domain(
        folder / "domain.png",
        args,
        float(tm["y_sheet"]),
        float(tm["x0"]),
        float(tm["x1"]),
    )

    def _coeff(pol: Polarization) -> str:
        reflected, transmitted = sheet_coefficients(
            tm["chi_ee"],
            tm["chi_mm"],
            tm["k"],
            theta=tm["theta"],
            te=pol is Polarization.TEZ,
            chi_mm_nn=tm["chi_nn"],
        )
        return f"R {_fmt(reflected)} T {_fmt(transmitted)}"

    if args.absorber:
        chi = tm["chi_ee"]
        prefix = f"chi_ee = chi_mm = {chi.real:.6g}{chi.imag:+.6g}j"
    else:
        prefix = (
            f"chi_ee={args.chi_ee:g} chi_mm={args.chi_mm:g} chi_nn={args.chi_nn:g}  "
            f"TMz {_coeff(Polarization.TMZ)}  TEz {_coeff(Polarization.TEZ)}"
        )
    print(
        f"{folder.name}  {prefix}  "
        f"TMz lit {tm['lit']:.3f} shadow {tm['shadow']:.3f}  "
        f"TEz lit {te['lit']:.3f} shadow {te['shadow']:.3f}"
    )


if __name__ == "__main__":
    main()
