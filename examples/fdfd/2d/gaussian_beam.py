"""TMz Gaussian beam launched across a total-field/scattered-field split.

The beam propagates in +y. Its waist sits at the center of the box. The
impressed current is confined to the two Ez rows that touch the split, and
a convolutional PML backs the PEC wall. The comparison is the total-field
side, outside the PML and at least half a wavelength from the split.
``--wavelength``, ``--width``, ``--height``, ``--dx``, ``--pml``, ``--w0``,
and ``--split`` set the free-space wavelength, the box, the cell size, the
PML depth, the waist radius, and the first total-field Ez row. A small
direct factor is left on one BLAS thread when that count is unset. Results
go in ``gaussian_beam_<parameters>/`` next to this script.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from maxwell_fd.analytics.beam2d import gaussian_beam_tmz  # noqa: E402
from maxwell_fd.drivers.fdfd2d import FDFDOperator, tfsf_y_current  # noqa: E402
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D  # noqa: E402
from maxwell_fd.materials.volume import UniformIsotropic  # noqa: E402
from maxwell_fd.operators.pml import PMLSpec  # noqa: E402
from maxwell_fd.utils.constants import C0  # noqa: E402

WAVELENGTH = 1.0
WIDTH = 8.0
HEIGHT = 6.0
DX = 0.05
PML = 8
W0 = 1.5
SPLIT = 1.5


def _cells(length: float, dx: float) -> int:
    if not np.isfinite(length) or not np.isfinite(dx) or dx <= 0.0 or length <= 0.0:
        raise ValueError(f"length and dx must be positive, got {length}, {dx}")
    count = int(round(length / dx))
    if count < 1 or abs(count * dx - length) > 1e-8 * max(length, dx):
        raise ValueError(f"{length:g} is not a multiple of dx={dx:g}")
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH)
    parser.add_argument("--width", type=float, default=WIDTH)
    parser.add_argument("--height", type=float, default=HEIGHT)
    parser.add_argument("--dx", type=float, default=DX)
    parser.add_argument("--pml", type=int, default=PML)
    parser.add_argument("--w0", type=float, default=W0)
    parser.add_argument("--split", type=float, default=SPLIT)
    args = parser.parse_args(argv)
    _validate(parser, args)
    return args


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not np.isfinite(args.wavelength) or args.wavelength <= 0.0:
        parser.error("wavelength must be positive")
    if not np.isfinite(args.w0) or args.w0 <= 0.0:
        parser.error("w0 must be positive")
    try:
        args.nx = _cells(args.width, args.dx)
        args.ny = _cells(args.height, args.dx)
    except ValueError as exc:
        parser.error(str(exc))
    if args.nx < 4 or args.ny < 4:
        parser.error("width and height must contain at least 4 cells")
    if isinstance(args.pml, bool) or args.pml < 1:
        parser.error("pml must be a positive cell count")
    if 2 * args.pml >= args.nx or 2 * args.pml >= args.ny:
        parser.error("pml must leave an interior")
    if not np.isfinite(args.split):
        parser.error("split must be finite")
    row = args.split / args.dx
    if abs(row - round(row)) > 1e-8:
        parser.error("split must lie on an Ez row")
    args.row = int(round(row))
    band = args.pml * args.dx
    clearance = 0.5 * args.wavelength
    if args.split < band + clearance or args.split > args.height - band - clearance:
        parser.error("leave 0.5 wavelengths between the split and the PML")
    waist_y = 0.5 * args.height
    waist_x = 0.5 * args.width
    if waist_y < args.split + clearance or waist_y > args.height - band - clearance:
        parser.error("the waist must sit 0.5 wavelengths inside the total-field region")
    if waist_x < band + clearance or waist_x > args.width - band - clearance:
        parser.error("leave 0.5 wavelengths between the waist and the PML")


def _solve(args: argparse.Namespace) -> dict[str, object]:
    grid = YeeGrid2D(args.nx, args.ny, args.dx, args.dx, Polarization.TMZ, Boundary.PEC)
    omega = 2.0 * np.pi * C0 / args.wavelength
    k = omega / C0
    operator = FDFDOperator(
        grid, UniformIsotropic().sample_tm(grid), PMLSpec.box(args.pml)
    )
    x, y = grid.coordinates("ez")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    origin = (0.5 * args.width, 0.5 * args.height)
    incident = gaussian_beam_tmz(xx, yy, k=k, w0=args.w0, origin=origin)
    current = tfsf_y_current(operator, omega, incident["ez"], args.row)
    solved = operator.solve(omega, current)
    ez = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    dof = operator.layout.e[0]
    ez[dof.i, dof.j] = solved
    inset = args.pml * args.dx
    edge = inset + 0.5 * args.wavelength
    total = (
        (xx >= edge)
        & (xx <= x[-1] - edge)
        & (yy >= y[args.row] + 0.5 * args.wavelength)
        & (yy <= y[-1] - edge)
    )
    exact = incident["ez"]
    error = float(
        np.linalg.norm(ez[total] - exact[total]) / np.linalg.norm(exact[total])
    )
    return {
        "x": x,
        "y": y,
        "field": ez,
        "exact": exact,
        "mask": total,
        "error": error,
        "inset": inset,
        "split": float(y[args.row]),
        "origin": origin,
    }


def _panel(ax: plt.Axes, result: dict[str, object], values: np.ndarray, limit: float):
    x = np.asarray(result["x"])
    y = np.asarray(result["y"])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mesh = ax.pcolormesh(
        xx, yy, values, shading="gouraud", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    inset = float(result["inset"])
    width = float(x[-1] - x[0])
    height = float(y[-1] - y[0])
    ax.add_patch(
        Rectangle(
            (inset, inset),
            width - 2.0 * inset,
            height - 2.0 * inset,
            fill=False,
            edgecolor="#d94801",
            lw=1.2,
            zorder=4,
        )
    )
    ax.axhline(float(result["split"]), color="0.2", ls="--", lw=0.8, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(float(y[0]), float(y[-1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def plot_fields(path: Path, args: argparse.Namespace) -> float:
    result = _solve(args)
    field = np.real(np.asarray(result["field"]))
    exact = np.real(np.asarray(result["exact"]))
    row = int(np.argmin(np.abs(np.asarray(result["y"]) - float(result["split"]))))
    exact[:, :row] = 0.0
    mask = np.asarray(result["mask"])
    error = float(result["error"])
    limit = max(
        float(np.max(np.abs(field[mask]))), float(np.max(np.abs(exact[mask]))), 1e-6
    )
    diff = field - exact
    diff_limit = max(float(np.max(np.abs(diff[mask]))), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0), constrained_layout=True)
    panels = (
        (axes[0], field, limit, "FD"),
        (axes[1], exact, limit, "Analytic"),
        (axes[2], diff, diff_limit, "FD − Analytic"),
    )
    for ax, values, scale, label in panels:
        image = _panel(ax, result, values, scale)
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
        rf"Gaussian beam, $w_0={args.w0:g}$, $\lambda={args.wavelength:g}$, "
        rf"${args.width:g}\times{args.height:g}$, $\Delta={args.dx:g}$, "
        f"PML {args.pml} cells",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return error


def plot_domain(path: Path, args: argparse.Namespace) -> None:
    band = args.pml * args.dx
    width = args.width
    height = args.height
    x_edges = np.linspace(0.0, width, args.nx + 1)
    y_edges = np.linspace(0.0, height, args.ny + 1)
    segments = [[(float(x), 0.0), (float(x), height)] for x in x_edges]
    segments += [[(0.0, float(y)), (width, float(y))] for y in y_edges]
    waist_x = 0.5 * width
    waist_y = 0.5 * height
    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), width, height, facecolor="#fdd0a2", edgecolor="none", zorder=0
        )
    )
    ax.add_patch(
        Rectangle(
            (band, band),
            width - 2.0 * band,
            height - 2.0 * band,
            facecolor="#ffffff",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_collection(
        LineCollection(segments, colors="#bdbdbd", linewidths=0.25, zorder=2)
    )
    ax.plot(
        [0.0, width],
        [args.split, args.split],
        color="#08519c",
        lw=1.4,
        ls="--",
        zorder=4,
    )
    ax.plot(
        [waist_x - args.w0, waist_x + args.w0],
        [waist_y, waist_y],
        color="#08519c",
        lw=2.4,
        solid_capstyle="round",
        zorder=5,
    )
    ax.add_patch(
        Rectangle(
            (band, band),
            width - 2.0 * band,
            height - 2.0 * band,
            fill=False,
            edgecolor="#e6550d",
            lw=1.3,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), width, height, fill=False, edgecolor="#252525", lw=1.0, zorder=5
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Mesh, PML, and Gaussian beam\n"
        rf"$\lambda={args.wavelength:g}$, $w_0={args.w0:g}$, "
        rf"$\Delta={args.dx:g}$, PML {args.pml} cells"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", lw=1.0, label="mesh"),
            Patch(
                facecolor="#fdd0a2",
                edgecolor="#e6550d",
                label=f"PML ({args.pml} cells)",
            ),
            Line2D([0], [0], color="#08519c", lw=1.4, ls="--", label="TF/SF split"),
            Line2D([0], [0], color="#08519c", lw=2.4, label=r"waist, $1/e$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
    )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    examples = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(examples))
    from run_dir import output_dir

    args = _parse_args()
    folder = output_dir(
        __file__,
        lam=args.wavelength,
        waist=args.w0,
        width=args.width,
        height=args.height,
        dx=args.dx,
        pml=args.pml,
        split=args.split,
    )
    plot_domain(folder / "domain.png", args)
    error = plot_fields(folder / "gaussian_beam.png", args)
    print(f"{folder.name}  L2 {error:.4%}")


if __name__ == "__main__":
    main()
