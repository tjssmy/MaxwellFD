"""Near-to-far pattern of a TMz line current.

The source is one Ez sample at the center of a PEC square. A convolutional
PML backs every wall. The contour is a square of half-width ``--half``,
centered on the source, with Ez on the nodes and tangential H averaged
from the two faces that touch each edge. The pattern is compared with the
Hankel far field at 20 wavelengths. ``--wavelength``, ``--side``, ``--dx``,
``--pml``, and ``--half`` set the free-space wavelength, the square, the
cell size, the PML depth, and the contour half-width. A small direct
factor is left on one BLAS thread when that count is unset. Results go in
``farfield_line_<parameters>/`` next to this script.
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

from maxwell_fd.analytics.farfield2d import (  # noqa: E402
    line_current_far_ez,
    tmz_far_ez,
    tmz_magnetic,
    tmz_rectangle,
)
from maxwell_fd.drivers.fdfd2d import FDFDOperator  # noqa: E402
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D  # noqa: E402
from maxwell_fd.materials.volume import UniformIsotropic  # noqa: E402
from maxwell_fd.operators.pml import PMLSpec  # noqa: E402
from maxwell_fd.utils.constants import C0  # noqa: E402

WAVELENGTH = 1.0
SIDE = 3.0
DX = 0.05
PML = 8
HALF = 0.8
RHO = 20.0
ANGLES = 72


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
    parser.add_argument("--side", type=float, default=SIDE)
    parser.add_argument("--dx", type=float, default=DX)
    parser.add_argument("--pml", type=int, default=PML)
    parser.add_argument("--half", type=float, default=HALF)
    args = parser.parse_args(argv)
    _validate(parser, args)
    return args


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not np.isfinite(args.wavelength) or args.wavelength <= 0.0:
        parser.error("wavelength must be positive")
    try:
        args.cells = _cells(args.side, args.dx)
    except ValueError as exc:
        parser.error(str(exc))
    if args.cells < 4:
        parser.error("side must contain at least 4 cells")
    if isinstance(args.pml, bool) or args.pml < 1 or 2 * args.pml >= args.cells:
        parser.error("pml must be a positive cell count that leaves an interior")
    if not np.isfinite(args.half) or args.half <= 0.0:
        parser.error("half must be positive")
    count = args.half / args.dx
    if abs(count - round(count)) > 1e-8:
        parser.error("half must be an integer number of cells")
    args.half_cells = int(round(count))
    if args.half < 0.5 * args.wavelength:
        parser.error("leave 0.5 wavelengths between the source and the contour")
    band = args.pml * args.dx
    outer = 0.5 * args.side + args.half + 0.5 * args.dx
    if outer > args.side - band:
        parser.error("leave the contour in the unstretched interior")
    source = args.cells // 2
    if source - args.half_cells < 1 or source + args.half_cells > args.cells - 1:
        parser.error("leave the contour in the unstretched interior")


def _pattern(args: argparse.Namespace) -> dict[str, np.ndarray | float]:
    grid = YeeGrid2D(
        args.cells, args.cells, args.dx, args.dx, Polarization.TMZ, Boundary.PEC
    )
    omega = 2.0 * np.pi * C0 / args.wavelength
    operator = FDFDOperator(
        grid, UniformIsotropic().sample_tm(grid), PMLSpec.box(args.pml)
    )
    i0 = args.cells // 2
    jz = np.zeros(grid.shapes()["ez"])
    jz[i0, i0] = 1.0 / (args.dx * args.dx)
    solved = operator.solve(omega, operator.layout.pack_e({"ez": jz}))
    ez = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    dof = operator.layout.e[0]
    ez[dof.i, dof.j] = solved
    hx, hy = tmz_magnetic(grid, ez, omega)
    half = args.half_cells
    contour = tmz_rectangle(
        grid, ez, hx, hy, i0 - half, i0 + half, i0 - half, i0 + half
    )
    x, y = grid.coordinates("ez")
    phi = np.linspace(0.0, 2.0 * np.pi, ANGLES, endpoint=False)
    rho = RHO * args.wavelength
    got = tmz_far_ez(
        contour["x"] - x[i0],
        contour["y"] - y[i0],
        contour["ez"],
        contour["hx"],
        contour["hy"],
        contour["nx"],
        contour["ny"],
        contour["dl"],
        phi=phi,
        rho=rho,
        omega=omega,
    )
    exact = line_current_far_ez(phi, rho, omega=omega)
    error = float(np.linalg.norm(got - exact) / np.linalg.norm(exact))
    return {
        "phi": phi,
        "field": got,
        "exact": exact,
        "error": error,
        "source": (x[i0], y[i0]),
    }


def plot_domain(path: Path, args: argparse.Namespace) -> None:
    band = args.pml * args.dx
    side = args.side
    source = (args.cells // 2) * args.dx
    half = args.half
    edges = np.linspace(0.0, side, args.cells + 1)
    segments = [[(float(x), 0.0), (float(x), side)] for x in edges]
    segments += [[(0.0, float(y)), (side, float(y))] for y in edges]
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), side, side, facecolor="#fdd0a2", edgecolor="none", zorder=0
        )
    )
    ax.add_patch(
        Rectangle(
            (band, band),
            side - 2.0 * band,
            side - 2.0 * band,
            facecolor="#ffffff",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_collection(
        LineCollection(segments, colors="#bdbdbd", linewidths=0.35, zorder=2)
    )
    ax.add_patch(
        Rectangle(
            (source - half, source - half),
            2.0 * half,
            2.0 * half,
            fill=False,
            edgecolor="#08519c",
            lw=1.5,
            ls="--",
            zorder=4,
        )
    )
    ax.plot(source, source, marker="o", color="#08519c", ms=5, zorder=5)
    ax.add_patch(
        Rectangle(
            (band, band),
            side - 2.0 * band,
            side - 2.0 * band,
            fill=False,
            edgecolor="#e6550d",
            lw=1.3,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), side, side, fill=False, edgecolor="#252525", lw=1.0, zorder=5
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(0.0, side)
    ax.set_ylim(0.0, side)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Mesh, PML, and near-to-far contour\n"
        rf"$\lambda={args.wavelength:g}$, side ${side:g}$, $\Delta={args.dx:g}$, PML {args.pml} cells"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6e6e6e", lw=1.0, label="mesh"),
            Patch(
                facecolor="#fdd0a2",
                edgecolor="#e6550d",
                label=f"PML ({args.pml} cells)",
            ),
            Line2D([0], [0], color="#08519c", lw=1.5, ls="--", label="contour"),
            Line2D([0], [0], color="#08519c", marker="o", lw=0, label="line current"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
    )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_pattern(path: Path, args: argparse.Namespace) -> float:
    result = _pattern(args)
    phi = np.degrees(np.asarray(result["phi"]))
    field = np.asarray(result["field"])
    exact = np.asarray(result["exact"])
    error = float(result["error"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    axes[0].plot(phi, np.abs(field), color="#08519c", lw=1.6, label="FD")
    axes[0].plot(phi, np.abs(exact), color="#d94801", lw=1.4, ls="--", label="Hankel")
    axes[0].set_xlabel(r"$\varphi$ (degrees)")
    axes[0].set_ylabel(r"$|E_z|$")
    axes[0].set_xlim(0.0, 360.0)
    axes[0].set_title(rf"$|E_z|$ at $\rho={RHO:g}\lambda$, L2 {error:.2%}")
    axes[0].legend(frameon=False)
    axes[1].plot(phi, np.unwrap(np.angle(field)), color="#08519c", lw=1.6, label="FD")
    axes[1].plot(
        phi,
        np.unwrap(np.angle(exact)),
        color="#d94801",
        lw=1.4,
        ls="--",
        label="Hankel",
    )
    axes[1].set_xlabel(r"$\varphi$ (degrees)")
    axes[1].set_ylabel(r"phase of $E_z$")
    axes[1].set_xlim(0.0, 360.0)
    axes[1].set_title("phase about the source")
    axes[1].legend(frameon=False)
    fig.suptitle(
        rf"Line-current far field, $\lambda={args.wavelength:g}$, "
        rf"side ${args.side:g}$, $\Delta={args.dx:g}$, half ${args.half:g}$",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return error


def main() -> None:
    examples = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(examples))
    from run_dir import output_dir

    args = _parse_args()
    folder = output_dir(
        __file__,
        lam=args.wavelength,
        side=args.side,
        dx=args.dx,
        pml=args.pml,
        half=args.half,
    )
    plot_domain(folder / "domain.png", args)
    error = plot_pattern(folder / "farfield_line.png", args)
    print(f"{folder.name}  L2 {error:.4%}")


if __name__ == "__main__":
    main()
