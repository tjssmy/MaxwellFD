"""TMz line current in a PEC box closed by a convolutional PML.

The source is one ``Ez`` sample. ``--pml`` is the cell count on each side.
``--side`` and ``--dx`` set the square, and ``--wavelength`` is the free-space
wavelength. Results go in ``pml_line_<parameters>/`` next to this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from maxwell_fd.analytics.green2d import line_current_ez
from maxwell_fd.drivers.fdfd2d import FDFDOperator
from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import UniformIsotropic
from maxwell_fd.operators.pml import PMLSpec
from maxwell_fd.utils.constants import C0

WAVELENGTH = 1.0
SIDE = 3.0
DX = 0.05
PML = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH)
    parser.add_argument("--side", type=float, default=SIDE)
    parser.add_argument("--dx", type=float, default=DX)
    parser.add_argument("--pml", type=int, default=PML, help="PML cells on each side")
    args = parser.parse_args()
    if args.wavelength <= 0.0 or args.side <= 0.0 or args.dx <= 0.0:
        parser.error("wavelength, side, and dx must be positive")
    n = args.side / args.dx
    if abs(n - round(n)) > 1e-8 or int(round(n)) < 2:
        parser.error("side must be an integer number of cells, at least 2")
    cells = int(round(n))
    if args.pml < 1 or 2 * args.pml >= cells:
        parser.error("pml must be a positive cell count that leaves an interior")
    args.cells = cells
    return args


def _solve(
    cells: int, dx: float, wavelength: float, pml_cells: int
) -> dict[str, object]:
    grid = YeeGrid2D(cells, cells, dx, dx, Polarization.TMZ, Boundary.PEC)
    omega = 2.0 * np.pi * C0 / wavelength
    operator = FDFDOperator(
        grid, UniformIsotropic().sample_tm(grid), PMLSpec.box(pml_cells)
    )
    i0 = cells // 2
    jz = np.zeros(grid.shapes()["ez"])
    jz[i0, i0] = 1.0 / (dx * dx)
    solved = operator.solve(omega, operator.layout.pack_e({"ez": jz}))
    ez = np.zeros(grid.shapes()["ez"], dtype=np.complex128)
    dof = operator.layout.e[0]
    ez[dof.i, dof.j] = solved
    x, y = grid.coordinates("ez")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    exact = line_current_ez(xx - x[i0], yy - y[i0], omega=omega, current=1.0)
    inset = pml_cells * dx
    rho = np.hypot(xx - x[i0], yy - y[i0])
    mask = (xx >= inset) & (xx <= x[-1] - inset) & (yy >= inset) & (yy <= y[-1] - inset)
    mask &= rho >= 0.5 * wavelength
    error = float(np.linalg.norm(ez[mask] - exact[mask]) / np.linalg.norm(exact[mask]))
    return {
        "x": x,
        "y": y,
        "field": ez,
        "exact": exact,
        "mask": mask,
        "error": error,
        "inset": inset,
        "side": cells * dx,
    }


def _panel(ax: plt.Axes, x, y, values, limit: float, inset: float, side: float):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mesh = ax.pcolormesh(
        xx,
        yy,
        np.real(values),
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    width = side - 2.0 * inset
    ax.add_patch(
        Rectangle(
            (inset, inset), width, width, fill=False, ec="#d94801", lw=1.2, zorder=4
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(float(y[0]), float(y[-1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def plot_fields(
    path: Path, result: dict[str, object], wavelength: float, pml_cells: int
) -> None:
    field = np.asarray(result["field"])
    exact = np.asarray(result["exact"])
    mask = np.asarray(result["mask"])
    shown = np.real(field)
    shown_exact = np.real(exact)
    shown_exact = np.where(np.isfinite(shown_exact), shown_exact, 0.0)
    limit = max(
        float(np.max(np.abs(shown[mask]))),
        float(np.max(np.abs(shown_exact[mask]))),
        1e-6,
    )
    diff = shown - shown_exact
    diff_limit = max(float(np.max(np.abs(diff[mask]))), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    panels = (
        (axes[0], shown, limit, "FDFD"),
        (axes[1], shown_exact, limit, "Hankel"),
        (axes[2], diff, diff_limit, "FDFD − Hankel"),
    )
    for ax, values, scale, label in panels:
        image = _panel(
            ax,
            result["x"],
            result["y"],
            values,
            scale,
            float(result["inset"]),
            float(result["side"]),
        )
        ax.set_title(label)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    error = float(result["error"])
    fig.suptitle(
        rf"TMz line current, $\lambda={wavelength:g}$, PML {pml_cells} cells, "
        rf"interior L2 {error:.2%}"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    examples = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(examples))
    from run_dir import output_dir

    args = _parse_args()
    folder = output_dir(
        __file__, lam=args.wavelength, side=args.side, dx=args.dx, pml=args.pml
    )
    result = _solve(args.cells, args.dx, args.wavelength, args.pml)
    plot_fields(folder / "pml_line.png", result, args.wavelength, args.pml)
    print(f"{folder.name}  L2 {float(result['error']):.4%}")


if __name__ == "__main__":
    main()
