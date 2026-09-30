"""Pure modes of a 3D PEC Yee cavity.

The figure shows the nonzero electric component of (1, 1, 0), (1, 0, 1),
and (0, 1, 1). Each panel is the plane in which that mode varies. The
script prints the relative Rayleigh residual of the real curl-curl
operator on the same grid. ``--dx`` is the spacing on every axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from maxwell_fd.analytics.cavity3d import cavity_mode
from maxwell_fd.drivers.fdfd3d import FDFDOperator3D
from maxwell_fd.grid.yee2d import Boundary
from maxwell_fd.grid.yee3d import YeeGrid3D
from maxwell_fd.materials.volume3d import UniformIsotropic3D

NX = 16
NY = 12
NZ = 10
DX = 1.0
_MODES = ((1, 1, 0, "ez"), (1, 0, 1, "ey"), (0, 1, 1, "ex"))
_TITLES = (
    r"$(1,1,0)$, $E_z$",
    r"$(1,0,1)$, $E_y$",
    r"$(0,1,1)$, $E_x$",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=NX)
    parser.add_argument("--ny", type=int, default=NY)
    parser.add_argument("--nz", type=int, default=NZ)
    parser.add_argument("--dx", type=float, default=DX)
    args = parser.parse_args()
    if min(args.nx, args.ny, args.nz) < 2:
        parser.error("nx, ny, and nz must be at least 2")
    if args.dx <= 0.0:
        parser.error("dx must be positive")
    return args


def _plane(
    grid: YeeGrid3D, field: np.ndarray, component: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    if component == "ez":
        x, y, _z = grid.coordinates("ez")
        return x, y, field[:, :, field.shape[2] // 2], "x", "y"
    if component == "ey":
        x, y, z = grid.coordinates("ey")
        return x, z, field[:, y.size // 2, :], "x", "z"
    x, y, z = grid.coordinates("ex")
    return y, z, field[x.size // 2, :, :], "y", "z"


def _residuals(grid: YeeGrid3D) -> tuple[list[dict[str, np.ndarray]], list[float]]:
    operator = FDFDOperator3D(grid, UniformIsotropic3D().sample(grid))
    stiffness, mass = operator.spatial_operator()
    fields: list[dict[str, np.ndarray]] = []
    residuals: list[float] = []
    for m, n, p, _component in _MODES:
        sampled = cavity_mode(grid, m, n, p)
        fields.append(sampled)
        mode = operator.layout.pack_e(sampled)
        stiffness_mode = stiffness @ mode
        mass_mode = mass @ mode
        rayleigh = float(mode @ stiffness_mode / (mode @ mass_mode))
        residual = np.linalg.norm(stiffness_mode - rayleigh * mass_mode)
        scale = np.linalg.norm(stiffness_mode)
        residuals.append(float(residual / scale))
    return fields, residuals


def plot_modes(
    path: Path, grid: YeeGrid3D, fields: list[dict[str, np.ndarray]]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)
    for ax, mode, sampled, title in zip(axes, _MODES, fields, _TITLES, strict=True):
        component = mode[3]
        horiz, vert, values, xlabel, ylabel = _plane(
            grid, sampled[component], component
        )
        hh, vv = np.meshgrid(horiz, vert, indexing="ij")
        limit = max(float(np.max(np.abs(values))), 1e-6)
        mesh = ax.pcolormesh(
            hh,
            vv,
            values,
            shading="gouraud",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_aspect("equal")
        ax.set_xlim(float(horiz[0]), float(horiz[-1]))
        ax.set_ylim(float(vert[0]), float(vert[-1]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        rf"PEC cavity, ${grid.nx}\times {grid.ny}\times {grid.nz}$, $\Delta={grid.dx:.12g}$"
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import sys

    examples = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(examples))
    from run_dir import output_dir

    args = _parse_args()
    folder = output_dir(__file__, nx=args.nx, ny=args.ny, nz=args.nz, dx=args.dx)
    grid = YeeGrid3D(args.nx, args.ny, args.nz, args.dx, args.dx, args.dx, Boundary.PEC)
    fields, residuals = _residuals(grid)
    plot_modes(folder / "cavity.png", grid, fields)
    report = "  ".join(
        f"({m},{n},{p}) {residual:.3e}"
        for (m, n, p, _component), residual in zip(_MODES, residuals, strict=True)
    )
    print(f"{folder.name}  {report}")


if __name__ == "__main__":
    main()
