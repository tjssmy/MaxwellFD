"""Zero-thickness sheets.

A resistive sheet is not a volumetric sample. Its surface current is placed
in the Ampere cell that contains it, by adding ``1/(Z_s Δy)`` to ``σ`` on the
tangential electric samples of that cell. That is eq:sigma-sheet. The cell
is the half-open interval ``y - Δy/2 <= y_sheet < y + Δy/2`` around each
tangential sample. ``E_z`` carries the TMz current and ``E_x`` carries the
TEz current. The normal component is left alone.

A symmetric GSTC sheet sits on a magnetic face, halfway between electric
rows. ``SymmetricSheet`` holds the tangential susceptibilities. The FDFD
driver replaces that face. This module does not paint ``σ`` for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import TEComponents, TMComponents


@dataclass(frozen=True)
class SymmetricSheet:
    """Tangential GSTC sheet on the magnetic face at ``y``.

    ``chi_ee`` and ``chi_mm`` are the tangential susceptibilities in
    eq:sc-h and eq:sc-e. The sheet sits halfway between electric rows.
    """

    y: float
    chi_ee: complex
    chi_mm: complex

    def __post_init__(self) -> None:
        if not np.isfinite(self.y):
            raise ValueError(f"sheet coordinate must be finite, got {self.y}")
        electric = complex(self.chi_ee)
        magnetic = complex(self.chi_mm)
        if not np.isfinite(electric) or not np.isfinite(magnetic):
            raise ValueError(f"chi must be finite, got {self.chi_ee}, {self.chi_mm}")
        object.__setattr__(self, "chi_ee", electric)
        object.__setattr__(self, "chi_mm", magnetic)


@dataclass(frozen=True)
class ResistiveSheet:
    """Horizontal electric sheet at ``y``, impedance ``zs`` ohms per square."""

    y: float
    zs: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.y):
            raise ValueError(f"sheet coordinate must be finite, got {self.y}")
        if not np.isfinite(self.zs) or self.zs <= 0.0:
            raise ValueError(f"zs must be positive, got {self.zs}")

    def apply(
        self, grid: YeeGrid2D, components: TMComponents | TEComponents
    ) -> TMComponents | TEComponents:
        """Return components whose sheet samples carry ``σ + 1/(Z_s Δy)``."""
        if grid.polarization is Polarization.TMZ:
            if not isinstance(components, TMComponents):
                raise TypeError("TMz sheet expects TMComponents")
            index = _sheet_row(grid, "ez", self.y)
            sigma = np.array(components.sigma_z, copy=True)
            sigma[:, index] += 1.0 / (self.zs * grid.dy)
            return TMComponents(
                eps_z=components.eps_z,
                mu_x=components.mu_x,
                mu_y=components.mu_y,
                sigma_z=sigma,
            )
        if not isinstance(components, TEComponents):
            raise TypeError("TEz sheet expects TEComponents")
        index = _sheet_row(grid, "ex", self.y)
        sigma = np.array(components.sigma_x, copy=True)
        sigma[:, index] += 1.0 / (self.zs * grid.dy)
        return TEComponents(
            eps_x=components.eps_x,
            eps_y=components.eps_y,
            mu_z=components.mu_z,
            sigma_x=sigma,
            sigma_y=components.sigma_y,
        )


def _sheet_row(grid: YeeGrid2D, component: str, y_sheet: float) -> int:
    y = grid.coordinates(component)[1]
    half = 0.5 * grid.dy
    hit = (y_sheet >= y - half) & (y_sheet < y + half)
    indices = np.flatnonzero(hit)
    if indices.size == 0:
        raise ValueError("sheet misses every electric sample")
    if indices.size != 1:
        raise RuntimeError(f"sheet landed in {indices.size} cells")
    index = int(indices[0])
    if grid.boundary is Boundary.PEC and index in (0, y.size - 1):
        raise ValueError("sheet lies on the PEC wall")
    return index
