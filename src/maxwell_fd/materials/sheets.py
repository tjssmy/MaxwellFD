"""Zero-thickness sheets.

A resistive sheet is not a volumetric sample. Its surface current is placed
in the Ampere cell that contains it, by adding ``1/(Z_s Δy)`` to ``σ`` on the
tangential electric samples of that cell. That is eq:sigma-sheet. The cell
is the half-open interval ``y - Δy/2 <= y_sheet < y + Δy/2`` around each
tangential sample. ``E_z`` carries the TMz current and ``E_x`` carries the
TEz current. The normal component is left alone.

A symmetric GSTC sheet sits on a magnetic face, halfway between electric
rows. ``SymmetricSheet`` holds the tangential susceptibilities and the
normal magnetic susceptibility ``chi_mm_nn``. ``chi_em`` and ``chi_me``
are the magneto-electric strengths in front of ``n̂ × H`` and ``n̂ × E``.
The FDFD driver replaces
that face. An optional ``x`` interval keeps a finite run of the magnetic
samples; the default is the whole face. ``SymmetricSheet.piecewise`` gives
each sample the susceptibilities of the piece that contains its ``x``.
The jumps do not change. This module does not paint ``σ`` for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxwell_fd.grid.yee2d import Boundary, Polarization, YeeGrid2D
from maxwell_fd.materials.volume import TEComponents, TMComponents


@dataclass(frozen=True)
class SymmetricSheet:
    """GSTC sheet on the magnetic face at ``y``.

    ``chi_ee`` and ``chi_mm`` are the tangential susceptibilities in
    eq:sc-h and eq:sc-e. ``chi_mm_nn`` is the normal magnetic
    susceptibility. It enters the TMz jump through ``∂_x H_y`` and is zero
    at normal incidence. ``chi_em`` multiplies ``n̂ × H`` in the electric
    surface polarization and ``chi_me`` multiplies ``n̂ × E`` in the
    magnetic one. A reciprocal sheet uses ``chi_me = chi_em``. The sheet
    sits halfway between electric rows.
    ``x0`` and ``x1`` select the half-open run ``x0 <= x < x1`` of magnetic
    samples on that face. Omitting both covers the whole face. ``pieces``
    replaces that one run: each piece carries its own five susceptibilities
    on its own half-open interval, and a sample outside every piece is not
    on the sheet.
    """

    y: float
    chi_ee: complex
    chi_mm: complex
    x0: float | None = None
    x1: float | None = None
    chi_mm_nn: complex = 0.0
    chi_em: complex = 0.0
    chi_me: complex = 0.0
    pieces: tuple[SymmetricSheet, ...] = ()

    @classmethod
    def piecewise(cls, y: float, pieces: tuple[SymmetricSheet, ...]) -> SymmetricSheet:
        """Sheet whose cut faces take χ from ``pieces``.

        Each piece carries its own susceptibilities on ``x0 <= x < x1``.
        The jumps are the uniform-sheet jumps. Pieces share ``y``, may
        leave a gap, and may not overlap.
        """
        return cls(y, 0.0, 0.0, pieces=tuple(pieces))

    def susceptibility(
        self, x: float
    ) -> tuple[complex, complex, complex, complex, complex] | None:
        """``(chi_ee, chi_mm, chi_mm_nn, chi_em, chi_me)`` at magnetic sample ``x``.

        ``None`` means the sample is not on the sheet.
        """
        if self.pieces:
            for piece in self.pieces:
                if float(piece.x0) <= x < float(piece.x1):
                    return (
                        piece.chi_ee,
                        piece.chi_mm,
                        piece.chi_mm_nn,
                        piece.chi_em,
                        piece.chi_me,
                    )
            return None
        if self.x0 is not None and not (float(self.x0) <= x < float(self.x1)):
            return None
        return (self.chi_ee, self.chi_mm, self.chi_mm_nn, self.chi_em, self.chi_me)

    def carries_normal(self) -> bool:
        """True when any face has a nonzero ``chi_mm_nn``."""
        if self.pieces:
            return any(piece.chi_mm_nn != 0 for piece in self.pieces)
        return self.chi_mm_nn != 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.y):
            raise ValueError(f"sheet coordinate must be finite, got {self.y}")
        electric = complex(self.chi_ee)
        magnetic = complex(self.chi_mm)
        normal = complex(self.chi_mm_nn)
        cross_h = complex(self.chi_em)
        cross_e = complex(self.chi_me)
        if not all(
            np.isfinite(value)
            for value in (electric, magnetic, normal, cross_h, cross_e)
        ):
            raise ValueError(
                "chi must be finite, "
                f"got {self.chi_ee}, {self.chi_mm}, {self.chi_mm_nn}, "
                f"{self.chi_em}, {self.chi_me}"
            )
        object.__setattr__(self, "chi_ee", electric)
        object.__setattr__(self, "chi_mm", magnetic)
        object.__setattr__(self, "chi_mm_nn", normal)
        object.__setattr__(self, "chi_em", cross_h)
        object.__setattr__(self, "chi_me", cross_e)
        try:
            packed = tuple(self.pieces)
        except TypeError as exc:
            raise ValueError("pieces must be a sequence of sheets") from exc
        if any(not isinstance(piece, SymmetricSheet) for piece in packed):
            raise ValueError("each piece must be a SymmetricSheet")
        object.__setattr__(self, "pieces", packed)
        if packed:
            if self.x0 is not None or self.x1 is not None:
                raise ValueError("piecewise sheet sets the extent on each piece")
            if any(
                value != 0 for value in (electric, magnetic, normal, cross_h, cross_e)
            ):
                raise ValueError("piecewise sheet carries chi on each piece")
            _validate_pieces(float(self.y), packed)
            return
        if (self.x0 is None) != (self.x1 is None):
            raise ValueError("sheet extent needs both x0 and x1")
        if self.x0 is None:
            return
        x0 = float(self.x0)
        x1 = float(self.x1)
        if not np.isfinite(x0) or not np.isfinite(x1):
            raise ValueError(f"sheet extent must be finite, got {self.x0}, {self.x1}")
        if not x0 < x1:
            raise ValueError(f"sheet extent must have x0 < x1, got {x0}, {x1}")
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "x1", x1)


def _validate_pieces(y: float, pieces: tuple[SymmetricSheet, ...]) -> None:
    spans: list[tuple[float, float]] = []
    for piece in pieces:
        if piece.pieces:
            raise ValueError("a piece cannot contain pieces")
        if piece.x0 is None or piece.x1 is None:
            raise ValueError("each piece needs x0 and x1")
        if float(piece.y) != y:
            raise ValueError("each piece uses the sheet coordinate")
        spans.append((float(piece.x0), float(piece.x1)))
    end: float | None = None
    for start, stop in sorted(spans):
        if end is not None and start < end:
            raise ValueError("sheet pieces overlap")
        end = stop


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
    y_wall = grid.boundary in (Boundary.PEC, Boundary.PERIODIC_X)
    if y_wall and index in (0, y.size - 1):
        raise ValueError("sheet lies on the PEC wall")
    return index
