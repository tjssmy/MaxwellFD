"""Volumetric constitutive laws, staircase conductors, and thin sheets."""

from maxwell_fd.materials.conductors import Circle, Conductors, YBand
from maxwell_fd.materials.dispersion import DebyeDielectric, LorentzDielectric
from maxwell_fd.materials.sheets import ResistiveSheet, SymmetricSheet
from maxwell_fd.materials.volume import (
    Disk,
    MappedPermittivity,
    SlabY,
    StaircaseIsotropic,
    TEComponents,
    TMComponents,
    UniformIsotropic,
)
from maxwell_fd.materials.volume3d import (
    Ball,
    FieldComponents3D,
    StaircaseIsotropic3D,
    UniformIsotropic3D,
)

__all__ = [
    "Ball",
    "Circle",
    "Conductors",
    "DebyeDielectric",
    "Disk",
    "LorentzDielectric",
    "YBand",
    "MappedPermittivity",
    "ResistiveSheet",
    "SymmetricSheet",
    "SlabY",
    "StaircaseIsotropic",
    "StaircaseIsotropic3D",
    "TEComponents",
    "TMComponents",
    "UniformIsotropic",
    "FieldComponents3D",
    "UniformIsotropic3D",
]
