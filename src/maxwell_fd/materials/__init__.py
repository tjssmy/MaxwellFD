"""Volumetric constitutive laws and staircase conductors."""

from maxwell_fd.materials.conductors import Circle, Conductors, YBand
from maxwell_fd.materials.dispersion import DebyeDielectric, LorentzDielectric
from maxwell_fd.materials.volume import (
    Disk,
    MappedPermittivity,
    SlabY,
    StaircaseIsotropic,
    TEComponents,
    TMComponents,
    UniformIsotropic,
)
from maxwell_fd.materials.volume3d import FieldComponents3D, UniformIsotropic3D

__all__ = [
    "Circle",
    "Conductors",
    "DebyeDielectric",
    "Disk",
    "LorentzDielectric",
    "YBand",
    "MappedPermittivity",
    "SlabY",
    "StaircaseIsotropic",
    "TEComponents",
    "TMComponents",
    "UniformIsotropic",
    "FieldComponents3D",
    "UniformIsotropic3D",
]
