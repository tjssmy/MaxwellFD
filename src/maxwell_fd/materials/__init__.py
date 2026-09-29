"""Volumetric constitutive laws."""

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

__all__ = [
    "DebyeDielectric",
    "Disk",
    "LorentzDielectric",
    "MappedPermittivity",
    "SlabY",
    "StaircaseIsotropic",
    "TEComponents",
    "TMComponents",
    "UniformIsotropic",
]
