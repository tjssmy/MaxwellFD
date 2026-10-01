"""Discrete curl operators and the convolutional PML stretch."""

from maxwell_fd.operators.curl2d import CurlOperators, CurlScale, build_curls
from maxwell_fd.operators.curl3d import CurlOperators3, CurlScale3
from maxwell_fd.operators.curl3d import build_curls as build_curls3
from maxwell_fd.operators.pml import (
    ConvolutionalPML,
    PMLProfile,
    PMLProfile3,
    PMLSpec,
    PMLSpec3,
)

__all__ = [
    "ConvolutionalPML",
    "CurlOperators",
    "CurlOperators3",
    "CurlScale",
    "CurlScale3",
    "PMLProfile",
    "PMLProfile3",
    "PMLSpec",
    "PMLSpec3",
    "build_curls",
    "build_curls3",
]
