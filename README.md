# MaxwellFD

Yee-cell finite-difference solver for 2D (and later 3D) electromagnetics, with a path to volumetric media and GSTC metasurfaces. One Cartesian grid feeds both an FDTD leapfrog and a sparse FDFD system. The time convention for phasors is \(e^{j\omega t}\), matching the sibling solver [MaxwellMOM](../MaxwellMOM).

The formulation note is `FD_LaTeX_Reference.tex`. The project brief is `FDPrompt.md`. The implemented solver, the example figures, and how to run them are described in [`docs/MaxwellFD_2D.pdf`](docs/MaxwellFD_2D.pdf); the index is [`docs/README.md`](docs/README.md).

## Install

```bash
conda env create -f environment.yml
conda activate maxwell-fd
```

Or, in an existing environment that already has NumPy and SciPy:

```bash
pip install -e .[dev]
```

## Tests

```bash
pytest
```

Milestone 1 covers the 2D free-space Yee cell: curl signs, PEC cavity eigenvalues, a manufactured FDFD solve, the CFL limit, and a periodic standing wave. Milestone 2 adds staircase and mapped permittivities, Debye and Lorentz media, and Fresnel and dielectric-cylinder checks posed with a Dirichlet trace. The convolutional PML stretches the derivatives on a PEC-backed band, in both drivers. Embedded PEC and PMC objects remove Yee samples, and an open PEC cylinder checks that reduction against the Mie series on the PML. Impedance sheets and GSTC are specified in the note and are not in this slice.

## Layout

```
src/maxwell_fd/
  grid/            YeeGrid2D
  materials/       Uniform, staircase, mapped, Debye, Lorentz, and conductor masks
  operators/       curl_e, curl_h, and the convolutional PML
  drivers/         FDTD step and FDFD sparse solve
  analytics/       cavity frequencies, leapfrog dispersion, Fresnel, Mie, line current
```
