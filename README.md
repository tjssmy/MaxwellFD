# MaxwellFD

Yee-cell finite-difference solver for 2D electromagnetics and a 3D FDFD cell, with a path to volumetric media and GSTC metasurfaces. One Cartesian grid feeds both an FDTD leapfrog and a sparse FDFD system in 2D. The 3D path is the same sparse system on a rectangular box. The time convention for phasors is \(e^{j\omega t}\), matching the sibling solver [MaxwellMOM](../MaxwellMOM).

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

Milestone 1 covers the 2D free-space Yee cell: curl signs, PEC cavity eigenvalues, a manufactured FDFD solve, the CFL limit, and a periodic standing wave. Milestone 2 adds staircase and mapped permittivities, Debye and Lorentz media, and Fresnel and dielectric-cylinder checks posed with a Dirichlet trace. The convolutional PML stretches the derivatives on a PEC-backed band, in both drivers. Embedded PEC and PMC objects remove Yee samples, and an open PEC cylinder checks that reduction against the Mie series on the PML. An open dielectric cylinder on the same PML keeps the staircase permittivity in the operator and drives the scattered field with the contrast current. The 3D Yee cell and a sparse FDFD operator check a rectangular PEC cavity: three pure modes against the semi-discrete frequency with the third centered difference. An open dielectric sphere on a PML small enough to factor is compared with the Mie series. A resistive sheet adds \(1/(Z_s\Delta y)\) to the conductivity of one Ampere cell and is compared with the analytic reflection. A symmetric GSTC sheet replaces the magnetic sample on one face with the pair \(H^-\), \(H^+\) and is compared with the analytic reflection and transmission.

## Layout

```
src/maxwell_fd/
  grid/            YeeGrid2D and YeeGrid3D
  materials/       Uniform, staircase, mapped, Debye, Lorentz, conductor masks, resistive and symmetric sheets, and 3D uniform and ball laws
  operators/       2D and 3D curls, the symmetric GSTC cell, and the convolutional PML
  drivers/         2D FDTD step, 2D FDFD solve, and a 3D FDFD solve
  analytics/       cavity frequencies, leapfrog dispersion, Fresnel, 2D and 3D Mie, sheets, line current
```
