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

Milestone 1 covers the 2D free-space Yee cell: curl signs, PEC cavity eigenvalues, a manufactured FDFD solve, the CFL limit, and a periodic standing wave. Milestone 2 adds staircase and mapped permittivities, Debye and Lorentz media, and Fresnel and dielectric-cylinder checks posed with a Dirichlet trace. The convolutional PML stretches the derivatives on a PEC-backed band, in both drivers. Embedded PEC and PMC objects remove Yee samples, and an open PEC cylinder checks that reduction against the Mie series on the PML. An open dielectric cylinder on the same PML keeps the staircase permittivity in the operator and drives the scattered field with the contrast current. The 3D Yee cell and a sparse FDFD operator check a rectangular PEC cavity: three pure modes against the semi-discrete frequency with the third centered difference. An open dielectric sphere on a PML small enough to factor is compared with the Mie series. A resistive sheet adds \(1/(Z_s\Delta y)\) to the conductivity of one Ampere cell and is compared with the analytic reflection. A symmetric GSTC sheet replaces the magnetic sample on one face with the pair \(H^-\), \(H^+\) and is compared with the analytic reflection and transmission. A finite run of that face, with \(\chi_{ee}=\chi_{mm}=-2j/k\), is a normal-incidence absorber: the unknown is the scattered field, and a PML backs all four sides. The normal magnetic susceptibility \(\chi_{mm}^{nn}\) adds \(\partial_x H_y\) to the TMz jump. One periodic cell, one transverse period wide, compares that term at an angle with the analytic reflection and transmission. The same susceptibilities on a \(10\lambda\) sheet at \(30^\circ\), with a PML on all four sides, are the open scattered-field run. \(\chi_{em}\) and \(\chi_{me}\) add the magneto-electric coupling on both polarizations. A reciprocal sheet uses the same value for both. One periodic cell compares that pair with the analytic reflection and transmission, and the same susceptibilities on a \(10\lambda\) sheet at \(30^\circ\) are the open scattered-field run. A \(z\)-directed Hertzian dipole places \(I\ell\) on one \(E_z\) sample and is compared with the outgoing spherical wave on a PML-backed cube. A 2D paraxial Gaussian beam is launched by the impressed current on the two \(E_z\) rows of a horizontal split, and the total-field side is compared with the beam. A 2D TMz near-to-far integral samples \(E_z\) and the averaged tangential \(H\) on a square contour around a line current and is compared with the leading Hankel wave. A Bloch phase \(e^{-j k_{B,x} a}\) multiplies the forward periodic \(x\) wrap. A cell \(0.8\lambda\) wide at \(30^\circ\), narrower than one transverse period, compares that phase with the analytic oblique sheet.

## Layout

```
src/maxwell_fd/
  grid/            YeeGrid2D and YeeGrid3D
  materials/       Uniform, staircase, mapped, Debye, Lorentz, conductor masks, resistive and symmetric sheets, and 3D uniform and ball laws
  operators/       2D and 3D curls, the symmetric GSTC cell, and the convolutional PML
  drivers/         2D FDTD step, 2D FDFD solve, and a 3D FDFD solve
  analytics/       cavity frequencies, leapfrog dispersion, Fresnel, 2D and 3D Mie, sheets, line current, Hertzian dipole, Gaussian beam, near-to-far integral
```
