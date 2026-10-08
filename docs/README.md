# MaxwellFD documentation

Formulation note, methods write-up, and the project brief. Install and tests:
[`README.md`](../README.md).

| Document | Role |
|----------|------|
| [`../FD_LaTeX_Reference.tex`](../FD_LaTeX_Reference.tex) / `.pdf` | Equation reference. The code cites these labels. The convolutional PML, the embedded PEC and PMC masks, the 3D rectangular cavity, an open dielectric sphere, a resistive sheet, and a symmetric GSTC sheet are implemented. The sphere comparison uses the Mie series; this note does not derive that series. The resistive sheet is the conductance on one Ampere cell. The symmetric cell replaces one magnetic face, the normal magnetic susceptibility enters the TMz jump, and the magneto-electric pair enters both polarizations. A Hertzian dipole is an impressed current on one electric sample, compared with the outgoing spherical wave. A 2D paraxial Gaussian beam is an impressed current on the two electric rows of a horizontal split, compared with the beam on the total-field side. A 2D TMz near-to-far integral samples a closed contour around a line current and is compared with the leading Hankel wave. |
| [`MaxwellFD_2D.tex`](MaxwellFD_2D.tex) / `.pdf` | Physics, discrete method, and code for the implemented solver, including the Fresnel, dielectric-cylinder, PML line-current, open PEC-cylinder, open dielectric-cylinder, resistive-sheet, symmetric-sheet, finite absorbing-sheet, periodic oblique-sheet, finite oblique-sheet, 3D PEC-cavity, open dielectric-sphere, Hertzian-dipole, Gaussian-beam, and line-current far-field examples. |
| [`../FDPrompt.md`](../FDPrompt.md) | Project brief. |

Runnable demos live under `examples/fdfd/2d/`, `examples/fdfd/3d/`, and `examples/fdtd/`. Result folders sit beside the script that wrote them.
