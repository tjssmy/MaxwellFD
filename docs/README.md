# MaxwellFD documentation

Formulation note, methods write-up, and the project brief. Install and tests:
[`README.md`](../README.md).

| Document | Role |
|----------|------|
| [`../FD_LaTeX_Reference.tex`](../FD_LaTeX_Reference.tex) / `.pdf` | Equation reference. The code cites these labels. The convolutional PML, the embedded PEC and PMC masks, the 3D rectangular cavity, an open dielectric sphere, and a resistive sheet are implemented. The sphere comparison uses the Mie series; this note does not derive that series. The sheet is the conductance on one Ampere cell. The GSTC cell is specified here and is not implemented. |
| [`MaxwellFD_2D.tex`](MaxwellFD_2D.tex) / `.pdf` | Physics, discrete method, and code for the implemented solver, including the Fresnel, dielectric-cylinder, PML line-current, open PEC-cylinder, open dielectric-cylinder, resistive-sheet, 3D PEC-cavity, and open dielectric-sphere examples. |
| [`../FDPrompt.md`](../FDPrompt.md) | Project brief. |

Runnable demos live under `examples/fdfd/2d/`, `examples/fdfd/3d/`, and `examples/fdtd/`. Result folders sit beside the script that wrote them.
