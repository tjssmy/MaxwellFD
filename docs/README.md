# MaxwellFD documentation

Formulation note, methods write-up, and the project brief. Install and tests:
[`README.md`](../README.md).

| Document | Role |
|----------|------|
| [`../FD_LaTeX_Reference.tex`](../FD_LaTeX_Reference.tex) / `.pdf` | Equation reference. The code cites these labels. The convolutional PML, the embedded PEC and PMC masks, and the 3D rectangular cavity are implemented from this note. Resistive sheets and the GSTC cell are specified here and are not implemented. A 3D sphere small enough to factor is still open. |
| [`MaxwellFD_2D.tex`](MaxwellFD_2D.tex) / `.pdf` | Physics, discrete method, and code for the implemented solver, including the Fresnel, dielectric-cylinder, PML line-current, open PEC-cylinder, open dielectric-cylinder, and 3D PEC-cavity examples. |
| [`../FDPrompt.md`](../FDPrompt.md) | Project brief. |

Runnable demos live under `examples/fdfd/2d/`, `examples/fdfd/3d/`, and `examples/fdtd/`. Result folders sit beside the script that wrote them.
