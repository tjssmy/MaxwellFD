# MaxwellFD documentation

Formulation note, methods write-up, and the project brief. Install and tests:
[`README.md`](../README.md).

| Document | Role |
|----------|------|
| [`../FD_LaTeX_Reference.tex`](../FD_LaTeX_Reference.tex) / `.pdf` | Equation reference. The code cites these labels. The convolutional PML and the embedded PEC and PMC masks are implemented from this note. Resistive sheets and the GSTC cell are specified here and are not implemented. |
| [`MaxwellFD_2D.tex`](MaxwellFD_2D.tex) / `.pdf` | Physics, discrete method, and code for the implemented 2D Yee solver, including the Fresnel, dielectric-cylinder, PML line-current, and open PEC-cylinder examples. |
| [`../FDPrompt.md`](../FDPrompt.md) | Project brief. |

Runnable demos live under `examples/fdfd/2d/` and `examples/fdtd/`. Result folders sit beside the script that wrote them.
