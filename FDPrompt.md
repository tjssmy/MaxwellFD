# MOM / BEM Solver for 3D Electromagnetic Scattering (Metasurface Focus)

**Objective**  
Develop a modular, research-grade Python MOM/BEM code for solving 3D scattering problems on piecewise planar surfaces in free space. The long-term target is efficient analysis of metasurface reflectors characterized by Generalized Sheet Transition Conditions (GSTC) and surface susceptibilities.

**Core Physics Requirements**
- Time convention: $e^{j\omega t}$ (phasor notation)
- Formulation: Start with the Electric Field Integral Equation (EFIE) for PEC/PMC surfaces. Later support combined-field formulations and GSTC.
- Geometry: Triangulated 2D surfaces in 3D space (initially flat finite plates, later spheres, cones, etc.)
- Basis functions: RWG (Rao-Wilton-Glisson) elements
- Sources: Plane wave, Hertzian dipoles, Gaussian beams
- Output: Scattered/total fields, surface current distributions, RCS

**Key Capabilities (Phased)**
1. PEC and PMC surfaces
2. Impedance surfaces
3. GSTC / bianisotropic metasurfaces (susceptibility tensors)
4. Efficient matrix assembly and solution (direct + iterative)
5. Post-processing and visualization (fields on surfaces + far-field)

**Development Strategy**
I will use **Grok 4.3** for:
- Rigorous derivation of the formulation and numerics
- Creation of a detailed LaTeX reference document (physics, math, discretization, quadrature, singularity treatment, validation cases)

I will use **Grok Build** for:
- Implementation, refactoring, performance optimization, and testing in the actual codebase

**Required LaTeX Document Sections** (create this first)
1. Mathematical formulation (EFIE, integral operators, Green's function)
2. Discretization with RWG basis functions and testing procedure
3. Singularity extraction / regularization techniques
4. Matrix assembly and conditioning considerations
5. Validation strategy and analytical test cases (flat plate, sphere via Mie series, etc.)
6. Extension roadmap to GSTC metasurfaces
7. Software architecture proposal (modules, data structures, class design)

**Software Architecture Goals**
- Clean separation: Geometry/Mesh → Operators → Solvers → Post-processing
- Extensible surface material model (PEC → Impedance → GSTC)
- Frequency-driven meshing with per-surface control
- Support for both dense direct solve and iterative solvers (GMRES + preconditioning)
- Clear path toward acceleration (Numba → CuPy / GPU matrix-vector products)

**Incremental Milestones**
1. Flat finite PEC plate + plane-wave incidence (EFIE, RWG, validation against analytical or reference data)
2. PEC sphere (Mie series validation)
3. Multiple PEC objects
4. Impedance surfaces
5. GSTC metasurface implementation (initially normal susceptibility)
6. Performance profiling + GPU acceleration experiments

**Non-Functional Requirements**
- Readable, well-documented research code (type hints, docstrings, modular)
- Reproducible test cases with known reference solutions
- Easy to extend for new surface types and excitations
- Visualization of currents and near/far fields

**Workflow**
1. Grok 4.3 creates the detailed physics + numerics LaTeX document + high-level implementation plan.
2. We review and refine the plan.
3. Grok Build implements the code incrementally following the milestones.
4. I test in Cursor / local environment and provide feedback.

Please begin by creating the LaTeX document with sections 1–7 above, focusing first on the EFIE + RWG formulation for PEC surfaces.