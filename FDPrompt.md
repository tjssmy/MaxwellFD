# MOM / BEM Solver for 3D Electromagnetic Scattering (Metasurface Focus)

**Objective**  
Develop a modular, research-grade Python FD codes for solving 3D scattering problems. The long-term target is efficient analysis of metasurface reflectors characterized by Generalized Sheet Transition Conditions (GSTC) and surface susceptibilities.

**Core Physics Requirements**
- Time convention: $e^{j\omega t}$ (phasor notation)
- Formulation: Yee Cell for FDTD or FD
- Geometry: Uniform grid in 2D or 3D
- Sources: Plane wave, Hertzian dipoles, Gaussian beams
- Output: Scattered/total fields, surface current distributions

**Key Capabilities (Phased)**
1. Specify permittivity as function of space, frequency and external quantities such as temperature, stress, etc. 
2. PEC and PMC surfaces
3. Impedance surfaces
4. GSTC / bianisotropic metasurfaces (susceptibility tensors)
5. Efficient solution
6. Post-processing and visualization (fields on surfaces + far-field)

**Development Strategy**
I will use **Grok 4.X** and **Grok Build** for:
- Rigorous derivation of the formulation and numerics
- Creation of a detailed LaTeX reference document (physics, math, discretization, validation cases)
- Implementation, refactoring, performance optimization, and testing in the actual codebase

**Required LaTeX Document Sections** (create this first)
1. Mathematical formulation 
2. Discretization
4. Matrix assembly and conditioning considerations
5. Validation strategy and test cases
6. Extension roadmap to GSTC metasurfaces
7. Software architecture proposal (modules, data structures, class design)

**Software Architecture Goals**
- Clean separation: Geometry/Mesh → Operators → Solvers → Post-processing
- Extensible surface material model (PEC → Impedance → GSTC)
- Support for both dense direct solve and iterative solvers (GMRES + preconditioning) for FD problem
- Clear path toward acceleration (Numba → CuPy / GPU matrix-vector products) for FDTD and FD

**Incremental Milestones**
- To be filled in
  
**Non-Functional Requirements**
- Readable, well-documented research code (type hints, docstrings, modular)
- Reproducible test cases with known reference solutions
- Easy to extend for new surface types and excitations
- Visualization of currents and near/far fields

**Workflow**
1. Grok creates the detailed physics + numerics LaTeX document + high-level implementation plan.
2. We review and refine the plan.
3. Grok Build implements the code incrementally following the milestones.
4. I test in local environment and provide feedback.

Please begin by creating the LaTeX document with sections 1–7 above, focusing first on the EFIE + RWG formulation for PEC surfaces.
