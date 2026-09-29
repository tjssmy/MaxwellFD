# Yee-Cell Finite-Difference Solver for 2D and 3D Electromagnetics (Metasurface Focus)

**Objective**  
Develop a modular, research-grade Python finite-difference code for time-domain (FDTD) and frequency-domain (FDFD) problems on a uniform Cartesian Yee grid, in 2D and in 3D. The long-term target is efficient analysis of inhomogeneous volumetric media and of metasurface reflectors characterized by Generalized Sheet Transition Conditions (GSTC) and surface susceptibilities. Where the physics overlaps, results should cross-check against the sibling MaxwellMOM solver and against the same analytical reductions (Fresnel, Mie, GSTC reflection and transmission).

**Core Physics Requirements**
- Time convention: $e^{j\omega t}$ (phasor notation). FDTD itself is a real-valued leapfrog update; spectra and every comparison with MaxwellMOM use this convention.
- Formulation: Yee staggered cell. One grid, one material sampling, and one boundary model serve both the explicit leapfrog scheme (FDTD) and the time-harmonic sparse scheme (FDFD).
- Geometry: uniform Cartesian grid. Begin in 2D (TMz and TEz), then the full 3D cell. Volumes are staircase-filled cells. Conductors and metasurfaces are faces or edges of that grid, not a triangle mesh.
- Materials: $\varepsilon$, $\mu$, and $\sigma$ as functions of position, frequency, and external quantities (temperature, stress, and other user-supplied maps). Scalar isotropic media first; tensors after those agree with analytics.
- Boundaries and sheets: PEC, PMC, impedance and other thin sheets, GSTC susceptibility tensors. Open domains closed with a PML.
- Sources: plane wave, Hertzian dipole, Gaussian beam.
- Output: total and scattered fields, reflection and transmission, surface currents on conductors and sheets, near-to-far fields and RCS.

**Key Capabilities (Phased)**
1. Permittivity, permeability, and conductivity as functions of space, frequency, and external quantities such as temperature and stress
2. PEC and PMC surfaces and objects
3. Impedance and other thin-sheet surfaces
4. GSTC / bianisotropic metasurfaces (susceptibility tensors), with a later path to spatially varying and spatially dispersive $\chi$
5. Efficient solution: explicit FDTD; sparse-direct FDFD where the system fits; iterative FDFD where it does not
6. Post-processing and visualization (volume fields, sheet currents, reflection/transmission, far field)

**Development Strategy**
I will use **Grok 4.X** and **Grok Build** for:
- Rigorous derivation of the formulation and numerics
- Creation of a detailed LaTeX reference document (physics, math, discretization, stability and dispersion, boundary and sheet models, validation cases)
- Implementation, refactoring, performance optimization, and testing in the actual codebase

**Required LaTeX Document Sections** (create this first)
1. Mathematical formulation (Maxwell's equations, $e^{j\omega t}$ phasors, Yee placement of $\mathbf{E}$ and $\mathbf{H}$, FDTD leapfrog and FDFD curl operators, total-field/scattered-field source injection)
2. Discretization (centered staggered differences, 2D TMz and TEz reductions, sampling and averaging of materials on the cell, staircase geometry)
3. Stability, numerical dispersion, and domain closure (CFL limit, numerical dispersion relation, PEC/PMC enforcement, PML: convolutional PML in time and complex-stretched coordinates in frequency)
4. Linear algebra and thin sheets (FDTD field update with no stored matrix; FDFD sparse assembly, conditioning, sparse-direct vs GMRES or BiCGSTAB with preconditioning; subcell update for an impedance sheet)
5. Validation strategy and test cases (cavity or parallel-plate modes, dielectric slab vs Fresnel, 2D cylinder and 3D sphere vs Mie, sheet reflection/transmission vs the MaxwellMOM analytic reductions)
6. Extension roadmap to GSTC metasurfaces (uniform tangential $\chi_{ee}$ and $\chi_{mm}$, then normal susceptibilities and magneto-electric coupling, spatial $\chi(\mathbf{r})$, Bloch-periodic cells, dispersion in $\chi$)
7. Software architecture proposal (modules, data structures, class design), including which ideas carry over from MaxwellMOM (material models, excitations, analytical checks) and which are replaced (Yee grid and update in place of RWG and Green's functions)

**Software Architecture Goals**
- Clean separation: Grid → Materials and boundary models → Operators → Drivers → Solvers → Post-processing
- One Yee grid shared by two drivers: an FDTD time stepper and an FDFD sparse system. A validation case runs in time or in frequency from the same geometry description.
- Two material paths, kept separate in the API: a volumetric constitutive law $\varepsilon,\mu,\sigma(\mathbf{r},\omega,\text{external state})$, and a surface model that extends PEC → PMC → impedance → GSTC
- External state (temperature, stress, and similar) is a prescribed field sampled onto the grid. A coupled thermal or mechanical solve can feed that interface later.
- FDFD storage is sparse: direct factorization for 2D and small 3D, iterative solvers (GMRES or BiCGSTAB, with preconditioning) for larger systems
- Acceleration path: Numba kernels, then CuPy, for the FDTD field update and the FDFD matrix-vector product

**Incremental Milestones**
1. 2D Yee grid shared by FDTD and FDFD. Free-space plane wave, CFL check, numerical-dispersion check, and a PEC cavity or parallel-plate mode against analytic eigenvalues.
2. Volumetric materials on the staggered locations: $\varepsilon(\mathbf{r})$, then $\varepsilon(\omega)$ (a complex permittivity in FDFD; Debye or Lorentz auxiliaries in FDTD). Dielectric slab against Fresnel coefficients, dielectric cylinder against the 2D Mie series. Temperature, stress, and similar quantities enter as prescribed maps that set those constitutive values.
3. PML truncation and embedded PEC/PMC objects. Open-region 2D scattering against Mie. Then the 3D cell: a rectangular cavity, then a dielectric or PEC sphere against the Mie series, at a size the FDFD system can still factor.
4. Impedance / thin-sheet boundary. Normal-incidence reflection and transmission against analytics and against a MaxwellMOM impedance-sheet run of the same case.
5. GSTC subcell for a flat sheet: uniform tangential $\chi_{ee}$ and $\chi_{mm}$ first, checked on reflection and transmission against the closed forms MaxwellMOM uses. Then $\chi_{mm}^{nn}$ and magneto-electric terms.
6. Hertzian dipole and Gaussian-beam sources, total versus scattered field, and a near-to-far transformation. Bloch-periodic boundaries for a metasurface unit cell, compared with MaxwellMOM periodic reflection/transmission on cases both codes can run.
7. Profiling, then Numba, iterative FDFD, and GPU experiments. Spatially varying $\chi(\mathbf{r})$ and spatially dispersive $\chi$ come after the uniform-sheet subcell matches analytics.

**Non-Functional Requirements**
- Readable, well-documented research code (type hints, docstrings, modular)
- Reproducible test cases with known reference solutions, analytical where they exist and MaxwellMOM where that solver can run the same configuration
- Easy to extend for new constitutive laws, surface types, and excitations
- Visualization of volume fields, sheet and surface currents, and near- and far-field quantities

**Workflow**
1. Grok creates the detailed physics + numerics LaTeX document + high-level implementation plan.
2. We review and refine the plan.
3. Grok Build implements the code incrementally following the milestones.
4. I test in the local environment and provide feedback.

Please begin by creating the LaTeX document with sections 1–7 above. Derive the 2D Yee cell first, for free space and isotropic dielectrics, with the FDTD and FDFD forms side by side. Carry the GSTC subcell in the extension roadmap, and derive that operator once the baseline discretization is in place.
