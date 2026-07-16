References
==========

Background and canonical references for VMEC and related equilibrium methods:

1. S. P. Hirshman and J. C. Whitson, “Steepest-descent moment method for
   three-dimensional magnetohydrodynamic equilibria,” *Physics of Fluids* 26
   (1983).

2. S. P. Hirshman, W. I. van Rij, and P. Merkel, “Three-dimensional free
   boundary calculations using a spectral Green’s function method,” *Computer
   Physics Communications* 43 (1986).

3. P. Merkel, “Solution of stellarator boundary value problems with external
   currents,” *Nuclear Fusion* 27 (1987).

4. VMEC2000 reference documentation and ``wout`` file format notes (VMEC/LIBSTELL
   distribution and Princeton VMEC resources).

5. VMEC++ numerics notes (local copy):
   ``vmecpp/docs/the_numerics_of_vmecpp.pdf``.

6. VMEC++ Fourier basis implementation note (local copy):
   ``vmecpp/docs/fourier_basis_implementation.md``.

7. VMEC2000 solver core (residuals, bcovar, preconditioner):
   ``STELLOPT/VMEC2000/Sources/General/funct3d.f`` and
   ``STELLOPT/VMEC2000/Sources/General/bcovar.f``.

8. VMEC2000 time-step control and restart logic:
   ``STELLOPT/VMEC2000/Sources/TimeStep/evolve.f`` and
   ``STELLOPT/VMEC2000/Sources/TimeStep/restart.f``.

9. VMEC2000 diagnostic scalars and Mercier stability:
   ``STELLOPT/VMEC2000/Sources/Input_Output/eqfor.f`` and
   ``STELLOPT/VMEC2000/Sources/Input_Output/mercier.f``.

10. A. H. Glasser, J. M. Greene, and J. L. Johnson, “Resistive instabilities
    in general toroidal plasma configurations,” *Physics of Fluids* 18(7),
    875-888 (1975).

11. M. Landreman and R. Jorge, “Magnetic well and Mercier stability of
    stellarators near the magnetic axis,” *Journal of Plasma Physics* 86(5),
    905860510 (2020), arXiv:2006.14881.

12. VMEC++ solver/restart structure and parity-relevant control flow:
    ``vmecpp/src/vmecpp/cpp/vmecpp/vmec/vmec/vmec.cc``.

13. VMEC++ output-quantity and near-axis extrapolation notes:
    ``vmecpp/src/vmecpp/cpp/vmecpp/vmec/output_quantities/output_quantities.cc``.

14. P. Kim, R. Jorge, and W. Dorland, “The On-Axis Magnetic Well and
    Mercier's Criterion for Arbitrary Stellarator Geometries,” *Journal of
    Plasma Physics* 87(4), 905870409 (2021), arXiv:2011.07416.

15. J. Schilling et al., “Magnetohydrodynamic equilibrium and stability
    properties of the Infinity Two fusion pilot plant,” *Journal of Plasma
    Physics* 90(6), 905900615 (2024), Appendix B.

16. J. Schilling et al., “VMEC++: The Numerics of VMEC,” arXiv:2502.04374 —
    hot restart, JSON input schema, zero-crash policy, and the wout
    validation methodology adopted here.

17. C. S. Skene and K. J. Burns, “Fast automated adjoints for spectral PDE
    solvers,” arXiv:2506.14792 — adjoints reusing the forward spectral
    machinery; the template for the implicit-differentiation module.

18. M. Blondel et al., “Efficient and Modular Implicit Differentiation,”
    NeurIPS 2022 (jaxopt) — the implicit-function-theorem ``custom_vjp``
    formulation used for equilibrium gradients.

Confinement objectives and optimization:

19. M. Landreman and E. Paul, “Magnetic fields with precise quasisymmetry for
    plasma confinement,” *Physical Review Letters* 128, 035001 (2022),
    arXiv:2108.03711 — the two-term quasisymmetry ratio residual and the
    precise-QA/QH configurations (:doc:`confinement`).

20. A. Goodman et al., “Constructing precisely quasi-isodynamic magnetic
    fields,” *Journal of Plasma Physics* 89(5), 905890504 (2023),
    arXiv:2211.09829 — the constructed-QI target implemented by
    :class:`~vmec_jax.core.omnigenity.QIResidual`.

21. J. R. Cary and S. G. Shasharina, “Omnigenity and quasihelicity in helical
    plasma confinement systems,” *Physics of Plasmas* 4, 3323 (1997) — the
    bounce-integral formulation of omnigenity.

22. D. Dudt et al., “Magnetic fields with general omnigenity,” *Journal of
    Plasma Physics* 90(1), 905900120 (2024), arXiv:2305.08026 — omnigenity
    optimization in a differentiable (DESC) framework.

23. A. Redl et al., “A new set of analytical formulae for the computation of
    the bootstrap current and the neoclassical conductivity in stellarators,”
    *Physics of Plasmas* 28, 022502 (2021) — the Redl bootstrap closure.

24. M. Landreman, S. Buller, and M. Drevlak, “Optimization of quasi-symmetric
    stellarators with self-consistent bootstrap current and energetic particle
    confinement,” *Physics of Plasmas* 29, 082501 (2022), arXiv:2205.02914 —
    the self-consistent bootstrap iteration reproduced in
    ``examples/optimization/*_bootstrap_selfconsistent.py``.

25. R. Jorge, A. Goodman, M. Landreman, J. Rodrigues, and F. Wechsung,
    “Single-stage stellarator optimization: combining coils with fixed
    boundary equilibria,” *Plasma Physics and Controlled Fusion* 65, 074003
    (2023), arXiv:2302.10622 — the combined plasma–coil objective
    ``J = J_plasma + w_coils J_coils`` and the two-stage vs single-stage
    comparison protocol used by the single-stage examples.

26. R. Jorge, A. Giuliani, and J. Loizu, “Simplified and flexible coils for
    stellarators using single-stage optimization,” arXiv:2406.07830 (2024) —
    cold-start single-stage optimization with staged Fourier-mode release.

27. F. Wechsung et al., “Precise stellarator quasi-symmetry can be achieved
    with electromagnetic coils,” *PNAS* 119(13), e2202084119 (2022) — coil
    regularization set (length, curvature, coil–coil distance) and the
    normalized ``max |B·n|/|B|`` reporting convention.
