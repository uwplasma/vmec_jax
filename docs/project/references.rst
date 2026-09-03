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
    875-888 (1975), doi:10.1063/1.861224.

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

19. A. H. Boozer, “Plasma equilibrium with rational magnetic surfaces,”
    *Physics of Fluids* 24, 1999 (1981), doi:10.1063/1.863297 — the Boozer
    coordinates in which quasisymmetry and the fast-ion proxies are stated.

20. R. Sanchez et al., “Ballooning stability optimization of low-aspect-ratio
    stellarators,” *Plasma Physics and Controlled Fusion* 42, 641 (2000),
    doi:10.1088/0741-3335/42/6/303 — the ballooning-optimization companion to
    the COBRA formulation used by :mod:`vmex.core.stability`.

Confinement objectives and optimization:

21. M. Landreman and E. Paul, “Magnetic fields with precise quasisymmetry for
    plasma confinement,” *Physical Review Letters* 128, 035001 (2022),
    arXiv:2108.03711, doi:10.1103/PhysRevLett.128.035001 — the two-term
    quasisymmetry ratio residual (the residual inside eq. 1) and the
    precise-QA/QH configurations (:doc:`/explanation/confinement`).

22. A. Goodman et al., “Constructing precisely quasi-isodynamic magnetic
    fields,” *Journal of Plasma Physics* 89(5), 905890504 (2023),
    arXiv:2211.09829 — the constructed-QI target implemented by
    :class:`~vmex.core.qi.ConstructedQIResidual`; the lightweight
    :class:`~vmex.core.omnigenity.QIResidual` is a level-set surrogate.

23. J. R. Cary and S. G. Shasharina, “Omnigenity and quasihelicity in helical
    plasma confinement systems,” *Physics of Plasmas* 4, 3323 (1997) — the
    bounce-integral formulation of omnigenity.

24. D. Dudt et al., “Magnetic fields with general omnigenity,” *Journal of
    Plasma Physics* 90(1), 905900120 (2024), arXiv:2305.08026 — omnigenity
    optimization in a differentiable (DESC) framework.

25. A. Redl, C. Angioni, E. Belli, and O. Sauter, “A new set of analytical
    formulae for the computation of the bootstrap current and the neoclassical
    conductivity in tokamaks,” *Physics of Plasmas* 28, 022502 (2021),
    doi:10.1063/5.0012664 — the Redl bootstrap closure (eqs. 10-16, 19-21).

26. M. Landreman, S. Buller, and M. Drevlak, “Optimization of quasi-symmetric
    stellarators with self-consistent bootstrap current and energetic particle
    confinement,” *Physics of Plasmas* 29, 082501 (2022), arXiv:2205.02914,
    doi:10.1063/5.0098166 —
    the self-consistent bootstrap iteration reproduced in
    ``benchmarks/*_bootstrap_selfconsistent.py``; the compact finite-beta
    optimization workflows are ``examples/optimization/*_optimization_bootstrap.py``.

27. R. Jorge, A. Goodman, M. Landreman, J. Rodrigues, and F. Wechsung,
    “Single-stage stellarator optimization: combining coils with fixed
    boundary equilibria,” *Plasma Physics and Controlled Fusion* 65, 074003
    (2023), arXiv:2302.10622 — the combined plasma–coil objective
    ``J = J_plasma + w_coils J_coils`` and the two-stage vs single-stage
    comparison protocol used by the single-stage examples.

28. R. Jorge, A. Giuliani, and J. Loizu, “Simplified and flexible coils for
    stellarators using single-stage optimization,” arXiv:2406.07830 (2024) —
    cold-start single-stage optimization with staged Fourier-mode release.

29. F. Wechsung et al., “Precise stellarator quasi-symmetry can be achieved
    with electromagnetic coils,” *PNAS* 119(13), e2202084119 (2022) — coil
    regularization set (length, curvature, coil–coil distance) and the
    normalized ``max |B·n|/|B|`` reporting convention.

30. K. Unalmis et al., “Spectrally accurate, reverse-mode differentiable
    bounce-averaging algorithm and its applications,” *Journal of Plasma
    Physics* (2026), doi:10.1017/S0022377826101652 — endpoint-regularized
    quadrature and the independent DESC oracle for
    :mod:`vmex.core.bounce`.

31. E. Rodríguez, P. Helander, and A. G. Goodman, “The maximum-J property
    in quasi-isodynamic stellarators,” *Journal of Plasma Physics* 90,
    905900212 (2024), doi:10.1017/S0022377824000345 — the signed
    outward :math:`\partial\mathcal J_\parallel/\partial s<0` condition implemented
    by :class:`~vmex.core.maxj.MaximumJResidual`.

32. E. Rodríguez and G. G. Plunk, “Near-axis quasi-isodynamic database,”
    *Journal of Plasma Physics* (2026),
    doi:10.1017/S0022377826101688 — defines the bounce-time-weighted
    Maxwellian maximum-J fraction :math:`f_J`, which is distinct from a
    uniform resolved-orbit count.

33. D. A. Spong and J. H. Harris, “New QP/QI symmetric stellarator
    configurations,” *Plasma and Fusion Research* 5, S2039 (2010),
    doi:10.1585/pfr.5.S2039 — motivates quasi-poloidal symmetry as a practical
    precursor to poloidally closed-contour quasi-isodynamic configurations.

34. R. Conlin, P. Kim, D. W. Dudt, D. Panici, and E. Kolemen, “Stellarator
    Optimization with Constraints,” *Journal of Plasma Physics* 90,
    905900501 (2024), arXiv:2403.11033 — hard shaping constraints and
    augmented-Lagrangian methods for stellarator design.

35. B. Jang, R. Conlin, and M. Landreman, “Exponential Spectral Scaling:
    Robust and Efficient Stellarator Boundary Optimization via Mode-Dependent
    Scaling,” arXiv:2509.16320 (2025) — the direct full-spectrum variable
    scaling exposed by ``use_ess=True``.

36. D. Panici, B. Jang, R. Conlin, D. Dudt, Y. G. Elmacioglu, and E. Kolemen,
    “Deflation Techniques for Stellarator Equilibrium and Optimization,”
    arXiv:2602.09957 (2026) — a systematic route to distinct minima when a
    single deterministic continuation path is insufficient.

37. H. Chen *et al.*, “Direct Optimization of Stellarator Omnigenity from the
    Second Adiabatic Invariant,” arXiv:2608.02418 (2026) — recent direct
    bounce-action optimization complementary to constructed geometric QI
    targets.

38. V. V. Nemov, S. V. Kasilov, W. Kernbichler, and M. F. Heyn,
    “Evaluation of :math:`1/\nu` neoclassical transport in stellarators,”
    *Physics of Plasmas* 6, 4622–4632 (1999), doi:10.1063/1.873749 — defines
    the effective-ripple transport measure reported by NEO as
    :math:`\epsilon_{\mathrm{eff}}^{3/2}`.

39. V. V. Nemov et al., “Poloidal motion of trapped particle orbits in
    real-space coordinates,” *Physics of Plasmas* 15, 052501 (2008),
    doi:10.1063/1.2912456 — defines the :math:`\Gamma_c` fast-ion proxy
    (eq. 61) evaluated by :class:`~vmex.core.gammac.GammaC`.

40. J. L. Velasco et al. and the W7-X Team, “A model for the fast evaluation
    of prompt losses of energetic ions in stellarators,” *Nuclear Fusion* 61,
    116059 (2021), doi:10.1088/1741-4326/ac2994 — the :math:`\Gamma_c`
    organization used here (eq. 16) and the approximately linear prompt-loss
    relation of eqs. 20-21.

41. P. Helander, *Reports on Progress in Physics* 77, 087001 (2014),
    doi:10.1088/0034-4885/77/8/087001 — the review behind the statement that
    the two-term residual vanishes exactly for a quasisymmetric :math:`|B|`.

42. O. Sauter, C. Angioni, and Y. R. Lin-Liu, *Physics of Plasmas* 6, 2834
    (1999), doi:10.1063/1.873240 (erratum *Physics of Plasmas* 9, 5140
    (2002), doi:10.1063/1.1517052) — the collisionality formulas (eqs.
    18b-18e) that the Redl closure reuses in :mod:`vmex.core.bootstrap`.

43. Y. R. Lin-Liu and R. L. Miller, *Physics of Plasmas* 2, 1666 (1995),
    doi:10.1063/1.871315 — the trapped-fraction groundwork underlying the
    Sauter and Redl fits.

44. A. Bader et al., “Stellarator equilibria with reactor relevant energetic
    particle losses,” *Journal of Plasma Physics* 85, 905850508 (2019),
    doi:10.1017/S0022377819000680 — energetic-particle losses measured
    against proxy values.

45. A. Bader et al., “Modeling of energetic particle transport in optimized
    stellarators,” *Nuclear Fusion* 61, 116060 (2021),
    doi:10.1088/1741-4326/ac2991 — documents the imperfect correlation
    between :math:`\Gamma_c` and simulated energetic-particle losses.

Numerical implementation:

46. JAX documentation, “The Autodiff Cookbook” — the wide-Jacobian cost model
    (``jacfwd`` versus ``jacrev``) used to select ``jacrev`` for the local
    three-surface raw-force blocks,
    https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html.

47. X. S. Li, “An Overview of SuperLU: Algorithms, Implementation, and User
    Interface,” *ACM Transactions on Mathematical Software* 31, 302–325
    (2005) — globally pivoted sparse factorization used for the equilibrated
    block-banded radial solve, doi:10.1145/1089014.1089017.
