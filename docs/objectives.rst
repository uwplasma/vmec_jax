Objectives library
==================

Everything you can put in front of the optimizer, in one place.  An
objective in ``vmex`` is an ordinary function (or a small class) of a
*converged* equilibrium — there is no objective base class to subclass, no
registration step.  This page catalogs the built-in objectives by physics
area, shows a minimal usage snippet for each, and ends with the one table
that matters in practice: which objectives support exact implicit gradients
(``jac="implicit"``) and which need finite differences (``jac=None``).  The
first-principles derivations of the metrics themselves — Boozer coordinates,
the quasisymmetry two-term residual, the constructed-QI target, Mercier and
the magnetic well — are on :doc:`confinement`.

.. contents:: On this page
   :local:
   :depth: 1

How objectives plug in
----------------------

:func:`~vmex.core.optimize.least_squares` takes simsopt-style
``(function, target, weight)`` terms; each term contributes
``weight * (function(eq) - target)`` rows to the stacked residual.  Two
calling conventions are recognized automatically:

- **one positional argument** — the term receives the converged
  :class:`~vmex.core.optimize.Equilibrium` (which carries ``state``,
  ``runtime``, ``wout``, and the input).  Residual-class instances
  (:class:`~vmex.core.optimize.QuasisymmetryRatioResidual`,
  :class:`~vmex.core.omnigenity.QIResidual`,
  :class:`~vmex.core.bootstrap.RedlBootstrapMismatch`) are callable this
  way, as is any user lambda;
- **two positional arguments** — the term is treated as a pure traceable
  ``(state, runtime)`` function (the scalar targets below).

For ``jac="implicit"`` every term must be traceable: either a
two-positional ``(state, runtime)`` callable, or an object exposing a
``residuals_state(state, runtime)`` method (the residual classes do — the
optimizer picks it up automatically, so the *same* term list works in both
gradient modes).  Terms that evaluate wout tables on host NumPy
(:func:`~vmex.core.optimize.d_merc`,
:func:`~vmex.core.optimize.l_grad_b`, the wout-lane QI residual, the
eigenvector-weighted turbulence proxies) work with ``jac=None`` only —
use :func:`~vmex.core.optimize.mercier_stability_residual` for Mercier and
:func:`~vmex.core.optimize.l_grad_b_state` for ``L_grad_B`` with
``jac="implicit"``.

.. code-block:: python

   import numpy as np
   import vmex as vj
   from vmex import optimize as opt

   inp = vj.VmecInput.from_file("input.minimal_seed_nfp2")
   qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10),
                                       helicity_m=1, helicity_n=0)
   result = opt.least_squares(
       [(qs, 0.0, 1.0),                    # residual class instance
        (opt.aspect_ratio, 6.0, 1.0),      # traceable scalar target
        (opt.mean_iota, 0.42, 1.0)],
       inp, max_mode=5, jac="implicit", use_ess=True)

Quasisymmetry
-------------

:class:`~vmex.core.optimize.QuasisymmetryRatioResidual` is the
Landreman–Paul two-term quasisymmetry ratio residual, sampled pointwise on
the requested flux surfaces (full Gauss–Newton residual geometry, not a
pre-summed scalar).  The helicity pair selects the symmetry family:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - family
     - ``(helicity_m, helicity_n)``
     - contours of ``|B|`` in Boozer coordinates
   * - QA (quasi-axisymmetric)
     - ``(1, 0)``
     - poloidally closed (tokamak-like)
   * - QH (quasi-helical)
     - ``(1, -1)`` or ``(1, 1)``
     - diagonal, pitch set by ``nfp``
   * - QP (quasi-poloidal)
     - ``(0, 1)``
     - toroidally closed

``helicity_n`` is in units of ``nfp`` (the simsopt convention).  Measured
campaign results from a near-circular seed are on :doc:`optimization`; the
runnable scripts are ``examples/optimization/QA_optimization_ess.py`` and
friends.

Scalar geometry and profile targets
-----------------------------------

All of these are pure ``(state, runtime)`` functions — traceable, cheap,
and composable with both gradient modes:

- :func:`~vmex.core.optimize.aspect_ratio` — the VMEC/simsopt effective
  aspect ratio;
- :func:`~vmex.core.optimize.volume` — plasma volume;
- :func:`~vmex.core.optimize.mean_iota` /
  :func:`~vmex.core.optimize.edge_iota` — rotational-transform targets
  (a floor on ``|iota|`` avoids the rational surfaces near zero transform);
- :func:`~vmex.core.optimize.mirror_ratio` — ``(Bmax - Bmin)/(Bmax +
  Bmin)`` on a flux surface, the practical QI knob;
- :func:`~vmex.core.optimize.magnetic_well` — the standard vacuum-well
  measure (positive = well, stabilizing).

Two reporting diagnostics run on the host-side wout engine:
:func:`~vmex.core.optimize.d_merc` (Mercier interchange criterion) and
:func:`~vmex.core.optimize.l_grad_b` (the ``L_grad_B`` coil-complexity proxy).
Their live-state counterparts :func:`~vmex.core.stability.d_merc_state`
(``lasym = False``), :func:`~vmex.core.stability.jdotb_state`,
:func:`~vmex.core.stability.jdotb_residual`,
:func:`~vmex.core.stability.glasser_d_r_state`, and
:func:`~vmex.core.optimize.l_grad_b_state` are pure JAX and can be composed
with implicit differentiation.  ``jdotb_state`` reproduces the VMEC
``jdotb = <J.B>`` WOUT profile; ``jdotb_residual`` selects ``[2:-1]`` for a
least-squares current target.  ``glasser_d_r_state`` evaluates the
Glasser--Greene--Johnson resistive-interchange parameter, for which
``D_R <= 0`` is the necessary stability condition on nonzero-shear surfaces
provided the ideal prerequisite ``DMerc > 0`` also holds.
The convenience
:func:`~vmex.core.optimize.mercier_stability_residual` selects
``DMerc[2:-1]`` and returns
``smoothing * softplus((margin - DMerc) / smoothing)``.  Positive ``DMerc``
is stable, so target this residual to zero; the default ``margin=0`` and
``smoothing=1e-6`` give a smooth approximation to the instability hinge
``max(-DMerc, 0)``.  Softplus is strictly positive at finite arguments, so a
stable surface approaches rather than reaches zero; its residual decreases
exponentially as ``DMerc - margin`` grows.

.. code-block:: python

   terms = [(opt.mercier_stability_residual, 0.0, 100.0)]
   result = opt.least_squares(terms, inp, max_mode=3, jac="implicit")

The analogous :func:`~vmex.core.optimize.glasser_stability_residual`
penalizes positive ``D_R[2:-1]``.  Exact zero-shear surfaces are physically
outside the GGJ criterion; the residual uses ``shear_epsilon=1e-8`` by
default so zero-shear seeds have finite optimization rows.  The reporting
profile keeps the strict zero default. Always include
:func:`~vmex.core.optimize.mercier_stability_residual` in the same objective
so ``DMerc > 0`` is enforced rather than treating ``D_R`` as a standalone
criterion. Then post-check
:func:`~vmex.core.optimize.mercier_shear_state` and require
``abs(S) >> shear_epsilon`` on every target surface; a regularized zero-shear
result is numerically defined but not a valid GGJ stability claim.

``L_grad_B`` additionally has a fully traceable ``(state, runtime)`` lane,
:func:`~vmex.core.optimize.l_grad_b_state` — same convention
(``L_grad_B = |B| sqrt(2 / ||grad B||_F^2)``, same sampling grid and radial
stencils, wout-lane parity to float round-off), rebuilt from the state-field
chain in pure JAX, so it works under ``jac="implicit"``.  The default hard
minimum over the surface grid is exact but has a jumping gradient when the
minimizing gridpoint switches; passing ``softmin_k=k`` selects the smooth
soft minimum ``-logsumexp(-k L)/k`` (a lower bound within
``log(ntheta*nphi)/k``) — optimize with the smooth form, report the hard
minimum (gradient validated against the frozen-path FD to 1.7e-6):

.. code-block:: python

   import functools
   lgradb = functools.partial(opt.l_grad_b_state, softmin_k=50.0)
   result = opt.least_squares(
       [(qs, 0.0, 1.0), (lgradb, 0.35, 1.0)],
       inp, max_mode=3, jac="implicit")

Omnigenity and quasi-isodynamicity
----------------------------------

:class:`~vmex.core.omnigenity.QIResidual` is the traceable
quasi-isodynamic objective (plan R26h.h2): the Goodman *et al.* (JPP 2023)
constructed-QI-target distance — bounce-distance uniformity, extremum
alignment, and squash monotonicity, each term an exact zero of an exactly
QI field — evaluated in level space on a pure-JAX Boozer ``|B|`` spectrum
(:func:`~vmex.core.omnigenity.boozer_bmnc_state`, parity ~1e-6 against
``booz_xform_jax`` on the dominant modes).  Because the whole chain is
traceable, QI optimization now runs with exact implicit gradients:

.. code-block:: python

   from vmex.core.omnigenity import QIResidual

   qi = QIResidual(np.linspace(0.15, 0.95, 6))
   result = opt.least_squares(
       [(qi, 0.0, 10.0),
        (opt.mirror_ratio, 0.20, 2.0),
        (opt.mean_iota, 0.12, 1.0)],
       inp, max_mode=6, jac="implicit", use_ess=True)

Sanity anchors (CI-gated): an analytically QI field scores ``< 1e-24``, the
bundled ``nfp1_QI`` deck scores 36x below a circular tokamak and 138x below
the (QA, deliberately non-QI) Landreman–Paul configuration.  The measured
single-call campaign — seed 4.5e-1 to 1.8e-2 (25x) in 17.3 minutes — is in
:doc:`optimization`.  The earlier Goodman-style *wout-lane* residual
(:func:`~vmex.core.optimize.quasi_isodynamic_residual`, host NumPy,
``jac=None``) remains available for diagnostics and cross-checks.

.. note::

   The traceable ``QIResidual`` is tuned for *gradient flow*, not for
   *labeling*: driven hard it reports an optimistic absolute value.  When
   quoting how QI a finished configuration is, use the fully-resolved
   wout/Boozer residual
   :func:`~vmex.core.optimize.quasi_isodynamic_residual_from_wout` —
   optimize with the traceable form, report with the wout form.  See
   :ref:`confinement-qi-fidelity`.

Bootstrap current (Redl)
------------------------

:mod:`vmex.core.bootstrap` implements the Redl (2021) analytic
bootstrap formula, differentiably, plus the machinery to make an
equilibrium's current profile self-consistent with it (plan R26g,
reproducing the workflow of Landreman–Buller–Drevlak, arXiv:2205.02914):

- :class:`~vmex.core.bootstrap.KineticProfiles` — prescribed
  ``n_e/T_e/T_i/Z_eff`` polynomials in ``s`` (objective parameters, not
  VMEC inputs);
- :class:`~vmex.core.bootstrap.RedlBootstrapMismatch` — the paper's
  ``f_boot``: the normalized mismatch between the equilibrium's
  ``<J.B>`` (:func:`~vmex.core.bootstrap.vmec_j_dot_B`, via the MHD
  identity) and the Redl prediction
  (:func:`~vmex.core.bootstrap.j_dot_B_redl`).  Dual-lane: a
  wout-table lane with simsopt ``VmecRedlBootstrapMismatch`` parity, and a
  traceable ``residuals_state`` lane for ``jac="implicit"``.  Evaluated on
  the published optima of arXiv:2205.02914, ``f_boot`` lands at 2.5e-4 (QA,
  2.5% beta), 3.5e-5 (QH 2.5%), 1.3e-4 (QH 5%);
- ``least_squares(..., current_dofs=k)`` — frees the first ``k`` ``AC``
  power-series coefficients plus ``CURTOR`` alongside the boundary
  harmonics, in both gradient modes — the dof set a self-consistent
  bootstrap optimization needs;
- :func:`~vmex.core.bootstrap.self_consistent_bootstrap` — a
  fixed-boundary Picard loop that iterates the current profile to
  bootstrap consistency (hot-restarted solves; a tokamak test case
  converges in 9 iterations, ``f_boot`` 0.37 to 3.4e-3).

.. code-block:: python

   from vmex.core.bootstrap import KineticProfiles, RedlBootstrapMismatch

   profiles = KineticProfiles(                      # paper profiles:
       ne_coeffs=4.13e20 * np.array([1, 0, 0, 0, 0, -1]),  # n0 (1 - s^5)
       Te_coeffs=12.0e3 * np.array([1, -1]),               # T0 (1 - s)
       Ti_coeffs=12.0e3 * np.array([1, -1]))
   boot = RedlBootstrapMismatch(profiles, helicity_n=0)    # 0 = QA
   result = opt.least_squares(
       [(qs, 0.0, 1.0), (boot, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0)],
       inp, max_mode=4, jac="implicit",
       current_dofs=6)          # free AC[0:6] + CURTOR with the boundary

MHD stability
-------------

:mod:`vmex.core.stability` provides the infinite-n ideal-ballooning
objective (plan R26h.h1) — a JAX port of the COBRA eigenproblem in the Gaur
*et al.* (arXiv:2302.07673) formulation, with field-line coefficients per
simsopt's COBRA-validated ``vmec_fieldlines`` conventions and a batched
symmetric-tridiagonal ``eigvalsh`` solve:

- :func:`~vmex.core.stability.ballooning_lambda` — the most-unstable
  eigenvalue per (surface, field line, ballooning parameter); ``λ > 0`` is
  unstable, with ``λ = (γ a_N/v_A)²``;
- :func:`~vmex.core.stability.ballooning_growth_rate` — a smooth
  scalar reduction (``softmax`` upper bound of λ over all lines), built to
  be *driven negative* as a stable-by-construction constraint.

.. code-block:: python

   from vmex.core.stability import ballooning_growth_rate

   terms = [(qs, 0.0, 1.0),
            (ballooning_growth_rate, -0.01, 5.0)]   # keep λ_max below zero
   result = opt.least_squares(terms, inp, max_mode=4, jac="implicit")

Everything inside is JAX AD (geometry derivatives included), so it composes
with both gradient modes; ``d(growth)/d(pres_scale)`` matches finite
differences to 4.7e-9 in CI, and the objective destabilizes monotonically
with pressure on the solovev family, in sign agreement with Mercier.  For
interchange stability, combine with
:func:`~vmex.core.optimize.magnetic_well` (traceable) or
:func:`~vmex.core.optimize.mercier_stability_residual` (traceable); retain
:func:`~vmex.core.optimize.d_merc` for wout-lane reporting or ``jac=None``.

Turbulence proxies (SPECTRAX-GK)
--------------------------------

:mod:`vmex.core.turbulence` wires the gyrokinetic proxies of
`SPECTRAX-GK <https://github.com/uwplasma/spectrax-gk>`_ (uwplasma's
JAX-native Hermite–Laguerre flux-tube solver; ``pip install spectraxgk``,
optional dependency) into the objective protocol (plan R26h.h4):

- :func:`~vmex.core.turbulence.gk_fieldline_geometry` /
  :func:`~vmex.core.turbulence.flux_tube_geometry` — sample one field
  line of the converged interior solution into GS2/GX-normalized flux-tube
  geometry (``bmag``, ``gds2/gds21/gds22``, curvature/grad-B drifts, …),
  pure JAX, no spectraxgk import needed;
- :func:`~vmex.core.turbulence.turbulent_growth_rate` — the dominant
  linear ITG/TEM growth rate on that flux tube.  Fully differentiable in
  *both* gradient modes (validated 0.44 ``v_th/L`` at the Cyclone-base
  drive ``R/L_Ti = 6.9`` versus ~0 below the critical gradient; AD vs FD
  2.9e-8);
- :func:`~vmex.core.turbulence.quasilinear_flux_proxy` and
  :func:`~vmex.core.turbulence.nonlinear_heat_flux_proxy` — the
  mixing-length and saturation-rule heat-flux surrogates.  These weight the
  dominant *eigenvector*, whose derivatives JAX declines for non-symmetric
  operators — value-level objectives, ``jac=None``.

.. code-block:: python

   from vmex.core.turbulence import turbulent_growth_rate

   terms = [(qs, 0.0, 1.0),
            (turbulent_growth_rate, 0.0, 0.5)]      # push gamma toward 0
   result = opt.least_squares(terms, inp, max_mode=3, jac="implicit")

Which objectives differentiate how
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 14 16 32

   * - objective
     - ``jac=None``
     - ``jac="implicit"``
     - why
   * - :class:`~vmex.core.optimize.QuasisymmetryRatioResidual`
     - yes
     - yes
     - traceable ``residuals_state`` lane
   * - scalar targets (aspect, volume, iota, mirror, well)
     - yes
     - yes
     - pure ``(state, runtime)`` functions
   * - :class:`~vmex.core.omnigenity.QIResidual`
     - yes
     - yes
     - traceable Boozer transform + smooth level-space residual
   * - :class:`~vmex.core.bootstrap.RedlBootstrapMismatch`
     - yes
     - yes
     - dual lane (wout parity / traceable)
   * - :func:`~vmex.core.stability.ballooning_growth_rate`
     - yes
     - yes
     - all-JAX eigenproblem, softmax reduction
   * - :func:`~vmex.core.turbulence.turbulent_growth_rate`
     - yes
     - yes
     - eigenvalue-only reduction carries JVP + VJP
   * - :func:`~vmex.core.optimize.l_grad_b_state`
     - yes
     - yes
     - traceable ``L_grad_B`` lane (soft-min via ``softmin_k``)
   * - :func:`~vmex.core.optimize.d_merc_state`,
       :func:`~vmex.core.optimize.mercier_stability_residual`
     - yes
     - yes
     - traceable Mercier profile / smooth interior instability hinge
       (stellarator-symmetric states only)
   * - :func:`~vmex.core.optimize.jdotb_state`,
       :func:`~vmex.core.optimize.jdotb_residual`,
       :func:`~vmex.core.optimize.mercier_shear_state`,
       :func:`~vmex.core.optimize.glasser_d_r_state`,
       :func:`~vmex.core.optimize.glasser_stability_residual`
     - yes
     - yes
     - traceable VMEC ``<J.B>`` and GGJ resistive-interchange profile /
       smooth upper-bound residual (stellarator-symmetric states only)
   * - :func:`~vmex.core.optimize.d_merc`,
       :func:`~vmex.core.optimize.l_grad_b`
     - yes
     - no
     - host-NumPy wout engine (``l_grad_b``: use
       :func:`~vmex.core.optimize.l_grad_b_state` instead)
   * - :func:`~vmex.core.optimize.quasi_isodynamic_residual` (wout lane)
     - yes
     - no
     - host-NumPy Boozer tables (use
       :class:`~vmex.core.omnigenity.QIResidual` instead)
   * - :func:`~vmex.core.turbulence.quasilinear_flux_proxy`,
       :func:`~vmex.core.turbulence.nonlinear_heat_flux_proxy`
     - yes
     - no
     - eigenvector weights have no nonsymmetric-eig derivative

``jac="implicit"`` requires a fixed-boundary problem. Its boundary parameter
map supports both symmetric and ``LASYM = T`` equilibria, but individual
objectives can be symmetry-limited; in particular the traceable Mercier and
Glasser objectives and quasisymmetry currently require ``LASYM = F``.
The traceable ``jdotb`` objective supports both symmetry modes. See
:doc:`optimization` for the gradient machinery and measured cost of each
piece.

Writing your own objective
--------------------------

Any function of the converged equilibrium is already an objective.  For
finite differences, one argument is enough:

.. code-block:: python

   terms = [(lambda eq: float(eq.wout.b0), 1.0, 1.0)]   # target B0 = 1 T

For implicit gradients, write it as a pure two-positional
``(state, runtime)`` JAX function — the scalar targets in
:mod:`vmex.core.optimize` (~10 lines each) are the templates to copy.
If it returns a vector, each entry becomes a Gauss–Newton residual row;
give a class a ``residuals_state(state, runtime)`` method and a
``J(eq)``/``__call__`` pair and it will work in both modes, like the
built-in residual classes.
