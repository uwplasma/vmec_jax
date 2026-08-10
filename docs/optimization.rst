Optimization and differentiability
==================================

``vmex`` turns VMEC into a differentiable building block: converged
equilibria expose exact gradients with respect to boundary shape, profile,
and coil parameters, and a weighted least-squares driver uses them to
run whole stellarator-design campaigns in minutes on a CPU.  This page
covers the driver and the gradient machinery; the catalog of objective
functions (quasisymmetry, omnigenity, bootstrap, stability, turbulence, …)
lives on its own page, :doc:`objectives`, and every worked example is a
runnable script in ``examples/optimization/`` (see :doc:`tutorials`).

.. contents:: On this page
   :local:
   :depth: 1

The optimizer-neutral problem
-----------------------------

VMEX owns the equilibrium, objective composition, and derivatives; it does
not require users to adopt one optimization algorithm.  Construct a
:class:`~vmex.core.problem.VmecProblem` and pass its ordinary callables to
SciPy, JAXopt, Optax, or user code:

.. code-block:: python

   from dataclasses import replace
   import jax.numpy as jnp
   import numpy as np
   import scipy.optimize
   import vmex as vj
   from vmex import optimize as opt

   inp = vj.VmecInput.from_file("input.minimal_seed_nfp2")
   max_mode = 5
   mpol = max(max_mode + 2, 5)
   ntor = mpol
   ntheta = 2 * mpol + 6
   nzeta = 2 * ntor + 4
   inp = replace(inp, delt=0.5)
   inp = inp.change_resolution(
       mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nzeta
   )
   qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10),
                                       helicity_m=1, helicity_n=0)

   def elongation_excess(state, runtime):
       return jnp.maximum(opt.max_elongation(state, runtime) - 8.0, 0.0)

   problem = opt.VmecProblem.from_tuples(
       inp,
       [(qs, 0.0, 1.0),
        (opt.aspect_ratio, 6.0, 1.0),
        (opt.mean_iota, 0.42, 1.0),
        (elongation_excess, 0.0, 1.0)],
       max_mode=max_mode,
       progress=True,                 # report elapsed construction time
   )
   problem.compile_residual_and_jacobian()  # optional visible first compilation

   result = scipy.optimize.least_squares(
       problem.residual,
       problem.x0,
       jac=problem.residual_jac,
       x_scale=problem.scales,
   )
   optimized_input = problem.input_from_x(result.x)
   optimized_equilibrium = problem.equilibrium_from_x(result.x)

The optimization script owns its numerical policy.  It selects ``DELT``,
``MPOL``, ``NTOR``, ``NTHETA``, and ``NZETA`` explicitly.  The immutable
:meth:`~vmex.core.input.VmecInput.change_resolution` operation only changes
the four resolution fields, preserving every representable axis and boundary
coefficient and zeroing newly exposed modes.

:func:`~vmex.core.optimize.make_problem` with ``progress=True`` reports seed
validation and structure preparation;
:meth:`~vmex.core.problem.FunctionProblem.compile_residual_and_jacobian` or
:meth:`~vmex.core.problem.FunctionProblem.compile_value_and_gradient` makes
the first derivative compilation visible before entering an optimizer.  The
compile calls are optional for correctness: without one, the optimizer
performs the same work on its first evaluation.  Each operation prints a
short label, elapsed-time heartbeats, and completion time.  For a
CPU-only job where cold latency matters more than peak warm-kernel speed, set
``VMEX_FAST_COMPILE=1`` *before* importing VMEX; the machine-local persistent
compilation cache is already enabled by default.

The factory's four ordinary defaults are:

* ``derivative_method="implicit"`` -- exact derivatives of the converged
  fixed-boundary equilibrium;
* ``implicit_jacobian_method="auto"`` -- a reverse adjoint for one residual
  row and the block-tridiagonal response for a residual vector;
* ``jacobian_batch_size=1`` -- one response column per batch, minimizing cold
  compilation complexity and peak memory;
* ``weight_semantics="cost"`` -- tuple weight ``w`` multiplies
  ``0.5 * (value - target)**2``.

Beginner QI and QS scripts should omit all four.  ``jacobian_batch_size`` is
the main compile-latency/warm-throughput control.  Isolated cold-cache tests on
an Apple M3 Max (JAX/JAXlib 0.9.2) covered QI, QS, and four scalar geometry /
profile terms at ``max_mode`` 1, 3, and 5.  Batch size 1 shortened the first
Jacobian compilation by 12.9--14.2 s in every case.  Complete 12-evaluation
mode-5 runs took 99.5 s versus 107.7 s for QI and 84.0 s versus 93.8 s for QS,
with final costs agreeing within ``1e-10`` relative.  Use
``jacobian_batch_size="auto"`` for long repeated campaigns at one array shape,
where its faster warm stages amortize the larger initial compilation.  The
explicit ``"block_tridiagonal"``, ``"forward_gmres"``, and
``"reverse_adjoint"`` methods are advanced numerical controls rather than
alternative beginner settings.  These timings are benchmark observations,
not CI thresholds.  Reproduce individual cold rows with
``VMEX_COMPILATION_CACHE=disabled python
benchmarks/optimization_defaults.py --input INPUT --case qi --max-mode 5
--batch 1``; the complete recorded matrix is
``benchmarks/optimization_defaults.json`` (the measurement platform is
recorded inside the artifact).

The same object provides ``fun``/``grad``/``value_and_grad`` for scalar
optimizers and ``jax_fun``/``jax_value_and_grad``/``jax_residual`` for JAX
libraries.  The decision vector is always explicit; evaluation does not
mutate a hidden ``problem.x``.  ``J(x)`` and ``dJ(x)`` are aliases for users
familiar with SIMSOPT.

For a continuation stage, ``problem.x_from_input(inp)`` maps the current input
through the problem's exact boundary/profile parameterization.  There is no
ambiguous mutable ``inp.x`` because the selected variables depend on
``max_mode`` and ``current_dofs``.  After optimization,
``problem.equilibrium_from_x(result.x)`` returns the accepted converged state
already used to evaluate that point.  Reuse it when reporting or refining a
strongly shaped result instead of cold-solving from a reconstructed magnetic
axis:

.. code-block:: python

   final_input = replace(
       optimized_input,
       ns_array=np.array([101]),
       ftol_array=np.array([1.0e-14]),
       niter_array=np.array([8000]),
   )
   final_equilibrium = opt.solve_equilibrium(
       final_input,
       initial_state=optimized_equilibrium.state,
       verbose=True,                    # print the VMEC iteration table
       raise_on_max_iterations=True,
   )
   final_input.to_indata("input.optimized_final")
   vj.write_wout("wout_optimized_final.nc", final_equilibrium.wout)

Mode ladders and resolution convergence
---------------------------------------

ESS is a variable scale, not a continuation method.  Releasing all modes at
once is useful for a fast survey because it pays one JAX compilation.  A mode
ladder can reach another basin by first solving the low-order shape, then
carrying that boundary into successively richer spaces.  The explicit pattern
is deliberately ordinary Python:

.. code-block:: python

   for max_mode in [1, 2, 3, 4, 5]:
       mpol = max(max_mode + 2, 5)
       inp = inp.change_resolution(
           mpol=mpol,
           ntor=mpol,
           ntheta=2 * mpol + 6,
           nzeta=2 * mpol + 4,
       )
       problem = opt.VmecProblem.from_tuples(
           inp, terms, max_mode=max_mode
       )
       result = scipy.optimize.least_squares(
           problem.residual,
           problem.x_from_input(inp),
           jac=problem.residual_jac,
           x_scale=problem.scales,
           max_nfev=15,
       )
       inp = problem.input_from_x(result.x)
       equilibrium = problem.equilibrium_from_x(result.x)
       inp.to_indata(f"input.qi_max_mode_{max_mode:03d}")

On the bundled ``input.nfp2_QI_seed`` case, direct ``max_mode=5`` stopped after 47
evaluations at cost 0.03742.  The ``[1, 2, 3, 4, 5]`` ladder, capped at 15
evaluations per stage (75 total), reached 0.02582 -- 31% lower.  This is a
basin result, not a claim that every QS/QI problem needs a ladder.  A stage
endpoint can be higher than the previous endpoint because changing ``mpol``,
``ntor``, and JAX shapes changes the discretized problem.  Within each fixed
stage, accepted least-squares costs remain monotone.

Radial resolution should be treated as a convergence parameter, not a magic
minimum.  Holding that ladder boundary fixed and resolving it at increasingly
fine ``NS`` gave:

.. list-table:: Fixed-boundary radial convergence (strict equilibrium solve)
   :header-rows: 1
   :widths: 15 25 25

   * - ``NS``
     - QI total
     - total least-squares cost
   * - 15
     - 0.025795
     - 0.025394
   * - 19
     - 0.026356
     - 0.025714
   * - 25
     - 0.026463
     - 0.025814
   * - 35
     - 0.026265
     - 0.025776
   * - 51
     - 0.026407
     - 0.025893
   * - 75
     - 0.026525
     - 0.025983
   * - 101
     - 0.026479
     - 0.025977

Thus ``NS=25`` is the measured minimum suitable for inexpensive optimization
of this smooth, vacuum, ``max_mode <= 5`` case (QI differs by 0.06% from
``NS=101``); ``NS=35`` is the safer default when boundaries become strongly
shaped or rejected equilibria appear.  ``NS=15`` is a coarse scouting grid,
not a final optimization grid.  The ``NS=101`` run does not change the
boundary variables, but it recomputes the equilibrium and objectives at a
finer radial discretization, so it is validation/refinement rather than a
guarantee that a coarse-grid minimum is unchanged.

Likewise, ``mpol=ntor=max_mode+2`` is an efficient optimization resolution,
not a spectral-convergence certificate.  On the same optimized mode-5
boundary, raising both solver truncations from 7 to 9 changed the recomputed
QI total by about 2.1%.  Validate final candidates with at least one larger
``mpol``/``ntor`` buffer; if the target metric moves materially, polish the
optimization at that resolution.  Keep the real-space grids above the
spectral anti-aliasing threshold when doing so.

At ``NS=101`` and ``FTOL=1e-14`` the hot solve converged in 2,656 iterations
versus 3,046 from the automatic cold-axis path.  Both succeeded; carrying the
accepted state saved 13% of the iterations.  Resolution and spectral-shape
changes limit restart reuse, so the larger speedups occur between nearby
optimization trials of the same shape.  ``verbose=True`` prints the live
iteration table; the measured 2,656 iterations also shows why
``NITER_ARRAY=8000`` is a conservative final-run limit here.

The factory accepts ``weight_semantics="cost"`` (the default), for which a
tuple weight ``w`` contributes ``sqrt(w) * (f - target)`` residual rows and
therefore multiplies the squared cost.  ``weight_semantics="residual"`` makes
``w`` multiply the residual row itself.  These names state the formula and do
not depend on knowledge of another package or an older VMEX interface.

``derivative_method="implicit"`` means exact differentiation of the
converged fixed-boundary equilibrium, not differentiation through its solver
iterations.  This first factory supports that method.  Complete user-supplied
x-level derivatives use ``FunctionProblem.from_functions``; the parallel
finite-difference provider is added in the later derivative-provider layer.

``implicit_jacobian_method="auto"`` is the normal choice.  Advanced
comparisons can select ``"block_tridiagonal"``, ``"forward_gmres"``, or
``"reverse_adjoint"``.  These names describe the mathematical assembly path;
the older compatibility drivers retain their established ``jac_solver``
spelling.

Accepted-iteration monitoring
-----------------------------

Objective evaluation is silent by default.  Printing from an objective is
misleading because trust-region and line-search methods deliberately evaluate
rejected points, so their raw costs need not decrease monotonically.  Attach
an :class:`~vmex.core.monitoring.OptimizationMonitor` to report optimizer
callbacks instead:

.. code-block:: python

   from vmex import OptimizationMonitor

   monitor = OptimizationMonitor(problem)
   result = scipy.optimize.least_squares(
       problem.residual,
       problem.x0,
       jac=problem.residual_jac,
       x_scale=problem.scales,
       callback=monitor,
   )

The stable columns are iteration, cost, accepted reduction, optimality when
the backend supplies it, equilibrium solves, and failed/rejected trials.
``monitor.records`` contains immutable records for plotting or tests.
JAXopt, Optax, and custom loops call ``monitor.record(...)`` with the value
and optimality they already computed, so monitoring never dictates the
optimizer.  The compatibility least-squares driver uses SciPy's concise
accepted-iteration table for ``verbose=1``; scalar ``minimize`` uses this
same monitor.

External optimizers and supplied derivatives
--------------------------------------------

The files ``QI_optimization_scipy.py``, ``QI_optimization_jaxopt.py``, and
``QI_optimization_optax.py`` in ``examples/optimization`` construct the same
QI :class:`~vmex.core.problem.VmecProblem`.  They demonstrate SciPy nonlinear
least squares, BFGS and L-BFGS-B; JAXopt LBFGS and Levenberg--Marquardt; and an
arbitrary Optax transformation chain.  VMEX contains no method registry or
hand-written clone of those algorithms.

Opaque host objectives use the same public object with a different derivative
provider:

.. code-block:: python

   problem = opt.VmecProblem.from_tuples(
       inp,
       [(host_objective, target, weight)],
       derivative_method="finite_difference",  # parallel equilibrium re-solves
       fd_method="3-point",
       workers=None,                 # automatic workstation default
   )

Central probes are independent and run through
:func:`vmex.core.parallel.map_ensemble`; ``workers=1`` selects a serial
reference.  Traceable objectives should keep
``derivative_method="implicit"`` (exact converged-equilibrium derivatives): a
scalar exact gradient uses one adjoint instead of two equilibrium solves per
degree of freedom.  Users with complete x-level derivatives construct
:class:`~vmex.core.problem.FunctionProblem` directly with ``value_and_grad``
or ``residual_and_jac``.  Thus exact, finite-difference, and supplied
derivatives all present identical optimizer-facing names.

``benchmarks/optimization_derivatives.py`` records exact/parallel-FD timing,
work count, and numerical agreement under one input and objective.
``benchmarks/qi_optimizer_acceptance.py`` reproduces the precise-QI objective
tuple on the bundled ``input.nfp2_QI_seed``,
checks finite derivatives at ``max_mode=1`` or 5, and can run every documented
backend with a fixed iteration budget.  Reports include package versions and
failure/fallback counters.  SIMSOPT and DESC versions are recorded for
provenance, but VMEX does not claim a cross-code speed ratio unless equation,
resolution, variables, objective, tolerances, and warm-cache state match; a
shorter run of different physics is not a defensible benchmark.

The committed 2026-08-08 measurements (platform recorded inside each
artifact) are in ``benchmarks/optimization_derivatives.json`` and
``benchmarks/qi_optimizer_acceptance_mode5.json``.  On the two-variable Solovev
gate, after both lanes were warmed, the exact Jacobian took 1.52 ms versus
277.83 ms for four parallel central probes (183x), agreeing to 1.23e-10
relative error.  More importantly for scaling, the exact scalar-gradient work
does not grow to 196 equilibrium probes at the 98-variable precise-QI
``max_mode=5`` point: its gradient was finite in 47.8 s, and one accepted
least-squares step reduced cost from 1.225319 to 0.928739 in another 40.8 s
with no failed trial or derivative fallback.  The recorded local SIMSOPT
1.10.7 development build and DESC 0.17.1 development build solve different
discretizations/objectives, so no fabricated wall-time ratio is attached to
their names.

The compatibility drivers
-------------------------

:func:`~vmex.core.optimize.least_squares` is a thin
``scipy.optimize.least_squares`` driver over the boundary Fourier degrees
of freedom (:func:`~vmex.core.optimize.pack_boundary` /
:func:`~vmex.core.optimize.unpack_boundary`; ``RBC(0,0)`` stays fixed),
taking weighted ``(callable, target, weight)`` terms:

.. code-block:: python

   result = opt.least_squares(
       [(qs, 0.0, 1.0),
        (opt.aspect_ratio, 6.0, 1.0),
        (opt.mean_iota, 0.42, 1.0)],
       inp, max_mode=5,
       jac="implicit",       # exact implicit-differentiation Jacobians
       use_ess=True)         # spectral trust-region scaling (below)
   result.input.to_indata("input.QA_optimized")

Repeated trial solves are cheap by construction: solver executables are
cached per structure, so only the first solve of a stage compiles, and every
trial is warm-started from the previous converged state (``hot_restart``,
sharpened by the perturbation seed below).  Failed trial solves return a
large finite residual — the trust region backs off instead of crashing.

Two gradient modes share the same term list.  ``jac=None`` uses scipy
``"2-point"`` finite differences — one full equilibrium solve per degree of
freedom per Jacobian, works with *every* objective.  ``jac="implicit"``
computes the exact residual Jacobian by implicit differentiation (one
amortized linear-algebra pass instead of ~2N solves) and requires traceable
terms and a fixed-boundary problem — the
compatibility table is in :doc:`objectives`.  ``current_dofs=k``
additionally frees the first ``k`` current-profile (``AC``) coefficients
plus ``CURTOR`` in either mode — the dof set of the self-consistent
bootstrap objective.

Bounded-storage profile optimization
------------------------------------

Vector profile objectives normally require their complete residual Jacobian
for a Gauss--Newton step.  At high Fourier resolution those radial
block-tridiagonal factors can dominate memory.  :func:`~vmex.core.optimize.minimize`
instead minimizes the identical scalar cost

.. math::

   \Phi(p) = \frac{1}{2}\sum_i r_i(p)^2

with scipy L-BFGS-B.  Reverse implicit differentiation evaluates
``grad(Phi)`` with one matrix-free adjoint, independent of the number of
profile samples, and does not form the dense residual Jacobian:

.. code-block:: python

   result = opt.minimize(
       [(opt.mercier_stability_residual, 0.0, 1.0),
        (opt.glasser_stability_residual, 0.0, 1.0),
        (opt.jdotb_residual, 0.0, 1e-6)],
       inp, max_mode=5,
       bounds=bounds,
       options={"maxiter": 100})

This is opt-in because it changes the step model from Gauss--Newton
trust-region to limited-memory quasi-Newton, although the objective and its
unconstrained minimizers are unchanged.  Existing
:func:`~vmex.core.optimize.least_squares` behavior is unaffected.  Plain
state hot restarts remain enabled; the perturbation warm start is unavailable
because it would require the forward state-response columns this path avoids.

Single-call ESS optimization (the recommended pattern)
------------------------------------------------------

The classic way to keep a shape optimization from tearing itself apart is
*staged continuation*: optimize at ``max_mode = 1``, then 2, … releasing
finer boundary harmonics only after the coarse shape has settled
(``max_mode=(1, 2, 3, 4, 5)`` runs that ladder automatically).  The
recommended pattern since R26 makes the ladder unnecessary: hand the
optimizer **all** the harmonics at once and let **Exponential Spectral
Scaling** (``use_ess=True``) impose the coarse-to-fine ordering through the
trust region itself.  Each dof's trust radius is scaled by

.. math::

   \mathrm{x\_scale}_i
   = \frac{e^{-\alpha\,\max(|m_i|,\,|n_i|)}}{e^{-\alpha}},

so high harmonics move on exponentially shorter leashes — the optimizer
explores the same hierarchy the ladder enforced, in a single
``least_squares`` call, with no stage boundaries for the objective to
stall at.

.. figure:: _static/figures/ess_x_scale.png
   :alt: ESS trust-region scale versus harmonic level for alpha 0.7 and 1.2
   :align: center
   :width: 78%

   The ESS trust-region weight per harmonic level
   (:math:`\alpha = 0.7` as in the bundled examples, :math:`\alpha = 1.2`
   the ``ess_alpha`` default).  At :math:`\alpha = 1.2` a ``max_mode``-6
   dof moves ~400x more cautiously than a ``max_mode``-1 dof.

Measured on a 36-core CPU from a near-circular torus seed (single call, all
harmonics released at once; scripts
``examples/optimization/QA_optimization_ess.py`` and
``QI_optimization_ess.py``):

.. list-table::
   :header-rows: 1
   :widths: 10 8 18 14 14 12 12 12

   * - class
     - nfp
     - residual
     - seed
     - achieved
     - max_mode
     - dofs
     - wall
   * - QA
     - 2
     - QS (1, 0)
     - 2.04e-01
     - **7.2e-06**
     - 5
     - 120
     - **14.5 min**
   * - QI
     - 1
     - omnigenity
     - 4.52e-01
     - **1.81e-02** (25x)
     - 6
     - 168
     - **17.3 min**

The staged ladder remains available (``max_mode=(1, ..., 5)``) and reaches
comparable precision — QA at QS 3.7e-7 in 25.5 min — but takes ~1.8x
longer for the same precision class.  Both patterns ship as side-by-side
example scripts so the two patterns can be compared directly.

Gradients (:mod:`vmex.core.implicit`)
-----------------------------------------

Derivatives through the equilibrium use **implicit differentiation** of the
converged fixed point: the solve is wrapped in ``jax.custom_vjp``; the
forward pass runs the fast (opaque) host solver, and the backward pass
solves the adjoint linear system matrix-free with preconditioned GMRES —
O(1) memory in the iteration count, no unrolling, no finite-difference
step-size to tune.  Coarse multigrid stages are just an initializer and are
stop-gradient by construction.  See the *Implicit differentiation* section
of :doc:`algorithms` for the formulation and cost analysis.

.. code-block:: python

   import jax
   from vmex.core import implicit
   from vmex.core.input import VmecInput

   inp = VmecInput.from_file("input.solovev")
   gpu = jax.devices("gpu")[0]
   from vmex.core.device import device_scope

   with device_scope(gpu):
       p0 = implicit.params_from_input(inp)
       sol = implicit.run(inp, p0)                    # ImplicitSolution pytree
       grad = jax.grad(lambda p: implicit.run(inp, p).wb)(p0)

Pass ``device="cpu"`` / ``"gpu"`` (or a ``jax.Device``) to
:func:`~vmex.core.implicit.params_from_input` or
:func:`~vmex.core.implicit.run` to select the implicit-gradient hardware
without environment variables. Omitted ``device`` and ``device=None`` leave
placement to JAX; ``device="auto"`` requests VMEX's CPU preference for this
launch-bound path. High-level optimization entry points use ``"auto"`` by
default. Hold :func:`~vmex.core.device.device_scope` around parameter creation
and differentiation when selecting a non-default accelerator.

:func:`~vmex.core.implicit.run` is the differentiable member of the
entry-point family (see *Choosing an entry point* in :doc:`quickstart`; the
non-differentiable Python default is
:func:`~vmex.core.optimize.solve_equilibrium`).  Besides the scalar
outputs, the returned solution carries the internally built evaluation
context as ``sol.runtime`` (a non-pytree convenience attribute), so custom
objectives can evaluate further ``(state, runtime)`` targets —
``opt.mean_iota(sol.state, sol.runtime)`` — without rebuilding the runtime
per evaluation.

Gradient accuracy is validated in CI against central finite differences for
fixed-boundary degrees of freedom — boundary Fourier coefficients,
``phiedge``, and profile parameters (``pres_scale``) — on a 2D (solovev)
and a 3D (li383) case, with agreement at the 1e-6 relative level (2D) and
at the finite-difference noise floor (3D)
(``tests/test_implicit_grad.py``).  For *solver-sensitive* metrics (iota,
mirror ratio, magnetic well, the QI residual) a naive re-solving finite
difference is **not** a valid reference — it perturbs the solver's discrete
convergence path, not just the fixed point — and the frozen-path FD
(:func:`~vmex.core.implicit.frozen_path_directional_fd`) must be used
instead; see *Gradient checking: solver-sensitive metrics and the frozen
path* in :doc:`algorithms`.

Implicit gradients are not merely faster than finite differences here —
on the flagship campaigns they are *necessary*: the exact-axisymmetric seed
is a saddle of the QS residual where finite differences stall, and on the
QP class the implicit path selects a better basin.

The high-level optimization APIs use ``adjoint_tol=1e-6`` and
``adjoint_maxiter=300`` by default.  The tolerance is checked against the true
linear residual, so VMEX never silently returns an unconverged adjoint.  The
larger restart ceiling fixes stiff QI objectives without charging easier
objectives for unused iterations; both controls remain ordinary keyword
arguments for stricter studies.

An optimizer's initial point is solved strictly before the run starts.  Later
invalid trial boundaries use the status-returning implicit lane: the host
callback returns a fixed-shape fallback and a status code, and the objective
selects a finite differentiable penalty pointing back toward the valid seed.
No Python exception crosses ``jax.pure_callback``, so rejected high-mode
trials do not produce an opaque JAX traceback.  Use
``problem.evaluate(x).status`` and its ``diagnostics`` mapping when a driver
needs the typed failure rather than only the numerical penalty.

The vector-Jacobian path normally uses the amortized block factorization.  If
its warm corrector is non-finite at a difficult accepted point, VMEX retries
the independent certified per-column GMRES lane before the optimizer sees the
Jacobian.  ``derivative_fallbacks`` and ``failed_trials`` are exposed on the
compatibility ``OptimizeResult`` and in ``problem.evaluate(x).diagnostics``;
the fast path and its defaults are unchanged.

The gradient stack: what makes a Jacobian cheap
-----------------------------------------------

A trust-region iteration spends its time in two places: the **Jacobian**
(one linear solve per dof) and the **trial solves** (one equilibrium per
proposed boundary).  The R25 work attacked both, and all of it is on by
default:

.. figure:: _static/figures/gradient_stack_speedup.png
   :alt: measured before/after of the three gradient-stack optimizations
   :align: center
   :width: 100%

   Measured on the nfp2 minimal-seed deck (Jacobian phase and trial
   iterations) and the full QA campaign (right); 2026-07-12, CPU.
   Reproduce the campaign numbers with the two
   ``examples/optimization/QA_optimization*.py`` scripts.

- **Direction-adaptive Jacobians** (``jac_solver="auto"``, default). A scalar
  residual uses one matrix-free reverse adjoint and never assembles dense
  angular blocks. Vector residuals use the block path below; ``"reverse"``
  and ``"block"`` remain explicit controls.
- **Block-tridiagonal Jacobian factorization** (``jac_solver="block"``).
  The raw force Jacobian ``dF/dz`` is *exactly*
  block-tridiagonal in radius (nearest-neighbor coupling; verified to
  1e-14), so ``ns`` dense ``(3mn, 3mn)`` blocks are assembled with
  3-colored ``jax.jvp`` probes — a cost independent of the dof count —
  factored once (``solvax.block_thomas``), and back-substituted for every
  dof right-hand side.  A warm-started GMRES corrector uses the configured
  budget and certifies each column to the same ``adjoint_tol`` as the old
  path.  Measured: the warm
  Jacobian phase drops 20.35 s to 0.61 s (**33x**); ``jac_solver="gmres"``
  (one preconditioned GMRES per dof) remains as a fallback.
  A true width-three nonlinear row kernel is parity-tested, but its direct
  streamed SOLVAX assembly is not selected: the high-resolution HSX resource
  gate was slower and used more peak memory than this three-color path.
- **Converged-state memo.**  scipy's trust-region drivers call ``jac(x)``
  at exactly the ``x`` that ``fun(x)`` just converged; a one-entry
  params-keyed memo removes that redundant solve per accepted iterate, and
  the final diagnostic ``result.equilibrium`` hot-seeds from the last trial
  state instead of re-solving cold.
- **Perturbation warm start** (``warm_start="perturbation"``, default).
  Each *trial* solve is seeded with the DESC-style first-order prediction
  :math:`z_{\mathrm{ref}} + \sum_j (dx)_j\, dz_j` — the per-dof state
  responses :math:`dz_j` are exactly the columns the Jacobian already
  computed, so the linearization is stashed for free at each ``jac(x)``
  call.  Measured: total forward-solve iterations drop **3.7x** (23,685 to
  6,364 over 20 trials), and the predicted seed rescued a trial whose plain
  hot restart hit a Jacobian-sign failure.  ``"state"`` (plain hot restart)
  and ``None`` are the fallbacks; all three converge to identical fixed
  points.
- **Memory** stays bounded by column chunking. ``jac_chunk_size="auto"`` caps
  SOLVAX's device-aware choice by the square-root width, so accelerator
  capacity cannot accidentally request one full probe ``vmap``. Chunked
  columns are identical to float64 round-off.
- **Krylov recycling** (``recycle=True``) carries a GCROT deflation space
  across the per-dof solves.  It is **off by default for a measured
  reason**: solvax v0.1's FIFO recycle space *slows* warm-started columns
  (1.7–3.4x more iterations on the benchmark operator).  The plumbing is
  exact and ready for a harmonic-Ritz GCRO-DR upgrade in SOLVAX; benchmark
  per-column iteration counts before enabling it.

The end-to-end effect on the profiled production ``opt_step`` and the CPU
versus GPU placement question are quantified in :doc:`performance`.

For implicit Jacobians, ``jac_chunk_size=None`` means one full-width
``lax.map`` batch rather than a bare ``vmap``.  This is a correctness
workaround for JAX 0.6.2: the bare transform produced an incorrect nested
iterative-solve column, while the full-width batch, smaller chunks, and
independent central finite differences agree.  The memory meaning remains
“unchunked”; ``"auto"`` or an integer supplies the research-grade bounded
path.

Free-boundary gradients (virtual casing)
----------------------------------------

External-field optimization on a specified plasma boundary is differentiable
through a different route
(:mod:`vmex.core.freeboundary_diff`): rather than differentiating the
NESTOR vacuum solve, the plasma-boundary contribution to the vacuum field
is computed by **virtual casing**, which is a smooth function of the coil /
``extcur`` parameters and the plasma surface.  Coil-parameter derivatives
of this boundary residual are finite-difference-validated.  They are not the
derivative of a fully reconverged NESTOR equilibrium.  (The two scopes are
complementary: the
fixed-boundary implicit adjoint is validated to ~1e-6 relative; the
virtual-casing residual path is FD-validated.)  See the
*Differentiable free boundary* section of :doc:`algorithms`.

True single-stage (plasma boundary **and** coils at once)
---------------------------------------------------------

The virtual-casing residual is differentiable not only in the coils but also
in the *plasma boundary*: :func:`~vmex.core.freeboundary_diff.surface_field_data_from_state`
rebuilds the virtual-casing surface field traceably straight from a live
equilibrium state (bit-exact against the wout path), so a single
``jax.value_and_grad`` of a combined objective

.. code-block:: text

    J(boundary, extcur) = w_bn * <(B_ext . n)^2>        # coil<->plasma consistency
                        + w_phys * physics(boundary)     # e.g. QS / iota / well

threads through the **implicit adjoint** (boundary dofs) *and* **virtual
casing** (coil dofs) simultaneously — the coil residual is evaluated on the
boundary the plasma solve just produced, and that coupling is differentiated
too.  Both blocks are finite-difference validated (coil dofs to ~1e-10;
boundary dofs to the FD floor of the re-solved equilibrium).

Making virtual casing differentiable in the boundary needs its adaptive
quadrature/patch precision frozen to static values:
:func:`~vmex.core.freeboundary_diff.plan_vc_precision` selects it once from
the starting (concrete) boundary and returns a ``PrecisionPlan`` to pass back
via ``precision=`` (the auto-selection otherwise concretizes traced surface
values).  ``examples/single_stage_simultaneous_opt.py`` runs the full loop:
from the bundled CTH-like free-boundary equilibrium it reshapes the boundary to
move the edge rotational transform while re-tuning the coils to keep
``<(B.n)^2>`` small — one exact gradient over both dof families.
``examples/single_stage_essos_coils_opt.py`` is the coil-agnostic variant of
the same loop, driving the coil half straight from an ESSOS
``essos.coils.Coils`` Biot–Savart callable (no mgrid file), and
``examples/single_stage_free_boundary_opt.py`` is the coil-only half in
isolation — perturb the confining coil currents of the bundled CTH-like case
and recover them by minimizing ``<(B.n)^2>`` with the exact virtual-casing
gradient (finite-difference-validated to ~1e-9).

Worked results
--------------

From a near-circular torus seed, ``jac="implicit"`` with ESS reaches
precise quasisymmetry and strong quasi-isodynamicity (measured on an office
CPU): QA (nfp 2) QS **7.2e-6** in one 14.5-minute call (the staged ladder
reaches 3.7e-7 in 25.5 min), QH (nfp 4) QS **5.83e-5**, QP (nfp 2) QS
9.4e-2 in the single-call budget (the hardest class — basin-limited;
the shipped deck reaches **3.3e-2** after an extended ladder plus warm-start
refinement), and QI (nfp 1)
omnigenity residual **1.81e-2**, 25x below the seed, in one 17.3-minute
call.  The complete scripts are in ``examples/optimization/``
(``QA``/``QH``/``QP``/``QI``, each with an ``_ess`` single-call variant
where measured).

.. figure:: _static/figures/readme_optimization.png
   :alt: QA/QH/QP seed vs optimized boundary, 3-D |B|, and Boozer |B| on the LCFS
   :align: center
   :width: 100%

   Quasisymmetry (QA/QH/QP): seed (grey) vs optimized (blue) boundary cross
   sections (top), the optimized LCFS in 3-D coloured by ``|B|`` (middle), and
   ``|B|`` in Boozer coordinates on the LCFS (jet line contours, bottom), whose
   contour geometry reads off the symmetry family. The label is the QS residual
   measured on the plotted equilibrium. Reproduce with
   ``benchmarks/make_readme_figures.py --only optimization`` from the decks in
   ``benchmarks/opt_decks/``.

.. figure:: _static/figures/readme_qi.png
   :alt: QI equilibria at nfp 1-4: boundary, 3-D |B|, and Boozer |B| on the LCFS
   :align: center
   :width: 100%

   Quasi-isodynamic (QI) equilibria at nfp 1/2/3/4 (bundled decks in
   ``examples/data/``): boundary cross sections, 3-D ``|B|`` geometry, and
   ``|B|`` in Boozer coordinates on the LCFS (jet). The label is the QI
   (omnigenity) residual, not QS. Reproduce with
   ``benchmarks/make_readme_figures.py --only qi``.
