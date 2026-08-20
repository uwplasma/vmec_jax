Optimization API
================

VMEX separates equilibrium physics and derivatives from the optimization
algorithm. :class:`vmex.core.problem.VmecProblem` contains a decision vector,
callables, metadata, and immutable input conversions; it does not own an
optimizer.

Problem construction
--------------------

Weighted objective tuples are the shortest interface:

.. code-block:: python

   problem = opt.VmecProblem.from_tuples(
       inp,
       [(qi, 0.0, 1.0),
        (opt.aspect_ratio, 5.0, 0.01),
        (iota_floor, 0.0, 10.0)],
       max_mode=5,
       use_ess=True,
   )

Each tuple is ``(function, target, weight)``. By default the row is
``sqrt(weight) * (function - target)``, so ``weight`` multiplies the squared
cost. Negative cost weights are rejected. Set
``weight_semantics="residual"`` only when importing a definition in which the
weight itself multiplies each residual row.

``weight`` may also be a one-dimensional array with one entry per residual
row. This is the general radial-weight interface: for a profile sampled on
``s``, an edge-emphasized cost can use, for example,
``weight = w0 * (1 + 9*s**4)``. The same mechanism applies to Mercier,
Glasser, magnetic-well, QS/QI, and user-defined vector objectives; no
objective-specific weighting class is needed.

Use :meth:`~vmex.core.problem.VmecProblem.from_loss` for one traceable scalar
``loss(equilibrium_state, solver_context)``. ``equilibrium_state`` contains
the solved VMEC spectral coefficients; ``solver_context`` contains its grids,
profiles, and transforms (it is not elapsed run time). Use
:meth:`~vmex.core.problem.FunctionProblem.from_functions` when the user already
has decision-vector-level functions and derivatives.

Callable contracts
------------------

.. list-table::
   :header-rows: 1

   * - Consumer
     - Value
     - Derivative
   * - ``scipy.optimize.least_squares``
     - ``problem.residual``
     - ``problem.residual_jac``
   * - ``scipy.optimize.minimize``
     - ``problem.fun``
     - ``problem.grad`` or ``problem.value_and_grad`` with ``jac=True``
   * - JAXopt / Optax
     - ``problem.jax_fun``
     - ``problem.jax_value_and_grad``
   * - SIMSOPT-style user code
     - ``problem.J``
     - ``problem.dJ``

For tuple problems, every scalar interface is defined by the certified
least-squares pair

.. math::

   \Phi(x) = \tfrac12 r(x)^T r(x), \qquad \nabla\Phi(x) = J(x)^T r(x).

The SciPy and JAX callables therefore return the same value and gradient.
VMEX maintains one exact-key host cache to avoid repeated work when an
optimizer requests the value and derivative separately.

``problem.dof_names`` is ordered exactly like ``problem.x0`` and every
optimizer vector passed to the problem. For example,
``dict(zip(problem.dof_names, result.x))`` labels an optimized SciPy result.
The older ``problem.names`` spelling remains available as the underlying
immutable tuple.

``RBC(0,0)`` is fixed by default because changing it mainly changes the major
radius. Pass ``vary_major_radius=True`` to release that coefficient explicitly;
VMEX does not add the identically-zero ``ZBS(0,0)`` direction.

Derivative methods
------------------

``derivative_method="implicit"`` is the default. It differentiates the
converged fixed point, requires traceable
``(equilibrium_state, solver_context)`` objectives, and
normally costs far less than one equilibrium solve per decision variable.
``implicit_jacobian_method="auto"`` uses a reverse adjoint for one residual row
and the block-tridiagonal forward response for a residual vector. Advanced
choices are ``"block_tridiagonal"``, ``"forward_gmres"``, and
``"reverse_adjoint"``.

``jacobian_batch_size=1`` minimizes cold compilation complexity and peak
memory for the usual QI/QS problems through ``max_mode=5``.
``jacobian_batch_size="auto"`` may improve warm throughput in long campaigns
that reuse one array shape. ``adjoint_tol`` and ``adjoint_maxiter`` control the
certified Krylov solves.

``derivative_method="finite_difference"`` accepts opaque host objectives. It
uses independent equilibrium probes and ``workers=None`` automatically uses
the CPUs available to the process. Select ``fd_method="2-point"`` or
``"3-point"`` and set ``workers=1`` for a serial reference.

Forward solves and FSQ certification
------------------------------------

The input's ``NS_ARRAY``, ``FTOL_ARRAY``, and ``NITER_ARRAY`` define the
multigrid solve. The same schedule is used by implicit and finite-difference
problems. ``forward_ftol`` and ``forward_max_iterations`` are concise
overrides for the final stage. :func:`vmex.core.optimize.solve_equilibrium`
accepts the same two names for one-off forward solves:

.. code-block:: python

   problem = opt.VmecProblem.from_tuples(
       inp, terms, max_mode=5,
       forward_ftol=1e-12,
       forward_max_iterations=5500,
       max_fsq_ratio=1e6,
   )

VMEC reports ``FSQ = fsqr + fsqz + fsql``. A converged trial is always
derivative-certified. If a trial exhausts its iteration budget, VMEX only
differentiates it when ``FSQ / forward_ftol <= max_fsq_ratio``; otherwise all
scalar interfaces return the same smooth rejection wall. The default
``1e6`` is deliberately tolerant of nearly converged optimization trials.
Reduce it for stricter studies after profiling the intended configurations.

Inspect the policy instead of guessing:

.. code-block:: python

   evaluation = problem.evaluate(x)
   print(evaluation.status, evaluation.diagnostics)

Diagnostics include ``fsq``, ``fsq_ratio``, ``max_fsq_ratio``,
``derivative_certified``, solve/iteration totals, rejected trials, and
derivative fallbacks. ``benchmarks/optimization.py`` profiles the QI, QA, QH,
QP, and scalar contracts over NFP 1--5 and accepts ``--max-fsq-ratio`` without
turning machine-specific results into a package default.

SciPy
-----

.. code-block:: python

   result = scipy.optimize.least_squares(
       problem.residual, problem.x0,
       jac=problem.residual_jac,
       x_scale=problem.scales,
       max_nfev=50,
       verbose=2,
   )

   result = scipy.optimize.minimize(
       problem.value_and_grad, problem.x0,
       jac=True, method="L-BFGS-B",
       bounds=problem.bounds,
       options={"maxiter": 100},
   )

``BFGS`` and ``L-BFGS-B`` use the same smooth rejected-trial scalar pair as
the least-squares-derived objective. Bounds and line-search options remain
ordinary SciPy choices. :class:`vmex.core.monitoring.OptimizationMonitor`
records accepted iterations without changing the objective. For a tuple
problem it also separates the weighted cost by term; saving and plotting the
history needs only the callback plus two output calls:

The SciPy examples optimize normalized variables ``u`` with
``x = x0 + step * u``. A bound of ``[-1, 1]`` therefore means one declared
parameter step, not a universal physical limit. The QA/QP boundary examples
use that conservative box; the QI and coil-shape examples use ``[-3, 3]``
because the narrower box changed the line-search path or pinned necessary coil
motion in representative scans. Keep ``PARAMETER_BOUND`` next to the other
driver inputs and validate it for a new normalization.

.. code-block:: python

   monitor = opt.OptimizationMonitor(problem, stream=None)
   result = scipy.optimize.least_squares(
       problem.residual, problem.x0, jac=problem.residual_jac,
       callback=monitor)
   monitor.save("objectives.csv")
   monitor.plot("objectives.png")

Continuation stages may restart the optimizer iteration counter; the monitor
keeps the combined saved history strictly increasing. Exact-zero terms are
drawn at a relative numerical display floor instead of forcing a meaningless
``1e-308`` axis. For joint surface/coil objectives, accepted vectors are also
available as ``monitor.x_history``. For a joint normalized surface/coil driver,
the optional movie call applies ``x = x0 + step*u`` and can color the surface::

   monitor.movie_surface_coils(
       "optimization.gif", objects_from_x, x0=x0, scales=step,
       surface_color="B.n/B", plasma_problem=problem,
       external_field=lambda objects: coil_field(objects[1]), max_frames=50)

Pass ``color_factory=`` to color the first surface by a scalar field at every
accepted iterate. The single-stage examples expose one top-level choice:
``None``, ``"absB"``, ``"B.n/B"``, or a user callable (for example a
bootstrap diagnostic). The movie uses one color scale across all frames.

For a custom JAX scalar objective, keep differentiation visible in the driver.
Return ``(cost, {name: term_cost})`` as auxiliary data, call
``jax.value_and_grad`` directly, and cache that already-computed evaluation for
the plotting callback:

.. code-block:: python

   value_and_grad = jax.value_and_grad(objective, has_aux=True)

   def scipy_value_and_grad(x):
       (value, terms), gradient = value_and_grad(jnp.asarray(x))
       monitor.cache_evaluation(x, value, gradient, terms)
       return float(value), np.asarray(gradient)

   result = scipy.optimize.minimize(
       scipy_value_and_grad, x0, jac=True, method="BFGS", callback=monitor)

The objective, its exact derivative, and SciPy's ``jac=True`` contract are all
explicit. Rejected line-search evaluations remain outside the accepted-iterate
history because only SciPy calls ``monitor`` as the callback. Large VMEC,
virtual-casing, and coil terms may likewise be differentiated separately and
their values and gradients added before ``cache_evaluation``; this produces
smaller XLA executables without changing the optimizer contract.

Joint VMEX--ESSOS objectives
----------------------------

ESSOS owns coil geometry and its variable convention. ``coils.dofs`` and
``coils.dof_names`` have identical ordering, while ``coils.with_dofs(x)``
constructs a traceable trial set without mutation. The same package provides
``surfacerzfourier_from_boundary(rbc, zbs, nfp, ...)``,
``loss_coil_separation``, ``loss_coil_surface_distance``, and
``Coils.from_simsopt``. VMEX therefore needs no duplicate coil indexing or
distance implementation.

For a finite-beta prescribed boundary, virtual casing gives the field of the
enclosed plasma currents. The physical exterior field is that contribution
plus the actual ESSOS coil field. The two interface residuals used by the
example are

.. math::

   \mathbf B_{\rm out}\!\cdot\mathbf n,
   \qquad
   \frac{|\mathbf B_{\rm out}|^2-|\mathbf B_{\rm in}|^2
   -2\mu_0p_{\rm edge}}{B_{\rm ref}^2}.

The second is a normalized total-pressure jump, not a pressure-profile error.
It supplies the tangential-field magnitude condition that ``B.n/B`` alone
does not constrain, even when the input pressure vanishes at the LCFS. The
fixed-boundary examples vary boundary and coil variables together; they do
not call NESTOR. A coil-only free-boundary optimization must instead
differentiate the fully reconverged NESTOR root. The experimental public path
keeps the construction visible in the driver::

   from vmex.core import implicit as im

   params = im.params_from_input(inp)
   config = vj.make_free_boundary_config(
       inp, BiotSavart(coils0), field_from_parameters=field_from_u)
   solver_context = im.runtime_from_params(params, config.implicit)

   def objective(u):
       equilibrium_state, status, _, _ = vj.solve_free_boundary_implicit_status(
           params, u, config)

       def accepted(_):
           residual = opt.residuals_from_tuples(
               equilibrium_state, solver_context, tuples)
           return 0.5 * jnp.vdot(residual, residual)

       # This visible wall lets a line search backtrack from an invalid trial.
       rejected = lambda _: 1e3 * (1 + jnp.linalg.norm(u))**2
       return jax.lax.cond(status == 0, accepted, rejected, None)

   value_and_grad = jax.value_and_grad(objective)
   result = scipy.optimize.minimize(
       value_and_grad, u0, jac=True, method="L-BFGS-B")

Here NESTOR moves the LCFS and only the ESSOS coil vector is optimized. See
``single_stage_free_boundary_optimization.py`` for coil geometry terms and
its finite-beta counterpart for beta and Redl bootstrap terms. The current
path is reverse-mode only. Status 0 is derivative-certified, 1 is a failed
solve, and 2 is an under-converged solve; only status 0 enters the adjoint.
The certified whole-state GCROT transpose remains the default.
``adjoint_solver="boundary_schur"`` selects the advanced boundary-Schur lane, which
eliminates the block-tridiagonal radial bulk and solves only the evolved-edge
correction before checking the full coupled residual. Local three-surface
rows and a pivoted sparse band solve reduce its cold cost substantially, but
local-row compilation still keeps it opt-in. ``device="auto"`` uses the CPU
for this response on accelerator hosts; an explicit ``device="gpu"`` or
process-wide JAX placement overrides that measured lower-memory default.

Use :class:`vmex.core.monitoring.EquilibriumReporter` for the compact physics
summary shared by the examples.  Each entry accepts either VMEX's
``function(equilibrium_state, solver_context)`` convention or a host
``function(equilibrium)``;
the call prints one line and returns the same values by label::

   report = opt.EquilibriumReporter(
       ("QS total", qs.total, ".6e"),
       ("aspect", opt.aspect_ratio, ".4f"),
       ("mean iota", opt.mean_iota, ".4f"))
   values = report("final", equilibrium)

JAXopt and Optax
----------------

Install ``vmex[optimizers]`` and pass the JAX pair directly:

.. code-block:: python

   solver = jaxopt.LBFGS(
       problem.jax_value_and_grad,
       value_and_grad=True,
       jit=False,
       maxiter=100,
   )
   result = solver.run(jnp.asarray(problem.x0))

   transform = optax.adam(1e-2)
   x, state = jnp.asarray(problem.x0), transform.init(problem.x0)
   for _ in range(100):
       value, gradient = problem.jax_value_and_grad(x)
       updates, state = transform.update(gradient, state, x)
       x = optax.apply_updates(x, updates)

The equilibrium uses a host callback, so an outer JAXopt solver should use
``jit=False``; VMEX still JIT-compiles the numerical kernels. See the three
``QI_optimization_{scipy,jaxopt,optax}.py`` examples, which share one problem
definition.

Resolution, continuation, and ESS
---------------------------------

Optimization scripts should show their numerical resolution explicitly:

.. code-block:: python

   mpol = max(max_mode + 2, minimum_mpol)
   inp = replace(inp, delt=0.5).change_resolution(
       mpol=mpol, ntor=mpol,
       ntheta=2 * mpol + 6,
       nzeta=2 * mpol + 4,
   )

``max_mode`` selects decision variables; ``mpol`` and ``ntor`` select the
equilibrium representation. They are related but not interchangeable.
Real-space grids must resolve the retained spectrum. Converge representative
results in radial and angular resolution rather than treating one formula as
a proof of adequacy.

``use_ess=True`` supplies exponential spectral scales to the optimizer. It
allows high modes to be present while low modes take larger steps. ESS is a
scaling policy, not a global optimizer: a mode ladder can reach a different
basin because every stage solves a different restricted problem. Carry a
stage forward with ``inp = problem.input_from_x(result.x)`` and construct the
next problem from that input.

``ess_alpha`` controls that separation explicitly. A mode of level
``k=max(|m|, |n|)`` is scaled by ``exp(-ess_alpha*(k-1))``. The default
``ess_alpha=1.2`` is a conservative starting point for crude seeds; values near
``0.7``--``0.9`` allow modes 3--5 to move more during basin exploration, while
larger values suppress them more strongly. There is no configuration-independent
best value: compare candidates at equal solve budgets and recheck the winner at
the final VMEC resolution. ``QA_optimization_global.py`` shows bounded basin
hopping with ``ess_alpha=0.7`` followed by exact least-squares polishing.

Hot restart and final output
----------------------------

Optimization trials hot-restart by default. The exact accepted state is
available without another cold solve:

.. code-block:: python

   inp = problem.input_from_x(result.x)
   equilibrium = problem.equilibrium_from_x(result.x)

   final_input = replace(
       inp,
       ns_array=np.array([101]),
       ftol_array=np.array([1e-14]),
       niter_array=np.array([8000]),
   )
   final_equilibrium = opt.solve_equilibrium(
       final_input,
       initial_state=equilibrium.state,
       verbose=True,
       raise_on_max_iterations=True,
   )
   final_input.to_indata("input.optimized")
   vj.write_wout("wout_optimized.nc", final_equilibrium.wout)
   vj.plot_wout("wout_optimized.nc", "figures")

``verbose=True`` shows whether the final run needs a larger iteration budget.
The hot seed is especially important for strongly shaped boundaries whose
cold magnetic-axis guess may be poor.

Pointwise fields and VJPs
-------------------------

Use :func:`~vmex.core.optimize.solve_equilibrium` for ordinary field queries.
When parameter VJPs are needed,
:meth:`~vmex.core.problem.VmecProblem.from_input` supplies the boundary/current
parameterization without inventing an optimization objective, and
``problem.equilibrium_from_x`` retains it. Set Cartesian or VMEC flux points
once and evaluate the field inside the LCFS:

.. code-block:: python

   final_equilibrium = problem.equilibrium_from_x(result.x)
   final_equilibrium.set_points_xyz([[x, y, z]])
   # Or: final_equilibrium.set_points_flux([[s, theta, phi]])

   B = final_equilibrium.B()
   absB = final_equilibrium.absB()
   gradB = final_equilibrium.gradB()
   gradgradB = final_equilibrium.gradgradB()
   gradgradgradB = final_equilibrium.gradgradgradB()

   dBdx = final_equilibrium.B_vjp(jnp.ones_like(B))
   dgradBdx = final_equilibrium.gradB_vjp(jnp.ones_like(gradB))
   d2Bdx = final_equilibrium.gradgradB_vjp(jnp.ones_like(gradgradB))
   d3Bdx = final_equilibrium.gradgradgradB_vjp(
       jnp.ones_like(gradgradgradB))

``B`` and all derivative components/axes are Cartesian for either point-input
route: ``gradB[..., i, j] = d B_i / d x_j``. VJPs return one value per
``problem.dof_names`` and include selected current parameters as well as
boundary modes; a flux-coordinate point is first mapped to Cartesian space,
and that physical point is held fixed during the parameter VJP. VMEX inverts
Cartesian points to ``(s, theta, phi)`` with a differentiable Newton solve,
evaluates angular dependence spectrally, and interpolates the radial mesh.
Points outside the LCFS return NaNs; use ``problem.exterior_field`` or
``equilibrium.exterior_field`` there. Equilibria returned by
``problem.equilibrium_from_x`` retain the same exterior-field VJPs. High
spatial derivative orders are
substantially more expensive and radial derivatives are piecewise smooth at
the VMEC mesh surfaces, so converge them in ``NS_ARRAY``.

Runnable vacuum and finite-beta examples, including Cartesian/cylindrical
queries, flux-coordinate inversion, derivatives through third order, and
parameter VJPs inside and outside the LCFS, are
``examples/vmex_get_B_gradB.py`` and
``examples/vmex_get_B_outside_plasma.py``. Both print all three Cartesian
spatial derivative orders and their VJPs; the exterior example returns VMEX
boundary modes followed by named ESSOS coil modes. The vacuum and finite-beta
``vmex_fieldline_tracing_*.py`` examples plot 3-D trajectories and toroidal
Poincare sections inside and just outside the LCFS. The seed radius is sampled
continuously across the LCFS. VMEX flux-coordinate traces use toroidal angle,
Cartesian coil/exterior traces use arclength, and a signed-distance event
terminates unbounded exterior trajectories.

To include coil parameters in an exterior-field VJP, pass the same functional
ESSOS update used by an optimization:

.. code-block:: python

   field = problem.exterior_field(
       result.x,
       external_parameters=coils.dofs,
       external_field_from_parameters=lambda x: coil_field(coils.with_dofs(x)),
       external_dof_names=coils.dof_names,
   )

``field.dof_names`` then lists VMEX variables followed by ESSOS variables.
The factored reverse pass differentiates the equilibrium and coil data once,
rather than nesting the implicit equilibrium solve inside each Cartesian
spatial derivative. Third spatial derivatives and their parameter VJPs remain
expensive; the examples print progress before each order and use compact grids
for a bounded cold run.

Resources and reproducibility
-----------------------------

A single equilibrium uses XLA threading. Parallel finite differences and
ensembles use process-available CPUs by default, respecting scheduler
affinity; set ``workers`` explicitly when sharing a node. Device
selection is controlled by ``device=`` and the policies in
:doc:`/howto/run-on-gpu`.

JAX compilation is structural. A new resolution or objective shape compiles
new executables; repeated equal-shape stages reuse them and the persistent
machine-local cache. Optional ``compile_residual_and_jacobian`` and
``compile_value_and_gradient`` calls merely make the first compilation
visible with elapsed-time heartbeats.

API summary
-----------

The main entry points are :func:`vmex.core.optimize.make_problem`,
:class:`vmex.core.problem.FunctionProblem`,
:class:`vmex.core.problem.VmecProblem`,
:class:`vmex.core.problem.Evaluation`, and
:class:`vmex.core.monitoring.OptimizationMonitor`, and
:class:`vmex.core.monitoring.EquilibriumReporter`.
