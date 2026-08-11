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

Use :meth:`~vmex.core.problem.VmecProblem.from_loss` for one traceable scalar
``loss(state, runtime)``. Use
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

Derivative methods
------------------

``derivative_method="implicit"`` is the default. It differentiates the
converged fixed point, requires traceable ``(state, runtime)`` objectives, and
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
records accepted iterations without changing the objective.

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

Resources and reproducibility
-----------------------------

A single equilibrium uses XLA threading. Parallel finite differences and
ensembles use process-available CPUs by default, respecting scheduler and
container affinity; set ``workers`` explicitly when sharing a node. Device
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
:class:`vmex.core.monitoring.OptimizationMonitor`.
