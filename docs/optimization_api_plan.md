# Optimizer-agnostic optimization API

Status: approved implementation plan, 2026-08-08

## Purpose

VMEX should own the differentiable equilibrium problem, not the optimization
algorithm.  A user should be able to assemble physics residuals or a scalar
loss once, obtain accurate derivatives, and pass the resulting callables to
SciPy, JAXopt, Optax, or a custom optimizer without changing VMEX.

The design follows three rules:

1. The optimization variables and equilibrium solve are explicit.
2. Objective composition and differentiation are independent of the optimizer.
3. Convenience solvers are thin adapters over the same public callables.

The resulting source must remain sober and easy to audit.  New abstractions are
accepted only when they remove duplicated work or expose a capability that is
otherwise private.  There will be no optimizer registry, plugin framework, or
central method-name dispatch table.

## Public contract

The central object is an immutable problem description with an explicit
decision vector.  It exposes the names used by SciPy and JAX rather than hiding
them inside a solver driver:

```python
from dataclasses import replace

max_mode = 5
mpol = max(max_mode + 2, 5)
ntor = mpol
ntheta = 2 * mpol + 6
nzeta = 2 * ntor + 4
inp = replace(inp, delt=0.5)
inp = inp.change_resolution(
    mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nzeta
)
problem = opt.VmecProblem.from_tuples(
    inp,
    [(qi, 0.0, 1.0),
     (opt.aspect_ratio, 6.0, 0.1),
     (iota_shortfall, 0.0, 10.0)],
    max_mode=max_mode,
    progress=True,
)

problem.x0
problem.names
problem.bounds
problem.scales

problem.fun(x)                 # scalar host value
problem.grad(x)                # scalar host gradient
problem.value_and_grad(x)      # (value, gradient)
problem.residual(x)            # residual vector
problem.residual_jac(x)        # residual Jacobian
problem.residual_and_jac(x)    # (residual, Jacobian)

problem.jax_fun(x)             # scalar JAX value
problem.jax_value_and_grad(x)  # JAX (value, gradient)
problem.jax_residual(x)        # JAX residual vector

problem.input_from_x(x)
problem.x_from_input(inp)      # inverse mapping for continuation
problem.equilibrium_from_x(x) # reuse the accepted converged state
problem.evaluate(x)            # value, residual, status, diagnostics
problem.compile_residual_and_jacobian()  # optional visible first compilation
```

`J(x)` and `dJ(x)` are concise aliases for users familiar with SIMSOPT.  The
decision vector is nevertheless always an argument; a mutable `problem.x`
would make concurrent evaluation and cache correctness needlessly difficult.

Two constructors cover the common extension points:

- `VmecProblem.from_tuples(...)` composes residual terms.
- `VmecProblem.from_loss(...)` accepts a pure `(state, runtime) -> scalar`
  function, with an optional supplied `value_and_grad` implementation.

`FunctionProblem.from_functions(...)` is a small optimizer-neutral container
for users who already have x-level callables.  It performs no equilibrium or
optimizer work.

Tuple weights use ``weight_semantics="cost"`` by default:

```text
residual_i = sqrt(weight_i) * (value_i - target_i)
cost       = 0.5 * sum_i weight_i * (value_i - target_i)^2
```

The compatibility `least_squares()` and `minimize()` wrappers retain their
existing residual-scale convention.  The distinction is explicit in their
docstrings and tests; published VMEX inputs must not change meaning silently.

The other beginner defaults are ``derivative_method="implicit"`` (exact
derivatives of the converged equilibrium),
``implicit_jacobian_method="auto"`` (select the exact scalar/vector assembly
from objective shape), and ``jacobian_batch_size=1`` (short cold compilation
for the usual QI/QS ``max_mode <= 5`` problem).  Documentation, rather than
ordinary scripts, lists the advanced alternatives.  ``"auto"`` batching is
reserved for long same-shape campaigns that amortize its larger first
compilation.

## Internal boundaries

The implementation separates five responsibilities:

1. **Parameterization** owns `x0`, names, bounds, scales, and conversions to a
   `VmecInput` or implicit-parameter pytree.
2. **Evaluation** solves the equilibrium once for a particular `x`, records a
   typed success/failure status, and memoizes only exact-key repeat calls.
3. **Objective composition** maps a solved state/runtime to residuals or a
   scalar loss.
4. **Derivative providers** produce residual Jacobians or scalar gradients.
5. **Optimizer adapters and examples** pass these public callables to external
   libraries and contain no equilibrium-specific derivative logic.

The public data containers live in `vmex/core/problem.py`.  Physics objectives
remain in their present modules.  Existing implicit differentiation machinery
is extracted behind the public factory rather than copied.  Source movement is
kept mechanical and separately committed when it helps review.

## Derivative policy

### Built-in exact derivatives

For a converged fixed point `F(z, p) = 0`, VMEX differentiates the fixed point,
not the iterations used to reach it.  A scalar objective uses one reverse
implicit adjoint.  A vector residual uses a block/multi-right-hand-side forward
linearization when that is cheaper.  Direction choice is based on problem
dimensions, never on the selected optimizer.

The exact lane covers, and must remain covered by tests for:

- quasisymmetry residuals;
- constructed and J-invariant quasi-isodynamic residuals;
- maximum-J residuals;
- smooth `L_grad_B`;
- `DMerc` and its smooth stability-margin residual;
- `jdotb`, Mercier shear, and Glasser `D_R`;
- Redl bootstrap mismatch;
- differentiable ballooning and turbulence objectives where JAX provides a
  mathematically defined derivative.

Host WOUT/Boozer functions remain reporting oracles.  Optimization uses their
traceable state/runtime counterparts.  Nonsmooth reductions such as a hard
minimum are reported as such; their smooth variants are the default for
gradient-based optimization.

### User derivatives and finite differences

Users may provide complete `value_and_grad` or `residual_and_jac` callables.
Traceable user objectives may instead rely on VMEX implicit differentiation.
Opaque host objectives use a finite-difference provider with configurable
scheme, step, and workers.  A hybrid problem may concatenate exact, supplied,
and finite-difference row blocks without changing the optimizer-facing API.

### Failure contract

Exceptions must not cross `jax.pure_callback`.  A host equilibrium callback
returns a fixed-shape state, diagnostics, and a success code.  Failed trials
select a finite differentiable penalty branch.  Host adapters expose the same
condition through `Evaluation.status` and never substitute a stale derivative
without saying so.

The initial point remains strict: an invalid seed raises a concise VMEX error.
Once a valid residual size and derivative shape are known, invalid trial points
are rejected with a finite penalty and recorded failure diagnostics.

## External optimizers

The same problem object is used directly:

```python
scipy.optimize.least_squares(
    problem.residual, problem.x0, jac=problem.residual_jac,
    bounds=problem.bounds,
)

scipy.optimize.minimize(
    problem.value_and_grad, problem.x0, jac=True,
    method="L-BFGS-B", bounds=problem.bounds,
)

jaxopt.LBFGS(
    problem.jax_value_and_grad, value_and_grad=True,
).run(problem.x0)

jaxopt.LevenbergMarquardt(
    problem.jax_residual,
    jac_fun=problem.jax_residual_jac,
).run(problem.x0)
```

Optax remains deliberately explicit: the example creates any desired gradient
transformation and applies updates from `problem.jax_value_and_grad`.  VMEX
does not reimplement Adam or restrict the transformations a user may compose.

Existing `least_squares()` and `minimize()` functions become thin compatibility
adapters.  Any future convenience adapter must accept a problem object and
contain no objective, equilibrium, or derivative implementation.

## Parallel and performance policy

`parallel="auto"` selects work appropriate to the derivative:

- scalar exact gradients use one reverse adjoint;
- vector exact Jacobians use device-aware multi-RHS batches;
- finite differences use worker-local evaluators and multiple host workers;
- ensembles and multistart campaigns use `evaluate_many`;
- real multi-device runs may use `shard_map`;
- a single CPU device uses host concurrency, not artificial `pmap` devices.

Advanced users can override workers, batch size, derivative direction, device
mesh, Krylov tolerance, iteration budgets, and cache policy with ordinary
arguments.  Defaults avoid CPU oversubscription between JAX, BLAS, and worker
pools.

Performance is compared with local VMEX, SIMSOPT, and DESC installations using
the same input, variables, objective values, derivative accuracy, tolerances,
and warm/cold cache state.  Claims are made only from recorded reproducible
benchmarks.  The intended advantages are fewer equilibrium solves than finite
differences, one adjoint for scalar loss, batched linear solves for residual
Jacobians, and exact reuse of a primal solve between value and derivative.

No CI test uses a wall-clock threshold.  Regression gates count equilibrium
solves, compilations, objective evaluations, and Krylov work.  Benchmark JSON
records timing separately.

## Monitoring and logging

Problem evaluation is silent.  An optional monitor reports accepted optimizer
iterations rather than every trial evaluation:

```text
iter  cost        reduction   optimality  eq solves  rejected
0     1.2253e+00  -           2.31e+00    1          0
1     4.8172e-01  7.44e-01    8.02e-01    3          1
```

Higher-cost trial evaluations are valid in trust-region and line-search
methods, and Adam is not monotonic.  Tests therefore validate derivatives and
accepted progress rather than requiring every function evaluation to decrease.

The repeated PjRt executable-version cache warning is suppressed through the
documented JAX logging configuration, with an explicit user override.  VMEX
warnings and failed-trial diagnostics remain visible.

## Test strategy

Tests are divided by what they prove.

### API and composition tests

- scalar, vector, tuple, and supplied-derivative construction;
- shapes, dtypes, aliases, names, bounds, and scales;
- exact cost-weight tuple semantics;
- compatibility-wrapper residual-weight behavior;
- value/gradient and residual/Jacobian consistency;
- explicit input-resolution changes preserve existing Fourier coefficients;
- input/vector round trips and accepted-equilibrium reuse preserve continuation
  state for a high-resolution final solve;
- each named compilation method selects only its stated derivative pair,
  populates the exact-key cache, and emits concise progress records;
- a repeated `fun(x)`/`grad(x)` pair reuses one equilibrium evaluation;
- the same unchanged problem is accepted by SciPy, JAXopt, and Optax;
- optional backends use `pytest.importorskip` and remain optional dependencies.

Most interface tests use an analytic quadratic or Rosenbrock evaluator and run
without a VMEC solve.

### Physics-value tests

- QS state residual agrees with the host/WOUT lane and symmetry expectations;
- QI state residual agrees with the Boozer reference and known physical
  ordering;
- `L_grad_B_state` agrees with `l_grad_b` on multiple equilibria and grids;
- `d_merc_state` agrees with WOUT `DMerc` on validated surfaces;
- `D_R` agrees with its published relation and independent DCON samples;
- bootstrap and other residual classes retain existing independent oracles.
- boundary elongation is grid-converged, JAX-traceable for symmetric and
  asymmetric states, and manually cross-checked against the existing SIMSOPT
  ``MaxElongationPen`` on the optimized precise-QI nfp=2 result.

### Derivative tests

- JVP/VJP duality at the state and full equilibrium levels;
- Taylor remainder convergence away from nondifferentiable ties;
- central frozen-path finite differences of the converged fixed point;
- a tighter re-solved finite difference as a secondary end-to-end check;
- scalar `grad == residual_jac.T @ residual` for a least-squares objective;
- batch-size and forward/reverse direction invariance to solver tolerance;
- finite values and certified residuals for every returned derivative;
- QI gets a full equilibrium-to-boundary scalar-adjoint test, not only a
  state-gradient smoke test.

Tolerances are determined by step-size convergence and the equilibrium solve
floor for each objective.  A single arbitrary relative threshold is not used
for quantities near zero.

### Failure and output tests

- invalid high-mode trial returns status and finite penalty;
- invalid initial seed raises a concise typed error;
- no exception is raised inside a pure callback;
- rejected trials do not overwrite a valid cache entry;
- accepted-iteration monitor output has stable columns and reductions;
- persistent-cache execution emits no PjRt version-compatibility warning;
- `verbose=0` produces no optimization output.

### End-to-end acceptance

Using the precise-QI nfp=2 workflow and its near-circular seed:

1. `max_mode=1`: the complete QI tuple objective passes a directional Taylor
   test and all supported optimizers reduce the same objective in a small
   budget.
2. `max_mode=5`: invalid boundaries are rejected without a
   `VmecJacobianError` traceback and at least one accepted improvement is
   obtained under the documented continuation/scaling policy.
3. Direct SciPy least-squares, SciPy BFGS/L-BFGS-B, JAXopt LBFGS/LM, and Optax
   Adam examples require no VMEX source changes.
4. Accepted-iteration output is concise and the `W0807` lines are absent.
5. Results and Jacobians are invariant to automatic versus fixed batching
   within the certified tolerance.

The full precise-QI acceptance script is a manual/benchmark workflow, not part
of every pull-request CI shard.

### Keeping CI time flat

- Reuse one module-scoped low-resolution equilibrium and its linearization.
- Parameterize several objective checks over that fixture.
- Replace redundant real-solve smokes instead of only adding new ones.
- Use analytic fake evaluators for backend contracts.
- Keep production-resolution campaigns under existing `full`/`weekly` marks.
- Inspect `pytest --durations` and move no new long test into the default lane
  without removing equivalent redundant work.

## Documentation requirements

The README optimization section will show the direct SciPy contract and link to
the detailed guide.  `docs/optimization.rst` will explain the problem object,
derivative choices, failures, caching, parallel policy, and monitoring.
`docs/objectives.rst` will retain a generated/audited differentiation matrix and
show custom objectives.  Each external backend gets a runnable example and a
short dependency note.  Public classes and methods receive concise docstrings
with shapes, units, purity, exceptions, and cache behavior.

Documentation examples are smoke-tested so API drift fails CI.

## Pull-request sequence

1. **Expose optimizer-agnostic VMEC problem callables.**  Add this plan, the
   public containers/factory, tuple composition, compatibility adapters,
   quiet-by-default JAX logging, and fast API tests.  Include the status-safe
   callback because a public optimization problem must reject invalid trial
   boundaries without emitting a ``pure_callback`` traceback.
2. **Certify implicit derivatives and failure diagnostics.**  Harden QI
   adjoints, expose certified linear-solve budgets, complete the objective
   derivative matrix, and consolidate numerical and diagnostic tests.
3. **Add backend-neutral monitoring and runtime diagnostics.**  Separate trials
   from accepted iterations, harden derivative fallbacks, and document status.
4. **Add SciPy, JAXopt, and Optax examples and automatic parallel evaluation.**
   Add examples, optional integration tests, worker policies, precise-QI
   acceptance scripts, and reproducible comparison benchmarks.

Each PR is independently reviewable, contains its tests and documentation, and
uses no speculative abstraction needed only by a later PR.

## Definition of done

- No optimizer implementation owns private access to a VMEX derivative.
- All optimizer-facing callables are public, documented, and directly tested.
- Built-in differentiable physics objectives have value and derivative gates.
- Failed high-mode trials are recoverable and free of callback tracebacks.
- Default output is quiet; optional progress is iteration-oriented.
- External SciPy, JAXopt, and Optax examples run from the same problem object.
- Parallel defaults are useful on a workstation and configurable by experts.
- Default CI time does not materially increase.
- Reproducible benchmarks substantiate performance claims against the local
  SIMSOPT and DESC installations.
