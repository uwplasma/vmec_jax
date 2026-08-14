# Set up a boundary optimization

Use a visible numerical setup, ordinary objective tuples, and the external optimizer that fits the problem. The complete API is {doc}`/reference/optimization`; objective choices are {doc}`/reference/objectives`.

## Build the problem

```python
from dataclasses import replace
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity import QIResidual

inp = vj.VmecInput.from_file("examples/data/input.minimal_seed_nfp2")
max_mode = 5
mpol = max(max_mode + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

qi = QIResidual(np.linspace(0.1, 1.0, 6))

def iota_floor(state, runtime):
    return jnp.maximum(0.33 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

def elongation_excess(state, runtime):
    return jnp.maximum(opt.max_elongation(state, runtime) - 8.0, 0.0)

terms = [
    (qi, 0.0, 1.0),
    (opt.aspect_ratio, 5.0, 0.005),
    (iota_floor, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0),
]
problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=max_mode, use_ess=True)
```

The script—not a preparation helper—chooses `DELT`, `MPOL`, `NTOR`, `NTHETA`, and `NZETA`. `max_mode` controls the free boundary coefficients; equilibrium resolution must be converged independently.

## Choose an optimizer

For residual objectives, SciPy least squares supplies a trust region and a useful iteration table:

```python
result = scipy.optimize.least_squares(
    problem.residual, problem.x0, jac=problem.residual_jac,
    x_scale=problem.scales, max_nfev=50, verbose=2)
```

For BFGS or L-BFGS-B, use the scalar pair:

```python
result = scipy.optimize.minimize(
    problem.value_and_grad, problem.x0, jac=True, method="L-BFGS-B",
    bounds=problem.bounds, options={"maxiter": 100})
```

The same problem exposes `jax_value_and_grad` for JAXopt and Optax. See `examples/optimization/QI_optimization_scipy.py`, `QI_optimization_jaxopt.py`, and `QI_optimization_optax.py`.

## Continue in mode number when useful

QI and QS landscapes are non-convex. ESS improves scaling when all modes are released together, but it does not reproduce the basin selection of a mode ladder. A transparent continuation loop is:

```python
for max_mode, max_nfev in zip([1, 3, 5], [20, 30, 50]):
    mpol = max(max_mode + 2, 5)
    inp = inp.change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=max_mode, use_ess=True)
    result = scipy.optimize.least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=max_nfev, verbose=2)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    inp.to_indata(f"input.QI_max_mode_{max_mode:03d}")
```

A short QP stage can select a poloidally closed-|B| basin before a constructed-QI stage; `QI_optimization.py` shows that workflow. Treat it as a basin strategy, not a universal guarantee.

## Control convergence explicitly

The input schedule applies to both implicit and finite-difference derivatives. For concise overrides use:

```python
problem = opt.VmecProblem.from_tuples(
    inp, terms, max_mode=5,
    forward_ftol=1e-12,
    forward_max_iterations=5500,
    max_fsq_ratio=1e6)
```

`max_fsq_ratio` is the largest iteration-limited `FSQ / ftol` that VMEX will differentiate. Inspect `problem.evaluate(x).diagnostics` and calibrate stricter values on the intended NFP, objective family, and resolution. `benchmarks/optimization.py` provides a reproducible profiler for QI, QA, QH, QP, SciPy/JAX agreement, and finite differences.

## Refine and save the result

```python
final_input = replace(inp,
    ns_array=np.array([101]), ftol_array=np.array([1e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state,
    verbose=True, raise_on_max_iterations=True)

final_input.to_indata("input.QI_optimized")
vj.write_wout("wout_QI_optimized.nc", final_equilibrium.wout)
vj.plot_wout("wout_QI_optimized.nc", "figures")
```

The accepted optimization state hot-starts the high-resolution solve and avoids rebuilding a poor cold magnetic axis. `verbose=True` prints force residuals and the iteration count so the final budget can be increased deliberately.

## Shared and HPC machines

Leave `workers=None` to use the CPUs visible to the process; VMEX respects scheduler and container affinity. Set an explicit smaller value when sharing a node. One equilibrium already uses XLA threading, while `parallel.solve_ensemble` and finite-difference probes parallelize independent solves. Select accelerators with `device=` and follow {doc}`run-on-gpu`.
