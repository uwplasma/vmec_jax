# Optimize a boundary

This lesson builds an optimizer-neutral VMEC problem and passes its residual
and exact Jacobian directly to SciPy. The optimization reshapes a circular
tokamak toward aspect ratio 4.

## Choose the VMEC and boundary resolutions

```python
from dataclasses import replace
import scipy.optimize

import vmex as vj
from vmex import optimize as opt

inp = vj.VmecInput.from_file("input.circular_tokamak")
max_mode = 1
mpol = max(max_mode + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
```

The script explicitly owns `DELT`, `MPOL`, `NTOR`, and the real-space grids.
`max_mode` separately selects which boundary coefficients the optimizer may
change. `RBC(0,0)` remains fixed by default; pass
`vary_major_radius=True` to release it without also introducing the null
`ZBS(0,0)` direction. `problem.dof_names` shows the exact resulting order.

## Build and solve the problem

```python
terms = [(opt.aspect_ratio, 4.0, 1.0)]  # function, target, cost weight
problem = opt.VmecProblem.from_tuples(
    inp, terms, max_mode=max_mode, use_ess=True)

result = scipy.optimize.least_squares(
    problem.residual, problem.x0, jac=problem.residual_jac,
    x_scale=problem.scales, max_nfev=20, verbose=2)
```

Each tuple contributes `weight * (function - target)**2 / 2` to the cost.
The first residual/Jacobian call may compile JAX executables; later calls with
the same array structure reuse them. Failed trial equilibria receive a finite,
consistent rejection residual so the trust region can shorten its step.

## Inspect and save the result

```python
optimized_input = problem.input_from_x(result.x)
equilibrium = problem.equilibrium_from_x(result.x)
equilibrium_state, solver_context = equilibrium.state, equilibrium.runtime
aspect = float(opt.aspect_ratio(equilibrium_state, solver_context))
print(f"final cost = {result.cost:.6e}, aspect = {aspect:.6f}")

optimized_input.to_indata("input.aspect_optimized")
vj.write_wout("wout_aspect_optimized.nc", equilibrium.wout)
```

`problem.value_and_grad` fits SciPy BFGS/L-BFGS-B, while
`problem.jax_value_and_grad` fits JAXopt and Optax. Add QI, QS, mirror,
elongation, iota, or stability tuples without changing the optimizer wiring;
the complete pattern is in {doc}`/howto/optimize-a-boundary`.
