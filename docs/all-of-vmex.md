# All of VMEX

One page, the whole mental model: an input deck goes in, a solver iterates
Fourier moments to force balance, a `wout_*.nc` file comes out — and because
the solve is a JAX fixed point, you can differentiate through it and optimize
the boundary. Everything else in these docs is detail on one of those steps.

## The pipeline: input → solve → wout

```{list-table}
:header-rows: 1
:widths: 18 40 42

* - stage
  - object
  - notes
* - input
  - {class}`~vmex.core.input.VmecInput`
  - frozen dataclass with VMEC2000 semantics; reads `&INDATA` namelists and
    structured JSON ({doc}`reference/input-file`), builds from keyword
    arguments, round-trips to either format
* - solve
  - {func}`~vmex.core.multigrid.solve_multigrid` →
    {class}`~vmex.core.solver.SolveResult`
  - the `NS_ARRAY` multigrid ladder over the VMEC2000 iteration
    ({doc}`explanation/iteration`); free boundary via
    {func}`~vmex.core.multigrid.solve_free_boundary_multigrid`
* - output
  - {func}`~vmex.core.wout.wout_from_state` /
    {func}`~vmex.core.wout.write_wout` → `wout_*.nc`
  - the VMEC2000 variable set ({doc}`reference/wout-file`); loads unchanged
    in simsopt and booz_xform
```

The `vmex` command wraps the pipeline end to end and adds plotting, the
Boozer transform, scaling, and diagnostics ({doc}`reference/cli`).

## Choosing an entry point

Four solve entry points share the same numerics; pick by what you need back:

```{list-table}
:header-rows: 1
:widths: 30 24 46

* - entry point
  - returns
  - use when
* - {func}`vmex.core.optimize.solve_equilibrium`
  - `Equilibrium` (spectral coefficients + solver context + lazy `.wout`)
  - **Default for Python work**: analysis, objectives, anything that reads
    wout tables or the `(equilibrium_state, solver_context)` scalar targets
* - {func}`vmex.core.multigrid.solve_multigrid`
  - `SolveResult` (state + convergence data)
  - you only need the converged state / iteration diagnostics — the engine
    behind the CLI and `solve_equilibrium`
* - {func}`vmex.core.implicit.run`
  - `ImplicitSolution` (differentiable pytree, carries `.runtime`)
  - gradients: wrap it in `jax.grad`/`jax.value_and_grad` — the
    implicit-adjoint path of {doc}`explanation/adjoint-gradients`
* - {func}`vmex.core.solver.solve`
  - `SolveResult` (one grid stage)
  - low-level single-`ns` building block (no `NS_ARRAY` ladder); mainly for
    solver development and tests
```

## The two lanes

The same jitted physics runs through two lanes (`vmex --mode cli|jit`): the
default **CLI lane** — a Python loop over a jitted N-iteration block with
host residual checks, exact-`ftol` early exit, and live VMEC2000-format
printing — and the **JIT lane**, one `lax.while_loop`, fully traceable, the
forward solver inside the differentiable API. A regression test pins the two
lanes to machine-precision agreement ({doc}`explanation/iteration`). Device
placement (CPU vs GPU) follows the measured policy of
{mod}`vmex.core.device` ({doc}`howto/run-on-gpu`).

## The multigrid ladder

`NS_ARRAY = 5 17 51` solves at ns=5, interpolates to 17, re-solves, then 51 —
VMEC2000's exact `interp.f` transfer ({doc}`explanation/multigrid`). The same
seam gives hot restart: seed any solve from a previous state
(`initial_state=`) or from any wout file (`restart_from=` / `--restart`),
skipping rungs the seed already covers
({doc}`howto/restart-from-previous-run`).

## What is differentiable

Fixed-boundary equilibria are differentiable in boundary Fourier
coefficients, profiles, `phiedge`, `pres_scale`, and `curtor` through the
implicit function theorem on the converged fixed point — validated against
finite differences at 2e-9 relative on the bundled Solovev case
(`examples/take_gradients.py`). Coil/`extcur` derivatives on a specified
boundary go through the virtual-casing residual; the NESTOR free-boundary
fixed point itself is not differentiated. The per-configuration contract is
{doc}`reference/capabilities`; the machinery is
{doc}`explanation/adjoint-gradients`.

## Where objectives plug in

Objectives are plain functions of `(equilibrium_state, solver_context)` —
the solved spectral coefficients and the grids/profile data used to evaluate
them — including quasisymmetry,
omnigenity/QI, aspect ratio, iota, Mercier, bootstrap, turbulence proxies
({doc}`reference/objectives`). The driver
{func}`vmex.core.optimize.least_squares` takes simsopt-style
`(function, target, weight)` terms over the boundary dofs with exact implicit
Jacobians (`jac="implicit"`); {doc}`howto/optimize-a-boundary` is the
campaign recipe.

## A worked 20-line script

```python
import jax
import numpy as np
import vmex as vj
from vmex.core import implicit
from vmex.core import optimize as opt

# 1. input: read a deck (or build VmecInput(**fields) from scratch)
inp = vj.VmecInput.from_file("input.circular_tokamak")

# 2. solve: full multigrid ladder, VMEC2000-format iteration printout
eq = opt.solve_equilibrium(inp)
print("converged:", eq.result.converged, "aspect:", float(eq.wout.aspect))

# 3. output: standard wout, plus figures
vj.write_wout("wout_circular_tokamak.nc", eq.wout)
vj.plot_wout("wout_circular_tokamak.nc", outdir=".")

# 4. differentiate: exact d(wb)/d(params) via the implicit adjoint
p0 = implicit.params_from_input(inp)
grad = jax.grad(lambda p: implicit.run(inp, p).wb)(p0)

# 5. optimize: reshape the boundary toward aspect ratio 4
result = opt.least_squares([(opt.aspect_ratio, 4.0, 1.0)],
                           inp, max_mode=1, jac="implicit")
```

Each numbered step has a tutorial: {doc}`tutorials/first-equilibrium`,
{doc}`tutorials/plots-and-boozer`, {doc}`tutorials/first-gradient`,
{doc}`tutorials/first-optimization`.

## The other lane: mirrors

`vmex.mirror` is a separate spline-native lane for open-mirror and
stellarator-mirror-hybrid equilibria (`mout_*.nc` output, its own implicit
adjoint): {doc}`howto/mirror-machines` and
{doc}`explanation/mirror-geometry`.
