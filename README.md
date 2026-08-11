# VMEX

VMEX is a JAX implementation of VMEC for stellarator and tokamak ideal-MHD equilibria. It reads standard VMEC input files, solves fixed- and free-boundary problems, writes standard `wout_*.nc` files, and provides exact implicit derivatives of converged fixed-boundary equilibria for optimization.

![VMEX equilibria and diagnostics](docs/_static/figures/readme_equilibrium_showcase.png)

## Install

```console
pip install vmex
vmex --doctor
vmex --test
```

Python 3.10+ is supported. VMEX installs CPU JAX, SciPy, plotting, NetCDF, and `booz_xform_jax`; install an accelerator-enabled JAX wheel separately using the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). Optional integrations are `vmex[optimizers]` for JAXopt/Optax, `vmex[freeb]` for differentiable virtual casing, `vmex[coils]` for ESSOS, and `vmex[turbulence]` for GKX.

An editable source install remains connected to its checkout, so `pip install -e .` only needs to be repeated when packaging metadata or dependencies change—not after each `git fetch` or checkout.

## Solve and inspect an equilibrium

```python
import vmex as vj

inp = vj.VmecInput.from_file("input.circular_tokamak")
result = vj.solve_multigrid(inp, verbose=True)
wout = vj.wout_from_state(inp=inp, state=result.state,
                           fsqr=result.fsqr, fsqz=result.fsqz, fsql=result.fsql,
                           niter=result.iterations, converged=result.converged)
vj.write_wout("wout_circular_tokamak.nc", wout)
vj.plot_wout("wout_circular_tokamak.nc", "figures")
```

The CLI provides the same workflow:

```console
vmex input.circular_tokamak
vmex --plot wout_circular_tokamak.nc
vmex input.nearby --restart wout_circular_tokamak.nc
```

VMEX uses the input file's `NS_ARRAY`, `FTOL_ARRAY`, and `NITER_ARRAY`. `verbose=True` prints the VMEC iteration table; typed errors distinguish invalid inputs, Jacobian failures, non-convergence, and numerical failures.

The common CLI operations are:

| Command | Result |
|---|---|
| `vmex input.X` | solve INDATA or JSON and write `wout_X.nc` |
| `vmex input.X --plot --booz` | solve, write all WOUT plots, transform to Boozer coordinates, and plot `|B|` |
| `vmex --plot wout_X.nc` | plot the equilibrium summary, surfaces, `|B|`, profiles, and 3-D LCFS |
| `vmex input.X --restart wout_Y.nc` | hot-restart a fixed- or free-boundary solve from a saved equilibrium |
| `vmex --scale input.X [B R]` | scale field and length by optional factors; without them target 5.7 T and 1.7 m |
| `vmex --doctor` / `vmex --test` | inspect the installation / run the bundled quick start |

See the [CLI reference](https://vmex.readthedocs.io/en/latest/reference/cli.html) for resolution, device, convergence, coil, plotting, and Boozer options.

## Hot restart

Pass a previous state or wout to initialize a nearby run. VMEX adapts the boundary and skips completed multigrid rungs when possible.

```python
base = vj.solve_multigrid(inp)
nearby = vj.solve_multigrid(changed_input, initial_state=base.state)
from_file = vj.solve_multigrid(changed_input, restart_from="wout_base.nc")
```

The CLI equivalent is `vmex input.changed --restart wout_base.nc`; a deck may instead set `RESTART_WOUT`. Optimization trial solves hot-restart automatically. See the [restart guide](https://vmex.readthedocs.io/en/latest/howto/restart-from-previous-run.html) for grid changes and validation rules.

## Optimizer-neutral problems

Objective tuples use `(function, target, weight)`, with `weight` multiplying the squared cost by default. The resulting problem works directly with SciPy, JAXopt, Optax, or a user optimizer.

```python
from dataclasses import replace
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from vmex import optimize as opt
from vmex.core.omnigenity import QIResidual

max_mode = 5
mpol = max(max_mode + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
qi = QIResidual(np.linspace(0.1, 1.0, 6))

def iota_floor(state, runtime):
    return jnp.maximum(0.33 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

problem = opt.VmecProblem.from_tuples(inp, [
    (qi, 0.0, 1.0),
    (opt.aspect_ratio, 5.0, 0.005),
    (iota_floor, 0.0, 10.0),
], max_mode=max_mode, use_ess=True)

result = least_squares(problem.residual, problem.x0,
    jac=problem.residual_jac, x_scale=problem.scales, max_nfev=50, verbose=2)
optimized_input = problem.input_from_x(result.x)
optimized_equilibrium = problem.equilibrium_from_x(result.x)
```

The defaults are exact implicit derivatives, automatic Jacobian direction, one-column Jacobian batches, hot restarts, and cost weights. Advanced controls include:

- `derivative_method="finite_difference"` for opaque host objectives;
- `implicit_jacobian_method` and `jacobian_batch_size` for response assembly and memory/compile tradeoffs;
- `forward_ftol` and `forward_max_iterations` for the final forward-solve stage;
- `max_fsq_ratio` for the largest under-converged `FSQ / ftol` that may be differentiated;
- `workers` for parallel finite differences, scans, and ensembles. `None` uses the CPUs available to the process and respects scheduler or container limits.

`problem.value_and_grad` and `problem.jax_value_and_grad` expose the same scalar contract. `problem.evaluate(x)` reports solve effort, failed trials, derivative fallbacks, `fsq`, `fsq_ratio`, and whether the implicit derivative was certified. The runnable examples show SciPy least squares, BFGS/L-BFGS-B, JAXopt, Optax Adam, QI/QS objectives, high-accuracy final solves, input/wout output, and plotting.

## QA, QH, QP, and QI examples

The scripts in `examples/optimization/` optimize QA (NFP=2), QH (NFP=4), QP (NFP=2), and QI (NFP=2) from simple seeds; each writes an optimized input, WOUT, and standard plots. Run `QA_optimization.py`, `QH_optimization.py`, `QP_optimization.py`, or `QI_optimization.py`, then `python examples/plot_optimized_families.py` to reproduce the composites below. Each column shows four toroidal cuts separated by `π/(2 NFP)`, the 3-D LCFS colored by `|B|`, and LCFS `|B|` in Boozer coordinates.

![QA, QH, and QP optimization examples](docs/_static/figures/readme_optimization.png)

Validated QI inputs spanning NFP=1–4 are bundled in `examples/data/`; the same plotting script reads them directly.

![QI equilibria at NFP 1 through 4](docs/_static/figures/readme_qi.png)

## Finite beta, free boundary, and mirrors

`examples/free_boundary_essos_coils.py` holds the Landreman–Paul QA coil currents fixed while increasing beta and re-solving the NESTOR free boundary. The magnetic-axis displacement is the expected Shafranov shift.

![Free-boundary beta ramp and Shafranov shift](docs/_static/figures/readme_essos_beta_scan.png)

VMEX also solves open-ended mirrors. `examples/mirror_fixed_boundary_nonaxisymmetric.py` compares an axisymmetric mirror with a non-axisymmetric rotating ellipse; `examples/mirror_free_boundary_beta_scan.py` continues an ESSOS-coil free boundary from 0% to 80% central beta. The latter plots the solved on-axis field against the MHD paraxial scaling `B/Bvac = sqrt(1-beta)` implied by `p + B²/(2 μ0) = Bvac²/(2 μ0)`. The 0–10% lane is supported; higher-beta points remain clearly marked as extended validation pending refined-grid promotion.

![Axisymmetric and rotating-ellipse fixed-boundary mirrors](docs/_static/figures/mirror_fixed_boundary_3d.png)

![Free-boundary mirror beta scan](docs/_static/figures/mirror_free_boundary_beta_scan.png)

## Equilibrium and kinetic diagnostics

`vmex --plot wout_X.nc` produces the compact summary below: iota, pressure, current, Mercier `DMerc` and Glasser `DR`, magnetic well, second adiabatic-invariant contours, Boozer `|B|`, and scalar equilibrium checks. VMEX additionally provides the infinite-n ballooning eigenvalue and growth-rate objectives; positive `ballooning_lambda` is unstable. Trapped-particle fraction is differentiable and validated against analytic, DESC, and near-axis QI references, including its finite on-axis QI limit.

![VMEX equilibrium, stability, and Boozer diagnostics](docs/_static/figures/readme_diagnostics_summary.png)

`examples/optimization/QA_bootstrap_selfconsistent.py` and `QH_bootstrap_selfconsistent.py` iterate VMEC and the Redl model to a self-consistent bootstrap-current profile and compare against the published equilibrium and SFINCS data.

![Self-consistent QA and QH bootstrap current](docs/_static/figures/readme_bootstrap.png)

## Physics and interoperability

VMEX includes VMEC pressure/current/iota profiles, multigrid continuation, NESTOR free boundary, mgrid and direct coil fields, Boozer transforms, QI/QS and maximum-J objectives, Mercier and ballooning diagnostics, bootstrap-current objectives, dimensional scaling, mirror equilibria, and standard wout/mout output. The [capability reference](https://vmex.readthedocs.io/en/latest/reference/capabilities.html) states the validation level and limitations of each path.

VMEX outputs are intended for existing VMEC workflows: `wout_*.nc` files load in SIMSOPT, `booz_xform`, and other downstream tools. VMEC2000 compatibility and deliberate differences are documented in the [compatibility reference](https://vmex.readthedocs.io/en/latest/reference/vmec2000-compatibility.html).

### Solver feature comparison

This concise matrix was checked on 2026-08-11 against current [STELLOPT/VMEC2000](https://github.com/PrincetonUniversity/STELLOPT) and [VMEC++](https://github.com/proximafusion/vmecpp) sources. ✓ denotes a public path, △ a documented limitation, and — no public path; the linked VMEX capability contract above defines validation scope.

| Capability | VMEC2000 | VMEC++ | VMEX |
|---|:---:|:---:|:---:|
| symmetric fixed-boundary toroidal equilibria | ✓ | ✓ | ✓ |
| non-stellarator-symmetric (`LASYM`) fixed boundary | ✓ | — | ✓ |
| 3-D NESTOR free boundary | ✓ | ✓ | ✓ |
| axisymmetric NESTOR free boundary | ✓ | — | ✓ |
| hot restart from a saved equilibrium | — | ✓ Python | ✓ Python/CLI |
| INDATA / JSON input | ✓ / — | ✓ / ✓ | ✓ / ✓ |
| exact equilibrium derivatives and optimizer interface | — | — | ✓ fixed; △ free |
| accelerator execution | CPU/MPI | CPU/OpenMP | CPU/GPU |
| open mirrors and stellarator–mirror hybrids | — | — | ✓ / △ |

### Convergence parity and implementation size

On the bundled NFP=4 QH case at `ns=51`, VMEX follows VMEC2000 and VMEC++ through the full force-residual trace (fresh local run: VMEX `d7347c9`, VMEC2000 `512375c`, VMEC++ 0.5.3). Reproduce it with `python benchmarks/make_readme_figures.py --only convergence`; the benchmark discovers local solver installations or accepts `VMEX_XVMEC2000` and `VMEX_VMECPP_PY`.

![VMEX, VMEC2000, and VMEC++ convergence trace](docs/_static/figures/readme_convergence.png)

The following `cloc 2.11` snapshot counts implementation code and comments, excluding tests, generated code, and third-party sources. VMEX counts `vmex/core` (the toroidal solver); VMEC2000 counts `VMEC2000/Sources` but not shared STELLOPT libraries; VMEC++ counts `src/vmecpp` C++/headers/Python. These scopes make the comparison reproducible, not a claim of identical feature breadth.

| Solver and revision | Files | Code lines | Comment lines |
|---|---:|---:|---:|
| VMEC2000 `aeb0261` | 115 | 24,164 | 8,451 |
| VMEC++ `d83035b` | 146 | 38,338 | 9,661 |
| VMEX `d7347c9` | 46 | 21,189 | 7,857 |

VMEX reduces duplication by expressing spectral operators as vectorized JAX array programs and using the same equations for CPU, accelerators, and automatic differentiation. It also deliberately omits some legacy modes, so the smaller codebase reflects both architecture and narrower compatibility surface.

## Performance and parallelism

JAX compilation is paid once per array structure and reused from a machine-local cache. Warm runs are the relevant measure for continuation, parameter scans, and optimization.

![VMEX runtime comparison](docs/_static/figures/readme_runtime_compare.png)

Independent solves use `vj.parallel.solve_ensemble(inputs, workers=None)`. A single equilibrium already uses XLA's internal threading; ensemble workers are therefore bounded by both the number of cases and the CPUs made available by the host scheduler. Explicit `workers=1` gives a reproducible serial baseline, and GPU/device placement can be selected with `device=`.

Reproducible performance artifacts live in `benchmarks/`; `benchmarks/optimization.py` profiles QI, QA, QH, QP, scalar objectives, SciPy/JAX contract agreement, finite differences, optimizer choices, and the `max_fsq_ratio` policy without committing machine-specific scans or decorative plots.

## Documentation and development

The [documentation](https://vmex.readthedocs.io/) is organized as tutorials, task-focused how-to guides, API/reference pages, and numerical explanations. Start with:

- [first equilibrium](https://vmex.readthedocs.io/en/latest/tutorials/first-equilibrium.html)
- [first gradient](https://vmex.readthedocs.io/en/latest/tutorials/first-gradient.html)
- [first optimization](https://vmex.readthedocs.io/en/latest/tutorials/first-optimization.html)
- [optimization reference](https://vmex.readthedocs.io/en/latest/reference/optimization.html)
- [objectives reference](https://vmex.readthedocs.io/en/latest/reference/objectives.html)
- [parallel and HPC usage](https://vmex.readthedocs.io/en/latest/howto/parallel-ensembles.html)

For development:

```console
git clone https://github.com/uwplasma/vmex
cd vmex
pip install -e ".[dev]"
pytest -q -m "not full and not weekly"
python -m ruff check vmex tests examples benchmarks
```

See [contributing](https://vmex.readthedocs.io/en/latest/project/contributing.html), the [test manifest](tests/manifest.json), and the [changelog](docs/project/changelog.md). VMEX is released under the MIT license.

## Roadmap

- Differentiate the complete reconverged NESTOR plasma–vacuum root, then promote free-boundary plasma-and-coil single-stage optimization beyond the current virtual-casing derivative lane.
- Promote rotating-ellipse stellarator–mirror hybrids from extended validation with refinement, independent force checks, and practical optimization examples.
- Broaden trapped-particle-fraction benchmarks against near-axis theory across QA/QH/QP/QI, retaining the physically nonzero on-axis QI trapped fraction.
- Implement differentiable effective ripple `epsilon_eff` and `Gamma_c`, then add Eduardo Lascas Neto’s associated diagnostic plots. J-contour plotting and the max-J objective already exist and will be integrated into that common diagnostic workflow.
