# VMEX

VMEX is a clean-room, JAX-native reimplementation of the VMEC2000 ideal-MHD
equilibrium code for stellarators and tokamaks. It solves fixed- and
free-boundary equilibria with VMEC2000-derived numerics, writes standard
`wout_*.nc` files that load unchanged in simsopt and booz_xform, and — unlike
the Fortran original — differentiates converged fixed-boundary equilibria via
implicit differentiation. It runs on CPUs and GPUs.

```console
pip install vmex
vmex --test                        # bundled QH case: solve + wout + plots
vmex input.circular_tokamak        # run any VMEC input deck
vmex --plot wout_circular_tokamak.nc
```

The same solve from Python, with an exact gradient at the end:

```python
import jax
import vmex as vj
from vmex.core import implicit

inp = vj.VmecInput.from_file("input.circular_tokamak")
result = vj.solve_multigrid(inp)               # converged equilibrium

p0 = implicit.params_from_input(inp)           # differentiable parameters
grad = jax.grad(lambda p: implicit.run(inp, p).wb)(p0)
```

New here? {doc}`all-of-vmex` is the whole mental model on one page;
{doc}`installation` covers CPU/GPU installs and `vmex --doctor`.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Learn by doing: [your first equilibrium](tutorials/first-equilibrium.md),
[plots and Boozer coordinates](tutorials/plots-and-boozer.md),
[a first gradient](tutorials/first-gradient.md),
[a first optimization](tutorials/first-optimization.md).
:::

:::{grid-item-card} How-to guides
:link: howto/index
:link-type: doc

Task recipes: [run on GPU](howto/run-on-gpu.md),
[restart from a previous run](howto/restart-from-previous-run.md),
[free boundary](howto/free-boundary.md),
[optimization campaigns](howto/optimize-a-boundary.md),
[troubleshooting](howto/troubleshoot.md).
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

Every [CLI flag](reference/cli.rst), [input key](reference/input-file.rst),
[wout variable](reference/wout-file.rst),
[objective](reference/objectives.rst); the
[VMEC2000 compatibility contract](reference/vmec2000-compatibility.rst), the
[capability contract](reference/capabilities.rst), and the
[API](reference/api/basic.rst).
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

The theory: [the variational problem](explanation/variational-problem.rst),
[spectral representation](explanation/spectral-representation.rst),
[preconditioners](explanation/preconditioners.rst),
[NESTOR](explanation/nestor-vacuum.rst),
[adjoint gradients and SOLVAX](explanation/adjoint-gradients.md).
:::

::::

```{figure} /_static/figures/readme_runtime_compare.png
:alt: Runtime comparison of VMEX against VMEC2000 and VMEC++
:width: 95%

Benchmark-suite runtimes: vmex (cold and warm) versus VMEC2000 and a
VMEC++. Warm (compiled-cache) solves are the relevant
number for optimization loops; the full generated table is
{doc}`reference/performance`.
```

```{toctree}
:hidden:
:caption: Start here

all-of-vmex
installation
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:caption: How-to guides

howto/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:caption: Explanation

explanation/index
```

```{toctree}
:hidden:
:caption: Project

project/contributing
project/references
project/changelog
```
