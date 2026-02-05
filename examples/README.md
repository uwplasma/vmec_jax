# Examples

This folder contains runnable scripts and bundled reference data used by the
test suite. We keep **all scripts inside subfolders** so each topic is isolated
and easy to discover; this also avoids ambiguous top-level entrypoints.

## Layout and intent

- `tutorial/`: step-by-step scripts (00–09) that introduce the main kernels.
- `solvers/`: convergence experiments and solver-focused scripts.
- `gradients/`: autodiff + implicit differentiation demos (requires JAX).
- `validation/`: parity checks vs bundled `wout_*.nc` (and reporting utilities).
- `visualization/`: figure generation + VTK export scripts.
- `data/`: bundled `input.*` and `wout_*_reference.nc` files used in CI tests.
- `compat/`: thin wrappers that forward to the tutorial scripts (for legacy paths).
- `outputs/`: default location where some scripts write `.npz` artifacts.

## Quick run

Most scripts accept an input path:

```bash
python examples/tutorial/00_parse_and_boundary.py examples/data/input.LandremanSenguptaPlunk_section5p3_low_res --out boundary.npz --verbose
python examples/tutorial/08_solve_fixed_boundary.py examples/data/input.circular_tokamak --verbose
```
