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
  - keeps plotting-focused scripts; VMEC parity/validation runners live under `validation/`.
- `data/`: bundled `input.*` and `wout_*_reference.nc` files used in CI tests.
- `compat/`: thin wrappers that forward to the tutorial scripts (for legacy paths).
- `outputs/`: default location where some scripts write `.npz` artifacts.

### n3are parity runner

- Main parity comparison entrypoint:
  - `validation/n3are_vmec_vs_vmecjax.py`
- Backward-compatible wrapper:
  - `visualization/n3are_vmec2000_vs_vmecjax.py`

## Quick run

Most scripts accept an input path. Solver-style scripts are **verbose by default**
(to mimic VMEC2000 terminal output); use `--quiet` to suppress iteration logs:

```bash
python examples/tutorial/00_parse_and_boundary.py examples/data/input.li383_low_res --out boundary.npz --verbose
python examples/tutorial/08_solve_fixed_boundary.py examples/data/input.circular_tokamak
```
