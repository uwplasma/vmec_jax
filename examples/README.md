# Examples

This folder contains runnable scripts and bundled reference data used by the
test suite. The public examples are intentionally few and opinionated: they use
the same small, high-level API surface so you can copy/paste and adapt quickly.

Exception:
- `showcase_axisym_input_to_wout.py` is intentionally top-level so users have a
  single minimal "input -> wout + plots + parity" script to start from.

## Layout and intent

- `tutorial/`: minimal low-level kernel demos (start here only if you want to learn internals).
- `gradients/`: small autodiff examples (requires JAX).
- `validation/`: parity checks vs bundled `wout_*.nc`.
- `visualization/`: VMEC-style figure generation + VTK export.
- `data/`: bundled `input.*` and `wout_*_reference.nc` files used in CI tests.
- `outputs/`: default location where some scripts write `.npz` artifacts.
 
Developer-only diagnostics and research scripts live under `tools/diagnostics/`
(they are not part of the user-facing examples set).

## Quick run

Most scripts accept an input path. Solver-style scripts are **verbose by default**
(to mimic VMEC2000 terminal output); use `--quiet` to suppress iteration logs:

```bash
python examples/showcase_axisym_input_to_wout.py
python examples/tutorial/00_parse_and_boundary.py examples/data/input.circular_tokamak --out boundary.npz --verbose
```
