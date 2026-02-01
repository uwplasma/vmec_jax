# Contributing

Contributions are welcome, especially those that improve:

- correctness / parity with VMEC2000,
- performance (without harming readability),
- documentation and examples.

## Development setup

```bash
pip install -e .[jax,netcdf,dev]
pytest -q
```

We recommend keeping `JAX_ENABLE_X64=1` for parity work.

## Style and quality

- Keep kernels **pure** (no I/O, no global state).
- Prefer small PRs that add:
  1) a kernel,
  2) an example or diagnostic,
  3) a regression test.
- Avoid expanding dependencies in the core runtime.
  Put optional features behind extra requirements (`docs`, `netcdf`, plotting).

## Adding a new porting step

1. Identify the corresponding VMEC2000 routine(s).
2. Add a minimal JAX/Numpy kernel with clear input/output shapes.
3. Add an example script that:
   - runs end-to-end on a small case,
   - writes an `.npz` artifact with useful arrays,
   - prints min/max/means and sanity checks.
4. Add a regression test (fast) using the bundled low-res case.

## Large changes

For large refactors or algorithmic changes, prefer to:

- introduce compatibility shims first,
- keep old APIs temporarily,
- update docs alongside code.

