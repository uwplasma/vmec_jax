# Performance notes

This page collects practical advice for using `vmec-jax` efficiently.

## Enable float64

VMEC2000 is float64-first. For parity, enable x64 in JAX:

```bash
export JAX_ENABLE_X64=1
```

## JIT boundaries and compile latency

On CPU, compilation can dominate runtime for moderate problem sizes. `vmec-jax` uses:

- a jitted geometry kernel (`eval_geom`)
- non-jitted solver gradients by default (to reduce compile latency)

Solver functions accept `jit_grad=True` to trade longer compile time for faster iterations.

## Preconditioning

VMEC2000 relies heavily on preconditioning. `vmec-jax` currently includes a very lightweight
mode-diagonal preconditioner (`preconditioner="mode_diag"`) to stabilize early solver experiments.
VMEC’s radial block-tridiagonal preconditioner is a future milestone.

## Static precomputation

Use `VMECStatic` to avoid rebuilding:

- mode tables,
- angle grids,
- Fourier basis tensors,
- radial grid.

## Avoid Python objects in jitted functions

JAX `jit` requires inputs to be arrays or PyTrees.
`vmec-jax` makes the key containers PyTrees:

- `VMECState`
- `HelicalBasis`
- `Geom`

If you build your own containers, follow the same approach.

## Memory considerations

The current Fourier implementation stores `(K, ntheta, nzeta)` basis tensors for cos/sin phases.
This is acceptable for low-resolution validation cases, but will become heavy for larger `mpol/ntor`.

Planned upgrades:

- factored/FFT-based angular transforms
- chunked evaluation in `θ`/`ζ` to reduce peak memory
