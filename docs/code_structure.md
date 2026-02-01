# Code structure

At a high level, the package is organized around a few core concepts:

- **configuration / inputs**: parse VMEC `&INDATA` and derive grid sizes
- **static data**: precomputed tables for angles and Fourier phases
- **state**: the Fourier coefficients that define the equilibrium fields
- **kernels**: pure functions that evaluate geometry/fields/energies on grids
- **solvers**: optimization loops that update state to reduce an objective

## Key modules

- `vmec_jax/namelist.py`:
  robust parsing of `&INDATA` from VMEC input files.

- `vmec_jax/config.py`:
  `VMECConfig` and `load_config()`; derives grid sizes and normalizes options.

- `vmec_jax/modes.py`:
  VMEC mode ordering (`xm/xn`) and default grid size logic.

- `vmec_jax/grids.py`:
  angle grid construction (`AngleGrid`).

- `vmec_jax/fourier.py`:
  helical Fourier basis and synthesis/derivative kernels.

- `vmec_jax/state.py`:
  `VMECState` (PyTree), packing/unpacking, and shape helpers.

- `vmec_jax/static.py`:
  `VMECStatic` container and `build_static()` precomputation.

- `vmec_jax/init_guess.py`:
  axis-regular initial guess from boundary coefficients.

- `vmec_jax/coords.py`:
  evaluate `(R,Z,λ)` and angular derivatives on the `(s,θ,ζ)` grid.

- `vmec_jax/geom.py`:
  metric tensor + signed Jacobian `sqrtg` (step-2 geometry).

- `vmec_jax/profiles.py`:
  1D profile evaluation (pressure/iota/current) from input coefficients.

- `vmec_jax/integrals.py`:
  volume integrals from `sqrtg`.

- `vmec_jax/field.py`:
  contravariant field components `(bsupu, bsupv)` and helper conventions (`lamscale`).

- `vmec_jax/energy.py`:
  magnetic energy functional (`wb`) and basic flux-profile helpers.

- `vmec_jax/solve.py`:
  step-5/6/7 solver loops (lambda-only, fixed-boundary GD, fixed-boundary L-BFGS).

- `vmec_jax/wout.py`:
  minimal `wout_*.nc` reader for regression and parity checks.

## Examples and tests

- `examples/` contains stepwise scripts that produce `.npz` dumps and (optionally) figures.
- `tests/` contains quick regression tests intended to keep `pytest -q` fast.

