# vmec-jax (validated through step 7)

This is an incremental JAX port of **VMEC2000** (fixed-boundary first). The
equilibrium solve (force-balance / energy minimization) is **not implemented
yet** (R/Z solve + pressure/forces), but the repo is validated through:

- Step-0: INDATA parsing + boundary eval
- Step-1: initial guess + full coords kernel (+ autodiff demo)
- Step-2: radial FD + metric/Jacobian (`sqrtg`) + rough volume sanity checks
- Step-3: input profiles (pressure/iota/current) + volume profile from `sqrtg`
- Step-4: B-field components + magnetic energy (`wb`) regression vs VMEC2000 `wout`
- Step-5: lambda-only solver (R/Z fixed) regression toward VMEC2000 `wout`
- Step-6: basic fixed-boundary solver (R/Z/lambda) with monotone energy decrease
- Step-7: fixed-boundary solver option: L-BFGS (no external deps)

## Install

```bash
pip install -e .
# add JAX
pip install -e .[jax]
# (optional) for wout_*.nc IO / baseline comparisons
pip install -e .[netcdf]
```

## Quick start

Run the validated example chain on the bundled low-res case:

```bash
python examples/00_parse_and_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out boundary_step0.npz --verbose
python examples/02_init_guess_and_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out coords_step1.npz --verbose
python examples/04_geom_metrics.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out geom_step2.npz --verbose
python examples/05_profiles_and_volume.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out profiles_step3.npz --verbose
python examples/06_field_and_energy.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
python examples/07_solve_lambda.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
python examples/08_solve_fixed_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose
python examples/09_solve_fixed_boundary_lbfgs.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose
```

Solver note: the optimization routines default to `jit_grad=False` to reduce compilation latency; set `jit_grad=True` for faster per-iteration runtime once shapes are stable.

Optional: autodiff demo through the full coords kernel:

```bash
python examples/03_grad_full_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose --topk 12
```

## Step-3 outputs

The step-3 script writes a `.npz` with:
- `pressure(s)` (Pa)
- `iota(s)` and/or `current(s)` (depending on the input)
- `dV/ds` and `V(s)` computed from `sqrtg` (per field period; multiply by `NFP` for the full torus)

## Coordinate conventions (matches VMEC internal)

- `theta` is poloidal angle in `[0, 2π)`.
- `zeta` is the *field-period* toroidal angle in `[0, 2π)`, i.e. VMEC typically represents one field period.
- The Fourier phase is `m*theta - n*zeta`.
- Derivatives w.r.t the *physical* toroidal angle `phi_phys` pick up `NFP`:
  `∂/∂phi_phys = NFP * ∂/∂zeta`.

## Next step

Implement VMEC-quality fixed-boundary convergence (VMEC-style preconditioning + force residual parity),
then add implicit differentiation (custom VJP) for cheap outer gradients.
