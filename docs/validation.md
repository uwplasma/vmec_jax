# Validation and regression testing

`vmec-jax` is developed using a “regression-first” workflow: each porting step introduces a small
kernel and validates it against VMEC2000 outputs (typically via `wout_*.nc`).

## Bundled regression case

The repo includes several small, low-resolution reference cases used in examples and tests:

- 3D stellarator (vacuum):
  - input: `examples/input.LandremanSenguptaPlunk_section5p3_low_res`
  - reference output: `examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc`

- Tokamak sanity cases (vacuum):
  - `examples/input.circular_tokamak` + `examples/wout_circular_tokamak_reference.nc`
  - `examples/input.up_down_asymmetric_tokamak` + `examples/wout_up_down_asymmetric_tokamak_reference.nc`

- Finite-beta case:
  - `examples/input.li383_low_res` + `examples/wout_li383_low_res_reference.nc`

This case is meant to be:

- small enough to run on a laptop,
- rich enough to exercise 3D mode ordering and the field-period angle conventions.

## What is validated today

The tests in `tests/` cover:

- correct INDATA parsing (`test_namelist.py`)
- boundary evaluation and agreement with the `s=1` state surface (`test_boundary_eval.py`, `test_coords_kernel.py`)
- metric/Jacobian positivity and shape checks (`test_geom_metrics.py`)
- step-3 volume/profile regression (`test_step3_profiles.py`)
- step-4 `bsup*` and `wb` regression vs `wout` (`test_step4_field_energy.py`)
- step-5 lambda-only solve moves `wb` toward equilibrium (`test_step5_solve_lambda.py`)
- step-6/7 fixed-boundary solvers decrease the energy while preserving constraints (`test_step6_*`, `test_step7_*`)

## Running tests

```bash
pytest -q
```

If you do not have `netCDF4` installed, tests that require `wout` I/O will be skipped.

## Comparing against VMEC2000 on new inputs

Recommended workflow for a new VMEC input file:

1. Run VMEC2000 to produce a `wout_*.nc`.
2. Run `vmec-jax` step scripts and compare:
   - mode ordering and sizes (`mpol`, `ntor`, `ns`, `nfp`),
   - total volume and `dV/ds`,
   - `bsup*` fields,
   - `wb`.

The scripts `examples/06_field_and_energy.py` and `examples/07_solve_lambda.py` accept `--wout`
to compare against a supplied `wout_*.nc`.

## Tolerances and why they start loose

In early porting steps, we sometimes compare quantities that depend on a radial derivative scheme.
VMEC2000 uses carefully tuned half-mesh operators; `vmec-jax` currently uses a simpler finite
difference on the coefficient arrays. This can introduce systematic differences that are *not*
errors, but do limit pointwise parity.

To keep progress moving:

- we validate “stable” intermediate quantities first (e.g. `wout` Nyquist `sqrtg`),
- use RMS comparisons in real space when grids differ,
- tighten tolerances only once upstream kernels match VMEC more closely.
