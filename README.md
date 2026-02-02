# vmec-jax (validated through step 7)

This is an incremental JAX port of **VMEC2000** (fixed-boundary first). The
repo includes early fixed-boundary energy-minimization solvers, but it is **not
yet VMEC-quality** (full force-balance residual parity + VMEC preconditioners).
The repo is validated through:

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
# (optional) plotting (publication-ready figures in examples)
pip install -e .[plots]
# (optional) build docs locally
pip install -e .[docs]
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

## Documentation

Sphinx docs live in `docs/` and are configured for ReadTheDocs via `.readthedocs.yaml`.

Build locally:

```bash
python -m sphinx -b html docs docs/_build/html
```

## Examples (structured)

In addition to the stepwise scripts in `examples/`, there are curated example sets:

- `examples/1_Simple/`: quick demos + figures (e.g. boundary plots)
- `examples/2_Intermediate/`: multi-kernel workflows + figures
- `examples/3_Advanced/`: solver experiments + convergence plots + ParaView VTK export

Optional: autodiff demo through the full coords kernel:

```bash
python examples/03_grad_full_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose --topk 12
```

## ParaView visualization (VTK export)

Export a surface `B` field and a field-line trace for ParaView (requires the bundled `wout_*.nc` reference, so `netCDF4` must be installed):

```bash
python examples/3_Advanced/02_vtk_field_and_fieldlines.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --hi-res --outdir vtk_out
```

## Step-9: implicit differentiation (lambda-only)

Differentiate an outer objective through the lambda-only solve (no backprop through iterations), producing a publication-ready figure:

```bash
python examples/2_Intermediate/02_implicit_lambda_gradients.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --outdir figures_implicit_lambda
```

## Step-9: implicit differentiation (fixed-boundary)

Implicitly differentiate a *geometric* quantity through the full fixed-boundary equilibrium solve (advanced; requires `matplotlib`):

```bash
python examples/3_Advanced/03_implicit_fixed_boundary_sensitivity.py examples/input.circular_tokamak --outdir figures_implicit_fixed_boundary
```

## Step-10: VMEC2000 parity diagnostics (contravariant B)

Compare contravariant B components (``bsupu``, ``bsupv``) reconstructed by vmec_jax against the `wout` contravariant fields (writes figures; requires `netCDF4` + `matplotlib`):

```bash
python examples/2_Intermediate/05_bsup_parity_figures.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --outdir figures_bsup_parity
```

## Step-10: VMEC2000 parity diagnostics (covariant B)

Compare covariant B components (``bsubu``, ``bsubv``) reconstructed from the metric and `wout` contravariant fields against the `wout` covariant fields (writes figures; requires `netCDF4` + `matplotlib`):

```bash
python examples/2_Intermediate/03_bsub_parity_figures.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --outdir figures_bsub_parity
```

## Step-10 target: force-like residual diagnostics

Print vmec_jax force-like residual scalars (derived from objective gradients) alongside VMEC2000 `wout` scalars `fsqr/fsqz/fsql`:

```bash
python examples/3_Advanced/05_force_residual_report.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --hi-res
```

## Step-10: VMEC2000 parity diagnostics (|B|)

Compare `wout` |B| Fourier coefficients (`bmnc/bmns`) against |B| reconstructed from `wout` Nyquist `bsup*` and the metric (writes figures; requires `netCDF4` + `matplotlib`):

```bash
python examples/2_Intermediate/04_bmag_parity_figures.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --outdir figures_bmag_parity
```

## Step-10: VMEC2000 output parity (wout writer)

Roundtrip a reference ``wout_*.nc`` file through vmec_jax's minimal writer (requires `netCDF4`):

```bash
python examples/3_Advanced/06_wout_roundtrip.py --wout examples/wout_circular_tokamak_reference.nc --out wout_roundtrip.nc
```

## Step-10 (WIP): VMEC `forces` kernel (R/Z)

Run the initial port of VMEC2000’s real-space `forces` (R/Z) kernel and dump diagnostics (requires `netCDF4`):

```bash
python examples/3_Advanced/10_vmec_forces_rz_kernel_report.py
```

## Step-3 outputs

The step-3 script writes a `.npz` with:
- `pressure_pa(s)` (Pa) and `pressure(s)` (VMEC internal units, `mu0*Pa`)
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
