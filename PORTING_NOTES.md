# VMEC2000 → vmec_jax: porting notes (through step 7)

This repo snapshot is validated through:
- Step-0: INDATA parsing + boundary evaluation
- Step-1: initial guess + coords kernel
- Step-2: metric/Jacobian (`sqrtg`) via radial FD + Cartesian embedding
- Step-3: 1D profiles + volume integrals from `sqrtg`
- Step-4: contravariant B + magnetic energy (`wb`) vs `wout`
- Step-5: lambda-only solve (R/Z fixed) vs `wout`
- Step-6: basic fixed-boundary solve (R/Z/lambda) with monotone energy decrease
- Step-7: fixed-boundary solve option: L-BFGS (no external deps)

## Kernel mapping (Fortran → Python/JAX)

| VMEC2000 (Fortran) | Role | vmec_jax (this step) |
|---|---|---|
| `readin.f` / `vmec_input.f` | read &INDATA (profiles, boundary, controls) | `vmec_jax/namelist.py`, `vmec_jax/config.py` |
| `fixaray.f` | choose `ntheta`, `nzeta`; precompute sin/cos tables | `vmec_jax/modes.py`, `vmec_jax/grids.py`, `vmec_jax/fourier.py` |
| `totzsp_mod.f` / `tomnsp_mod.f` | Fourier transforms | `vmec_jax/fourier.py` (`eval_fourier*`) |
| boundary arrays `rbc/rbs/zbc/zbs` | LCFS shape | `vmec_jax/boundary.py` |
| `funct3d.f` (geometry bits) | coordinates + derivatives + metric/Jacobian | `vmec_jax/coords.py`, `vmec_jax/radial.py`, `vmec_jax/geom.py` |
| `profile_functions.f` | pmass/piota/pcurr profile parameterizations | `vmec_jax/profiles.py` (power-series only for now) |
| (postprocessing) | volume integrals from `sqrt(g)` | `vmec_jax/integrals.py` |
| `bcovar.f` + `add_fluxes.f90` | contravariant B components (bsup*) | `vmec_jax/field.py` |
| (diagnostics) `wb` | magnetic energy integral | `vmec_jax/energy.py` |
| `read_wout_mod.f90` | read `wout_*.nc` (subset) | `vmec_jax/wout.py` |
| (inner solve) | optimize lambda at fixed R/Z | `vmec_jax/solve.py` |
| (outer solve) | optimize R/Z/lambda (fixed boundary) | `vmec_jax/solve.py` |

## What to do next (step-4/5)

1. **VMEC-quality fixed-boundary solve**: add VMEC-style preconditioning and converge to force-balance parity.
   - mode-space diagonal / block-diagonal,
   - later radial block-tridiagonal,
   - include pressure/force residual diagnostics (not just energy).

Notes:
- `vmec_jax.solve` now supports a lightweight radial tri-diagonal smoother preconditioner (`radial_tridi`) that can be combined with the existing mode-diagonal scaling (`mode_diag+radial_tridi`).
- `vmec_jax.wout.read_wout` parses VMEC force residual scalars (`fsqr/fsqz/fsql`) when present, for solver diagnostics and future parity work.

Notes on conventions:
- VMEC input pressure coefficients are in Pa, but VMEC’s internal pressure used in the energy
  functional is in `mu0*Pa` (B² units). `vmec_jax.profiles.eval_profiles` returns both:
  `pressure_pa` (Pa) and `pressure` (`mu0*Pa`) for parity.
- VMEC `wout` files store some fields on the radial half mesh; regression tests against `wout`
  should use VMEC-style half-mesh radial integration where appropriate (e.g. volume/`wp`).

3. **Verification harness**:
   - extend `.npz` stage dumps to include field and force quantities
   - compare norms and key 1D profiles against VMEC2000 `wout_*.nc`

## Performance tactics to keep it laptop-fast

- Keep all static tables (`m,n`, trig basis) as **constants** captured by a `jit`ted function.
- Use `lax.scan` over radius and chunked evaluation over angles to control memory.
- Implement VMEC's **radial preconditioner** first (cheap and strong), then consider
  quasi-Newton (L-BFGS) on top.

## Documentation and examples

- Sphinx docs live in `docs/` and are configured for ReadTheDocs via `.readthedocs.yaml`.
- Curated examples (including figure generation) live in:
  - `examples/1_Simple/`
  - `examples/2_Intermediate/`
  - `examples/3_Advanced/`
