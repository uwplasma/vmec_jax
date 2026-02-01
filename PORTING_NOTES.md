# VMEC2000 → vmec_jax: porting notes (through step 5)

This repo snapshot is validated through:
- Step-0: INDATA parsing + boundary evaluation
- Step-1: initial guess + coords kernel
- Step-2: metric/Jacobian (`sqrtg`) via radial FD + Cartesian embedding
- Step-3: 1D profiles + volume integrals from `sqrtg`
- Step-4: contravariant B + magnetic energy (`wb`) vs `wout`
- Step-5: lambda-only solve (R/Z fixed) vs `wout`

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

## What to do next (step-4/5)

1. **Full fixed-boundary solve**: extend the current lambda-only solve to update R/Z as well.
   - start with a laptop-friendly outer loop (L-BFGS / Anderson / quasi-Newton)
   - then add VMEC-inspired mode-space + radial preconditioning

3. **Verification harness**:
   - extend `.npz` stage dumps to include field and force quantities
   - compare norms and key 1D profiles against VMEC2000 `wout_*.nc`

## Performance tactics to keep it laptop-fast

- Keep all static tables (`m,n`, trig basis) as **constants** captured by a `jit`ted function.
- Use `lax.scan` over radius and chunked evaluation over angles to control memory.
- Implement VMEC's **radial preconditioner** first (cheap and strong), then consider
  quasi-Newton (L-BFGS) on top.
