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

Step-9 (implicit differentiation) notes:
- `vmec_jax.implicit.solve_lambda_state_implicit` adds a custom-VJP wrapper for the **lambda-only** sub-solve.
  The backward pass solves a damped linear system using CG and Hessian-vector products computed via `jax.jvp`.
  This is the first building block for implicit differentiation through full fixed-boundary equilibria.
- `vmec_jax.implicit.solve_fixed_boundary_state_implicit` extends the same approach to the full fixed-boundary
  optimization over (R, Z, lambda), exposing implicit gradients w.r.t. 1D profiles/fluxes.

Step-10 (parity diagnostics) notes:
- `vmec_jax.field.bsup_from_geom` can reconstruct contravariant components (bsupu, bsupv) from the metric/Jacobian, flux functions, and lambda derivatives; see `tests/test_step10_bsup_parity.py` and `examples/2_Intermediate/05_bsup_parity_figures.py`.
- `vmec_jax.field.bsub_from_bsup` computes covariant components (bsubu, bsubv) from the metric and contravariant field.
- `vmec_jax.wout.read_wout` now reads `bsubumn*`/`bsubvmn*` (Nyquist) from `wout_*.nc` for parity tests and examples.
- `vmec_jax.residuals.force_residuals_from_state` provides a first force-like residual proxy from the total-objective gradient. This is not yet VMEC `residue/getfsq` parity, but supports regression tests and solver diagnostics while the real-space force kernels are ported.
- VMEC-style half-mesh staggering for the Jacobian is implemented in `vmec_jax.vmec_jacobian` (ports `jacobian.f`), with helper parity kernels in `vmec_jax.vmec_parity`.
- Half-mesh metric + B ingredients needed by VMEC's force kernels are implemented in `vmec_jax.vmec_bcovar` (ports core `bcovar` algebra used before `forces`).
- An initial direct port of VMEC's `forces` (R/Z) kernel lives in `vmec_jax.vmec_forces`. This is intentionally a parity/debug kernel and is not yet full `residue.f90` / `getfsq.f` parity.
- Important VMEC convention: after `bcovar` runs, VMEC overwrites `guu/guv/gvv` with the **B-product tensors** `(B^i B^j)*sqrt(g)` (used by `forces.f`). The Step-10 JAX force kernel therefore must use `bc.gij_b_uu/gij_b_uv/gij_b_vv` for these terms, not the metric elements `g_uu/g_uv/g_vv`.
- Important VMEC axis convention (vmec_params.f): internal odd-m fields use `jmin1` rules:
  only the `m=1` contribution is extrapolated to the axis; odd `m>=3` contributions are zero on axis. Implementing this is essential for `fsqr/fsqz/fsql` parity near the magnetic axis.

Notes on conventions:
- VMEC input pressure coefficients are in Pa, but VMEC’s internal pressure used in the energy
  functional is in `mu0*Pa` (B² units). `vmec_jax.profiles.eval_profiles` returns both:
  `pressure_pa` (Pa) and `pressure` (`mu0*Pa`) for parity.
- VMEC `wout` files store some fields on the radial half mesh; regression tests against `wout`
  should use VMEC-style half-mesh radial integration where appropriate (e.g. volume/`wp`).

3. **Verification harness**:
   - extend `.npz` stage dumps to include field and force quantities
   - compare norms and key 1D profiles against VMEC2000 `wout_*.nc`

## VMEC++ numerics notes (Schilling 2025)
Key formulas in **The Numerics of VMEC++** (Sec. 5) that matter for Step-10 parity:

- **Half-grid metric interpolation with parity**: even/odd-`m` contributions are interpolated
  separately with explicit `√s` factors (Eq. 5.118–5.120). This matches VMEC’s current
  implementation (attempting the “direct” interpolation in Eq. 5.113 degrades convergence).
- **Contravariant B on half-grid**: derived from `λ_θ`/`λ_ζ` with even/odd separation and
  `√s` normalization (Eq. 5.135–5.136).
- **Covariant B from metric**: `B_θ = g_θθ B^θ + g_θζ B^ζ`, `B_ζ = g_θζ B^θ + g_ζζ B^ζ`
  (Eq. 5.142–5.143), with axisymmetric simplifications when `g_θζ=0`.
- **Lambda-force handling**:
  - odd-`m` λ-forces scale as `√s` times even-`m` forces (Eq. 5.176–5.179),
  - full-grid `B_ζ` uses a **hybrid blend** of `B_ζ` from half-grid interpolation and a
    `g_θζ B^θ + g_ζζ B^ζ` reconstruction (Eq. 5.181–5.185),
  - blend factor `ε_blend = 2*pdamp*(1-s)` (Eq. 5.182).
- **Force decomposition**: VMEC organizes `F_R/F_Z` into `A - ∂_θ B + ∂_ζ C` components,
  with `P = R*(|B|^2/(2 μ0) + p)` (Eq. 5.159–5.173). These formulas govern the `forces.f`
  port and are the target for the remaining Step-10 parity work.

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
