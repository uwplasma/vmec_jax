# vmec_jax porting resume (for Codex)

## 0) Overall goal
Port **VMEC2000 (Fortran)** to a **fast, laptop-friendly, end-to-end differentiable** JAX codebase called **`vmec_jax`**.

Guiding constraints:
- **Minimal dependencies**: `jax` (+ `jaxlib`), `numpy`, optional `netCDF4` for `wout` IO.
- **Fixed-boundary first** (no free-boundary, no MPI/parallelization initially).
- **End-to-end differentiable**: avoid non-differentiable control flow; later use **implicit differentiation** to avoid backprop through iterations.
- **Stepwise validated port**: each step adds a small kernel + diagnostics + regression test.
- **Source-of-truth priority**: prefer **STELLOPT/VMEC2000** (more up-to-date) and **VMEC++ numerics** for algorithmic details.

## 0b) Key decisions from VMEC++ numerics + STELLOPT/VMEC2000
- **Fourier transforms**: VMEC-style **DFT with precomputed trig/weight tables** is canonical. FFTs are a later optimization only if we reproduce VMEC scaling/weights exactly.
- **Constraint pipeline**: Step-10 parity hinges on the **constraint force multiplier (`tcon`)**, **de-alias/bandpass filtering (`alias`)**, and the **`faccon`/`xmpq` scalings** used in `tomnsps`.
- **Solver**: VMEC’s convergence relies on **radial preconditioners (R/Z + lambda, plus m=1 special)** and the **Garabedian-style preconditioned descent / time-stepper**. These are required for VMEC-quality fixed-boundary convergence.
- **Differentiability**: the solver should be made differentiable via **implicit differentiation (custom VJP)** instead of backprop through many iterations.

## 1) Where we started
We began from a raw Fortran **VMEC2000** distribution, and created a new Python package skeleton `vmec_jax` intended to eventually mirror VMEC functionality but using JAX for speed + autodiff.

## 2) Current state (what already works)
This repo version corresponds to **Step-7** of the port plan.

### Step-0: INDATA parsing + boundary evaluation
- Robust **INDATA parser** for VMEC-like input files.
- Builds a **helical Fourier basis** (m,n) and evaluates boundary surfaces:
  - `R(theta,ζ)`, `Z(theta,ζ)` on a uniform `(ntheta,nzeta)` grid (one field period).
- Computes angular derivatives `∂/∂theta` and `∂/∂φ_phys`.

Script:
- `examples/tutorial/00_parse_and_boundary.py`

Validation:
- Prints min/max/mean of R,Z and derivatives; saves `boundary.npz`.

### Step-1: Interior initial guess + full coords kernel + gradients
- Constructs an axis-regular initial guess for interior Fourier coefficients:
  - for m>0 harmonics use `rho**m` with `rho = sqrt(s)` (VMEC/VMEC++ convention),
  - for m=0 R coefficients: linear blend between axis and boundary when axis inputs are provided,
  - other m=0 components scale with `s` for regularity at the axis,
  - λ initialized to 0.
- Evaluates full coordinates on `(s,theta,ζ)` grid:
  - `R(s,theta,ζ)`, `Z(s,theta,ζ)`, `λ(s,theta,ζ)` and `R_theta, R_φ, ...`.
- Demonstrates autodiff through geometry (grad demo).

Scripts:
- `examples/tutorial/02_init_guess_and_coords.py`
- `examples/tutorial/03_grad_full_coords.py`

Validation:
- **Boundary consistency** at s=1 (matches Step-0 boundary eval to ~1e-16).
- Grad demo produces nonzero gradients.

### Step-2: Radial derivatives + metric tensor + Jacobian
- Adds finite-difference radial derivative operator on the s grid.
- Builds covariant basis vectors and computes:
  - covariant metric elements `g_ss, g_stheta, g_sφ, g_thetatheta, g_thetaφ, g_φφ`,
  - Jacobian `sqrtg = e_s dot (e_theta × e_φ)`.
- Prints diagnostics and rough volume integral.

Script:
- `examples/tutorial/04_geom_metrics.py`

Validation:
- `sqrtg` and `g_thetatheta` are zero only on the axis surface `s=0` (expected coordinate singularity).
- For `s>=1`, `sqrtg>0` and metrics are positive where expected.

### Step-3: Profiles + volume profile (from sqrtg)
- Implements VMEC-style **power-series** profiles (pressure, iota, current) from `&INDATA`.
- Pressure convention matches VMEC: inputs `AM`/`PRES_SCALE` are in Pa, while the solver/energy
  uses `pressure = mu0 * pressure_pa` (B² units).
- Computes `dV/ds` and `V(s)` from `sqrtg` by integrating over angles and cumulative trapezoid in `s`.

Script:
- `examples/tutorial/05_profiles_and_volume.py`

Validation:
- Produces a finite, positive volume; prints both **per field period** and **full torus** volumes.
- `pytest -q` includes a regression check of total volume against the bundled VMEC2000 `wout_*.nc`.

### Step-4: B-field + magnetic energy functional (wb)
- Computes contravariant field components `(bsupu, bsupv)` from `sqrtg`, flux functions, and `lambda` derivatives.
- Computes VMEC-normalized magnetic energy `wb`.

Script:
- `examples/tutorial/06_field_and_energy.py`

Validation:
- `pytest -q` includes a regression check of `wb` and B-field consistency against the bundled VMEC2000 `wout_*.nc`.

### Step-5: Lambda-only solver (R/Z fixed)
- Implements a first fixed-boundary optimization loop that updates only the `lambda` coefficients
  (in VMEC's scaled convention), holding R/Z fixed.
- Uses gradient descent + backtracking line search on the magnetic energy `wb`.

Script:
- `examples/tutorial/07_solve_lambda.py`

Validation:
- `pytest -q` includes a regression check that starting from `lambda=0` moves `wb` toward the bundled
  VMEC2000 `wout_*.nc` equilibrium.

### Step-6: Basic fixed-boundary solver (R/Z + lambda)
- Adds a first end-to-end optimization loop over **all** Fourier coefficients:
  - holds the *edge* R/Z coefficients fixed (prescribed boundary),
  - enforces simple axis regularity (m>0 coefficients are 0 at s=0),
  - uses gradient descent + backtracking line search to monotonically decrease a VMEC-style energy objective.

Script:
- `examples/tutorial/08_solve_fixed_boundary.py`

Validation:
- `pytest -q` includes a regression check that the solver decreases the energy while preserving the boundary
  coefficients exactly.

### Step-7: Fixed-boundary solver option: L-BFGS
- Adds an L-BFGS variant of the fixed-boundary solver (no external deps), useful for experimentation.
- Both Step-6 and Step-7 solvers preserve the fixed boundary (edge coefficients) and enforce simple axis regularity.

Script:
- `examples/tutorial/09_solve_fixed_boundary_lbfgs.py`

Validation:
- `pytest -q` includes a regression check that L-BFGS decreases the energy while preserving boundary constraints.

## 3) Key JAX gotchas we hit & fixed
1) **`jit` cannot accept arbitrary Python objects**:
   - `HelicalBasis` and later `VMECState` were passed into jitted functions.
   - Fix: make them **JAX PyTrees** (`tree_flatten/tree_unflatten`).

2) **Duplicate PyTree registration**:
   - `register_pytree_node_class` caused crashes if a class was registered twice.
   - Fix: ensure each class is registered exactly once; add idempotent guards.

3) **x64 warnings / dtype truncation**:
   - JAX defaults to float32 unless x64 enabled.
   - Fix: example scripts enable x64 when JAX is present; internal dtype selection avoids warning spam.

4) **Running scripts from `examples/`**:
   - Users commonly `cd examples` and run scripts.
   - Fix: examples add repo root to `sys.path` so imports work without installation.

5) Naming mismatch (`R_s` vs `Rs`) in geometry outputs:
   - Fix: provide consistent property aliases so debug scripts save/load reliably.

## 4) How to run the current validated steps
From repo root:

```bash
python examples/tutorial/00_parse_and_boundary.py examples/data/input.li383_low_res --out boundary.npz --verbose
python examples/tutorial/02_init_guess_and_coords.py examples/data/input.li383_low_res --out coords_step1.npz --verbose --dump_coeffs
python examples/tutorial/03_grad_full_coords.py examples/data/input.li383_low_res --verbose --topk 12
python examples/tutorial/04_geom_metrics.py examples/data/input.li383_low_res --out geom_step2.npz --verbose --dump_full
python tools/inspect_npz.py geom_step2.npz
```

## 5) What we explicitly left out so far
- Free-boundary VMEC.
- Parallelization (MPI/OpenMP).
- Full VMEC solver loop (Richardson / steepest descent) and preconditioners.
- Writing a full `wout_*.nc` parity output.
- Lasym (non-stellarator-symmetric) parity cases; deferred until tomnspa conventions are reconciled.

## 6) Near-term plan (next milestones)
### Step-8: VMEC-quality fixed-boundary solve
- Add VMEC-style preconditioning:
  - mode-space diagonal / block-diagonal,
  - later radial block-tridiagonal,
- Add force residual parity diagnostics (not just energy), and converge to VMEC-like equilibria.

Current incremental progress toward Step-8:
- Added a lightweight, JAX-friendly radial tri-diagonal smoother preconditioner (`preconditioner="radial_tridi"` or `"mode_diag+radial_tridi"`).
- Added a regression that the VMEC2000 `wout` reference equilibrium is *nearly stationary* for our total-energy objective (gradient RMS is small), and now parse `fsqr/fsqz/fsql` from `wout_*.nc` for context.

### Step-9: Implicit differentiation
- Replace backprop through iterations with implicit diff (custom VJP):
  - solve linear system for adjoint.
  - reuse preconditioner/Krylov.

Current incremental progress toward Step-9:
- Added `vmec_jax.implicit.solve_lambda_state_implicit`, a custom-VJP lambda-only solve that uses
  conjugate gradients + Hessian-vector products (via `jax.jvp`) in the backward pass.
- Added a regression test comparing the implicit gradient to a finite-difference gradient for a simple outer objective.
- Extended implicit differentiation to the full fixed-boundary solve via `vmec_jax.implicit.solve_fixed_boundary_state_implicit`
  (custom VJP with CG/HVP in the backward pass), plus a regression test against finite differences.

## 10) Step-10: VMEC2000 parity diagnostics (forces/field components)
Current incremental progress toward Step-10:
- Added parity checks for contravariant field components (`bsupu`, `bsupv`) reconstructed by vmec_jax against `wout`'s stored `bsup*` on Nyquist modes (outer surfaces; axis-sensitive).
- Added parity checks for covariant field components (`bsubu`, `bsubv`) reconstructed from the metric and `wout` contravariant fields, compared against `wout`'s stored `bsub*` on Nyquist modes.
- Added a figure-generating example script for visual parity inspection (relative error maps on a selected flux surface).
- Added a first force-like residual diagnostic based on objective gradients (`vmec_jax.residuals.force_residuals_from_state`) and a report example that prints these alongside VMEC2000 `wout` scalars (`fsqr/fsqz/fsql`).
- Added VMEC-style half-mesh Jacobian parity kernels (`vmec_jax.vmec_jacobian`, `vmec_jax.vmec_parity`) and a regression vs `wout` Nyquist `gmnc/gmns` (`tests/test_step10_vmec_jacobian_parity.py`).
- Added a half-mesh bcovar ingredient kernel (`vmec_jax.vmec_bcovar`) and a smoke regression (`tests/test_step10_vmec_bcovar_smoke.py`).
- Added an initial port of the VMEC `forces` (R/Z) kernel in array form (`vmec_jax.vmec_forces`) plus a smoke test and a diagnostics example (`examples/3_Advanced/10_vmec_forces_rz_kernel_report.py`). This is the starting point for full `residue/getfsq` parity.
- Added VMEC `fixaray` + `tomnsps` trig/normalization tables (`vmec_jax.vmec_tomnsp`) and a differentiability demo that backprops through a `tomnsps`-based scalar (`examples/2_Intermediate/07_grad_vmec_tomnsps_residual.py`).
- Added a VMEC-like scalar normalization + residual computation scaffold (`vmec_jax.vmec_residue`) and wired it into the Step-10 parity regression (`tests/test_step10_residue_getfsq_parity.py`).
- Fixed a major Step-10 force-kernel convention mismatch: in VMEC `bcovar` overwrites `guu/guv/gvv` with **B-product tensors** `GIJ = (B^i B^j)*sqrt(g)` (used by `forces.f`), which is *not* the metric. After switching the JAX force kernel to use `bc.gij_b_uu/gij_b_uv/gij_b_vv`, the parity diagnostic improved by ~9 orders of magnitude (see `examples/3_Advanced/10_vmec_forces_rz_kernel_report.py` and `tests/test_step10_residue_getfsq_parity.py`).
- Fixed the remaining dominant near-axis mismatch by implementing VMEC's mode-dependent axis rule for internal odd-m fields (jmin1): only the `m=1` contribution is extrapolated to the axis; odd `m>=3` contributions are zero on axis. With this fix, `tests/test_step10_residue_getfsq_parity.py` is now a passing regression (tight parity for the circular tokamak baseline).
- Wired the **constraint-force pipeline** into the R/Z force kernels and the reference-field parity path, using `alias → gcon` with `tcon` from the VMEC preconditioner. For reference fields, the lambda-force kernels now use `wout` `bsub*` averaged to the full mesh to keep `fsql` parity stable (li383 low-res within ~10% relative error).
- Remaining Step-10 gap: for some 3D stellarator-symmetric cases with `nfp>1` (e.g. `li383_low_res`, `n3are`), `bsup*` matches tightly but `bsub*` still shows O(1-8%) RMS differences, likely tied to VMEC’s real-space synthesis / half-mesh metric conventions.
- Matched the current STELLOPT/VMEC2000 convention by **removing the lasym-specific `tcon` halving** (the line is commented out in `bcovar.f`), so `tcon` is now applied uniformly for `lasym=True` and `False`.
- Shifted parity validation to **symmetric cases only** (circular tokamak + li383); lasym regression inputs remain bundled but are excluded from automated tests for now.
- Added a residual decomposition report (`examples/validation/residual_decomposition_report.py`) that breaks `fsqr/fsqz/fsql` into component-only norms (A/B/C/constraint) and top `(m,n)` contributors to guide parity debugging.
- Added a reference-vs-full field comparison report (`examples/validation/residual_compare_fields_report.py`) to isolate differences between the `wout`-driven parity path and the fully derived `vmec-jax` field path.
- Added a `lasym` block report (`examples/validation/lasym_block_report.py`) to separate `tomnsps` vs `tomnspa` contributions and highlight dominant asymmetric modes in 3D parity work.
- Aligned `fixaray` normalization (`dnorm`) with VMEC (reduced-interval normalization independent of `lasym`), added `dnorm3`, and marked lasym Step-10 parity cases as `xfail` while `tomnspa`/`symforce` mismatches are resolved.
- Added a lasym mode trace report (`examples/validation/lasym_mode_trace_report.py`) to isolate A/B/C/constraint contributions for a specific `(m,n)` mode.
- Matched VMEC's `scalxc` convention: after `tomnsps`, VMEC scales the Fourier-space forces by `scalxc`
  (constructed in `profil3d.f`) before calling `residue/getfsq` (see `funct3d.f`). Applying this scaling
  in `vmec_jax.vmec_residue` removed the remaining Step-10 scalar mismatch on symmetric cases.
- Added **JAX-traceable** VMEC force-normalization scalars (`vp`, `wb`, `wp`, `fnorm`, `fnormL`) computed
  directly from bcovar fields (`vmec_force_norms_from_bcovar_dynamic`) plus a dynamic `getfsq` wrapper
  (`vmec_fsq_from_tomnsps_dynamic`). The Step-10 scalar parity regression now uses this dynamic path,
  removing the dependency on `wout.vp/wb/wp` for scalar residual computation.
- Expanded Step-10 scalar parity coverage beyond the initial two cases by bundling additional
  **stellarator-symmetric** `input.*` / `wout_*.nc` pairs (see `examples/data/`) and validating them in
  `tests/test_step10_residue_getfsq_parity.py`.

## 11) Updated priorities (post-VMEC++ numerics review)
1) **Lock down VMEC numerics** (DFT basis, weights, mode ordering, half-mesh conventions) before any FFT acceleration.
2) **Port the constraint-force pipeline** fully (`tcon`, `alias` bandpass, `faccon` scalings, m=1 constraint timing).
3) **Implement full radial preconditioners** (R/Z + lambda + m=1) and update cadence (every ~25 iterations as in VMEC++).
4) **Implement VMEC time-stepper** (Garabedian-style preconditioned descent with adaptive damping).
5) **Only then** tune performance (FFT, fused kernels, `lax.scan`, static shapes) and expand parity/benchmark coverage.

## 7) Longer-term roadmap
- Match VMEC2000 feature set (non-stellarator-symmetric, free-boundary, etc.).
- Performance upgrades:
  - FFT-based angular transforms,
  - fused kernels, careful JIT boundaries,
  - optional GPU support.
- Integrate with SIMSOPT/DESC-style optimization pipelines.

## 8) Documentation
Sphinx documentation lives in `docs/` and is configured for ReadTheDocs via `.readthedocs.yaml`.

Local build:

```bash
pip install -e .[docs]
python -m sphinx -b html docs docs/_build/html
```

Notes:
- Docs sources are reStructuredText (`.rst`), so local builds do not require MyST/Markdown support.
- `docs/conf.py` falls back to the built-in `alabaster` theme if `furo` is not installed.
- Intersphinx is enabled only on ReadTheDocs (`READTHEDOCS=True`) to keep offline builds working.

## 9) Example structure (curated)
In addition to the stepwise scripts in `examples/`, curated examples live in:

- `examples/1_Simple/`
- `examples/2_Intermediate/`
- `examples/3_Advanced/`
