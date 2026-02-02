# vmec_jax porting resume (for Codex)

## 0) Overall goal
Port **VMEC2000 (Fortran)** to a **fast, laptop-friendly, end-to-end differentiable** JAX codebase called **`vmec_jax`**.

Guiding constraints:
- **Minimal dependencies**: `jax` (+ `jaxlib`), `numpy`, optional `netCDF4` for `wout` IO.
- **Fixed-boundary first** (no free-boundary, no MPI/parallelization initially).
- **End-to-end differentiable**: avoid non-differentiable control flow; later use **implicit differentiation** to avoid backprop through iterations.
- **Stepwise validated port**: each step adds a small kernel + diagnostics + regression test.

## 1) Where we started
We began from a raw Fortran **VMEC2000** distribution, and created a new Python package skeleton `vmec_jax` intended to eventually mirror VMEC functionality but using JAX for speed + autodiff.

## 2) Current state (what already works)
This repo version corresponds to **Step-7** of the port plan.

### Step-0: INDATA parsing + boundary evaluation
- Robust **INDATA parser** for VMEC-like input files.
- Builds a **helical Fourier basis** (m,n) and evaluates boundary surfaces:
  - `R(θ,ζ)`, `Z(θ,ζ)` on a uniform `(ntheta,nzeta)` grid (one field period).
- Computes angular derivatives `∂/∂θ` and `∂/∂φ_phys`.

Script:
- `examples/00_parse_and_boundary.py`

Validation:
- Prints min/max/mean of R,Z and derivatives; saves `boundary.npz`.

### Step-1: Interior initial guess + full coords kernel + gradients
- Constructs an axis-regular initial guess for interior Fourier coefficients:
  - for m>0 harmonics use `s**m` scaling (regular on axis),
  - for m=0 keep constant in s (initial conservative behavior),
  - λ initialized to 0.
- Evaluates full coordinates on `(s,θ,ζ)` grid:
  - `R(s,θ,ζ)`, `Z(s,θ,ζ)`, `λ(s,θ,ζ)` and `R_θ, R_φ, ...`.
- Demonstrates autodiff through geometry (grad demo).

Scripts:
- `examples/02_init_guess_and_coords.py`
- `examples/03_grad_full_coords.py`

Validation:
- **Boundary consistency** at s=1 (matches Step-0 boundary eval to ~1e-16).
- Grad demo produces nonzero gradients.

### Step-2: Radial derivatives + metric tensor + Jacobian
- Adds finite-difference radial derivative operator on the s grid.
- Builds covariant basis vectors and computes:
  - covariant metric elements `g_ss, g_sθ, g_sφ, g_θθ, g_θφ, g_φφ`,
  - Jacobian `sqrtg = e_s · (e_θ × e_φ)`.
- Prints diagnostics and rough volume integral.

Script:
- `examples/04_geom_metrics.py`

Validation:
- `sqrtg` and `g_θθ` are zero only on the axis surface `s=0` (expected coordinate singularity).
- For `s>=1`, `sqrtg>0` and metrics are positive where expected.

### Step-3: Profiles + volume profile (from sqrtg)
- Implements VMEC-style **power-series** profiles (pressure, iota, current) from `&INDATA`.
- Pressure convention matches VMEC: inputs `AM`/`PRES_SCALE` are in Pa, while the solver/energy
  uses `pressure = mu0 * pressure_pa` (B² units).
- Computes `dV/ds` and `V(s)` from `sqrtg` by integrating over angles and cumulative trapezoid in `s`.

Script:
- `examples/05_profiles_and_volume.py`

Validation:
- Produces a finite, positive volume; prints both **per field period** and **full torus** volumes.
- `pytest -q` includes a regression check of total volume against the bundled VMEC2000 `wout_*.nc`.

### Step-4: B-field + magnetic energy functional (wb)
- Computes contravariant field components `(bsupu, bsupv)` from `sqrtg`, flux functions, and `lambda` derivatives.
- Computes VMEC-normalized magnetic energy `wb`.

Script:
- `examples/06_field_and_energy.py`

Validation:
- `pytest -q` includes a regression check of `wb` and B-field consistency against the bundled VMEC2000 `wout_*.nc`.

### Step-5: Lambda-only solver (R/Z fixed)
- Implements a first fixed-boundary optimization loop that updates only the `lambda` coefficients
  (in VMEC's scaled convention), holding R/Z fixed.
- Uses gradient descent + backtracking line search on the magnetic energy `wb`.

Script:
- `examples/07_solve_lambda.py`

Validation:
- `pytest -q` includes a regression check that starting from `lambda=0` moves `wb` toward the bundled
  VMEC2000 `wout_*.nc` equilibrium.

### Step-6: Basic fixed-boundary solver (R/Z + lambda)
- Adds a first end-to-end optimization loop over **all** Fourier coefficients:
  - holds the *edge* R/Z coefficients fixed (prescribed boundary),
  - enforces simple axis regularity (m>0 coefficients are 0 at s=0),
  - uses gradient descent + backtracking line search to monotonically decrease a VMEC-style energy objective.

Script:
- `examples/08_solve_fixed_boundary.py`

Validation:
- `pytest -q` includes a regression check that the solver decreases the energy while preserving the boundary
  coefficients exactly.

### Step-7: Fixed-boundary solver option: L-BFGS
- Adds an L-BFGS variant of the fixed-boundary solver (no external deps), useful for experimentation.
- Both Step-6 and Step-7 solvers preserve the fixed boundary (edge coefficients) and enforce simple axis regularity.

Script:
- `examples/09_solve_fixed_boundary_lbfgs.py`

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
python examples/00_parse_and_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out boundary.npz --verbose
python examples/02_init_guess_and_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out coords_step1.npz --verbose --dump_coeffs
python examples/03_grad_full_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose --topk 12
python examples/04_geom_metrics.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out geom_step2.npz --verbose --dump_full
python tools/inspect_npz.py geom_step2.npz
```

## 5) What we explicitly left out so far
- Free-boundary VMEC.
- Parallelization (MPI/OpenMP).
- Full VMEC solver loop (Richardson / steepest descent) and preconditioners.
- Writing a full `wout_*.nc` parity output.

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
- Added parity checks for covariant field components (`bsubu`, `bsubv`) reconstructed from the metric and `wout` contravariant fields, compared against `wout`'s stored `bsub*` on Nyquist modes.
- Added a figure-generating example script for visual parity inspection (relative error maps on a selected flux surface).

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
