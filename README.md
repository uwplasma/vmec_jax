# vmec-jax

Laptop-friendly, end-to-end differentiable (JAX) rewrite of **VMEC2000** (fixed-boundary first).

![VMEC Step-10 parity pipeline](docs/_static/step10_pipeline.svg)

![LCFS cross-sections (one field period)](docs/_static/figures/lcfs_cross_sections.png)

![|B| parity error vs VMEC2000 wout](docs/_static/figures/bmag_parity_error.png)

Figures (top to bottom):
- **Step-10 parity pipeline**: the VMEC-style path we regress against (`bcovar → forces → tomnsps → getfsq`).
- **LCFS cross-sections**: last-closed flux surface slices across one field period.
- **|B| parity error**: pointwise relative error vs a bundled VMEC2000 `wout`.

`vmec-jax` aims to:
- reproduce VMEC2000 equilibria for the same inputs (output parity via `wout_*.nc` regressions),
- expose a clean, composable Python/JAX API (grad/JIT/vmap-ready),
- make sensitivity analysis and optimization workflows first-class (autodiff + implicit differentiation),
- remain hackable and readable as a research codebase.

Project status: validated geometry/field/energy kernels, a working fixed-boundary pipeline,
and an increasingly complete VMEC-style force/residual path. Constraint-force plumbing
(`tcon`/`alias`) is now wired into the parity kernels; **lambda residual parity** and
VMEC-quality fixed-boundary convergence are still in progress (see `CODEX_RESUME.md`).

## Key capabilities

- VMEC-style INDATA parsing and boundary evaluation.
- Differentiable geometry kernel on `(s,θ,ζ)` grids: metrics + Jacobian.
- VMEC-style profiles (pressure / iota / current) and volume integrals.
- Contravariant/covariant magnetic field components and VMEC-normalized magnetic energy `wb`.
- VMEC-style DFT trig/weight tables (`fixaray`) and `tomnsps` transforms for parity work.
- Constraint-force pipeline (`tcon` + `alias`) integrated into force kernels.
- Fixed-boundary solvers:
  - lambda-only solve,
  - full `(R,Z,λ)` energy minimization,
  - L-BFGS variant (no external optimizer dependency).
- Parity tooling vs VMEC2000 `wout_*.nc` (Nyquist fields, scalar integrals, diagnostics figures).
- Step-10 parity (baseline): VMEC-style `forces` + `tomnsps` + `getfsq` scalars
  match the bundled circular tokamak `wout` to a few percent (see
  `examples/validation/vmec_forces_rz_kernel_report.py` and
  `tests/test_step10_residue_getfsq_parity.py`).
- Advanced: implicit differentiation demos (custom VJP) for solver-aware gradients.

## Current parity status (Step-10 scalar residuals)

`vmec-jax` includes a Step-10 parity regression that compares VMEC-style scalar residuals
(`fsqr`, `fsqz`, `fsql`) computed from the ported `bcovar → forces → tomnsps → getfsq`
pipeline against bundled VMEC2000 reference `wout_*.nc` outputs.

The current relative errors are tracked in `docs/validation.rst`. Snapshot from
`examples/validation/step10_getfsq_parity_cases.py` (relative error
`|f̂-f|/max(|f|,ε)`):

| Case | fsqr | fsqz | fsql |
| --- | ---: | ---: | ---: |
| circular_tokamak | 4.9e-2 | 4.6e-2 | 4.8e-3 |
| up_down_asymmetric_tokamak | 4.1e-2 | 1.3e-2 | 3.2e-2 |
| li383_low_res | 1.6e-1 | 1.2e-1 | 1.1e-1 |
| LandremanSenguptaPlunk_section5p3_low_res | 1.1e-1 | 8.9e-2 | 1.6e-2 |

Not yet implemented (planned):
- Full VMEC-quality fixed-boundary convergence (VMEC-style preconditioners + force/residue parity).
- Free-boundary VMEC.
- MPI/parallelization.

## Parity matrix (high level)

Status key: `OK` (covered by tests), `Partial` (matches in some cases / loose tolerances), `Planned`.

| Area | Axisym (ntor=0) | 3D (lasym=F) | 3D (lasym=T) | Notes |
| --- | --- | --- | --- | --- |
| INDATA parsing + boundary | OK | OK | OK | `tests/` + `examples/tutorial/00_*` |
| Geometry (metrics + sqrtg) | OK | OK | OK | Nyquist `gmnc/gmns` parity tests |
| B field (`bsup*`, `bsub*`, `\`|B|\``) | OK | OK | OK | Nyquist parity; figures under `examples/validation/` |
| Energy scalars (`wb`, `wp`, volume) | OK | OK | OK | `tests/test_step10_energy_integrals_parity.py` + `wout.vp` checks |
| `wout` I/O (read + minimal write) | OK | OK | OK | `tests/test_step10_wout_roundtrip.py` |
| Constraint pipeline (`tcon/alias/gcon`) | Partial | Partial | Partial | parity kernels + diagnostics wired |
| Step-10 `forces → tomnsps → getfsq` | Partial | Partial | Partial | scalar parity tracked in `docs/validation.rst` |
| Step-10 `tomnspa` (lasym) blocks | n/a | n/a | Partial | `fsql` is the most sensitive |
| Fixed-boundary solvers | Partial | Partial | Partial | monotone energy decrease; not VMEC-quality yet |
| Implicit differentiation | OK | OK | OK | example coverage; solver parity still WIP |
| Free-boundary VMEC | Planned | Planned | Planned | not implemented |

## Next steps toward full VMEC2000 parity

Concrete milestones (correctness-first):

- Lock down VMEC numerics from the up-to-date sources (STELLOPT/VMEC2000 + VMEC++):
  - keep VMEC-style **DFT with precomputed trig/weight tables** as the canonical transform for parity,
  - verify mode ordering, `mscale/nscale`, and half/full mesh conventions used by `tomnsps`/`alias`.
- Tighten Step-10 scalar parity on 3D cases:
  - isolate which residual blocks dominate the remaining `fsqr/fsqz/fsql` gaps (per-case decomposition by `(m,n)` and by kernel source: `A/B/C` vs constraint terms),
  - use `vmec_jax.vmec_residue.vmec_fsq_sums_from_tomnsps` (and `tests/test_step10_getfsq_block_sums.py`) to attribute scalar changes to individual tomnsps/tomnspa blocks before/after each plumbing tweak,
  - match VMEC’s constraint-force pipeline end-to-end (especially `tcon(js)` from `bcovar/precondn` and the `alias → gcon` operator), since this is a major lever for 3D near-axis behavior.
- Finish the missing VMEC2000 “plumbing” that affects Step-10 scalars:
  - remaining `bcovar` details that influence `forces` (e.g. exact half/full mesh handling for quantities consumed by `forces.f`),
  - confirm axis rules (`jmin1/jmin2/jlam`) and `LCONM1` constraint behavior match `residue.f90` in the converged regime.
- Complete lambda residual parity:
  - finish the VMEC hybrid lambda force path,
  - tighten `fl*` block parity (including `tomnspa` for `lasym=True`).
- Move from “parity kernels” to a VMEC-quality fixed-boundary solver:
  - port the 1D (and later 2D) preconditioners and use them in a Newton / quasi-Newton / preconditioned descent loop that converges comparably to VMEC2000 on the bundled cases,
  - ensure solver stopping criteria and reported diagnostics match VMEC (including how `fsq*` are computed during the iteration history).

## Performance roadmap (JAX-first)

The current code is intentionally explicit and validation-driven. For performance, the biggest wins will come from swapping dense mode-sums for FFT-based spectral transforms and carefully structuring JAX compilation:

- Replace “basis-matrix” Fourier evaluation/projection only **after parity**:
  - VMEC++ uses DFTs with precomputed tables because of grid/weight conventions; FFTs are a later optimization and must reproduce the same scaling (`mscale/nscale`, endpoint weights, `ntheta1/2/3`) exactly.
  - candidate path: factorized DFTs (theta/phi separable) first, then FFT with explicit normalization.
- Fuse and batch the radial work:
  - use `jax.lax.scan` over `s` for recurrences and half-mesh operations to reduce memory pressure and compile time,
  - keep array layouts stable (`(ns, ntheta, nzeta)` contiguous) and avoid Python-side loops inside hot paths.
- Make compilation predictable:
  - enforce static shapes for `(ns, ntheta, nzeta, mpol, ntor)` in the core kernels,
  - use `jax.jit` with `static_argnames` (and small dataclass “static” bundles) to avoid recompiles.
- Solver-level algorithms that map well to JAX:
  - preconditioned nonlinear least-squares (Gauss–Newton / Levenberg–Marquardt) with matrix-free JVP/VJP and iterative linear solves (CG) for inner steps,
  - quasi-Newton (L-BFGS) as a fallback; optionally wire `jaxopt`/`optax` for robust line-searches and schedules while keeping end-to-end differentiability.
- GPU/TPU readiness:
  - avoid host callbacks in hot loops; keep diagnostics optional/off by default,
  - favor XLA-friendly primitives (`lax` control flow, `vmap`, FFTs) and keep `float64` supported (VMEC parity needs it; GPU `x64` can be slower but remains valuable for validation).

## Installation

Create an environment with Python ≥ 3.10.

Regular users (non-editable install):

```bash
git clone https://github.com/uwplasma/vmec_jax.git
cd vmec_jax
python -m pip install -U pip
python -m pip install .
```

Developers (editable install):

```bash
python -m pip install -e .
```

Recommended extras:

```bash
# JAX runtime (CPU)
python -m pip install ".[jax]"

# Read VMEC2000 `wout_*.nc` reference files
python -m pip install ".[netcdf]"

# Publication-ready figures in examples
python -m pip install ".[plots]"

# Build docs locally
python -m pip install ".[docs]"

# Dev tools
python -m pip install -e ".[dev]"
```

VMEC is typically run in float64. Enable x64 for JAX:

```bash
export JAX_ENABLE_X64=1
```

## Quickstart

Run a small validated workflow (inputs + reference `wout` files are bundled under `examples/data/`):

```bash
python examples/tutorial/00_parse_and_boundary.py examples/data/input.LandremanSenguptaPlunk_section5p3_low_res --out boundary.npz --verbose
python examples/tutorial/02_init_guess_and_coords.py examples/data/input.LandremanSenguptaPlunk_section5p3_low_res --out coords_step1.npz --verbose
python examples/tutorial/04_geom_metrics.py examples/data/input.LandremanSenguptaPlunk_section5p3_low_res --out geom_step2.npz --verbose
python examples/tutorial/05_profiles_and_volume.py examples/data/input.LandremanSenguptaPlunk_section5p3_low_res --out profiles_step3.npz --verbose
python examples/tutorial/06_field_and_energy.py examples/data/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/data/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
```

Compatibility wrappers live under `examples/compat/` and forward to `examples/tutorial/`.

## Examples

Examples are organized into:
- `examples/tutorial/`: step-by-step scripts (00–09).
- `examples/validation/`: parity checks vs bundled `wout_*.nc` + reports.
- `examples/visualization/`: plotting + VTK export.
- `examples/gradients/`: autodiff + implicit differentiation demos.
- `examples/solvers/`: solver experiments / convergence scripts.
- `examples/data/`: bundled regression inputs + reference `wout` files.

ParaView export (VTK surface fields + field lines):

```bash
python examples/visualization/vtk_field_and_fieldlines.py examples/data/input.LandremanSenguptaPlunk_section5p3_low_res --hi-res --outdir vtk_out
```

## Documentation

Sphinx docs live in `docs/`.

Build locally:

```bash
python -m sphinx -b html docs docs/_build/html
```

## Testing

```bash
pytest -q
```

If `netCDF4` is not installed, tests requiring `wout_*.nc` I/O are skipped.

## Contributing

Contributions are welcome. Practical ways to help:
- add parity regressions vs VMEC2000 (new cases, tighter tolerances),
- improve kernels (correctness-first; then JIT/vmap performance),
- expand documentation (derivations, conventions, and references),
- add examples that demonstrate differentiability and optimization workflows.

See `docs/contributing.rst` for style and workflow.

## License

MIT. See `LICENSE`.

## References / background

See `docs/references.rst` and the original VMEC literature for algorithmic context.

## Roadmap / step log

The detailed step-by-step porting log and current parity status live in `CODEX_RESUME.md` and `PORTING_NOTES.md`.
