# vmec-jax

Laptop-friendly, end-to-end differentiable (JAX) rewrite of **VMEC2000** (fixed-boundary first).

![VMEC Step-10 parity pipeline](docs/_static/step10_pipeline.svg)

![LCFS cross-sections (one field period)](docs/_static/figures/lcfs_cross_sections.png)

![|B| parity error vs VMEC2000 wout](docs/_static/figures/bmag_parity_error.png)

`vmec-jax` aims to:
- reproduce VMEC2000 equilibria for the same inputs (output parity via `wout_*.nc` regressions),
- expose a clean, composable Python/JAX API (grad/JIT/vmap-ready),
- make sensitivity analysis and optimization workflows first-class (autodiff + implicit differentiation),
- remain hackable and readable as a research codebase.

Project status: this repo contains validated geometry/field/energy kernels and early fixed-boundary solvers.
Force/residue parity (`fsqr/fsqz/fsql`) is under active development (see `CODEX_RESUME.md`).

## Key capabilities

- VMEC-style INDATA parsing and boundary evaluation.
- Differentiable geometry kernel on `(s,θ,ζ)` grids: metrics + Jacobian.
- VMEC-style profiles (pressure / iota / current) and volume integrals.
- Contravariant/covariant magnetic field components and VMEC-normalized magnetic energy `wb`.
- Fixed-boundary solvers:
  - lambda-only solve,
  - full `(R,Z,λ)` energy minimization,
  - L-BFGS variant (no external optimizer dependency).
- Parity tooling vs VMEC2000 `wout_*.nc` (Nyquist fields, scalar integrals, diagnostics figures).
- Step-10 parity (baseline): VMEC-style `forces` + `tomnsps` + `getfsq` scalars (`fsqr/fsqz/fsql`) match the bundled circular tokamak `wout` to a few percent (see `examples/validation/vmec_forces_rz_kernel_report.py` and `tests/test_step10_residue_getfsq_parity.py`).
- Advanced: implicit differentiation demos (custom VJP) for solver-aware gradients.

## Current parity status (Step-10 scalar residuals)

`vmec-jax` includes a Step-10 parity regression that compares VMEC-style scalar residuals
(`fsqr`, `fsqz`, `fsql`) computed from the ported `bcovar → forces → tomnsps → getfsq`
pipeline against bundled VMEC2000 reference `wout_*.nc` outputs.

The current relative errors (as of the latest CI run) are summarized in `docs/validation.rst`.

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
| B field (`bsup*`, `bsub*`, `|B|`) | OK | OK | OK | parity figures under `examples/validation/` |
| Energy scalars (`wb`, `wp`, volume) | OK | OK | OK | regression tests vs bundled `wout` |
| `wout` I/O (read + minimal write) | OK | OK | OK | `tests/test_step10_wout_roundtrip.py` |
| Step-10 `forces → tomnsps → getfsq` | Partial | Partial | Partial | scalar parity tracked in `docs/validation.rst` |
| Step-10 `tomnspa` (lasym) blocks | n/a | n/a | Partial | `fsql` is the most sensitive |
| Fixed-boundary solvers | Partial | Partial | Partial | monotone energy decrease; not VMEC-quality yet |
| Implicit differentiation | OK | OK | OK | example coverage; solver parity still WIP |
| Free-boundary VMEC | Planned | Planned | Planned | not implemented |

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

Note: top-level scripts `examples/00_...py` etc exist as compatibility wrappers and forward to `examples/tutorial/`.

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
