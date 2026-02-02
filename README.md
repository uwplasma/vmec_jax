# vmec-jax

Laptop-friendly, end-to-end differentiable (JAX) rewrite of **VMEC2000** (fixed-boundary first).

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
- Step-10 parity (baseline): VMEC-style `forces` + `tomnsps` + `getfsq` scalars (`fsqr/fsqz/fsql`) match the bundled circular tokamak `wout` to a few percent (see `examples/3_Advanced/10_vmec_forces_rz_kernel_report.py` and `tests/test_step10_residue_getfsq_parity.py`).
- Advanced: implicit differentiation demos (custom VJP) for solver-aware gradients.

Not yet implemented (planned):
- Full VMEC-quality fixed-boundary convergence (VMEC-style preconditioners + force/residue parity).
- Free-boundary VMEC.
- MPI/parallelization.

## Installation

Create an environment with Python ≥ 3.10, then install editable:

```bash
pip install -e .
```

Recommended extras:

```bash
# JAX runtime (CPU)
pip install -e .[jax]

# Read VMEC2000 `wout_*.nc` reference files
pip install -e .[netcdf]

# Publication-ready figures in examples
pip install -e .[plots]

# Build docs locally
pip install -e .[docs]

# Dev tools
pip install -e .[dev]
```

VMEC is typically run in float64. Enable x64 for JAX:

```bash
export JAX_ENABLE_X64=1
```

## Quickstart

Run a small validated workflow (inputs are bundled under `examples/`):

```bash
python examples/1_Simple/00_parse_and_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out boundary.npz --verbose
python examples/1_Simple/02_init_guess_and_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out coords_step1.npz --verbose
python examples/2_Intermediate/04_geom_metrics.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out geom_step2.npz --verbose
python examples/2_Intermediate/05_profiles_and_volume.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out profiles_step3.npz --verbose
python examples/2_Intermediate/06_field_and_energy.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
```

Note: top-level scripts `examples/00_...py` etc exist as compatibility wrappers and forward to the categorized folders.

## Examples

Examples are organized into:
- `examples/1_Simple/`: short demos and quick plots.
- `examples/2_Intermediate/`: multi-kernel workflows + parity/diagnostics figures.
- `examples/3_Advanced/`: solver experiments, ParaView export (VTK), sensitivity studies.

ParaView export (VTK surface fields + field lines):

```bash
python examples/3_Advanced/02_vtk_field_and_fieldlines.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --hi-res --outdir vtk_out
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
