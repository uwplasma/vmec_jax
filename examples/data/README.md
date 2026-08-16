# Example data

This folder contains VMEC input decks and small checked-in fixtures used by
examples, tests, and documentation.

- `input.*`: VMEC input decks.
- `ESSOS_biot_savart_LandremanPaulQA.json`: vacuum coils for the low-resolution
  Landreman--Paul QA boundary.
- `input.LandremanPaul2021_QA_beta2p5_bootstrap` and the matching
  `ESSOS_biot_savart_LandremanPaulQA_beta2p5_bootstrap.json`: a 2.5%-beta QA
  equilibrium with self-consistent bootstrap current for the coupled
  free-boundary optimization example.
- The corresponding `beta0p5` pair is the low-beta, current-oriented fixture
  used for exterior field-line tracing and fixed/free comparison. Its coils
  are reproduced by ESSOS `optimize_coils_finite_beta_vmex.py` from an
  independent vacuum seed and align with the VMEX toroidal-field direction.
- `single_grid/`: fixed-boundary single-grid runtime inputs used by the README,
  docs, and optional cross-implementation comparisons. README runtime inputs are
  normalized to `NS_ARRAY=151`, `FTOL_ARRAY=1e-14`, and `NITER_ARRAY=5000`.
- Large reference WOUT, mgrid, Boozer, and JXB files are ignored by git and are
  fetched on demand with `python tools/fetch_assets.py`. The command verifies
  the release size and SHA-256 recorded in `assets/manifest.json`.
- `single_grid/` copies that duplicate a file here are not shipped in the
  release; `fetch_assets.py` re-creates them from this folder on extraction, so
  the mirrored `mgrid_cth_like_lasym_small.nc` is always the tracked file.

Keep new example inputs small.  Put generated output files in ignored output
directories, not in this data folder.
