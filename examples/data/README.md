# Example data

This folder contains VMEC input decks and small checked-in fixtures used by
examples, tests, and documentation.

- `input.*`: VMEC input decks.
- `ESSOS_biot_savart_LandremanPaulQA.json`: vacuum coils for the low-resolution
  Landreman--Paul QA boundary.
- `ESSOS_biot_savart_LandremanPaulQA_finite_beta.json`: coils optimized against
  the same boundary at 2% beta using virtual casing.  On an independent 24 x 24
  surface grid they give area-weighted RMS `B.n/B = 0.068%` and normalized
  total-pressure jump `0.094%` (maximum values `0.236%` and `0.428%`).
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
