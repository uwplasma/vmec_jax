# Example data

This folder contains VMEC input decks and small checked-in fixtures used by
examples, tests, and documentation.

- `input.*`: VMEC input decks.
- `single_grid/`: fixed-boundary single-grid runtime inputs used by the README,
  docs, and optional cross-implementation comparisons. README runtime inputs are
  normalized to `NS_ARRAY=151`, `FTOL_ARRAY=1e-14`, and `NITER_ARRAY=5000`.
- Large reference WOUT, mgrid, Boozer, and JXB files are ignored by git and are
  fetched on demand with `python tools/fetch_assets.py`. The command verifies
  the release size and SHA-256 recorded in `assets/manifest.json`.

Keep new example inputs small.  Put generated output files in ignored output
directories, not in this data folder.
