# Example data

This folder contains VMEC input decks and small checked-in fixtures used by
examples, tests, and documentation.

- `input.*`: VMEC input decks.
- `ESSOS_biot_savart_LandremanPaulQA.json`: vacuum coils for the low-resolution
  Landreman--Paul QA boundary. All three coil files use the public
  `dofs_curves` / `dofs_currents` schema supported by ESSOS 0.16.
- `input.LandremanPaul2021_QA_beta2p5_bootstrap` and the matching
  `ESSOS_biot_savart_LandremanPaulQA_beta2p5_bootstrap.json`: a 2.5%-beta QA
  equilibrium with self-consistent bootstrap current for the coupled
  free-boundary optimization example.
- The corresponding `beta0p5` pair is the low-beta, current-oriented fixture
  used for exterior field-line tracing and fixed/free comparison. Its coils
  are reproduced by ESSOS `optimize_coils_finite_beta_vmex.py` from an
  independent vacuum seed and align with the VMEX toroidal-field direction.
- `input.ncsx_c09r00_free_lowres` + `mgrid_ncsx_c09r00_small.nc`: second
  free-boundary geometry family (NCSX c09r00, nfp=3, the li383-class plasma of
  `input.li383_low_res`).  The deck is the published c09r00 free-boundary
  input (PrincetonUniversity/STELLOPT `BENCHMARKS/DIAGNO_TEST/input.ncsx`)
  with only the resolution reduced to NS 9/15/25, MPOL=7, NTOR=6.  The mgrid
  was generated with MAKEGRID `xgrid` (STELLOPT `v6.5.0-42-g9177f58`) from
  `BENCHMARKS/FIELDLINES_TEST/coils.NCSX` (added upstream in `2080076f`,
  sha256 `3c429da06f4c062887a497a16e2d2bd10f0ecb0b8858c252698631f3853da428`):
  scaled mode, stellarator symmetric, R [0.75, 2.0] x Z [-0.8, 0.8] m,
  ir=jz=28, kp=24 (= the deck's NZETA), ten coil groups (ModA-C, PF1-6, TF).
  VMEC2000 and vmex both converge the deck cleanly to fsq < 1e-10;
  `tests/test_ncsx_free_boundary_parity.py` pins the parity digest.
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
