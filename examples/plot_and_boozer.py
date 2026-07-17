#!/usr/bin/env python
"""All the built-in diagnostics: solve -> every ``plot_wout`` figure -> Boozer.

vmex ships its plotting and its Boozer transform in the box, so a single
converged equilibrium gives you the whole diagnostic set with no external
tooling.  This script walks the two calls that matter:

- ``vj.plot_wout`` writes the five standard figures (flux-surface summary,
  nested cross-sections, |B| on a surface, the radial profiles, and a 3D
  render) and returns ``{key: path}``;
- ``vj.run_booz_xform`` + ``vj.plot_boozmn`` transform to straight-field-line
  Boozer coordinates and plot the |B| spectrum on the LCFS -- the view used to
  judge quasisymmetry.

CLI equivalent: ``vmec examples/data/input.li383_low_res --plot --booz``.

Physics: LI383 (NCSX-class, nfp=3), zero pressure.  Expected runtime a few
seconds warm; the Boozer step needs the optional ``booz_xform_jax`` package and
is skipped with a message if it is absent.
"""

import dataclasses
import os
from pathlib import Path

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.li383_low_res"
OUT_DIR = Path("output_plot_and_boozer")
WHICH = ("summary", "surfaces", "modB", "profiles", "3d")  # all plot_wout kinds
RUN_BOOZER = True                                          # optional dep
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"         # smoke-test mode

# --------------------------- solve a small equilibrium ---------------------
inp = vj.VmecInput.from_file(INPUT_FILE)
if CI:  # single coarse grid keeps the smoke test to a few seconds
    inp = dataclasses.replace(inp, ns_array=[13], ftol_array=[1e-8],
                              niter_array=[2000])
result = vj.solve_multigrid(inp, verbose=not CI)
wout = vj.wout_from_state(
    inp=inp, state=result.state,
    fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
    niter=int(result.iterations), converged=bool(result.converged),
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
wout_path = vj.write_wout(OUT_DIR / "wout_li383_low_res.nc", wout)
print(f"converged = {result.converged}; wrote {wout_path}")

# --------------------------- every wout figure -----------------------------
# plot_wout accepts a WoutData or a path and returns {key: written_png_path}.
figures = vj.plot_wout(wout_path, OUT_DIR, which=WHICH)
for key, path in figures.items():
    print(f"  [{key:9s}] {path}")

# --------------------------- Boozer spectrum on the LCFS -------------------
# booz_xform_jax is optional; guard the import so the core workflow always runs.
if RUN_BOOZER:
    try:
        boozmn_path = vj.run_booz_xform(wout_path, outdir=OUT_DIR)
        print(f"wrote {boozmn_path}")
        for key, path in vj.plot_boozmn(boozmn_path, OUT_DIR).items():
            print(f"  [booz:{key:9s}] {path}")
    except ImportError as exc:
        print(f"skipping Boozer step ({exc}); pip install booz_xform_jax")
