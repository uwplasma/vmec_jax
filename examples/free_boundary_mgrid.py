#!/usr/bin/env python
"""Free-boundary equilibrium from an mgrid file: the plasma shape is *found*.

In a fixed-boundary run you prescribe the last closed flux surface.  In a
free-boundary run you prescribe the *coils* instead -- through their currents
(``EXTCUR``) and the vacuum field they produce on a grid (an ``mgrid`` file) --
and VMEC solves for the plasma boundary that balances against that external
field.  Each iteration the NESTOR vacuum solver recomputes the field outside the
plasma; the LCFS at the end is an output, not an input.

This runs the bundled CTH-like stellarator (nfp=5) against ``mgrid_cth_like.nc``.
CLI equivalent: ``vmec examples/data/input.cth_like_free_bdy``.

Physics: CTH-like torsatron, two coil circuits (``EXTCUR = 4700, 1000`` A-turns).
Runtime ~10 s warm (the NESTOR vacuum solve makes free boundary heavier than
fixed boundary).
"""

import dataclasses
import os
from pathlib import Path

import vmex as vj

# --------------------------- parameters ------------------------------------
DATA = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA / "input.cth_like_free_bdy"
MGRID_FILE = DATA / "mgrid_cth_like.nc"       # tabulated vacuum field from the coils
OUT_DIR = Path("output_free_boundary_mgrid")
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"

inp = vj.VmecInput.from_file(INPUT_FILE)
if CI:  # loosen the tolerance slightly for a faster smoke (still fully converges)
    inp = dataclasses.replace(inp, ftol_array=[1e-9], niter_array=[3000])
print(f"free boundary: nfp={inp.nfp}  EXTCUR={list(map(float, inp.extcur[:2]))} A-turns")
print(f"external field: {MGRID_FILE.name}")

# --------------------------- solve (NESTOR vacuum + plasma) -----------------
result = vj.solve_free_boundary(inp, mgrid_path=MGRID_FILE, verbose=not CI)
print(f"\nconverged = {result.converged} after {int(result.iterations)} "
      f"iterations; fsqr = {float(result.fsqr):.3e}")

# --------------------------- write + plot the found equilibrium ------------
wout = vj.wout_from_state(
    inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
    fsql=float(result.fsql), niter=int(result.iterations),
    converged=bool(result.converged))
OUT_DIR.mkdir(parents=True, exist_ok=True)
wout_path = vj.write_wout(OUT_DIR / "wout_cth_like_free_bdy.nc", wout)
print(f"aspect = {float(wout.aspect):.4f}, volume = {float(wout.volume_p):.4f} m^3 "
      "(the boundary was solved for, not prescribed)")
print(f"wrote {wout_path}")

if not CI:
    for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
        print(f"wrote {path}")
