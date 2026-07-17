#!/usr/bin/env python
"""Fixed-boundary VMEC run: input file -> solve -> wout -> plots -> Boozer.

The three steps every new user needs: read an ``&INDATA`` file, converge the
equilibrium on the NS_ARRAY multigrid ladder (VMEC2000-style progress
printing), and write/plot the results.  CLI equivalent of this script:
``vmec examples/data/input.li383_low_res --booz``.

Physics: the LI383 (NCSX-class, nfp=3) stellarator boundary at low
resolution, zero pressure.  Expected runtime: ~1 min on a laptop CPU on the
first run (XLA compilation, cached persistently), a few seconds afterwards.
Achieved: converges to FTOL = 1e-13; the wout scalars printed at the end
(aspect ~ 4.4, volume ~ 2.96 m^3) match VMEC2000 to Appendix-A tolerances.
"""

import dataclasses
import os
from pathlib import Path

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.li383_low_res"
OUT_DIR = Path("output_fixed_boundary_run")
RUN_BOOZER = True          # Boozer spectrum via booz_xform_jax (optional dep)
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"  # smoke-test mode

# --------------------------- read the input --------------------------------
inp = vj.VmecInput.from_file(INPUT_FILE)
if CI:  # reduced budget for the CI smoke test
    inp = dataclasses.replace(inp, ns_array=[13], ftol_array=[1e-8],
                              niter_array=[2000])
print(f"input: {INPUT_FILE.name}")
print(f"  nfp={inp.nfp}  mpol={inp.mpol}  ntor={inp.ntor}  lasym={inp.lasym}")
print(f"  ns_array={list(map(int, inp.ns_array))}  "
      f"ftol_array={[f'{f:.0e}' for f in inp.ftol_array]}")
print(f"  major radius RBC(0,0) = {inp.rbc[inp.ntor, 0]:.3f} m, "
      f"phiedge = {inp.phiedge:.3f} Wb")

# --------------------------- solve ------------------------------------------
result = vj.solve_multigrid(inp, verbose=True)  # prints VMEC2000-format tables
print(f"\nconverged = {result.converged} after {int(result.iterations)} "
      f"iterations; fsqr = {float(result.fsqr):.3e}, "
      f"fsqz = {float(result.fsqz):.3e}, fsql = {float(result.fsql):.3e}")

# --------------------------- write the wout file ----------------------------
wout = vj.wout_from_state(
    inp=inp, state=result.state,
    fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
    niter=int(result.iterations), converged=bool(result.converged),
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
wout_path = vj.write_wout(OUT_DIR / "wout_li383_low_res.nc", wout)
print(f"\nwout scalars: aspect = {float(wout.aspect):.4f}, "
      f"volume = {float(wout.volume_p):.4f} m^3, "
      f"B0 = {float(wout.b0):.4f} T, betatotal = {float(wout.betatotal):.3e}")
print(f"wrote {wout_path}")

# --------------------------- plots ------------------------------------------
figures = vj.plot_wout(wout_path, OUT_DIR)  # summary/surfaces/modB/profiles/3d
for key, path in figures.items():
    print(f"wrote {path}")

# --------------------------- Boozer spectrum (optional) ---------------------
if RUN_BOOZER and not CI:
    try:
        boozmn_path = vj.run_booz_xform(wout_path, outdir=OUT_DIR)
        for key, path in vj.plot_boozmn(boozmn_path, OUT_DIR).items():
            print(f"wrote {path}")
    except ImportError as exc:
        print(f"skipping Boozer step ({exc}); pip install booz_xform_jax")
