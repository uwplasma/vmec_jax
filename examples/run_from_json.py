#!/usr/bin/env python
"""VMEC++-style JSON input: convert an &INDATA deck, round-trip, solve.

vmex reads *both* the classic Fortran ``&INDATA`` namelist and the JSON
schema used by VMEC++ (``vmecpp.VmecInput``), and can write the JSON form.  So
it is a drop-in for either ecosystem: ``vmec input.json`` and
``vmec input.circular_tokamak`` both work, and this script shows the conversion
and that the two representations describe the same equilibrium.

Steps: read an INDATA deck -> ``inp.to_json`` -> read the JSON back with
``VmecInput.from_file`` (suffix/`{`-autodetected) -> solve -> compare.

Physics: circular tokamak (nfp=1), zero pressure.  Runtime a couple of seconds.
"""

import dataclasses
import os
from pathlib import Path

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.circular_tokamak"
OUT_DIR = Path("output_run_from_json")
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"


def _solve(inp):
    if CI:
        inp = dataclasses.replace(inp, ns_array=[15], ftol_array=[1e-10],
                                  niter_array=[2000])
    res = vj.solve_multigrid(inp, verbose=False)
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged))
    return wout


# --------------------------- read INDATA, write JSON -----------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
inp_indata = vj.VmecInput.from_file(INPUT_FILE)
json_path = inp_indata.to_json(OUT_DIR / "circular_tokamak.json")
print(f"read {INPUT_FILE.name} (&INDATA) -> wrote {json_path} (VMEC++ JSON)")

# --------------------------- read the JSON back ----------------------------
# from_file dispatches on the .json suffix (or a leading '{') to the JSON parser.
inp_json = vj.VmecInput.from_file(json_path)
print(f"read {json_path.name} back: nfp={inp_json.nfp} mpol={inp_json.mpol} "
      f"ntor={inp_json.ntor} lasym={inp_json.lasym}")

# --------------------------- solve both, compare ---------------------------
wout_indata = _solve(inp_indata)
wout_json = _solve(inp_json)
d_aspect = abs(float(wout_indata.aspect) - float(wout_json.aspect))
print(f"aspect: INDATA={float(wout_indata.aspect):.6f}  "
      f"JSON={float(wout_json.aspect):.6f}  |diff|={d_aspect:.2e}")

vj.write_wout(OUT_DIR / "wout_circular_tokamak.nc", wout_json)
print(f"wrote {OUT_DIR / 'wout_circular_tokamak.nc'} from the JSON input "
      "(the two input formats describe one equilibrium)")
