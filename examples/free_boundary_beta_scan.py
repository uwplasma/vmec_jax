#!/usr/bin/env python
"""Free-boundary pressure scan: beta from 0 to ~5% at fixed coil currents.

A free-boundary plasma responds to pressure both internally (Shafranov shift)
and at its edge -- the last closed flux surface moves as the plasma pushes
against the fixed external (coil) field.  This ramps the pressure of the
CTH-like free-boundary case, holding the coil currents (``EXTCUR``) fixed, and
reports beta and the plasma volume at each step.

Unlike the fixed-boundary pressure scan (``finite_beta_scan.py``), here the
boundary is recomputed by the NESTOR vacuum solve at every pressure, so each
point is a full free-boundary solve.  Each converged state hot-starts the next
pressure point.

Physics: CTH-like torsatron (nfp=5), parabolic pressure, reaching beta ~ 2.6%.
Heavier than the fixed-boundary scan (one NESTOR solve per point).
"""

import dataclasses
import os
from pathlib import Path

import vmex as vj

# --------------------------- parameters ------------------------------------
DATA = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA / "input.cth_like_free_bdy"
MGRID_FILE = DATA / "mgrid_cth_like.nc"
PRES_SCALES = [0.0, 2000.0, 4000.0, 6000.0]   # Pa scale of the parabolic pressure
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if CI:
    PRES_SCALES = [0.0, 3000.0, 6000.0]

base = vj.VmecInput.from_file(INPUT_FILE)
# parabolic pressure p(s) = PRES_SCALE * (1 - s); coil currents held fixed
base = dataclasses.replace(
    base, ftol_array=[1e-9], niter_array=[3000],
    pmass_type="power_series", am=[1.0, -1.0] + [0.0] * (len(base.am) - 2))

# --------------------------- pressure ramp ---------------------------------
print(f"free-boundary beta scan (EXTCUR held at {list(map(float, base.extcur[:2]))} A-turns)")
print(f"\n{'pres_scale':>11s} {'beta_tot':>10s} {'volume(m^3)':>12s} {'iters':>6s}")
print(f"{'-'*11} {'-'*10} {'-'*12} {'-'*6}")

betas = []
state = None
for ps in PRES_SCALES:
    inp = dataclasses.replace(base, pres_scale=ps)
    res = vj.solve_free_boundary(
        inp, mgrid_path=MGRID_FILE, initial_state=state,
        error_on_no_convergence=False)
    state = res.state
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged), vacuum_output=res.vacuum)
    beta = float(wout.betatotal)
    betas.append(beta)
    print(f"{ps:11.1f} {beta:10.3e} {float(wout.volume_p):12.4f} {int(res.iterations):6d}")

print(f"\nbeta ramps 0 -> {max(betas) * 100:.1f}% at fixed coil currents; the "
      "free boundary is re-solved (NESTOR) at every pressure.")
