#!/usr/bin/env python
"""Finite-beta pressure ramp: beta, Shafranov shift, and Mercier stability.

Raising the plasma pressure raises beta, pushes the magnetic axis outward (the
Shafranov shift), and eventually threatens interchange (Mercier) stability.
This script ramps the pressure scale of a shaped tokamak, hot-restarting each
step from the previous equilibrium, and reads three diagnostics straight from
the wout:

- ``betatotal`` -- the volume-averaged beta;
- ``raxis_cc[0]`` -- the magnetic-axis major radius; its growth over the
  zero-pressure axis is the Shafranov shift;
- ``DMerc`` -- the Mercier criterion profile (``> 0`` is stable); we report its
  interior minimum.

Physics: shaped tokamak, parabolic pressure ``p(s) = PRES_SCALE * (1 - s)``,
NCURR=0.  Runtime a few seconds (warm-restarted ramp).
"""

import dataclasses
import os
from pathlib import Path

import numpy as np

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure"
PRES_MULTIPLIERS = [0.0, 5.0, 10.0, 15.0, 20.0]   # scales the base PRES_SCALE (-> ~1.5% beta)
NS = 25
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if CI:
    NS = 15
    PRES_MULTIPLIERS = [0.0, 10.0, 20.0]

base = vj.VmecInput.from_file(INPUT_FILE)
base = dataclasses.replace(base, ns_array=[NS], ftol_array=[1e-11], niter_array=[5000])
pres0 = float(base.pres_scale)


def _solve(inp, seed):
    res = vj.solve_multigrid(inp, initial_state=seed, verbose=False)
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged))
    return res, wout


# --------------------------- pressure ramp ---------------------------------
print(f"{'pres_scale':>11s} {'beta_tot':>10s} {'R_axis(m)':>10s} "
      f"{'Shafranov':>10s} {'min DMerc':>11s}")
print(f"{'-'*11} {'-'*10} {'-'*10} {'-'*10} {'-'*11}")

seed = None
raxis_ref = None
for mult in PRES_MULTIPLIERS:
    inp = dataclasses.replace(base, pres_scale=pres0 * mult)
    res, wout = _solve(inp, seed)
    seed = res.state  # hot restart the next (higher-pressure) point

    beta = float(wout.betatotal)
    raxis = float(np.asarray(wout.raxis_cc)[0])
    if raxis_ref is None:
        raxis_ref = raxis                       # the zero-pressure axis
    shafranov = raxis - raxis_ref
    dmerc_min = float(np.min(np.asarray(wout.DMerc)[2:-1]))  # interior, skip endpoints
    print(f"{pres0 * mult:11.3e} {beta:10.3e} {raxis:10.4f} "
          f"{shafranov:+10.4f} {dmerc_min:+11.3e}")

print("\nAs pressure rises: beta grows, the magnetic axis shifts outward "
      "(Shafranov), and DMerc (>0 stable) tracks Mercier stability.")
