#!/usr/bin/env python
"""Profile representations: power series vs cubic spline (same physics).

VMEC prescribes the pressure and either the rotational transform (``NCURR=0``)
or the toroidal current (``NCURR=1``) as radial profiles in the normalized flux
``s``.  Each profile can be given as polynomial coefficients (``power_series``)
*or* as spline knots (``cubic_spline`` / ``akima_spline``).  This script solves
the same shaped tokamak twice -- once with power-series profiles, once with the
cubic-spline knots sampled from those very polynomials -- and shows the two
converge to the same equilibrium (aspect, volume, beta agree).  That is the
point: the representation is a modelling convenience, not different physics.

The base deck (``input.shaped_tokamak_pressure``) uses ``NCURR=0`` with a
parabolic pressure ``p(s) = PRES_SCALE * (1 - s)`` and a linear iota
``iota(s) = 1.05 - 0.35 s``.  The commented ``NCURR=1`` block at the bottom
shows the current-prescribed alternative.

Physics: shaped tokamak, finite beta (~few %).  Runtime a few seconds warm.
"""

import dataclasses
import os
from pathlib import Path

import numpy as np

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure"
KNOTS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])       # spline sample locations in s
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"   # smoke-test mode


def _polyval(coeffs, s):
    """VMEC power-series convention: profile(s) = sum_k coeffs[k] * s**k."""
    return sum(c * s**k for k, c in enumerate(coeffs))


def _terms(coeffs):
    """The meaningful (non-zero-padded) power-series coefficients as floats."""
    arr = np.asarray(coeffs, dtype=float)
    last = int(np.max(np.nonzero(arr))) if np.any(arr) else 0
    return arr[: last + 1].tolist()


def _solve(inp, label):
    if CI:
        inp = dataclasses.replace(inp, ns_array=[13], ftol_array=[1e-10],
                                  niter_array=[3000])
    res = vj.solve_multigrid(inp, verbose=False)
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged))
    print(f"  {label:14s} converged={bool(res.converged)!s:5s} "
          f"aspect={float(wout.aspect):.4f} volume={float(wout.volume_p):.4f} "
          f"beta={float(wout.betatotal):.4e}")
    return wout


# --------------------------- power-series baseline -------------------------
inp_power = vj.VmecInput.from_file(INPUT_FILE)
print(f"power-series profiles: AM={_terms(inp_power.am)}  AI={_terms(inp_power.ai)}")
wout_power = _solve(inp_power, "power_series")

# --------------------------- equivalent spline knots -----------------------
# Sample the SAME polynomials at the knot locations, then hand VMEC the knots.
am_knots = _polyval(np.asarray(inp_power.am), KNOTS)
ai_knots = _polyval(np.asarray(inp_power.ai), KNOTS)
inp_spline = dataclasses.replace(
    inp_power,
    pmass_type="cubic_spline", am_aux_s=KNOTS.copy(), am_aux_f=am_knots,
    piota_type="cubic_spline", ai_aux_s=KNOTS.copy(), ai_aux_f=ai_knots,
)
print(f"cubic-spline profiles: AM_AUX_F={np.round(am_knots, 4).tolist()}  "
      f"AI_AUX_F={np.round(ai_knots, 4).tolist()}")
wout_spline = _solve(inp_spline, "cubic_spline")

# --------------------------- compare ---------------------------------------
d_aspect = abs(float(wout_power.aspect) - float(wout_spline.aspect))
d_beta = abs(float(wout_power.betatotal) - float(wout_spline.betatotal))
print(f"\nrepresentation-independent: |d aspect| = {d_aspect:.2e}, "
      f"|d beta| = {d_beta:.2e} (the two profile forms describe one equilibrium)")

# --------------------------- NCURR=1 alternative (current-prescribed) -------
# To prescribe the toroidal current instead of iota, set NCURR=1 and give a
# current profile.  The equivalent in-code edit:
#
#   inp_current = dataclasses.replace(
#       inp_power, ncurr=1, pcurr_type="power_series",
#       ac=[1.0, -1.0],          # I'(s) shape; total set by CURTOR
#       curtor=<total toroidal current in A>)
#   wout_current = _solve(inp_current, "ncurr=1")
#
# VMEC then solves for iota self-consistently from the prescribed current.
