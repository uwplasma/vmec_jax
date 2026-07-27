#!/usr/bin/env python
"""Free-boundary pressure scan from ESSOS coils via an in-memory mgrid.

vmex is coil-agnostic: coils live in ESSOS (``essos.coils.Coils``), and the
free-boundary solver consumes only a magnetic-field grid.  Here we take the
Landreman & Paul (2021) precise-QA coil set as optimized in ESSOS
(github.com/uwplasma/ESSOS, bundled as a 3 KB JSON), tabulate its Biot-Savart
field once onto a cylindrical grid bracketing the plasma
(``essos.coils.Coils.to_mgrid``), and read it straight back into a
:class:`vmex.core.mgrid.MgridField` -- no standalone mgrid file left on disk,
no ESSOS import inside the solve.  That ``MgridField`` supplies the external
field for every NESTOR vacuum iteration.

Holding the coil currents fixed, we ramp a parabolic pressure
``p(s) = PRES_SCALE(1-s)`` and *calibrate* PRES_SCALE at each step so the
converged equilibrium's actual volume-average beta (wout ``betatotal``) lands on
0, 1, 2, 3 % -- a nominal pressure is not enough, because at fixed coil currents
the plasma dilates and shifts as beta rises, feeding back on <B^2>.  Each
pressure step warm-starts from the previous accepted boundary (how experiments
ramp, and much more robust than re-solving from the vacuum guess).

Physics: nfp=2 precise-QA plasma held by 16 modular coils; watch the
Shafranov shift (axis moves outboard) and the LCFS response as beta rises.
Runtime: ~4 min for the full scan (one NESTOR free-boundary solve per
calibration attempt); the CI budget solves a single beta point coarsely.
"""

import dataclasses
import os
import tempfile
from pathlib import Path

import numpy as np

import vmex as vj

# --------------------------- parameters ------------------------------------
DATA = Path(__file__).resolve().parent / "data"
COILS_JSON = DATA / "ESSOS_biot_savart_LandremanPaulQA.json"  # ESSOS coil DOFs
INPUT_FILE = DATA / "input.LandremanPaul2021_QA_lowres"       # plasma seed deck
OUT_DIR = Path("output_free_boundary_essos_coils")
TARGET_BETAS = [0.0, 1.0, 2.0, 3.0]   # actual volume-average beta targets [%]
BETA_TOL = 0.15                       # accept |betatotal - target| below this [%]
SLOPE = 1.45e-3                       # first-guess beta[%] per unit PRES_SCALE
NS, MPOL, NTOR = 51, 5, 5
NITER, FTOL = 20000, 1e-10
PHIEDGE = -0.025                      # toroidal flux matching the coil field [Wb]
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if CI:  # smoke budget: one finite-beta point on a coarse grid
    TARGET_BETAS, NS, NITER, FTOL = [1.0], 16, 4000, 1e-8

# --------------------------- coils -> external field ------------------------
from essos.coils import Coils  # noqa: E402 (optional heavy import)

if hasattr(Coils, "from_json"):
    coils = Coils.from_json(str(COILS_JSON))
else:  # legacy ESSOS predating the Coils.from_json classmethod
    from essos.coils import Coils_from_json

    coils = Coils_from_json(str(COILS_JSON))
currents = np.asarray(coils.currents)  # symmetry-expanded physical currents [A]
mean_current = float(np.mean(np.abs(currents)))
print(f"ESSOS coils: {currents.shape[0]} filaments after nfp={coils.nfp}/stellsym "
      f"expansion, I ~ {mean_current:,.0f} A")

# Tabulate the coil field once onto a cylindrical grid bracketing the plasma
# (R in [0.45, 1.55], Z in [-0.6, 0.6]) and read it straight back as an
# MgridField -- the external field vmex's free-boundary solver consumes.
with tempfile.TemporaryDirectory() as _tmp:
    _mgrid_path = Path(_tmp) / "essos_LP_QA_mgrid.nc"
    coils.to_mgrid(str(_mgrid_path), nr=96, nphi=32, nz=96,
                   rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6)
    coil_field = vj.MgridField.from_mgrid_data(vj.read_mgrid(_mgrid_path))

# --------------------------- plasma deck ------------------------------------
# The fixed-boundary LP-QA deck only seeds the initial guess; truncate it to
# the scan resolution and switch on free boundary with the direct coil field.
inp = vj.VmecInput.from_file(INPUT_FILE)
k = inp.ntor - NTOR
base = dataclasses.replace(
    inp, lfreeb=True, mgrid_file="essos_coils(direct)", mpol=MPOL, ntor=NTOR,
    rbc=inp.rbc[k:k + 2 * NTOR + 1, :MPOL], zbs=inp.zbs[k:k + 2 * NTOR + 1, :MPOL],
    rbs=inp.rbs[k:k + 2 * NTOR + 1, :MPOL], zbc=inp.zbc[k:k + 2 * NTOR + 1, :MPOL],
    raxis_c=inp.raxis_c[:NTOR + 1], zaxis_s=inp.zaxis_s[:NTOR + 1],
    raxis_s=inp.raxis_s[:NTOR + 1], zaxis_c=inp.zaxis_c[:NTOR + 1],
    phiedge=PHIEDGE, ns_array=[NS], niter_array=[NITER], ftol_array=[FTOL],
    pmass_type="power_series", am=[1.0, -1.0] + [0.0] * 19)  # p = PS * (1 - s)


def warm_boundary(inp_i, wout):
    """Seed the next step's boundary/axis guess from an accepted LCFS."""
    rbc, zbs = np.zeros_like(inp_i.rbc), np.zeros_like(inp_i.zbs)
    n_in = (np.asarray(wout.xn, dtype=float) / float(wout.nfp)).astype(int)
    for j, (m, n) in enumerate(zip(np.asarray(wout.xm, dtype=int), n_in)):
        if m < inp_i.mpol and abs(n) <= inp_i.ntor:
            rbc[n + inp_i.ntor, m] = np.asarray(wout.rmnc)[-1][j]
            zbs[n + inp_i.ntor, m] = np.asarray(wout.zmns)[-1][j]
    nax = inp_i.ntor + 1
    return dataclasses.replace(inp_i, rbc=rbc, zbs=zbs,
                               raxis_c=np.asarray(wout.raxis_cc)[:nax],
                               zaxis_s=np.asarray(wout.zaxis_cs)[:nax])


# --------------------------- calibrated pressure ramp -----------------------
print(f"\n{'nominal':>8s} {'PRES_SCALE':>11s} {'actual beta':>12s} {'iters':>6s} "
      f"{'fsq':>9s} {'aspect':>7s} {'axis R':>8s}")
rows, current = [], base
for target in TARGET_BETAS:
    ps = target / SLOPE
    for attempt in range(3):  # solve, read actual beta, rescale (~linear)
        inp_i = dataclasses.replace(current, pres_scale=ps)
        res = vj.solve_free_boundary(inp_i, external_field=coil_field,
                                     error_on_no_convergence=False)
        wout = vj.wout_from_state(
            inp=inp_i, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
            fsql=float(res.fsql), niter=int(res.iterations),
            converged=bool(res.converged))
        beta = 100.0 * float(wout.betatotal)
        if target == 0.0 or abs(beta - target) <= BETA_TOL:
            break
        ps *= target / max(beta, 1e-6)  # pressure rescale toward the target
    fsq = float(res.fsqr) + float(res.fsqz) + float(res.fsql)
    axis_r = float(np.sum(np.asarray(wout.raxis_cc)))  # axis R at phi = 0
    print(f"{target:7.1f}% {ps:11.1f} {beta:11.3f}% {int(res.iterations):6d} "
          f"{fsq:9.1e} {float(wout.aspect):7.3f} {axis_r:8.4f}")
    rows.append((target, ps, beta, axis_r, wout))
    current = warm_boundary(current, wout)  # ramp continuation

dev = max(abs(beta - target) for target, _ps, beta, _ar, _w in rows)
print(f"\nactual betatotal within {dev:.3f}% of every nominal target (tolerance {BETA_TOL}%)")
if len(rows) > 1:
    shift = rows[-1][3] - rows[0][3]
    print(f"magnetic axis Shafranov-shifted {shift * 100:+.2f} cm at fixed coil currents")

# --------------------------- figure (skipped in CI) -------------------------
if not CI:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vmex.core.plotting import surface_rz

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.8), dpi=110, width_ratios=[1.1, 1.0])
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    shades = ["#b5cde3", "#6d9dc9", "#2e6da4", "#0c3766"]  # light -> dark = rising beta
    for (_target, _ps, beta, axis_r, wout), color in zip(rows, shades):
        R, Z = surface_rz(wout, s_index=-1, theta=theta, phi=np.array([0.0]))
        ax.plot(R[:, 0], Z[:, 0], color=color, lw=2.0,
                label=f"$\\langle\\beta\\rangle$ = {beta:.2f}%")
        ax.plot(axis_r, 0.0, "o", ms=6, color=color)
        ax2.plot(beta, 100.0 * (axis_r - rows[0][3]), "o", ms=7, color=color, zorder=2)
    ax.set(xlabel="R [m]", ylabel="Z [m]", title="LCFS and magnetic axis at $\\phi=0$")
    ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.5)
    ax2.plot([r[2] for r in rows], [100.0 * (r[3] - rows[0][3]) for r in rows],
             "-", color="#9a9a9a", lw=1.0, zorder=1)
    ax2.set(xlabel="actual $\\langle\\beta\\rangle$ [%]",
            ylabel="axis Shafranov shift at $\\phi=0$ [cm]", title="Shafranov shift")
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Free-boundary LP-QA from ESSOS coils (tabulated to an in-memory mgrid)")
    fig.tight_layout()
    fig_path = OUT_DIR / "essos_beta_scan.png"
    fig.savefig(fig_path)
    print(f"wrote {fig_path}")
