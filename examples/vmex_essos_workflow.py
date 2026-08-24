#!/usr/bin/env python
"""Round trip across the two vmex--ESSOS Python seams.

The seams are directional, and this script walks both of them once:

1. **vmex -> ESSOS.**  ``vj.essos_vmec_field`` hands a solved equilibrium
   (in memory, or a ``wout_*.nc`` path) to ``essos.fields.Vmec``.  We check
   the tables crossed intact by rebuilding the LCFS from ESSOS' ``to_xyz``,
   then let ESSOS give something back: a field-line trace whose measured
   ``d(theta)/d(phi)`` is the rotational transform, compared against the wout
   ``iotaf`` vmex computed independently.
2. **ESSOS -> vmex.**  ``vj.MgridField.from_coils`` tabulates an ESSOS coil
   set's Biot-Savart field onto a cylindrical grid, giving the external field
   a free-boundary solve consumes.  We measure the tabulation error against
   direct Biot-Savart, then solve a vacuum free boundary with it -- the LCFS
   is found by the coils rather than prescribed.

Then step 1 runs again on the free-boundary equilibrium, closing the loop:
vmex -> ESSOS -> vmex -> ESSOS.

vmex owns the equilibrium, ESSOS owns coils and tracing, and neither imports
the other: the equilibrium crosses as a wout, the coil field crosses as a
tabulated grid.  Only the released ESSOS surface is used
(``Coils``, ``fields.Vmec``, ``fields.BiotSavart``, ``dynamics.Tracing``).

The two solves share a geometry and not a field scale: the bundled coil set
was optimized for the Landreman & Paul (2021) precise-QA boundary, so the
free-boundary LCFS should land close to the prescribed one, but PHIEDGE is
the flux those coil currents hold, which is not the deck's.  Both solves are
vacuum (AM = 0, CURTOR = 0).  For a pressure scan on the coil-held plasma see
``free_boundary_essos_coils.py``.

Runtime: 2--4 min for the full script (two NS ladders), 1--2 min under
``VMEX_EXAMPLES_CI=1`` (single coarse grid, no figure).
"""

import dataclasses
import os
from pathlib import Path

import numpy as np

import vmex as vj

# --------------------------- parameters ------------------------------------
DATA = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA / "input.LandremanPaul2021_QA_lowres"          # nfp=2 precise QA
COILS_JSON = DATA / "ESSOS_biot_savart_LandremanPaulQA.json"     # 16 ESSOS coils
OUT_DIR = Path("output_vmex_essos_workflow")
NS_LADDER = [16, 31, 50]
FTOL = [1e-8, 1e-11, 1e-11]
S_TRACE = 0.5              # flux surface carrying the traced field lines
N_LINES = 4                # field lines, launched at equally spaced theta
TRACE_TIME = 200.0         # field-line integration parameter (dx/dt = B)
MPOL, NTOR = 5, 5          # Fourier resolution of both solves
PHIEDGE_FREE = -0.025      # toroidal flux the bundled coil currents hold [Wb]
GRID = dict(rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6, ir=96, jz=96, kp=32)
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if CI:  # smoke budget: one coarse radial grid per solve, no figure
    NS_LADDER, FTOL = [16], [1e-10]

import jax  # noqa: E402  (after vmex, which configures the JAX environment)
import jax.numpy as jnp  # noqa: E402
from essos.coils import Coils  # noqa: E402  (optional heavy import)
from essos.dynamics import Tracing  # noqa: E402

from vmex.core.plotting import surface_rz  # noqa: E402


def solve_and_export(inp, external_field=None):
    """Solve the NS ladder and return ``(WoutData, SolveResult)``.

    ``external_field`` selects the free-boundary lane; ``res.vacuum`` is
    ``None`` for a fixed-boundary solve, which ``wout_from_state`` accepts.
    """
    if external_field is None:
        res = vj.solve_multigrid(inp, raise_on_max_iterations=False)
    else:
        res = vj.solve_free_boundary_multigrid(
            inp, external_field=external_field, raise_on_max_iterations=False)
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged), vacuum_output=res.vacuum)
    return wout, res


def report_handoff(wout, label):
    """Hand ``wout`` to ESSOS and report what ESSOS makes of it.

    Two independent readings of the same equilibrium: the LCFS rebuilt from
    ESSOS' ``to_xyz`` against vmex's own Fourier sum (a pure transfer check,
    so it must hold to machine precision), and the rotational transform
    measured from an ESSOS field-line trace against the wout ``iotaf`` vmex
    computed from the force balance.  Returns the traced trajectories.
    """
    field = vj.essos_vmec_field(wout)

    theta = np.linspace(0.0, 2.0 * np.pi, 33)[:-1]
    phi = np.array([0.0, 0.5 * np.pi / float(wout.nfp)])
    R, Z = surface_rz(wout, s_index=-1, theta=theta, phi=phi)
    corners = np.stack([np.ones(theta.size * phi.size),
                        np.repeat(theta, phi.size), np.tile(phi, theta.size)], axis=1)
    xyz = np.asarray(jax.vmap(field.to_xyz)(jnp.asarray(corners)))
    lcfs_error = float(
        np.max(np.abs(np.hypot(xyz[:, 0], xyz[:, 1]).reshape(R.shape) - R))
        + np.max(np.abs(xyz[:, 2].reshape(Z.shape) - Z)))

    # ESSOS integrates the field line in flux coordinates, where ds/dt is
    # identically zero and iota is the slope of theta against phi.  A least
    # squares fit over the whole trace averages the per-period ripple out.
    seeds = jnp.stack([jnp.full((N_LINES,), S_TRACE),
                       jnp.linspace(0.0, 2.0 * jnp.pi, N_LINES, endpoint=False),
                       jnp.zeros(N_LINES)], axis=1)
    tracing = Tracing(field=field, model="FieldLine", initial_conditions=seeds,
                      maxtime=TRACE_TIME, timestep=1e-2, times_to_trace=200)
    traj = np.asarray(tracing.trajectories)
    iota_traced = np.array([np.polyfit(traj[i, :, 2], traj[i, :, 1], 1)[0]
                            for i in range(N_LINES)])
    iota_wout = float(np.interp(S_TRACE, np.linspace(0.0, 1.0, int(wout.ns)),
                                np.asarray(wout.iotaf)))

    turns = float(np.abs(traj[0, -1, 2] - traj[0, 0, 2]) / (2.0 * np.pi))
    print(f"  {label}: LCFS transfer error {lcfs_error:.2e} m")
    print(f"  {label}: {N_LINES} field lines x {turns:.1f} toroidal turns at s={S_TRACE}")
    print(f"  {label}: iota traced {iota_traced.mean():.6f} +- {iota_traced.std():.1e}"
          f"  vs wout {iota_wout:.6f}  (rel. {abs(iota_traced.mean() / iota_wout - 1):.1e})")
    return traj


# --------------------------- 1. vmex -> ESSOS -------------------------------
print("1. vmex fixed-boundary solve -> essos.fields.Vmec")
inp = dataclasses.replace(
    vj.VmecInput.from_file(INPUT_FILE).change_resolution(mpol=MPOL, ntor=NTOR),
    ns_array=NS_LADDER, niter_array=[4000] * len(NS_LADDER), ftol_array=FTOL)
fixed, res = solve_and_export(inp)
print(f"  converged = {bool(res.converged)}, ns = {int(fixed.ns)}, "
      f"aspect = {float(fixed.aspect):.3f}")
traj_fixed = report_handoff(fixed, "fixed")

# --------------------------- 2. ESSOS -> vmex -------------------------------
print("\n2. essos.coils.Coils -> vj.MgridField -> vmex free boundary")
if hasattr(Coils, "from_json"):
    coils = Coils.from_json(str(COILS_JSON))
else:  # legacy ESSOS predating the Coils.from_json classmethod
    from essos.coils import Coils_from_json

    coils = Coils_from_json(str(COILS_JSON))
coil_field = vj.MgridField.from_coils(coils, **GRID)
print(f"  {np.asarray(coils.currents).size} filaments tabulated onto "
      f"{GRID['ir']}x{GRID['kp']}x{GRID['jz']} (R, phi, Z)")

# The free-boundary deck reuses the fixed-boundary shape only as an initial
# guess; PHIEDGE is the flux these coil currents actually hold, and NZETA is
# pinned to a divisor of the mgrid plane count (VMEC2000's mgrid_mod rule).
free_inp = dataclasses.replace(
    inp, lfreeb=True, mgrid_file="essos_coils(direct)", nzeta=GRID["kp"] // 2,
    phiedge=PHIEDGE_FREE)
free, res = solve_and_export(free_inp, external_field=coil_field)
print(f"  converged = {bool(res.converged)}, ns = {int(free.ns)}, "
      f"aspect = {float(free.aspect):.3f}, LCFS solved for by NESTOR")

# What did the tabulation cost in accuracy?  NESTOR only ever evaluates the
# external field on the plasma boundary, so compare the interpolated field
# with direct Biot-Savart there; this is the number that decides whether
# ir/jz/kp are enough.  Trilinear interpolation degrades close to a filament,
# which is why the grid is judged on the surface that uses it.
from essos.fields import BiotSavart  # noqa: E402

theta_s = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
phi_s = np.linspace(0.0, 2.0 * np.pi / float(free.nfp), 17)
R_s, Z_s = surface_rz(free, s_index=-1, theta=theta_s, phi=phi_s)
r_s = R_s.ravel()
z_s = Z_s.ravel()
p_s = np.tile(phi_s, (theta_s.size, 1)).ravel()
direct = np.asarray(jax.vmap(BiotSavart(coils).B)(jnp.asarray(
    np.stack([r_s * np.cos(p_s), r_s * np.sin(p_s), z_s], axis=1))))
b_direct = np.stack([direct[:, 0] * np.cos(p_s) + direct[:, 1] * np.sin(p_s),
                     -direct[:, 0] * np.sin(p_s) + direct[:, 1] * np.cos(p_s),
                     direct[:, 2]], axis=1)
b_grid = np.stack([np.asarray(c) for c in coil_field.b_cyl(
    jnp.asarray(r_s), jnp.asarray(p_s), jnp.asarray(z_s))], axis=1)
tab_error = (np.linalg.norm(b_grid - b_direct, axis=1)
             / np.linalg.norm(b_direct, axis=1))
print(f"  tabulation vs direct Biot-Savart on the LCFS: "
      f"{np.median(tab_error):.2e} median, {tab_error.max():.2e} max (relative)")

# These coils were optimized for this boundary, so the shape the solver finds
# should be the shape it was handed -- at a different field scale, since
# PHIEDGE differs.  The shape agreement is the physics check on the seam.
R_f, Z_f = surface_rz(fixed, s_index=-1, theta=theta_s, phi=phi_s)
shape_gap = float(np.max(np.hypot(R_s - R_f, Z_s - Z_f)))
print(f"  free LCFS sits {100.0 * shape_gap:.1f} cm from the prescribed one "
      f"({100.0 * shape_gap / float(free.Aminor_p):.0f}% of the minor radius); "
      f"|B| on axis {abs(float(free.b0)):.3f} T against {abs(float(fixed.b0)):.3f} T")

# --------------------------- 3. loop closed ---------------------------------
print("\n3. the coil-held equilibrium goes back across the same seam")
traj_free = report_handoff(free, "free")

# --------------------------- figure (skipped in CI) -------------------------
if not CI:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.4), dpi=110)
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    cuts = np.array([0.0, 0.5 * np.pi / float(fixed.nfp)])
    for wout, color, label in ((fixed, "#2e6da4", "fixed boundary (prescribed)"),
                               (free, "#c2543a", "free boundary (ESSOS coils)")):
        R, Z = surface_rz(wout, s_index=-1, theta=theta, phi=cuts)
        for j, style in enumerate(("-", "--")):
            ax.plot(R[:, j], Z[:, j], style, color=color, lw=1.8,
                    label=label if j == 0 else None)
        s_full = np.linspace(0.0, 1.0, int(wout.ns))
        ax2.plot(s_full, np.asarray(wout.iotaf), color=color, lw=1.8, label=label)
    ax.set(xlabel="R [m]", ylabel="Z [m]",
           title="LCFS at $\\phi = 0$ (solid) and a quarter period (dashed)")
    ax.title.set_fontsize(9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.5)
    for traj, color in ((traj_fixed, "#2e6da4"), (traj_free, "#c2543a")):
        ax2.plot(S_TRACE, np.polyfit(traj[0, :, 2], traj[0, :, 1], 1)[0], "o",
                 ms=8, mfc="none", mew=1.6, color=color)
    ax2.set(xlabel="s", ylabel="$\\iota$",
            title="$\\iota$ profile (line: vmex, circle: ESSOS trace)")
    ax2.title.set_fontsize(9)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("vmex $\\leftrightarrow$ ESSOS: equilibrium out, coil field back in")
    fig.tight_layout()
    fig_path = OUT_DIR / "vmex_essos_workflow.png"
    fig.savefig(fig_path)
    print(f"\nwrote {fig_path}")
