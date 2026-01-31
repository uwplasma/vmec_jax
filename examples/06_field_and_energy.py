"""Step-4: magnetic field components and magnetic energy.

This script adds the first physics quantities beyond geometry:
  - contravariant B components (bsupu, bsupv) in VMEC coordinates
  - magnetic energy `wb` (VMEC-reported normalization)

If a `wout_*.nc` is provided (and netCDF4 is installed), we compare against
VMEC2000's stored Fourier coefficients for `sqrtg` and `bsup*`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.diagnostics import print_summary, summarize_array
from vmec_jax.energy import magnetic_wb_from_state
from vmec_jax.field import bsup_from_sqrtg_lambda, lamscale_from_phips
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--wout", type=str, default="", help="Optional VMEC2000 wout_*.nc to compare against")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    cfg, indata = load_config(args.input)
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    # For objective/grad demos we keep signgs fixed (use initial guess orientation).
    g0 = eval_geom(st0, static)
    signgs0 = 1 if float(np.mean(np.asarray(g0.sqrtg)[1:])) >= 0 else -1

    wb0, diag0 = magnetic_wb_from_state(st0, static, indata, signgs=signgs0)

    print("\n==== vmec_jax step-4: field + energy (initial guess) ====")
    print(f"signgs={signgs0}  lamscale={float(np.asarray(diag0['lamscale'])):.6g}")
    print(f"wb (VMEC norm) = {float(np.asarray(wb0)):.8e}")

    if not args.wout:
        return

    try:
        wout = read_wout(args.wout)
    except ImportError as e:
        raise SystemExit(str(e)) from e

    st_eq = state_from_wout(wout)
    g_eq = eval_geom(st_eq, static)

    # Build nyquist basis for reference fields
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)

    sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))
    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))

    # lamscale from wout phips (VMEC definition)
    lamscale = float(np.asarray(lamscale_from_phips(wout.phips, static.s)))

    # lambda derivatives from our geom kernel (these are of the scaled lambda)
    lam_u = np.asarray(g_eq.L_theta)
    lam_v = np.asarray(g_eq.L_phi) / wout.nfp

    bsupu, bsupv = bsup_from_sqrtg_lambda(
        sqrtg=sqrtg_ref,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=wout.phipf,
        chipf=wout.chipf,
        signgs=wout.signgs,
        lamscale=lamscale,
    )
    bsupu = np.asarray(bsupu)
    bsupv = np.asarray(bsupv)

    # Compare away from the axis (sqrtg=0).
    mask = np.isfinite(sqrtg_ref) & (np.abs(sqrtg_ref) > 1e-14)
    mask[0, :, :] = False
    du = bsupu[mask] - bsupu_ref[mask]
    dv = bsupv[mask] - bsupv_ref[mask]

    print("\n==== comparisons vs VMEC2000 wout reference ====")
    print(f"wout: {wout.path.name}")
    print(f"signgs={wout.signgs}  lamscale={lamscale:.6g}")
    print(f"wb_ref (wout) = {wout.wb:.8e}")

    print_summary(summarize_array("bsupu_ref", bsupu_ref[mask]), indent="")
    print_summary(summarize_array("bsupv_ref", bsupv_ref[mask]), indent="")
    print_summary(summarize_array("bsupu - ref", du), indent="")
    print_summary(summarize_array("bsupv - ref", dv), indent="")

    # Energy check: use reference sqrtg (nyquist) to avoid sensitivity to our radial derivative.
    gtt = np.asarray(g_eq.g_tt)
    gtp = np.asarray(g_eq.g_tp)
    gpp = np.asarray(g_eq.g_pp)
    B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2

    ds = float(static.s[1] - static.s[0])
    dtheta = 2.0 * np.pi / cfg.ntheta
    dphi = 2.0 * np.pi / (wout.nfp * cfg.nzeta)
    jac = wout.signgs * sqrtg_ref
    E_per_period = float(np.sum(0.5 * B2 * jac) * ds * dtheta * dphi)
    E_total = E_per_period * float(wout.nfp)
    wb_calc = E_total / (2.0 * np.pi) ** 2

    print(f"wb_calc (from wout sqrtg + vmec_jax metric) = {wb_calc:.8e}")
    print(f"wb_rel_err = {(wb_calc - wout.wb) / wout.wb:.3e}")

    if args.verbose:
        print_summary(summarize_array("sqrtg_ref (mask)", sqrtg_ref[mask]), indent="")


if __name__ == "__main__":
    main()

