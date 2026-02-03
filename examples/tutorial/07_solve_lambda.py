"""Step-5: solve the lambda field with fixed R/Z.

This script demonstrates a first fixed-boundary "inner solve": optimize VMEC's
lambda coefficients (stored in scaled form) to reduce the magnetic energy `wb`,
while holding the surface geometry (R,Z) fixed.

For validation, we start from a VMEC2000 `wout_*.nc` equilibrium, zero out lambda,
and attempt to recover the reference `wb` using the same flux profiles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.field import bsup_from_sqrtg_lambda, lamscale_from_phips
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.solve import solve_lambda_gd
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


def _wb_from_geom_and_sqrtg(*, geom, sqrtg, phipf, chipf, signgs: int, lamscale, nfp: int, ds, dtheta, dphi):
    lam_u = np.asarray(geom.L_theta)
    lam_v = np.asarray(geom.L_phi) / int(nfp)
    bsupu, bsupv = bsup_from_sqrtg_lambda(
        sqrtg=sqrtg,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=phipf,
        chipf=chipf,
        signgs=signgs,
        lamscale=lamscale,
    )
    bsupu = np.asarray(bsupu)
    bsupv = np.asarray(bsupv)
    gtt = np.asarray(geom.g_tt)
    gtp = np.asarray(geom.g_tp)
    gpp = np.asarray(geom.g_pp)
    B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
    jac = signgs * np.asarray(sqrtg)
    E_per_period = float(np.sum(0.5 * B2 * jac) * ds * dtheta * dphi)
    E_total = E_per_period * float(nfp)
    return E_total / (2.0 * np.pi) ** 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA), used to set grid sizes")
    p.add_argument(
        "--wout",
        type=str,
        default="",
        help="VMEC2000 wout_*.nc (defaults to bundled low-res reference if omitted)",
    )
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--step-size", type=float, default=0.05)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    try:
        import netCDF4  # noqa: F401
    except Exception as e:
        raise SystemExit("netCDF4 is required for this example (pip install -e .[netcdf])") from e

    wout_path = Path(args.wout) if args.wout else (REPO_ROOT / "examples" / "wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc")
    wout = read_wout(wout_path)

    # Build static using the input file for angle-grid sizing (should match the wout case).
    from vmec_jax.config import load_config

    cfg, _indata = load_config(args.input)
    static = build_static(cfg)

    st_ref = state_from_wout(wout)

    # Nyquist basis for reference sqrtg.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)
    sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))

    # VMEC's lambda in wout is scaled; recover lamscale from phips.
    lamscale = float(np.asarray(lamscale_from_phips(wout.phips, static.s)))

    # Create an initial state with the same R/Z but zero lambda.
    st0 = st_ref.__class__(
        layout=st_ref.layout,
        Rcos=st_ref.Rcos,
        Rsin=st_ref.Rsin,
        Zcos=st_ref.Zcos,
        Zsin=st_ref.Zsin,
        Lcos=np.zeros_like(np.asarray(st_ref.Lcos)),
        Lsin=np.zeros_like(np.asarray(st_ref.Lsin)),
    )

    g_ref = eval_geom(st_ref, static)
    g0 = eval_geom(st0, static)

    ds = float(static.s[1] - static.s[0])
    dtheta = 2.0 * np.pi / cfg.ntheta
    dphi = 2.0 * np.pi / (wout.nfp * cfg.nzeta)

    wb_ref = _wb_from_geom_and_sqrtg(
        geom=g_ref,
        sqrtg=sqrtg_ref,
        phipf=wout.phipf,
        chipf=wout.chipf,
        signgs=wout.signgs,
        lamscale=lamscale,
        nfp=wout.nfp,
        ds=ds,
        dtheta=dtheta,
        dphi=dphi,
    )
    wb0 = _wb_from_geom_and_sqrtg(
        geom=g0,
        sqrtg=sqrtg_ref,
        phipf=wout.phipf,
        chipf=wout.chipf,
        signgs=wout.signgs,
        lamscale=lamscale,
        nfp=wout.nfp,
        ds=ds,
        dtheta=dtheta,
        dphi=dphi,
    )

    print("\n==== vmec_jax step-5: solve lambda (R/Z fixed) ====")
    print(f"wout: {wout.path.name}")
    print(f"signgs={wout.signgs}  lamscale={lamscale:.6g}")
    print(f"wb_ref (from wout sqrtg + vmec_jax metric) = {wb_ref:.8e}")
    print(f"wb0 (lambda=0) = {wb0:.8e}")

    res = solve_lambda_gd(
        st0,
        static,
        phipf=wout.phipf,
        chipf=wout.chipf,
        signgs=wout.signgs,
        lamscale=lamscale,
        sqrtg=sqrtg_ref,
        max_iter=int(args.max_iter),
        step_size=float(args.step_size),
        verbose=bool(args.verbose),
    )

    g_sol = eval_geom(res.state, static)
    wb_sol = _wb_from_geom_and_sqrtg(
        geom=g_sol,
        sqrtg=sqrtg_ref,
        phipf=wout.phipf,
        chipf=wout.chipf,
        signgs=wout.signgs,
        lamscale=lamscale,
        nfp=wout.nfp,
        ds=ds,
        dtheta=dtheta,
        dphi=dphi,
    )

    print("\n==== results ====")
    print(f"iters = {res.n_iter}")
    print(f"wb_sol = {wb_sol:.8e}")
    print(f"wb_rel_err_to_wout = {(wb_sol - wout.wb) / wout.wb:.3e}")
    print(f"wb_rel_err_to_ref  = {(wb_sol - wb_ref) / wb_ref:.3e}")

    if args.verbose:
        print("\n==== convergence history ====")
        for i, (wb, g) in enumerate(zip(res.wb_history, np.r_[res.grad_rms_history, np.nan])):
            if i == 0:
                print(f"  {i:03d}: wb={wb:.8e}")
            else:
                print(f"  {i:03d}: wb={wb:.8e}  grad_rms={g:.3e}")


if __name__ == "__main__":
    main()
