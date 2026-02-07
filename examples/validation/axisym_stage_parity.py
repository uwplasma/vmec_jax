"""Axisymmetric stage-by-stage parity against a bundled VMEC2000 `wout`.

Purpose
-------
When fixed-boundary parity is off, it is hard to tell whether the first mismatch
is coming from geometry, `bsup`, `bsub`, or later residual-scalar conventions.

This script runs the VMEC-style Step-10 pipeline on a *reference* VMEC state
(loaded from `wout`) and reports parity metrics at each stage, plus it computes
`getfsq` scalars both with:
  - vmec_jax-computed `bcovar` fields, and
  - reference `wout` Nyquist `bsup/bsub/|B|` fields (to isolate bcovar mismatch).

It is intended to be fast and to pinpoint the first failing stage on axisym.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.config import load_config
from vmec_jax.field import bsup_from_geom, chips_from_chipf, lamscale_from_phips
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


def _half_mesh_coeffs(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


def _step10_fsq_on_state(*, state, static, indata, wout, trig, use_wout_bsup: bool):
    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout,
        indata=indata,
        use_wout_bsup=bool(use_wout_bsup),
        use_vmec_synthesis=True,
        trig=trig,
    )
    rzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
        wout=wout,
        trig=trig,
    )
    frzl = TomnspsRZL(
        frcc=rzl.frcc,
        frss=rzl.frss,
        fzsc=rzl.fzsc,
        fzcs=rzl.fzcs,
        flsc=rzl.flsc,
        flcs=rzl.flcs,
        frsc=rzl.frsc,
        frcs=rzl.frcs,
        fzcc=rzl.fzcc,
        fzss=rzl.fzss,
        flcc=rzl.flcc,
        flss=rzl.flss,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=static.s, signgs=int(wout.signgs))
    scal = vmec_fsq_from_tomnsps_dynamic(frzl=frzl, norms=norms, lconm1=bool(getattr(static.cfg, "lconm1", True)))
    return float(scal.fsqr), float(scal.fsqz), float(scal.fsql)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="circular_tokamak")
    p.add_argument("--hi-res", action="store_true", help="Use a higher angular resolution for diagnostics.")
    args = p.parse_args()

    examples_dir = Path(__file__).resolve().parents[1]
    data_dir = examples_dir / "data"
    input_path = data_dir / f"input.{args.case}"
    wout_path = data_dir / f"wout_{args.case}_reference.nc"
    if not wout_path.exists():
        wout_path = data_dir / f"wout_{args.case}.nc"

    if not input_path.exists() or not wout_path.exists():
        raise SystemExit(f"Missing bundled files for case={args.case!r}")

    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    if int(wout.ntor) != 0:
        raise SystemExit("This harness is axisymmetric-only (ntor must be 0).")

    # VMEC internal (theta,zeta) conventions.
    grid_vmec = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))

    if args.hi_res:
        ntheta = max(int(cfg.ntheta), 4 * int(wout.mpol) + 32)
        ntheta = 2 * (ntheta // 2)
        nzeta = max(int(cfg.nzeta), 1)
        grid_vmec = vmec_angle_grid(ntheta=ntheta, nzeta=nzeta, nfp=int(wout.nfp), lasym=bool(wout.lasym))
        cfg = replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))

    static = build_static(cfg, grid=grid_vmec)
    st = state_from_wout(wout)

    # Nyquist basis for reference fields (gmnc, bsup*, bsub* are Nyquist in wout).
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid_nyq = AngleGrid(theta=np.asarray(grid_vmec.theta), zeta=np.asarray(grid_vmec.zeta), nfp=int(wout.nfp))
    basis_nyq = build_helical_basis(modes_nyq, grid_nyq)

    # Precompute VMEC trig tables used by the parity path.
    mmax = int(np.max(np.asarray(static.modes.m)))
    nmax = int(np.max(np.abs(np.asarray(static.modes.n))))
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=mmax,
        nmax=nmax,
        lasym=bool(wout.lasym),
        dtype=jnp.asarray(st.Rcos).dtype,
    )

    # Stage 2: bsup from geometry (outer surfaces only).
    st_half = replace(
        st,
        Rcos=_half_mesh_coeffs(np.asarray(st.Rcos)),
        Rsin=_half_mesh_coeffs(np.asarray(st.Rsin)),
        Zcos=_half_mesh_coeffs(np.asarray(st.Zcos)),
        Zsin=_half_mesh_coeffs(np.asarray(st.Zsin)),
        Lcos=np.asarray(st.Lcos),
        Lsin=np.asarray(st.Lsin),
    )
    g = eval_geom(st_half, static)

    lamscale = lamscale_from_phips(wout.phips, static.s)
    chips = chips_from_chipf(wout.chipf)
    bsupu_calc, bsupv_calc = bsup_from_geom(
        g,
        phipf=wout.phipf,
        chipf=chips,
        nfp=wout.nfp,
        signgs=wout.signgs,
        lamscale=lamscale,
    )
    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    js0 = max(1, int(0.25 * (int(wout.ns) - 1))) if int(wout.ns) >= 4 else 1
    err_bsup_u = _rel_rms(np.asarray(bsupu_calc)[js0:], bsupu_ref[js0:])
    err_bsup_v = _rel_rms(np.asarray(bsupv_calc)[js0:], bsupv_ref[js0:])

    # Stage 3: bcovar bsup/bsub on VMEC internal grid.
    bc = vmec_bcovar_half_mesh_from_wout(
        state=st,
        static=static,
        wout=wout,
        pres=None,
        use_wout_bsup=False,
        use_vmec_synthesis=True,
        trig=trig,
    )
    # Stage 1: sqrt(g) on VMEC half mesh (use the same parity+axis rules as bcovar).
    sqrtg_calc = np.asarray(bc.jac.sqrtg)
    sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))
    err_sqrtg = _rel_rms(sqrtg_calc[1:], sqrtg_ref[1:])
    err_bcovar_bsup_u = _rel_rms(np.asarray(bc.bsupu)[js0:], bsupu_ref[js0:])
    err_bcovar_bsup_v = _rel_rms(np.asarray(bc.bsupv)[js0:], bsupv_ref[js0:])

    bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))
    err_bsub_u = _rel_rms(np.asarray(bc.bsubu)[js0:], bsubu_ref[js0:])
    err_bsub_v = _rel_rms(np.asarray(bc.bsubv)[js0:], bsubv_ref[js0:])

    # Stage 4: getfsq scalars (computed two ways).
    fsq_calc = _step10_fsq_on_state(state=st, static=static, indata=indata, wout=wout, trig=trig, use_wout_bsup=False)
    fsq_ref_fields = _step10_fsq_on_state(state=st, static=static, indata=indata, wout=wout, trig=trig, use_wout_bsup=True)
    fsq_wout = (float(wout.fsqr), float(wout.fsqz), float(wout.fsql))

    def _fmt3(x):
        return f"({x[0]:.3e}, {x[1]:.3e}, {x[2]:.3e})"

    print("[vmec_jax] axisym stage parity")
    print(f"[vmec_jax] case={args.case} input={input_path.name} wout={wout_path.name}")
    print(f"[stage] sqrtg_halfmesh rel_rms(excl_axis) = {err_sqrtg:.3e}")
    print(f"[stage] bsup_from_geom rel_rms(outer)    = u {err_bsup_u:.3e} v {err_bsup_v:.3e}")
    print(f"[stage] bcovar_bsup   rel_rms(outer)    = u {err_bcovar_bsup_u:.3e} v {err_bcovar_bsup_v:.3e}")
    print(f"[stage] bcovar_bsub   rel_rms(outer)    = u {err_bsub_u:.3e} v {err_bsub_v:.3e}")
    print(f"[stage] getfsq(wout)                  = {_fmt3(fsq_wout)}")
    print(f"[stage] getfsq(vmec_jax bcovar)       = {_fmt3(fsq_calc)}")
    print(f"[stage] getfsq(using wout fields)     = {_fmt3(fsq_ref_fields)}")


if __name__ == "__main__":
    main()
