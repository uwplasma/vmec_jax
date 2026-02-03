"""Step-1 demo: build a regularity-aware initial guess and evaluate geometry.

This script exercises the new step-1 kernels:
  - build VMECStatic
  - boundary coefficients from &INDATA
  - initial guess coefficients for all radial surfaces
  - real-space evaluation of R,Z,lambda and their angular derivatives

Usage
-----
    python 02_init_guess_and_coords.py <input_file> --out coords.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from vmec_jax._compat import has_jax, enable_x64

from vmec_jax.config import load_config
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.static import build_static
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.coords import eval_coords
from vmec_jax.fourier import eval_fourier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="VMEC input namelist file")
    ap.add_argument("--out", default="coords_step1.npz", help="Output .npz file")
    ap.add_argument("--dump_coeffs", action="store_true", help="Also store (ns,K) Fourier coeff arrays")
    ap.add_argument("--verbose", action="store_true", help="Print extra debug information")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        ap.error(f"Input file not found: {args.input}")

    cfg, indata = load_config(str(inp))
    if has_jax():
        enable_x64(True)
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    state0 = initial_guess_from_boundary(static, bdy, indata)

    coords = eval_coords(state0, static.basis)
    R = np.asarray(coords.R)
    Z = np.asarray(coords.Z)

    # Consistency check: boundary surface from state should match direct boundary eval.
    Rb_direct = np.asarray(eval_fourier(bdy.R_cos, bdy.R_sin, static.basis))
    Zb_direct = np.asarray(eval_fourier(bdy.Z_cos, bdy.Z_sin, static.basis))
    err_R = np.max(np.abs(R[-1] - Rb_direct))
    err_Z = np.max(np.abs(Z[-1] - Zb_direct))

    print("\n==== vmec_jax step-1 coords kernel ====")
    print(f"mpol={cfg.mpol} ntor={cfg.ntor} nfp={cfg.nfp} ns={cfg.ns} ntheta={cfg.ntheta} nzeta={cfg.nzeta}")
    print(f"R(s=0): min={R[0].min():.6g} max={R[0].max():.6g}")
    print(f"Z(s=0): min={Z[0].min():.6g} max={Z[0].max():.6g}")
    print(f"R(s=1): min={R[-1].min():.6g} max={R[-1].max():.6g}")
    print(f"Z(s=1): min={Z[-1].min():.6g} max={Z[-1].max():.6g}")
    print(f"boundary consistency: max|ΔR|={err_R:.3e}  max|ΔZ|={err_Z:.3e}")

    if args.verbose:
        from vmec_jax.diagnostics import summarize_many

        print("\n-- angular derivatives (selected slices) --")
        summarize_many(
            [
                ("R_theta(s=0)", np.asarray(coords.R_theta)[0]),
                ("Z_theta(s=0)", np.asarray(coords.Z_theta)[0]),
                ("R_phi(s=0)", np.asarray(coords.R_phi)[0]),
                ("Z_phi(s=0)", np.asarray(coords.Z_phi)[0]),
                ("R_theta(s=1)", np.asarray(coords.R_theta)[-1]),
                ("Z_theta(s=1)", np.asarray(coords.Z_theta)[-1]),
                ("R_phi(s=1)", np.asarray(coords.R_phi)[-1]),
                ("Z_phi(s=1)", np.asarray(coords.Z_phi)[-1]),
            ],
            indent="  ",
        )

        # Summarize coefficient magnitudes at a few radii
        m = np.asarray(static.modes.m)
        n = np.asarray(static.modes.n)
        def _topk(label, arr, k=8):
            arr = np.asarray(arr)
            mag = np.sqrt(arr**2)
            idx = np.argsort(-mag)[:k]
            print(f"\n-- top {k} |{label}| coefficients --")
            print("  rank  (m,n)    value")
            for r, j in enumerate(idx, start=1):
                print(f"  {r:>4d}  ({int(m[j]):>2d},{int(n[j]):>3d})  {arr[j]:>+12.5e}")

        _topk("Rcos(s=0)", np.asarray(state0.Rcos)[0])
        _topk("Zsin(s=0)", np.asarray(state0.Zsin)[0])
        _topk("Rcos(s=1)", np.asarray(state0.Rcos)[-1])
        _topk("Zsin(s=1)", np.asarray(state0.Zsin)[-1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        R=np.asarray(coords.R),
        Z=np.asarray(coords.Z),
        L=np.asarray(coords.L),
        R_theta=np.asarray(coords.R_theta),
        Z_theta=np.asarray(coords.Z_theta),
        L_theta=np.asarray(coords.L_theta),
        R_phi=np.asarray(coords.R_phi),
        Z_phi=np.asarray(coords.Z_phi),
        L_phi=np.asarray(coords.L_phi),
        s=np.asarray(static.s),
        theta=np.asarray(static.grid.theta),
        zeta=np.asarray(static.grid.zeta),
        m=np.asarray(static.modes.m),
        n=np.asarray(static.modes.n),
    )
    if args.dump_coeffs:
        save_kwargs.update(
            Rcos=np.asarray(state0.Rcos),
            Rsin=np.asarray(state0.Rsin),
            Zcos=np.asarray(state0.Zcos),
            Zsin=np.asarray(state0.Zsin),
            Lcos=np.asarray(state0.Lcos),
            Lsin=np.asarray(state0.Lsin),
        )
    np.savez(out, **save_kwargs)
    print(f"saving: {out}")


if __name__ == "__main__":
    main()
