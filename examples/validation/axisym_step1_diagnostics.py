#!/usr/bin/env python3
"""Print VMEC++-style step-1 diagnostics for axisymmetric cases.

This does NOT run a fixed-boundary solve. It inspects the initial condition
and computes the first preconditioned update quantities used by VMEC++.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax.diagnostics import summarize_many
from vmec_jax.solve import vmecpp_first_step_diagnostics


def _top_modes(a: np.ndarray, *, k: int = 8):
    if a.ndim != 3:
        return []
    rms = np.sqrt(np.mean(a * a, axis=0))
    flat = rms.reshape(-1)
    if flat.size == 0:
        return []
    idx = np.argsort(flat)[-k:][::-1]
    mpol, nrange = rms.shape
    out = []
    for ii in idx:
        m = int(ii // nrange)
        n = int(ii % nrange)
        out.append((m, n, float(rms[m, n])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="circular_tokamak",
        choices=[
            "circular_tokamak",
            "shaped_tokamak_pressure",
            "vmecpp_solovev",
        ],
    )
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--vmecpp-precond", action="store_true")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / "data"
    input_path = data_dir / f"input.{args.case}"

    run = vj.run_fixed_boundary(input_path, use_initial_guess=True, verbose=False)
    cfg = run.cfg
    static = run.static
    indata = run.indata
    state0 = run.state
    signgs = int(run.signgs)

    diag = vmecpp_first_step_diagnostics(
        state0,
        static,
        indata=indata,
        signgs=signgs,
        step_size=args.step_size,
        include_edge=True,
        zero_m1=True,
        use_vmecpp_precond=bool(args.vmecpp_precond),
    )

    print(f"[axisym_step1] case={args.case} input={input_path}")
    print(
        f"[axisym_step1] fsqr={diag['fsqr']:.6e} fsqz={diag['fsqz']:.6e} fsql={diag['fsql']:.6e}"
    )
    print(
        f"[axisym_step1] fsqr1={diag['fsqr1']:.6e} fsqz1={diag['fsqz1']:.6e} fsql1={diag['fsql1']:.6e}"
    )
    print(
        f"[axisym_step1] gcr2_raw={diag['gcr2_raw']:.6e} gcz2_raw={diag['gcz2_raw']:.6e} "
        f"gcl2_raw={diag['gcl2_raw']:.6e}"
    )
    print(
        f"[axisym_step1] rz_norm={diag['rz_norm']:.6e} f_norm1={diag['f_norm1']:.6e}"
    )
    print(
        f"[axisym_step1] delt={diag['time_step']:.3e} dtau={diag['dtau']:.3e} "
        f"b1={diag['b1']:.6f} fac={diag['fac']:.6f}"
    )

    summarize_many(
        [
            ("scalxc", diag["scalxc"]),
            ("rz_scale", diag["rz_scale"]),
            ("l_scale", diag["l_scale"]),
            ("frcc_u", diag["frcc_u"]),
            ("fzsc_u", diag["fzsc_u"]),
            ("flsc_u", diag["flsc_u"]),
            ("dRcc", diag["dRcc"]),
            ("dZsc", diag["dZsc"]),
            ("dLsc", diag["dLsc"]),
        ],
        indent="  ",
    )

    top_r = _top_modes(diag["dRcc"])
    top_z = _top_modes(diag["dZsc"])
    top_fr = _top_modes(diag["frcc_u"])
    top_fz = _top_modes(diag["fzsc_u"])
    top_fl = _top_modes(diag["flsc_u"])
    if top_r:
        print("[axisym_step1] top dRcc (m,n,rms)")
        for m, n, val in top_r:
            print(f"  m={m:2d} n={n:2d} rms={val:.6e}")
    if top_z:
        print("[axisym_step1] top dZsc (m,n,rms)")
        for m, n, val in top_z:
            print(f"  m={m:2d} n={n:2d} rms={val:.6e}")
    if top_fr:
        print("[axisym_step1] top frcc_u (m,n,rms)")
        for m, n, val in top_fr:
            print(f"  m={m:2d} n={n:2d} rms={val:.6e}")
    if top_fz:
        print("[axisym_step1] top fzsc_u (m,n,rms)")
        for m, n, val in top_fz:
            print(f"  m={m:2d} n={n:2d} rms={val:.6e}")
    if top_fl:
        print("[axisym_step1] top flsc_u (m,n,rms)")
        for m, n, val in top_fl:
            print(f"  m={m:2d} n={n:2d} rms={val:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
