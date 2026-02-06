"""Decompose VMEC++ final-state `getfsq` parity on a case.

Runs VMEC++ for the given input, evaluates vmec_jax Step-10 force scalars on the
final state, and reports how `include_edge`, `scalxc`, and `m=1` constraints
change `(fsqr, fsqz, fsql)`.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=root / "examples/data/input.n3are_R7.75B5.7_lowres")
    p.add_argument("--out", type=Path, default=root / "examples/outputs/vmecpp_getfsq_decomposition_n3are.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        import vmecpp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"vmecpp import failed: {exc}")

    out = vmecpp.run(vmecpp.VmecInput.from_file(args.input), verbose=False)
    with tempfile.TemporaryDirectory() as td:
        wpath = Path(td) / f"wout_{args.input.name.replace('input.', '')}_vmecpp.nc"
        out.wout.save(str(wpath))
        wout = read_wout(wpath)

    cfg, _indata = load_config(str(args.input))
    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(cfg.nfp), lasym=bool(cfg.lasym))
    static = build_static(cfg, grid=grid)
    st = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )
    k = vmec_forces_rz_from_wout(
        state=st,
        static=static,
        wout=wout,
        indata=None,
        use_wout_bsup=True,
        use_vmec_synthesis=True,
        trig=trig,
    )
    rzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(cfg.ntheta),
        cfg_nzeta=int(cfg.nzeta),
        wout=wout,
        trig=trig,
    )
    fr = TomnspsRZL(
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

    variants: dict[str, dict[str, float]] = {}
    for include_edge in (False, True):
        for apply_scalxc in (False, True):
            for apply_m1_constraints in (False, True):
                fs = vmec_fsq_from_tomnsps_dynamic(
                    frzl=fr,
                    norms=norms,
                    lconm1=True,
                    apply_m1_constraints=apply_m1_constraints,
                    include_edge=include_edge,
                    apply_scalxc=apply_scalxc,
                    s=static.s,
                )
                key = f"edge={int(include_edge)}_scalxc={int(apply_scalxc)}_m1={int(apply_m1_constraints)}"
                variants[key] = {
                    "fsqr": float(np.asarray(fs.fsqr)),
                    "fsqz": float(np.asarray(fs.fsqz)),
                    "fsql": float(np.asarray(fs.fsql)),
                }

    report = {
        "input": str(args.input),
        "wout_ref": {"fsqr": float(wout.fsqr), "fsqz": float(wout.fsqz), "fsql": float(wout.fsql)},
        "variants": variants,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[vmec_jax] wrote {args.out}")


if __name__ == "__main__":
    main()
