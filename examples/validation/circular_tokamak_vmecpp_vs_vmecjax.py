"""Run a simple circular tokamak case and compare VMEC++ vs vmec_jax."""

from __future__ import annotations

import argparse
from pathlib import Path

from vmec_jax.config import load_config
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _fsq_from_state_vs_wout(state, static, wout, *, indata):
    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )
    # Important: do NOT mix a non-reference geometry state with bsup/bsub stored
    # in a reference wout. Using `use_wout_bsup=True` only makes sense when the
    # input `state` is itself derived from the same `wout` (state_from_wout).
    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout,
        indata=indata,
        use_wout_bsup=False,
        use_vmec_synthesis=True,
        trig=trig,
    )
    rzl = vmec_residual_internal_from_kernels(
        k, cfg_ntheta=int(static.cfg.ntheta), cfg_nzeta=int(static.cfg.nzeta), wout=wout, trig=trig
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


def _rel_err(val, ref):
    denom = max(abs(ref), 1e-20)
    return abs(val - ref) / denom


def _parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=root / "examples" / "data" / "input.circular_tokamak",
    )
    p.add_argument(
        "--wout",
        type=Path,
        default=root / "examples" / "data" / "wout_circular_tokamak_reference.nc",
    )
    p.add_argument("--solver", type=str, default="vmec_gn")
    p.add_argument("--max-iter", type=int, default=25)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--solve", action="store_true")
    p.add_argument("--no-solve", action="store_true")
    p.add_argument("--no-vmecpp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input
    wout_path = args.wout

    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg, grid=grid)

    do_solve = bool(args.solve) and not bool(args.no_solve)
    if not args.solve and not args.no_solve:
        do_solve = False

    print("[vmec_jax] running circular tokamak comparison")
    print(f"[vmec_jax] input={input_path}")
    if do_solve:
        print(f"[vmec_jax] solver={args.solver} max_iter={args.max_iter} step_size={args.step_size}")
        run = run_fixed_boundary(
            input_path,
            solver=str(args.solver),
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            vmecpp_reference_mode=True,
            vmecpp_use_restart_triggers=True,
            vmecpp_use_direct_fallback=False,
            verbose=True,
        )
        result = run.result
        state = run.state
    else:
        print("[vmec_jax] using wout reference state (no solve)")
        result = None
        state = state_from_wout(wout)

    fsqr, fsqz, fsql = _fsq_from_state_vs_wout(state, static, wout, indata=indata)
    print("[vmec_jax] fsq vs wout reference")
    print(f"  fsqr={fsqr:.6e} ref={wout.fsqr:.6e} rel={_rel_err(fsqr, wout.fsqr):.3e}")
    print(f"  fsqz={fsqz:.6e} ref={wout.fsqz:.6e} rel={_rel_err(fsqz, wout.fsqz):.3e}")
    print(f"  fsql={fsql:.6e} ref={wout.fsql:.6e} rel={_rel_err(fsql, wout.fsql):.3e}")

    if not args.no_vmecpp:
        try:
            import vmecpp  # type: ignore

            vmec_input = vmecpp.VmecInput.from_file(input_path)
            vmec_out = vmecpp.run(vmec_input, verbose=False, max_threads=1)
            print("[vmecpp] fsq (wout)")
            print(
                f"  fsqr={vmec_out.wout.fsqr:.6e} fsqz={vmec_out.wout.fsqz:.6e} fsql={vmec_out.wout.fsql:.6e}"
            )
        except Exception as exc:
            print(f"[vmecpp] skipped: {exc!r}")

    if result is not None:
        print(f"[vmec_jax] iterations={result.n_iter} last fsq={result.fsqr2_history[-1]:.6e}")


if __name__ == "__main__":
    main()
