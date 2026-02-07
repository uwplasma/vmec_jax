"""Run a fixed set of parity cases and export per-iteration vmec_jax traces.

This script is designed for stage-by-stage debugging of the fixed-boundary
update loop. It records residual/timestep/restart histories for vmec_jax and,
if vmecpp is available, also records VMEC++ terminal scalars and any exposed
iteration traces from the Python object.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from vmec_jax.driver import run_fixed_boundary
from vmec_jax.wout import read_wout


def _arr(x):
    if x is None:
        return None
    a = np.asarray(x)
    out = []
    for v in a.tolist():
        if isinstance(v, (float, int, str, bool)) or v is None:
            out.append(v)
            continue
        # Handle enum-like objects from vmecpp bindings.
        if hasattr(v, "name"):
            out.append(str(getattr(v, "name")))
            continue
        if hasattr(v, "value"):
            vv = getattr(v, "value")
            if isinstance(vv, (float, int, str, bool)) or vv is None:
                out.append(vv)
                continue
        out.append(str(v))
    return out


def _wout_scalar_summary(wout):
    return {
        "fsqr": float(getattr(wout, "fsqr", np.nan)),
        "fsqz": float(getattr(wout, "fsqz", np.nan)),
        "fsql": float(getattr(wout, "fsql", np.nan)),
        "niter": int(getattr(wout, "niter", -1)),
    }


def _maybe_vmecpp_trace(input_path: Path):
    try:
        import vmecpp  # type: ignore
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    try:
        out = vmecpp.run(vmecpp.VmecInput.from_file(input_path), verbose=False, max_threads=1)
        payload = {
            "available": True,
            "wout_scalars": _wout_scalar_summary(out.wout),
        }

        # Keep this robust to vmecpp binding changes.
        for key in (
            "force_residual_r",
            "force_residual_z",
            "force_residual_lambda",
            "restart_reasons",
            "mhd_energy",
        ):
            if hasattr(out.wout, key):
                payload[key] = _arr(getattr(out.wout, key))
        return payload
    except Exception as exc:  # pragma: no cover - optional integration
        return {"available": False, "error": repr(exc)}


def _run_case(input_path: Path, *, max_iter: int, step_size: float, reference_mode: bool):
    run = run_fixed_boundary(
        input_path,
        solver="vmecpp_iter",
        max_iter=max_iter,
        step_size=step_size,
        vmecpp_reference_mode=reference_mode,
        vmecpp_use_restart_triggers=None,
        vmecpp_use_direct_fallback=False,
        verbose=False,
    )
    res = run.result
    assert res is not None
    diag = dict(res.diagnostics)
    out = {
        "input": str(input_path),
        "n_iter": int(res.n_iter),
        "w_last": float(res.w_history[-1]) if len(res.w_history) else float("nan"),
        "histories": {
            "w": _arr(res.w_history),
            "fsqr": _arr(res.fsqr2_history),
            "fsqz": _arr(res.fsqz2_history),
            "fsql": _arr(res.fsql2_history),
            "step": _arr(res.step_history),
            "grad_rms": _arr(res.grad_rms_history),
            "fsqr1": _arr(diag.get("fsqr1_history")),
            "fsqz1": _arr(diag.get("fsqz1_history")),
            "fsql1": _arr(diag.get("fsql1_history")),
            "fsq1": _arr(diag.get("fsq1_history")),
            "step_status": _arr(diag.get("step_status_history")),
            "restart_reason": _arr(diag.get("restart_reason_history")),
            "pre_restart_reason": _arr(diag.get("pre_restart_reason_history")),
            "time_step": _arr(diag.get("time_step_history")),
            "res0": _arr(diag.get("res0_history")),
            "fsq_prev": _arr(diag.get("fsq_prev_history")),
            "bad_growth_streak": _arr(diag.get("bad_growth_streak_history")),
            "iter1": _arr(diag.get("iter1_history")),
            "include_edge": _arr(diag.get("include_edge_history")),
            "zero_m1": _arr(diag.get("zero_m1_history")),
            "dt_eff": _arr(diag.get("dt_eff_history")),
            "update_rms": _arr(diag.get("update_rms_history")),
            "w_curr": _arr(diag.get("w_curr_history")),
            "w_try": _arr(diag.get("w_try_history")),
            "w_try_ratio": _arr(diag.get("w_try_ratio_history")),
            "restart_path": _arr(diag.get("restart_path_history")),
        },
        "diag": {
            "ftol": float(diag.get("ftol", np.nan)),
            "ijacob": int(diag.get("ijacob", -1)),
            "bad_resets": int(diag.get("bad_resets", -1)),
            "iter1_final": int(diag.get("iter1_final", -1)),
            "res0": float(diag.get("res0", np.nan)),
            "vmecpp_reference_mode": bool(diag.get("vmecpp_reference_mode", False)),
            "use_vmecpp_restart_triggers": bool(diag.get("use_vmecpp_restart_triggers", False)),
            "use_direct_fallback": bool(diag.get("use_direct_fallback", False)),
        },
    }
    return out


def _parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        nargs="+",
        default=[
            "input.n3are_R7.75B5.7_lowres",
            "input.li383_low_res",
            "input.circular_tokamak",
        ],
    )
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--reference-mode", action="store_true")
    p.add_argument("--with-vmecpp", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=root / "examples/outputs/vmecpp_reference_trace_suite.json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "examples" / "data"

    report = {
        "max_iter": int(args.max_iter),
        "step_size": float(args.step_size),
        "reference_mode": bool(args.reference_mode),
        "cases": [],
    }

    for case in args.cases:
        ip = case if Path(case).is_absolute() else str(data_dir / case)
        input_path = Path(ip)
        case_report = _run_case(
            input_path,
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            reference_mode=bool(args.reference_mode),
        )
        if args.with_vmecpp:
            case_report["vmecpp"] = _maybe_vmecpp_trace(input_path)
        report["cases"].append(case_report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[vmec_jax] wrote {args.out}")


if __name__ == "__main__":
    main()
