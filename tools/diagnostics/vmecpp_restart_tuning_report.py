"""Summarize restart-trigger divergences for vmecpp_iter tuning.

This script consumes a JSON report from ``vmecpp_reference_trace_suite.py`` and
focuses on restart-policy discrepancies between vmec_jax and VMEC++.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _arr(x):
    if x is None:
        return []
    return list(x)


def _norm_restart_reason(v) -> str:
    s = str(v).upper()
    if ("NO_RESTART" in s) or ("NONE" == s):
        return "none"
    if "BAD_PROGRESS" in s:
        return "bad_progress"
    if ("BAD_JACOBIAN" in s) or ("HUGE_INITIAL_FORCES" in s):
        return "bad_jacobian"
    if "bad_progress" in str(v):
        return "bad_progress"
    if "bad_jacobian" in str(v):
        return "bad_jacobian"
    return str(v)


def _parse_restart_event(v):
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            idx = int(v[0])
            return idx, _norm_restart_reason(v[1])
        except Exception:
            return None
    s = str(v)
    m = re.search(r"\[\s*(\d+)\s*,", s)
    if m is None:
        return None
    try:
        idx = int(m.group(1))
    except Exception:
        return None
    return idx, _norm_restart_reason(s)


def _vmecpp_restart_series(vmecpp: dict, n_target: int):
    raw = _arr(vmecpp.get("restart_reasons"))
    if n_target <= 0:
        n_target = len(raw)
    events = []
    for v in raw:
        ev = _parse_restart_event(v)
        if ev is not None:
            events.append(ev)
    if events:
        out = ["none"] * n_target
        for idx, reason in events:
            if 0 <= int(idx) < n_target:
                out[int(idx)] = str(reason)
        return out
    return [_norm_restart_reason(v) for v in raw[:n_target]]


def _f(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _case_report(case: dict, start_iter: int, max_rows: int) -> dict:
    name = Path(case.get("input", "unknown")).name
    vmecpp = case.get("vmecpp", {})
    if not vmecpp.get("available", False):
        return {"case": name, "status": "no_vmecpp_trace"}

    h = case.get("histories", {})
    n = len(_arr(h.get("step_status")))
    if n <= 0:
        return {"case": name, "status": "no_history"}

    vmecpp_rr = _vmecpp_restart_series(vmecpp, n_target=n)
    pre_rr = [_norm_restart_reason(v) for v in _arr(h.get("pre_restart_reason"))]
    act_rr = [_norm_restart_reason(v) for v in _arr(h.get("restart_reason"))]
    step_status = _arr(h.get("step_status"))
    restart_path = _arr(h.get("restart_path"))
    w_try_ratio = _arr(h.get("w_try_ratio"))
    fsq1 = _arr(h.get("fsq1"))
    res0 = _arr(h.get("res0"))
    fsq_prev = _arr(h.get("fsq_prev"))

    n_overlap = min(
        len(vmecpp_rr),
        len(pre_rr),
        len(act_rr),
        len(step_status),
        len(restart_path),
        len(w_try_ratio),
        len(fsq1),
        len(res0),
        len(fsq_prev),
    )
    if n_overlap <= 0:
        return {"case": name, "status": "no_overlap"}

    mismatches = []
    for i in range(max(int(start_iter), 0), n_overlap):
        j_decision = pre_rr[i] if pre_rr[i] != "none" else act_rr[i]
        v_decision = vmecpp_rr[i]
        if j_decision != v_decision:
            mismatches.append(
                {
                    "iter": i,
                    "vmec_jax_decision": j_decision,
                    "vmec_jax_pre": pre_rr[i],
                    "vmec_jax_restart": act_rr[i],
                    "vmecpp_restart": v_decision,
                    "step_status": str(step_status[i]),
                    "restart_path": str(restart_path[i]),
                    "w_try_ratio": _f(w_try_ratio[i]),
                    "fsq1": _f(fsq1[i]),
                    "res0": _f(res0[i]),
                    "fsq_prev": _f(fsq_prev[i]),
                }
            )

    kind_counts = {}
    for m in mismatches:
        key = f"{m['vmec_jax_decision']}->{m['vmecpp_restart']}"
        kind_counts[key] = int(kind_counts.get(key, 0)) + 1

    return {
        "case": name,
        "status": "ok",
        "n_overlap": int(n_overlap),
        "n_mismatch": int(len(mismatches)),
        "mismatch_kinds": kind_counts,
        "first_mismatches": mismatches[: max(int(max_rows), 0)],
    }


def _parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trace-json",
        type=Path,
        default=root / "examples/outputs/vmecpp_reference_trace_suite.json",
    )
    p.add_argument("--case-substr", type=str, default="")
    p.add_argument("--start-iter", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=8)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data = json.loads(args.trace_json.read_text())
    reports = []
    for case in data.get("cases", []):
        name = str(case.get("input", ""))
        if args.case_substr and args.case_substr not in name:
            continue
        reports.append(_case_report(case, int(args.start_iter), int(args.max_rows)))

    payload = {
        "trace_json": str(args.trace_json),
        "start_iter": int(args.start_iter),
        "max_rows": int(args.max_rows),
        "reports": reports,
    }
    txt = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
        print(f"[vmec_jax] wrote {args.out}")
    print(txt)


if __name__ == "__main__":
    main()

