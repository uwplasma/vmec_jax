"""Compare vmec_jax decision-trace policies against VMEC++ restart reasons.

Consumes a JSON report produced by ``vmecpp_reference_trace_suite.py`` and
reports the first iteration where vmec_jax trigger logic diverges from VMEC++
restart behavior.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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


def _arr(x):
    if x is None:
        return []
    return list(x)


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _parse_restart_event(v):
    # Common vmecpp JSON shape after serialization can be either:
    #   [iter, reason]  (list)
    # or stringified list:
    #   "[iter, <RestartReason.BAD_JACOBIAN: 2>]"
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


def _analyze_case(case: dict, *, start_iter: int) -> dict:
    name = Path(case.get("input", "unknown")).name
    vmecpp = case.get("vmecpp", {})
    if not vmecpp.get("available", False):
        return {"case": name, "status": "no_vmecpp_trace"}

    hz = case.get("histories", {})
    n_target = len(_arr(hz.get("pre_restart_reason")))
    vmecpp_rr = _vmecpp_restart_series(vmecpp, n_target=n_target)
    n = min(
        len(_arr(hz.get("pre_restart_reason"))),
        len(_arr(hz.get("restart_reason"))),
        len(_arr(hz.get("fsq1"))),
        len(_arr(hz.get("res0"))),
        len(_arr(hz.get("fsq_prev"))),
        len(_arr(hz.get("bad_growth_streak"))),
        len(vmecpp_rr),
    )
    if n == 0:
        return {"case": name, "status": "no_overlap"}

    pre_rr = [_norm_restart_reason(v) for v in _arr(hz.get("pre_restart_reason"))]
    act_rr = [_norm_restart_reason(v) for v in _arr(hz.get("restart_reason"))]
    fsq1 = _arr(hz.get("fsq1"))
    res0 = _arr(hz.get("res0"))
    fsq_prev = _arr(hz.get("fsq_prev"))
    bad_growth = _arr(hz.get("bad_growth_streak"))
    iter1 = _arr(hz.get("iter1"))
    include_edge = _arr(hz.get("include_edge"))
    zero_m1 = _arr(hz.get("zero_m1"))
    dt_eff = _arr(hz.get("dt_eff"))
    update_rms = _arr(hz.get("update_rms"))

    first_mismatch = None
    for k in range(max(int(start_iter), 0), n):
        j_decision = pre_rr[k] if pre_rr[k] != "none" else act_rr[k]
        v_decision = vmecpp_rr[k]
        if j_decision != v_decision:
            first_mismatch = {
                "iter": k,
                "vmec_jax_decision": j_decision,
                "vmec_jax_pre_restart_reason": pre_rr[k],
                "vmec_jax_restart_reason": act_rr[k],
                "vmecpp_restart_reason": v_decision,
                "fsq1": _f(fsq1[k]),
                "res0": _f(res0[k]),
                "fsq_prev": _f(fsq_prev[k]),
                "bad_growth_streak": _i(bad_growth[k]),
                "iter1": _i(iter1[k]) if k < len(iter1) else -1,
                "include_edge": _i(include_edge[k]) if k < len(include_edge) else -1,
                "zero_m1": _i(zero_m1[k]) if k < len(zero_m1) else -1,
                "dt_eff": _f(dt_eff[k]) if k < len(dt_eff) else float("nan"),
                "update_rms": _f(update_rms[k]) if k < len(update_rms) else float("nan"),
            }
            break

    return {
        "case": name,
        "status": "ok",
        "n_overlap": n,
        "first_decision_mismatch": first_mismatch,
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
        reports.append(_analyze_case(case, start_iter=max(int(args.start_iter), 0)))

    payload = {
        "trace_json": str(args.trace_json),
        "start_iter": max(int(args.start_iter), 0),
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
