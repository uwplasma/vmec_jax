"""Find the first vmec_jax vs VMEC++ per-iteration trace mismatch.

Consumes a JSON report produced by ``vmecpp_reference_trace_suite.py`` and
reports the first mismatch iteration for scalar force residual traces.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _to_float_list(a):
    if a is None:
        return []
    out = []
    for v in a:
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return out


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


def _first_finite_at_or_after(a, i0: int):
    for i in range(max(int(i0), 0), len(a)):
        v = a[i]
        if math.isfinite(v):
            return i, v
    return None, float("nan")


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
    raw = vmecpp.get("restart_reasons") or []
    raw = list(raw)
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


def _first_scalar_mismatch(a, b, *, rtol: float, atol: float, start_iter: int, normalize_first: bool):
    n = min(len(a), len(b))
    if n <= 0:
        return None
    ai0, a0 = _first_finite_at_or_after(a, start_iter)
    bi0, b0 = _first_finite_at_or_after(b, start_iter)
    if (ai0 is None) or (bi0 is None):
        normalize_first = False
    if not (math.isfinite(a0) and math.isfinite(b0)):
        normalize_first = False
    for i in range(n):
        if i < start_iter:
            continue
        ai = a[i]
        bi = b[i]
        if not (math.isfinite(ai) and math.isfinite(bi)):
            return {"iter": i, "reason": "non_finite", "a": ai, "b": bi}
        if normalize_first:
            ai_cmp = ai / max(abs(a0), 1e-30)
            bi_cmp = bi / max(abs(b0), 1e-30)
        else:
            ai_cmp = ai
            bi_cmp = bi
        abs_err = abs(ai_cmp - bi_cmp)
        rel_err = abs_err / max(abs(bi_cmp), 1e-30)
        if (abs_err > atol) and (rel_err > rtol):
            return {
                "iter": i,
                "reason": "value",
                "normalize_base_iter_a": ai0,
                "normalize_base_iter_b": bi0,
                "a": ai,
                "b": bi,
                "a_cmp": ai_cmp,
                "b_cmp": bi_cmp,
                "abs_err": abs_err,
                "rel_err": rel_err,
            }
    return None


def _analyze_case(case: dict, *, rtol: float, atol: float, start_iter: int, normalize_first: bool):
    name = Path(case["input"]).name
    vmecpp = case.get("vmecpp", {})
    if not vmecpp.get("available", False):
        return {"case": name, "status": "no_vmecpp_trace"}

    hz = case["histories"]
    j_fsqr = _to_float_list(hz.get("fsqr"))
    j_fsqz = _to_float_list(hz.get("fsqz"))
    j_fsql = _to_float_list(hz.get("fsql"))
    v_fsqr = _to_float_list(vmecpp.get("force_residual_r"))
    v_fsqz = _to_float_list(vmecpp.get("force_residual_z"))
    v_fsql = _to_float_list(vmecpp.get("force_residual_lambda"))

    checks = [
        (
            "fsqr",
            _first_scalar_mismatch(
                j_fsqr, v_fsqr, rtol=rtol, atol=atol, start_iter=start_iter, normalize_first=normalize_first
            ),
        ),
        (
            "fsqz",
            _first_scalar_mismatch(
                j_fsqz, v_fsqz, rtol=rtol, atol=atol, start_iter=start_iter, normalize_first=normalize_first
            ),
        ),
        (
            "fsql",
            _first_scalar_mismatch(
                j_fsql, v_fsql, rtol=rtol, atol=atol, start_iter=start_iter, normalize_first=normalize_first
            ),
        ),
    ]
    checks = [c for c in checks if c[1] is not None]
    first_metric = None
    if checks:
        first_metric = min(checks, key=lambda x: int(x[1]["iter"]))

    rr_j = [_norm_restart_reason(v) for v in (hz.get("restart_reason") or [])]
    rr_v = _vmecpp_restart_series(vmecpp, n_target=len(rr_j))
    rr_mismatch = None
    for i in range(min(len(rr_j), len(rr_v))):
        if i < int(start_iter):
            continue
        if rr_j[i] != rr_v[i]:
            rr_mismatch = {"iter": i, "vmec_jax": rr_j[i], "vmecpp": rr_v[i]}
            break

    out = {
        "case": name,
        "status": "ok",
        "lengths": {
            "vmec_jax": len(j_fsqr),
            "vmecpp": len(v_fsqr),
        },
        "first_scalar_mismatch": (
            None
            if first_metric is None
            else {"metric": first_metric[0], **first_metric[1]}
        ),
        "first_restart_mismatch": rr_mismatch,
    }
    return out


def _parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trace-json",
        type=Path,
        default=root / "examples/outputs/vmecpp_reference_trace_suite.json",
    )
    p.add_argument("--case-substr", type=str, default="")
    p.add_argument("--rtol", type=float, default=0.2)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--start-iter", type=int, default=1)
    p.add_argument("--no-normalize-first", action="store_true")
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
        reports.append(
            _analyze_case(
                case,
                rtol=float(args.rtol),
                atol=float(args.atol),
                start_iter=max(int(args.start_iter), 0),
                normalize_first=not bool(args.no_normalize_first),
            )
        )

    payload = {
        "trace_json": str(args.trace_json),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "start_iter": max(int(args.start_iter), 0),
        "normalize_first": not bool(args.no_normalize_first),
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
