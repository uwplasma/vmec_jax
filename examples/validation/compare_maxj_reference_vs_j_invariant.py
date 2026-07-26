#!/usr/bin/env python
"""Compare ``qi_functions_mod.py`` maxJ reduction against smooth VMEX shared-J maxJ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity import boozer_bmnc_state
from vmex.core import omnigenity_j as qi_j

from compare_qi_reference_vs_j_invariant import (
    _reference_goodman_transform,
    _reference_j_pair,
)


def _reference_maxj_from_j(
    jc: np.ndarray,
    surfaces: np.ndarray,
    *,
    target_maxj: float,
    p_j: float,
    bounce_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal-ish maxJ reduction from ``qi_functions_mod.py`` J_C tensor."""

    jc_pow = np.power(jc, p_j)  # (nsurf, n_bounce, nalpha)
    nsurf, n_bounce, nalpha = jc_pow.shape
    if nsurf < 2:
        return np.zeros((0,), dtype=float), np.zeros((0, n_bounce, nalpha), dtype=float)

    slopes = np.zeros((nsurf - 1, n_bounce, nalpha), dtype=float)
    penalties = np.zeros((nsurf - 1,), dtype=float)
    ds = np.diff(surfaces)
    for si in range(nsurf - 1):
        delta_s = ds[si] if abs(ds[si]) > 0 else 1.0e-10
        for j in range(int(bounce_start), n_bounce):
            for ialpha_i in range(nalpha):
                accum = 0.0
                hi_val = jc_pow[si + 1, j, ialpha_i]
                for ialpha_j in range(nalpha):
                    lo_val = jc_pow[si, j, ialpha_j]
                    denom = delta_s * 0.5 * (hi_val + lo_val)
                    if abs(denom) < 1.0e-10:
                        continue
                    accum += (hi_val - lo_val) / denom
                slopes[si, j, ialpha_i] = accum / nalpha
                penalties[si] += max(0.0, slopes[si, j, ialpha_i] - target_maxj) ** 2
        penalties[si] = np.sqrt(penalties[si])
    return penalties, slopes


def _vmex_maxj_from_j(
    jc: np.ndarray,
    surfaces: np.ndarray,
    *,
    target_maxj: float,
    p_j: float,
    bounce_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Current VMEX maxJ reduction reproduced in NumPy for comparison."""

    jc_pow = np.power(jc, p_j)  # (nsurf, n_bounce, nalpha)
    nsurf, n_bounce, nalpha = jc_pow.shape
    if nsurf < 2:
        return np.zeros((0,), dtype=float), np.zeros((0, max(n_bounce - 1, 0), nalpha), dtype=float)

    ds = np.diff(surfaces)
    ds = np.where(np.abs(ds) > 0.0, ds, 1.0e-10)
    start = int(bounce_start)
    jc_lo = jc_pow[:-1, start:, :]
    jc_hi = jc_pow[1:, start:, :]
    slopes = np.zeros_like(jc_hi)
    penalties = np.zeros((nsurf - 1,), dtype=float)
    for si in range(nsurf - 1):
        for ialpha_hi in range(nalpha):
            hi_alpha = jc_hi[si, :, ialpha_hi]  # (n_bounce-1,)
            slope_terms = (hi_alpha[:, None] - jc_lo[si]) / (
                ds[si] * (0.5 * (hi_alpha[:, None] + jc_lo[si]) + 1.0e-10)
            )
            slopes[si, :, ialpha_hi] = np.mean(slope_terms, axis=1)
        violation = np.maximum(0.0, slopes[si] - target_maxj)
        penalties[si] = np.sqrt(np.sum(violation**2))
    return penalties, slopes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmec-input", type=Path, required=True)
    parser.add_argument("--surfaces", type=str, default="0.1,0.28,0.46,0.64,0.82,1.0")
    parser.add_argument("--mboz", type=int, default=18)
    parser.add_argument("--nboz", type=int, default=18)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--nphi", type=int, default=141)
    parser.add_argument("--nalpha", type=int, default=27)
    parser.add_argument("--n-bounce", type=int, default=51)
    parser.add_argument("--nphi-int", type=int, default=141)
    parser.add_argument("--p-j", type=float, default=1.0)
    parser.add_argument("--p-lambda", type=float, default=1.0)
    parser.add_argument("--target-maxj", type=float, default=-0.06)
    parser.add_argument("--bounce-start", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_compare_maxj_reference_vs_j_invariant"),
    )
    args = parser.parse_args()

    surfaces = np.asarray([float(x.strip()) for x in args.surfaces.split(",") if x.strip()], dtype=float)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = vj.VmecInput.from_file(args.vmec_input)
    eq = opt.solve_equilibrium(inp)
    booz = boozer_bmnc_state(
        eq.state,
        eq.runtime,
        surfaces=surfaces.tolist(),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        oversample=int(args.oversample),
    )

    phi, alpha, b_lines = qi_j._synthesize_boozer_field_lines(
        bmnc_b=booz["bmnc_b"],
        xm_b=booz["xm_b"],
        xn_b=booz["xn_b"],
        iota_b=booz["iota_b"],
        nfp=int(booz["nfp"]),
        nphi=int(args.nphi),
        nalpha=int(args.nalpha),
    )
    phi = np.asarray(phi, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    b_lines = np.asarray(b_lines, dtype=float)
    gi_all = np.asarray(booz["gi_b"], dtype=float)

    bj_norm = np.power(
        np.arange(int(args.n_bounce), dtype=float) / max(int(args.n_bounce) - 1, 1),
        float(args.p_lambda),
    )

    nsurf = len(surfaces)
    jc_ref = np.zeros((nsurf, int(args.n_bounce), int(args.nalpha)), dtype=float)
    jc_smooth = np.zeros_like(jc_ref)

    for isurf in range(nsurf):
        b_surface = b_lines[isurf]
        gi_value = float(gi_all[isurf])
        for ialpha in range(int(args.nalpha)):
            b_line = b_surface[ialpha]
            bmin = float(np.min(b_line))
            bmax = float(np.max(b_line))
            scale = max(bmax - bmin, 1.0e-14)
            b_norm = (b_line - bmin) / scale
            ref_target = _reference_goodman_transform(b_norm, phi)
            _, jc_r, _, _ = _reference_j_pair(
                phi,
                b_norm,
                ref_target,
                bj_norm,
                gi_value,
                scale=scale,
                bmin=bmin,
                nphi_int=int(args.nphi_int),
            )
            jc_ref[isurf, :, ialpha] = jc_r

            smooth_target = np.asarray(qi_j._apply_smooth_goodman_transform(b_norm, phi), dtype=float)
            bj_phys = bj_norm * scale + bmin
            _, jc_s = qi_j._compute_j_pair(
                phi,
                b_line,
                smooth_target * scale + bmin,
                bj_phys,
                gi_value,
                nphi_int=int(args.nphi_int),
            )
            jc_smooth[isurf, :, ialpha] = np.asarray(jc_s, dtype=float)

    ref_penalty, ref_slope = _reference_maxj_from_j(
        jc_ref,
        surfaces,
        target_maxj=float(args.target_maxj),
        p_j=float(args.p_j),
        bounce_start=int(args.bounce_start),
    )
    vmex_penalty, vmex_slope = _vmex_maxj_from_j(
        jc_smooth,
        surfaces,
        target_maxj=float(args.target_maxj),
        p_j=float(args.p_j),
        bounce_start=int(args.bounce_start),
    )

    ref_slope_cmp = ref_slope[:, int(args.bounce_start) :, :]
    slope_diff = vmex_slope - ref_slope_cmp

    max_flat = int(np.argmax(np.abs(slope_diff)))
    max_pair, max_bj_idx, max_alpha_idx = np.unravel_index(max_flat, slope_diff.shape)
    max_bj_full_idx = max_bj_idx + int(args.bounce_start)
    max_bj_value = float(bj_norm[max_bj_full_idx])
    max_ref_slope = float(ref_slope_cmp[max_pair, max_bj_idx, max_alpha_idx])
    max_vmex_slope = float(vmex_slope[max_pair, max_bj_idx, max_alpha_idx])
    max_abs_diff = float(abs(max_vmex_slope - max_ref_slope))

    summary = {
        "vmec_input": str(args.vmec_input),
        "surfaces": surfaces.tolist(),
        "reference_penalty": ref_penalty.tolist(),
        "vmex_penalty": vmex_penalty.tolist(),
        "bounce_start": int(args.bounce_start),
        "penalty_rel_diff_l2": float(
            np.linalg.norm(vmex_penalty - ref_penalty) / max(np.linalg.norm(ref_penalty), 1.0e-14)
        ),
        "jc_rms_diff": float(np.sqrt(np.mean((jc_ref - jc_smooth) ** 2))),
        "slope_rms_diff": float(np.sqrt(np.mean(slope_diff**2))),
        "slope_max_abs_diff": max_abs_diff,
        "worst_surface_pair_index": int(max_pair),
        "worst_surface_pair_values": [float(surfaces[max_pair]), float(surfaces[max_pair + 1])],
        "worst_bj_index": int(max_bj_full_idx),
        "worst_bj_norm": max_bj_value,
        "worst_alpha_index": int(max_alpha_idx),
        "worst_reference_slope": max_ref_slope,
        "worst_vmex_slope": max_vmex_slope,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        out_dir / "comparison_arrays.npz",
        surfaces=surfaces,
        phi=phi,
        alpha=alpha,
        bj_norm=bj_norm,
        jc_ref=jc_ref,
        jc_smooth=jc_smooth,
        ref_penalty=ref_penalty,
        vmex_penalty=vmex_penalty,
        ref_slope=ref_slope_cmp,
        vmex_slope=vmex_slope,
        slope_diff=slope_diff,
    )

    print(f"[compare-maxj] wrote {out_dir / 'summary.json'}")
    print(f"[compare-maxj] reference_penalty = {np.array2string(ref_penalty, precision=6)}")
    print(f"[compare-maxj] vmex_penalty      = {np.array2string(vmex_penalty, precision=6)}")
    print(f"[compare-maxj] penalty_rel_diff  = {summary['penalty_rel_diff_l2']:.6e}")
    print(f"[compare-maxj] jc_rms_diff       = {summary['jc_rms_diff']:.6e}")
    print(f"[compare-maxj] slope_rms_diff    = {summary['slope_rms_diff']:.6e}")
    print(f"[compare-maxj] slope_max_abs_diff= {summary['slope_max_abs_diff']:.6e}")
    print(
        "[compare-maxj] worst_slope_mismatch "
        f"pair={max_pair} surfaces=({surfaces[max_pair]:.3f},{surfaces[max_pair + 1]:.3f}) "
        f"bj_idx={max_bj_full_idx} bj_norm={max_bj_value:.4f} alpha={max_alpha_idx} "
        f"ref={max_ref_slope:.6e} vmex={max_vmex_slope:.6e}"
    )


if __name__ == "__main__":
    main()
