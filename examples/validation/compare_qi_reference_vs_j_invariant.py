#!/usr/bin/env python
"""Compare ``qi_functions_mod.py`` J-QI construction against smooth VMEX J-QI.

This script uses the same traceable Boozer field-line data for both paths, then
applies:

1. a direct NumPy/SciPy reimplementation of the ``qi_functions_mod.py``
   Goodman well construction + ``GetBranches`` + J-integral logic; and
2. the smooth JAX implementation from ``vmex.core.omnigenity_j``.

That isolates differences in well construction / branch selection / J
evaluation without confounding them with two different Boozer transforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import UnivariateSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import vmex as vj
from vmex.core.omnigenity import boozer_bmnc_state
from vmex.core import omnigenity_j as qi_j


def _reference_goodman_transform(b_line: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Literal well construction from ``qi_functions_mod.py`` on one line."""

    ba = np.asarray(b_line, dtype=float).copy()
    phisa = np.asarray(phi, dtype=float)
    indmin = int(np.argmin(ba))

    bl = ba[: indmin + 1].copy()
    phisl = phisa[: indmin + 1]
    phisr = phisa[indmin:]
    indmax_l = int(np.argmax(bl))
    bl[:indmax_l] = bl[indmax_l]
    for i in range(len(bl) - 1):
        if bl[i] <= bl[i + 1]:
            jf = len(bl) - 1
            for j in range(i + 1, len(bl)):
                if bl[j] < bl[i]:
                    jf = j
                    break
            bl[i:jf] = bl[i]

    br = ba[indmin:].copy()
    indmax_r = int(np.argmax(br))
    br[indmax_r:] = br[indmax_r]
    for j in range(len(br) - 1, 1, -1):
        if br[j - 1] >= br[j]:
            kf = 0
            for k in range(j - 1, 1, -1):
                if br[k] < br[j]:
                    kf = k
                    break
            br[kf + 1 : j] = br[j]

    pmax = 50
    pmin = 15
    x1l = (phisl - phisl[0]) / max(phisl[-1] - phisl[0], 1.0e-14)
    x1r = (phisr - phisr[0]) / max(phisr[-1] - phisr[0], 1.0e-14)
    fl = np.where(
        x1l < 0.5,
        (1.0 - bl[0]) * ((np.cos(2.0 * np.pi * x1l) + 1.0) / 2.0) ** pmax,
        (-bl[-1]) * ((np.cos(2.0 * np.pi * x1l) + 1.0) / 2.0) ** pmin,
    )
    fr = np.where(
        x1r < 0.5,
        (-br[0]) * ((np.cos(2.0 * np.pi * x1r) + 1.0) / 2.0) ** pmin,
        (1.0 - br[-1]) * ((np.cos(2.0 * np.pi * x1r) + 1.0) / 2.0) ** pmax,
    )
    bl = bl + fl
    br = br + fr
    bl = bl[:-1]
    return np.concatenate((bl, br))


def _reference_get_branches(phi_bs: np.ndarray, ba: np.ndarray, bj: float) -> tuple[float, float]:
    """Literal ``GetBranches`` from ``qi_functions_mod.py``."""

    bmax = 1.0
    bmin = 0.0
    diffs = ba - bj
    diffsgn = diffs[:-1] * diffs[1:]
    inds = np.where(diffsgn < 0)[0]
    inds = np.sort(inds)

    if bj <= bmin:
        imin = int(np.argmin(ba))
        phimin = phi_bs[imin]
        return float(phimin), float(phimin)
    if bj >= bmax:
        return float(phi_bs[0]), float(phi_bs[-1])

    if len(inds) < 2:
        inds = np.where(diffsgn <= 0)[0]
        for iind in range(1, len(inds)):
            if inds[iind] != inds[iind - 1] + 1:
                inds = [inds[iind - 1], inds[-1]]
                break
    if len(inds) > 2:
        inds = [inds[0], inds[-1]]
    ind1 = int(inds[0])
    ind2 = int(inds[1])

    dy1 = ba[ind1] - ba[ind1 + 1]
    dx1 = phi_bs[ind1] - phi_bs[ind1 + 1]
    m1 = dy1 / dx1
    b1 = ba[ind1] - m1 * phi_bs[ind1]
    phi1 = (bj - b1) / m1 if m1 != 0 else phi_bs[ind1]

    dy2 = ba[ind2] - ba[ind2 + 1]
    dx2 = phi_bs[ind2] - phi_bs[ind2 + 1]
    m2 = dy2 / dx2
    b2 = ba[ind2] - m2 * phi_bs[ind2]
    phi2 = (bj - b2) / m2 if m2 != 0 else phi_bs[ind2 + 1]
    return float(phi1), float(phi2)


def _reference_j_pair(
    phi: np.ndarray,
    b_input_norm: np.ndarray,
    b_target_norm: np.ndarray,
    bj_norm: np.ndarray,
    gi_value: float,
    *,
    scale: float,
    bmin: float,
    nphi_int: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ji = np.zeros((len(bj_norm),), dtype=float)
    jc = np.zeros((len(bj_norm),), dtype=float)
    p1 = np.zeros((len(bj_norm),), dtype=float)
    p2 = np.zeros((len(bj_norm),), dtype=float)
    for j, bj in enumerate(bj_norm):
        phi1, phi2 = _reference_get_branches(phi, b_target_norm, float(bj))
        p1[j] = phi1
        p2[j] = phi2
        if phi1 == phi2:
            continue
        phibounce = np.linspace(phi1, phi2, nphi_int)
        bc_phys = np.interp(phibounce, phi, b_target_norm) * scale + bmin
        bi_phys = np.interp(phibounce, phi, b_input_norm) * scale + bmin
        bj_phys = bj * scale + bmin
        bc1 = 1.0 - bc_phys / bj_phys
        bi1 = 1.0 - bi_phys / bj_phys
        bc1[np.abs(bc1) < 1.0e-10] = 0.0
        bi1[np.abs(bi1) < 1.0e-10] = 0.0
        jc_integrand = UnivariateSpline(
            phibounce, np.sqrt(np.maximum(bc1, 0.0)) * gi_value / bi_phys, k=1, s=0
        )
        ji_integrand = UnivariateSpline(
            phibounce, np.sign(bi1) * np.sqrt(np.abs(bi1)) * gi_value / bi_phys, k=1, s=0
        )
        ji[j] = float(ji_integrand.integral(phi1, phi2))
        jc[j] = float(jc_integrand.integral(phi1, phi2))
    return ji, jc, p1, p2


def _qi_surface_from_j(ji: np.ndarray, jc: np.ndarray, p_j: float) -> float:
    ji_pow = np.power(ji.T, p_j)  # (nalpha, n_bounce)
    jc_pow = np.power(jc.T, p_j)
    nalpha = ji_pow.shape[0]
    pair_sum = nalpha * np.sum(ji_pow * ji_pow + jc_pow * jc_pow, axis=0) - 2.0 * np.sum(
        ji_pow, axis=0
    ) * np.sum(jc_pow, axis=0)
    qi_num = np.sum(pair_sum)
    mean_denom = (np.sum(ji_pow + jc_pow) / (2.0 * ji_pow.shape[1])) ** 2
    return float(np.sqrt(max(qi_num, 0.0) / max(mean_denom, 1.0e-14)))


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--vmec-input", type=Path, required=True)
    parser.add_argument("--surface", type=float, default=0.5)
    parser.add_argument("--mboz", type=int, default=18)
    parser.add_argument("--nboz", type=int, default=18)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--nphi", type=int, default=141)
    parser.add_argument("--nalpha", type=int, default=27)
    parser.add_argument("--n-bounce", type=int, default=51)
    parser.add_argument("--nphi-int", type=int, default=141)
    parser.add_argument("--p-j", type=float, default=1.0)
    parser.add_argument("--p-lambda", type=float, default=1.0)
    parser.add_argument("--alpha-indices", type=str, default="")
    parser.add_argument("--bounce-indices", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_compare_qi_reference_vs_j_invariant"),
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = vj.VmecInput.from_file(args.vmec_input)
    eq = vj.solve_multigrid(inp, verbose=True)
    booz = boozer_bmnc_state(
        eq.state,
        eq.runtime,
        surfaces=[args.surface],
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
    b_surface = np.asarray(b_lines[0], dtype=float)
    gi_value = float(np.asarray(booz["gi_b"], dtype=float)[0])
    bj_norm = np.power(
        np.arange(int(args.n_bounce), dtype=float) / max(int(args.n_bounce) - 1, 1),
        float(args.p_lambda),
    )

    ji_ref = np.zeros((int(args.n_bounce), int(args.nalpha)), dtype=float)
    jc_ref = np.zeros_like(ji_ref)
    ji_smooth = np.zeros_like(ji_ref)
    jc_smooth = np.zeros_like(ji_ref)
    ref_target_lines = np.zeros_like(b_surface)
    smooth_target_lines = np.zeros_like(b_surface)
    ref_p1 = np.zeros((int(args.n_bounce), int(args.nalpha)), dtype=float)
    ref_p2 = np.zeros_like(ref_p1)
    smooth_p1 = np.zeros_like(ref_p1)
    smooth_p2 = np.zeros_like(ref_p1)

    for ialpha in range(int(args.nalpha)):
        b_line = b_surface[ialpha]
        bmin = float(np.min(b_line))
        bmax = float(np.max(b_line))
        scale = max(bmax - bmin, 1.0e-14)
        b_norm = (b_line - bmin) / scale

        ref_target = _reference_goodman_transform(b_norm, phi)
        smooth_target = np.asarray(qi_j._apply_smooth_goodman_transform(b_norm, phi), dtype=float)
        ref_target_lines[ialpha] = ref_target
        smooth_target_lines[ialpha] = smooth_target

        ji_r, jc_r, p1_r, p2_r = _reference_j_pair(
            phi,
            b_norm,
            ref_target,
            bj_norm,
            gi_value,
            scale=scale,
            bmin=bmin,
            nphi_int=int(args.nphi_int),
        )
        ji_ref[:, ialpha] = ji_r
        jc_ref[:, ialpha] = jc_r
        ref_p1[:, ialpha] = p1_r
        ref_p2[:, ialpha] = p2_r

        bj_phys = bj_norm * scale + bmin
        ji_s, jc_s = qi_j._compute_j_pair(
            phi,
            b_line,
            smooth_target * scale + bmin,
            bj_phys,
            gi_value,
            nphi_int=int(args.nphi_int),
        )
        ji_smooth[:, ialpha] = np.asarray(ji_s, dtype=float)
        jc_smooth[:, ialpha] = np.asarray(jc_s, dtype=float)
        p1_s, p2_s = zip(*[qi_j._branch_crossings(phi, smooth_target, bj) for bj in bj_norm], strict=False)
        smooth_p1[:, ialpha] = np.asarray(p1_s, dtype=float)
        smooth_p2[:, ialpha] = np.asarray(p2_s, dtype=float)

    qi_ref = _qi_surface_from_j(ji_ref, jc_ref, float(args.p_j))
    qi_smooth = _qi_surface_from_j(ji_smooth, jc_smooth, float(args.p_j))

    default_alphas = [0, int(args.nalpha) // 4, int(args.nalpha) // 2]
    alpha_indices = [idx for idx in _parse_int_list(args.alpha_indices, default_alphas) if 0 <= idx < int(args.nalpha)]
    default_bounces = [
        max(int(args.n_bounce) // 2, 0),
        max((3 * int(args.n_bounce)) // 4, 0),
        max(int(args.n_bounce) - 2, 0),
    ]
    bounce_indices = [
        idx for idx in _parse_int_list(args.bounce_indices, default_bounces) if 0 <= idx < int(args.n_bounce)
    ]

    for ialpha in alpha_indices:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(phi, (b_surface[ialpha] - np.min(b_surface[ialpha])) / max(np.ptp(b_surface[ialpha]), 1.0e-14), label="B_I", lw=2.0)
        ax.plot(phi, ref_target_lines[ialpha], label="B_C reference", lw=2.0)
        ax.plot(phi, smooth_target_lines[ialpha], label="B_C smooth", lw=2.0, ls="--")
        for ib in bounce_indices:
            bj = bj_norm[ib]
            ax.scatter([ref_p1[ib, ialpha], ref_p2[ib, ialpha]], [bj, bj], s=25, marker="o")
            ax.scatter([smooth_p1[ib, ialpha], smooth_p2[ib, ialpha]], [bj, bj], s=25, marker="x")
        ax.set_title(f"surface={args.surface:.3f}, alpha_index={ialpha}")
        ax.set_xlabel("phi")
        ax.set_ylabel("normalized |B|")
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"well_compare_alpha_{ialpha:03d}.png", dpi=180)
        plt.close(fig)

    summary = {
        "vmec_input": str(args.vmec_input),
        "surface": float(args.surface),
        "qi_surface_reference": qi_ref,
        "qi_surface_smooth": qi_smooth,
        "qi_surface_rel_diff": abs(qi_smooth - qi_ref) / max(abs(qi_ref), 1.0e-14),
        "ref_target_rms_diff": float(np.sqrt(np.mean((ref_target_lines - smooth_target_lines) ** 2))),
        "ji_rms_diff": float(np.sqrt(np.mean((ji_ref - ji_smooth) ** 2))),
        "jc_rms_diff": float(np.sqrt(np.mean((jc_ref - jc_smooth) ** 2))),
        "alpha_indices": alpha_indices,
        "bounce_indices": bounce_indices,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        out_dir / "comparison_arrays.npz",
        phi=phi,
        alpha=alpha,
        bj_norm=bj_norm,
        b_input=b_surface,
        ref_target=ref_target_lines,
        smooth_target=smooth_target_lines,
        ji_ref=ji_ref,
        jc_ref=jc_ref,
        ji_smooth=ji_smooth,
        jc_smooth=jc_smooth,
        ref_p1=ref_p1,
        ref_p2=ref_p2,
        smooth_p1=smooth_p1,
        smooth_p2=smooth_p2,
    )

    print(f"[compare-qi] wrote {out_dir / 'summary.json'}")
    print(f"[compare-qi] qi_surface_reference = {qi_ref:.6e}")
    print(f"[compare-qi] qi_surface_smooth    = {qi_smooth:.6e}")
    print(f"[compare-qi] rel_diff             = {summary['qi_surface_rel_diff']:.6e}")
    for ialpha in alpha_indices:
        print(f"[compare-qi] wrote {out_dir / f'well_compare_alpha_{ialpha:03d}.png'}")


if __name__ == "__main__":
    main()
