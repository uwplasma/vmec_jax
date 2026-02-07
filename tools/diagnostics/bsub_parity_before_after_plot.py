from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.config import load_config
from vmec_jax.field import bsub_from_bsup
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.vmec_tomnsp import vmec_angle_grid
from vmec_jax.wout import read_wout, state_from_wout


CASES = [
    ("li383_low_res", "examples/data/input.li383_low_res", "examples/data/wout_li383_low_res_reference.nc"),
    ("n3are_R7.75B5.7_lowres", "examples/data/input.n3are_R7.75B5.7_lowres", "examples/data/wout_n3are_R7.75B5.7_lowres.nc"),
]


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


def _half_mesh_coeffs(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


def _bsub_errors(case: str, input_rel: str, wout_rel: str):
    input_path = REPO_ROOT / input_rel
    wout_path = REPO_ROOT / wout_rel
    cfg, _ = load_config(str(input_path))
    wout = read_wout(wout_path)

    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg, grid=grid)
    st = state_from_wout(wout)

    st_half = replace(
        st,
        Rcos=_half_mesh_coeffs(np.asarray(st.Rcos)),
        Rsin=_half_mesh_coeffs(np.asarray(st.Rsin)),
        Zcos=_half_mesh_coeffs(np.asarray(st.Zcos)),
        Zsin=_half_mesh_coeffs(np.asarray(st.Zsin)),
        Lcos=np.asarray(st.Lcos),
        Lsin=np.asarray(st.Lsin),
    )
    geom = eval_geom(st_half, static)

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    basis_nyq = build_helical_basis(modes_nyq, AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp))
    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    bsubu_eval, bsubv_eval = bsub_from_bsup(geom, bsupu_ref, bsupv_ref)

    bc_vmec = vmec_bcovar_half_mesh_from_wout(state=st, static=static, wout=wout, use_wout_bsup=False, use_vmec_synthesis=True)

    sl = slice(1, None)
    err_eval_u = _rel_rms(np.asarray(bsubu_eval)[sl], bsubu_ref[sl])
    err_eval_v = _rel_rms(np.asarray(bsubv_eval)[sl], bsubv_ref[sl])
    err_vmec_u = _rel_rms(np.asarray(bc_vmec.bsubu)[sl], bsubu_ref[sl])
    err_vmec_v = _rel_rms(np.asarray(bc_vmec.bsubv)[sl], bsubv_ref[sl])

    return {
        "case": case,
        "eval_u": err_eval_u,
        "eval_v": err_eval_v,
        "vmec_u": err_vmec_u,
        "vmec_v": err_vmec_v,
    }


def main():
    enable_x64()
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit("This example requires matplotlib.") from e

    results = [_bsub_errors(*case) for case in CASES]

    labels = [r["case"] for r in results]
    eval_u = [r["eval_u"] for r in results]
    eval_v = [r["eval_v"] for r in results]
    vmec_u = [r["vmec_u"] for r in results]
    vmec_v = [r["vmec_v"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes[0].bar(x - width / 2, eval_u, width, label="eval_fourier")
    axes[0].bar(x + width / 2, vmec_u, width, label="vmec_synthesis")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("rel RMS error")
    axes[0].set_title("bsubu parity")
    axes[0].set_yscale("log")

    axes[1].bar(x - width / 2, eval_v, width, label="eval_fourier")
    axes[1].bar(x + width / 2, vmec_v, width, label="vmec_synthesis")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_title("bsubv parity")
    axes[1].set_yscale("log")

    axes[0].legend(loc="upper right")
    fig.tight_layout()

    out_path = REPO_ROOT / "docs/_static/figures/bsub_parity_before_after.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
