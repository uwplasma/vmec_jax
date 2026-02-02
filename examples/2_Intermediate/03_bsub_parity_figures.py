"""Intermediate example: covariant B (bsub*) parity vs VMEC2000 wout.

This script reconstructs (B_u, B_v) from VMEC `wout` contravariant fields
(B^u, B^v) and the metric, and compares against `wout`'s stored covariant fields
(bsubu, bsubv). It writes publication-ready error-map figures.

Requires:
- netCDF4 (to read wout)
- matplotlib (for figures)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.config import load_config
from vmec_jax.field import bsub_from_bsup
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required for this example (pip install -e .[plots])") from e


def _set_pub_style(plt):
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _hi_res_cfg(cfg, *, mpol: int, ntor: int):
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 16)
    ntheta = 2 * (ntheta // 2)
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 16)
    if nzeta <= 0:
        nzeta = 1
    return replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))


def _half_mesh_coeffs(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--wout", type=str, required=True, help="wout_*.nc file to compare against")
    p.add_argument("--outdir", type=str, default="figures_bsub_parity")
    p.add_argument("--s-index", type=int, default=-1, help="Surface index (default: edge)")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, _indata = load_config(args.input)
    wout = read_wout(args.wout)
    cfg_hi = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg_hi)

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
    g = eval_geom(st_half, static)

    # Nyquist basis for bsub/bsup fields.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)

    bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    bsubu_calc, bsubv_calc = bsub_from_bsup(g, bsupu, bsupv)
    bsubu_calc = np.asarray(bsubu_calc)
    bsubv_calc = np.asarray(bsubv_calc)

    s_index = int(args.s_index)
    if s_index < 0:
        s_index = int(cfg_hi.ns) - 1
    if s_index <= 0 or s_index >= int(cfg_hi.ns):
        raise SystemExit(f"s-index out of range (must be 1..{int(cfg_hi.ns)-1}): {s_index}")

    eu = (bsubu_calc[s_index] - bsubu_ref[s_index]) / (np.abs(bsubu_ref[s_index]) + 1e-14)
    ev = (bsubv_calc[s_index] - bsubv_ref[s_index]) / (np.abs(bsubv_ref[s_index]) + 1e-14)

    plt = _import_matplotlib()
    _set_pub_style(plt)

    # Figure: relative error maps on (theta,zeta) grid.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    im0 = axes[0].imshow(eu.T, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[0].set_title(r"rel. error $bsubu$")
    axes[0].set_xlabel(r"$\\theta$ index")
    axes[0].set_ylabel(r"$\\zeta$ index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(ev.T, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[1].set_title(r"rel. error $bsubv$")
    axes[1].set_xlabel(r"$\\theta$ index")
    axes[1].set_ylabel(r"$\\zeta$ index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"bsub parity on surface s_index={s_index} ({Path(args.wout).name})")
    fig.savefig(outdir / "bsub_parity_errors.png")
    fig.savefig(outdir / "bsub_parity_errors.pdf")
    plt.close(fig)

    print(f"wrote: {outdir / 'bsub_parity_errors.png'}")
    print(f"wrote: {outdir / 'bsub_parity_errors.pdf'}")


if __name__ == "__main__":
    main()

