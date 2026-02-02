"""Intermediate example: contravariant B (bsup*) parity vs VMEC2000 wout.

This script reconstructs (B^u, B^v) from vmec_jax using the geometry + lambda
field stored in a VMEC `wout` file, and compares against `wout`'s stored
contravariant fields (bsupu, bsupv). It writes publication-ready error-map
figures.

Notes
-----
Near-axis surfaces can be sensitive to coordinate singularities and VMEC's
axis expansions. This example is intended for visual inspection on outer
surfaces (default: edge).

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
from vmec_jax.field import bsup_from_geom, lamscale_from_phips
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


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--wout", type=str, required=True, help="wout_*.nc file to compare against")
    p.add_argument("--outdir", type=str, default="figures_bsup_parity")
    p.add_argument("--s-index", type=int, default=-1, help="Surface index (default: edge)")
    p.add_argument("--outer-frac", type=float, default=0.25, help="Radial fraction to include in RMS (default: 0.25)")
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

    # Nyquist basis for reference bsup fields.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)

    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))

    lamscale = lamscale_from_phips(wout.phips, static.s)
    bsupu_calc, bsupv_calc = bsup_from_geom(
        g,
        phipf=wout.phipf,
        chipf=wout.chipf,
        nfp=wout.nfp,
        signgs=wout.signgs,
        lamscale=lamscale,
    )
    bsupu_calc = np.asarray(bsupu_calc)
    bsupv_calc = np.asarray(bsupv_calc)

    s_index = int(args.s_index)
    if s_index < 0:
        s_index = int(cfg_hi.ns) - 1
    if s_index <= 0 or s_index >= int(cfg_hi.ns):
        raise SystemExit(f"s-index out of range (must be 1..{int(cfg_hi.ns)-1}): {s_index}")

    eu = (bsupu_calc[s_index] - bsupu_ref[s_index]) / (np.abs(bsupu_ref[s_index]) + 1e-14)
    ev = (bsupv_calc[s_index] - bsupv_ref[s_index]) / (np.abs(bsupv_ref[s_index]) + 1e-14)

    js0 = max(1, int(float(args.outer_frac) * (int(cfg_hi.ns) - 1)))
    err_u = _rel_rms(bsupu_calc[js0:], bsupu_ref[js0:])
    err_v = _rel_rms(bsupv_calc[js0:], bsupv_ref[js0:])
    print(f"outer-surface rel RMS (js>={js0}): bsupu={err_u:.3e} bsupv={err_v:.3e}")

    plt = _import_matplotlib()
    _set_pub_style(plt)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    im0 = axes[0].imshow(eu.T, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[0].set_title(r"rel. error $bsupu$")
    axes[0].set_xlabel(r"$\\theta$ index")
    axes[0].set_ylabel(r"$\\zeta$ index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(ev.T, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[1].set_title(r"rel. error $bsupv$")
    axes[1].set_xlabel(r"$\\theta$ index")
    axes[1].set_ylabel(r"$\\zeta$ index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"bsup parity on surface s_index={s_index} ({Path(args.wout).name})")
    fig.savefig(outdir / "bsup_parity_errors.png")
    fig.savefig(outdir / "bsup_parity_errors.pdf")
    plt.close(fig)

    print(f"wrote: {outdir / 'bsup_parity_errors.png'}")
    print(f"wrote: {outdir / 'bsup_parity_errors.pdf'}")


if __name__ == "__main__":
    main()

