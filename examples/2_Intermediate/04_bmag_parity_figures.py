"""Intermediate example: |B| parity vs VMEC2000 wout.

This script reconstructs |B| from wout Nyquist contravariant components and the
metric, and compares against wout's stored |B| Fourier coefficients (bmnc/bmns).
It writes publication-ready error-map figures.

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
    p.add_argument("--outdir", type=str, default="figures_bmag_parity")
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

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)

    bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    Bmag_ref = np.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))

    gtt = np.asarray(g.g_tt)
    gtp = np.asarray(g.g_tp)
    gpp = np.asarray(g.g_pp)
    B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
    Bmag_calc = np.sqrt(np.maximum(B2, 0.0))

    s_index = int(args.s_index)
    if s_index < 0:
        s_index = int(cfg_hi.ns) - 1
    if s_index <= 0 or s_index >= int(cfg_hi.ns):
        raise SystemExit(f"s-index out of range (must be 1..{int(cfg_hi.ns)-1}): {s_index}")

    e = (Bmag_calc[s_index] - Bmag_ref[s_index]) / (np.abs(Bmag_ref[s_index]) + 1e-14)

    plt = _import_matplotlib()
    _set_pub_style(plt)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(e.T, origin="lower", aspect="auto", cmap="RdBu_r")
    ax.set_title(r"rel. error $|B|$")
    ax.set_xlabel(r"$\theta$ index")
    ax.set_ylabel(r"$\zeta$ index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "bmag_parity_error.png")
    fig.savefig(outdir / "bmag_parity_error.pdf")
    plt.close(fig)

    print(f"wrote: {outdir / 'bmag_parity_error.png'}")
    print(f"wrote: {outdir / 'bmag_parity_error.pdf'}")


if __name__ == "__main__":
    main()
