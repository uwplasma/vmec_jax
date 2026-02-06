"""Intermediate example: profiles + volume + figures.

This script mirrors `examples/05_profiles_and_volume.py` but also creates a small set of
publication-style figures (requires matplotlib).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.integrals import volume_from_sqrtg
from vmec_jax.profiles import eval_profiles
from vmec_jax.static import build_static


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
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--outdir", type=str, default="figures_step3", help="Output directory for .npz and figures")
    p.add_argument("--save", action="store_true", help="Save figures instead of showing them.")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, indata = load_config(args.input)
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    g = eval_geom(st0, static)

    prof = eval_profiles(indata, static.s)
    pressure = np.asarray(prof["pressure"])  # VMEC internal units (mu0*Pa)
    pressure_pa = np.asarray(prof["pressure_pa"])
    iota = np.asarray(prof.get("iota")) if "iota" in prof else None
    current = np.asarray(prof.get("current")) if "current" in prof else None

    dvds, V = volume_from_sqrtg(g.sqrtg, static.s, static.grid.theta, static.grid.zeta, cfg.nfp)
    dvds = np.asarray(dvds)
    V = np.asarray(V)

    np.savez(
        outdir / "profiles_step3.npz",
        s=np.asarray(static.s),
        pressure=pressure,
        pressure_pa=pressure_pa,
        iota=iota if iota is not None else np.asarray([]),
        current=current if current is not None else np.asarray([]),
        dvds=dvds,
        V=V,
        nfp=int(cfg.nfp),
    )

    plt = _import_matplotlib()
    _set_pub_style(plt)

    s = np.asarray(static.s)

    # Figure 1: pressure
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(s, pressure_pa, lw=2.2)
    ax.set_xlabel("s")
    ax.set_ylabel("pressure [Pa]")
    ax.set_title("Input pressure profile")
    fig.tight_layout()
    if args.save:
        fig.savefig(outdir / "pressure_profile.png")
        fig.savefig(outdir / "pressure_profile.pdf")
        plt.close(fig)
    else:
        fig.show()

    # Figure 2: iota / current (if present)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if iota is not None:
        ax.plot(s, iota, lw=2.2, label="iota(s)")
    if current is not None:
        ax.plot(s, current, lw=2.2, label="current I(s)")
    ax.set_xlabel("s")
    ax.set_title("Rotational transform / current profile")
    ax.legend()
    fig.tight_layout()
    if args.save:
        fig.savefig(outdir / "iota_or_current.png")
        fig.savefig(outdir / "iota_or_current.pdf")
        plt.close(fig)
    else:
        fig.show()

    # Figure 3: volume profile
    fig, ax = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True)
    ax[0].plot(s, dvds, lw=2.2)
    ax[0].set_ylabel("dV/ds (per field period)")
    ax[1].plot(s, V, lw=2.2)
    ax[1].set_ylabel("V(s) (per field period)")
    ax[1].set_xlabel("s")
    V_total = float(V[-1]) * float(cfg.nfp)
    fig.suptitle(f"Volume profile (V_total={V_total:.3e})")
    fig.tight_layout()
    fig.savefig(outdir / "volume_profile.png")
    fig.savefig(outdir / "volume_profile.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
