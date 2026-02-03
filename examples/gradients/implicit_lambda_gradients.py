"""Intermediate example: implicit differentiation through the lambda-only solve.

This script demonstrates a Step-9 capability: differentiating an outer scalar
objective through an equilibrium sub-solve **without** backpropagating through
many optimizer iterations.

It solves for VMEC lambda at fixed R/Z and then differentiates a simple
diagnostic objective (lambda L2 norm) w.r.t. a scalar scale factor on the flux
profiles.

Outputs:
- figures_implicit_lambda/lambda_norm_vs_scale.(png|pdf)
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

from vmec_jax._compat import enable_x64, has_jax, jax, jnp
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import TWOPI, lamscale_from_phips, signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.implicit import solve_lambda_state_implicit
from vmec_jax.init_guess import initial_guess_from_boundary
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
    p.add_argument("--outdir", type=str, default="figures_implicit_lambda")
    p.add_argument("--n", type=int, default=9, help="Number of scale samples")
    p.add_argument("--scale-min", type=float, default=0.8)
    p.add_argument("--scale-max", type=float, default=1.2)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not has_jax():
        raise SystemExit("This example requires JAX (pip install -e .[jax])")
    enable_x64(True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, indata = load_config(args.input)
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)
    flux0 = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    phipf0 = jnp.asarray(flux0.phipf)
    chipf0 = jnp.asarray(flux0.chipf)
    s = jnp.asarray(static.s)

    def lambda_norm(scale):
        phipf = scale * phipf0
        chipf = scale * chipf0
        phips = (signgs * phipf) / TWOPI
        lamscale = lamscale_from_phips(phips, s)
        st = solve_lambda_state_implicit(
            st0,
            static,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs,
            lamscale=lamscale,
            sqrtg=jnp.asarray(g0.sqrtg),
            max_iter=int(args.max_iter),
            grad_tol=1e-10,
        )
        return jnp.sqrt(jnp.mean(st.Lcos**2 + st.Lsin**2))

    scale0 = 1.0
    g = float(jax.grad(lambda_norm)(scale0))
    if args.verbose:
        print(f"d/dscale ||lambda||_rms at scale=1: {g:.6e}")

    scales = np.linspace(float(args.scale_min), float(args.scale_max), int(args.n))
    norms = np.array([float(np.asarray(lambda_norm(float(a)))) for a in scales])

    plt = _import_matplotlib()
    _set_pub_style(plt)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(scales, norms, marker="o", ms=4.0, lw=1.8)
    ax.axvline(scale0, color="k", lw=1.0, alpha=0.3)
    ax.set_xlabel("flux scale factor")
    ax.set_ylabel(r"$\\|\\lambda\\|_{\\mathrm{rms}}$")
    ax.set_title("Implicit differentiation through lambda-only solve")
    ax.text(0.02, 0.98, f\"grad@1 = {g:.3e}\", transform=ax.transAxes, va=\"top\", ha=\"left\")
    fig.tight_layout()
    fig.savefig(outdir / "lambda_norm_vs_scale.png")
    fig.savefig(outdir / "lambda_norm_vs_scale.pdf")
    plt.close(fig)

    print(f"wrote: {outdir / 'lambda_norm_vs_scale.png'}")
    print(f"wrote: {outdir / 'lambda_norm_vs_scale.pdf'}")


if __name__ == "__main__":
    main()

