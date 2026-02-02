"""Advanced example: implicit differentiation through the full fixed-boundary solve.

This script demonstrates Step-9 implicit differentiation for the fixed-boundary
equilibrium solve over (R, Z, lambda). It computes the sensitivity of a purely
geometric quantity (total volume) w.r.t. a scale factor applied to ``chipf``
(equivalently, scaling iota for current-free cases).

Outputs:
- figures_implicit_fixed_boundary/volume_vs_iota_scale.(png|pdf)
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

from vmec_jax._compat import enable_x64, has_jax, jax, jnp
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.implicit import ImplicitFixedBoundaryOptions, solve_fixed_boundary_state_implicit
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.integrals import volume_from_sqrtg
from vmec_jax.profiles import eval_profiles
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
    p.add_argument("--wout", type=str, default="", help="Optional wout_*.nc to start from a converged state")
    p.add_argument("--outdir", type=str, default="figures_implicit_fixed_boundary")
    p.add_argument("--n", type=int, default=7, help="Number of scale samples")
    p.add_argument("--scale-min", type=float, default=0.9)
    p.add_argument("--scale-max", type=float, default=1.1)
    p.add_argument("--max-iter", type=int, default=18)
    p.add_argument("--hi-res", action="store_true", help="Increase angular resolution for smoother volume sensitivity")
    args = p.parse_args()

    if not has_jax():
        raise SystemExit("This example requires JAX (pip install -e .[jax])")
    enable_x64(True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, indata = load_config(args.input)
    if args.hi_res:
        ntheta = max(int(cfg.ntheta), 32)
        ntheta = 2 * (ntheta // 2)
        cfg = replace(cfg, ntheta=ntheta, nzeta=max(int(cfg.nzeta), 1))
    static = build_static(cfg)

    if args.wout:
        wout = read_wout(args.wout)
        st0 = state_from_wout(wout)
    else:
        bdy = boundary_from_indata(indata, static.modes)
        st0 = initial_guess_from_boundary(static, bdy, indata)

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    flux0 = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    prof = eval_profiles(indata, static.s)
    pressure = jnp.asarray(prof.get("pressure", np.zeros_like(np.asarray(static.s))))
    gamma = float(indata.get_float("GAMMA", 0.0))

    phipf0 = jnp.asarray(flux0.phipf)
    chipf0 = jnp.asarray(flux0.chipf)

    def V_equilibrium(alpha):
        st = solve_fixed_boundary_state_implicit(
            st0,
            static,
            phipf=phipf0,
            chipf=alpha * chipf0,
            signgs=signgs,
            lamscale=jnp.asarray(flux0.lamscale),
            pressure=pressure,
            gamma=gamma,
            jacobian_penalty=1e3,
            solver="lbfgs",
            max_iter=int(args.max_iter),
            step_size=1.0,
            history_size=8,
            grad_tol=1e-10,
            preconditioner="mode_diag+radial_tridi",
            precond_exponent=1.0,
            precond_radial_alpha=0.5,
            implicit=ImplicitFixedBoundaryOptions(cg_max_iter=60, cg_tol=1e-10, damping=1e-5),
        )
        g = eval_geom(st, static)
        _dvds, V = volume_from_sqrtg(g.sqrtg, static.s, static.grid.theta, static.grid.zeta, nfp=int(cfg.nfp))
        return V[-1] * float(cfg.nfp)

    g_at_1 = float(np.asarray(jax.grad(V_equilibrium)(1.0)))
    scales = np.linspace(float(args.scale_min), float(args.scale_max), int(args.n))
    vols = np.array([float(np.asarray(V_equilibrium(float(a)))) for a in scales])

    plt = _import_matplotlib()
    _set_pub_style(plt)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(scales, vols, marker="o", ms=4.0, lw=1.8)
    ax.axvline(1.0, color="k", lw=1.0, alpha=0.3)
    ax.set_xlabel("chipf scale factor (≈ iota scale)")
    ax.set_ylabel("total volume")
    ax.set_title("Implicit differentiation through fixed-boundary equilibrium")
    ax.text(0.02, 0.98, f\"dV/dα @ 1 = {g_at_1:.3e}\", transform=ax.transAxes, va=\"top\", ha=\"left\")
    fig.tight_layout()
    fig.savefig(outdir / "volume_vs_iota_scale.png")
    fig.savefig(outdir / "volume_vs_iota_scale.pdf")
    plt.close(fig)

    print(f"wrote: {outdir / 'volume_vs_iota_scale.png'}")
    print(f"wrote: {outdir / 'volume_vs_iota_scale.pdf'}")


if __name__ == "__main__":
    main()

