"""Advanced example: fixed-boundary solve convergence + figures.

This script runs a short fixed-boundary optimization and writes convergence plots.
It is intended for solver experimentation and performance/profiling work.
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
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.profiles import eval_profiles
from vmec_jax.solve import solve_fixed_boundary_gd, solve_fixed_boundary_lbfgs
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
    p.add_argument("--outdir", type=str, default="figures_solver", help="Output directory for figures")
    p.add_argument("--solver", choices=["gd", "lbfgs"], default="gd")
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--step-size", type=float, default=5e-3)
    p.add_argument("--history-size", type=int, default=10, help="L-BFGS history (if solver=lbfgs)")
    p.add_argument("--jit-grad", action="store_true", help="JIT objective+grad inside the solver")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, indata = load_config(args.input)
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    prof = eval_profiles(indata, static.s)
    pressure = prof.get("pressure", np.zeros_like(np.asarray(static.s)))
    gamma = indata.get_float("GAMMA", 0.0)

    if args.solver == "gd":
        res = solve_fixed_boundary_gd(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=gamma,
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            jacobian_penalty=1e3,
            jit_grad=bool(args.jit_grad),
            verbose=bool(args.verbose),
        )
    else:
        res = solve_fixed_boundary_lbfgs(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=gamma,
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            history_size=int(args.history_size),
            jit_grad=bool(args.jit_grad),
            verbose=bool(args.verbose),
        )

    np.savez(
        outdir / "solver_history.npz",
        w=res.w_history,
        wb=res.wb_history,
        wp=res.wp_history,
        grad_rms=res.grad_rms_history,
        step=res.step_history,
    )

    plt = _import_matplotlib()
    _set_pub_style(plt)

    it = np.arange(res.w_history.size)

    # Figure 1: total energy vs iteration
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(it, res.w_history, marker="o", ms=3.0, lw=1.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("W")
    ax.set_title(f"Fixed-boundary solve convergence ({args.solver})")
    fig.tight_layout()
    fig.savefig(outdir / "energy_convergence.png")
    fig.savefig(outdir / "energy_convergence.pdf")
    plt.close(fig)

    # Figure 2: components wb/wp
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(it, res.wb_history, marker="o", ms=3.0, lw=1.8, label="wb")
    ax.plot(it, res.wp_history, marker="o", ms=3.0, lw=1.8, label="wp")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title("Energy components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "energy_components.png")
    fig.savefig(outdir / "energy_components.pdf")
    plt.close(fig)

    # Figure 3: gradient RMS
    if res.grad_rms_history.size:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.semilogy(np.arange(1, res.grad_rms_history.size + 1), res.grad_rms_history, marker="o", ms=3.0, lw=1.8)
        ax.set_xlabel("iteration")
        ax.set_ylabel("grad RMS")
        ax.set_title("Gradient norm")
        fig.tight_layout()
        fig.savefig(outdir / "grad_rms.png")
        fig.savefig(outdir / "grad_rms.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()

