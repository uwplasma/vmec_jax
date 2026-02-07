"""Differentiate through a (small) fixed-boundary solve w.r.t. boundary coefficients.

Goal: compute d(mean(|B| on LCFS))/d(params), where params perturb boundary RBC/ZBS.

This example intentionally unrolls a tiny number of solve iterations and uses
plain `jax.grad` for clarity. For large runs, use the implicit-diff utilities
in `vmec_jax.implicit`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmec_jax as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="circular_tokamak")
    p.add_argument("--max-iter", type=int, default=2, help="Keep tiny for runtime; this is a demo.")
    p.add_argument("--step-size", type=float, default=5e-3)
    args = p.parse_args()

    from vmec_jax._compat import enable_x64, has_jax, jax, jnp

    if not has_jax():
        raise SystemExit("This example requires JAX (pip install -e '.[jax]').")
    enable_x64(True)

    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples" / "data" / f"input.{args.case}"

    cfg, indata = vj.load_config(input_path)
    static = vj.build_static(cfg)
    boundary0 = vj.boundary_from_indata(indata, static.modes)

    # Pick a few boundary coefficients to treat as parameters.
    # RBC(0,0), RBC(0,1), ZBS(0,1) for a tokamak-like shape.
    modes = static.modes
    k00 = int(np.where((np.asarray(modes.m) == 0) & (np.asarray(modes.n) == 0))[0][0])
    k10 = int(np.where((np.asarray(modes.m) == 1) & (np.asarray(modes.n) == 0))[0][0])
    idx = jnp.asarray([k00, k10], dtype=int)

    def objective(params):
        # params = [dRBC(0,0), dRBC(0,1), dZBS(0,1)]
        dR00, dR10, dZ10 = params
        Rcos = jnp.asarray(boundary0.R_cos).at[k00].add(dR00).at[k10].add(dR10)
        Zsin = jnp.asarray(boundary0.Z_sin).at[k10].add(dZ10)
        bdy = vj.BoundaryCoeffs(R_cos=Rcos, R_sin=jnp.asarray(boundary0.R_sin), Z_cos=jnp.asarray(boundary0.Z_cos), Z_sin=Zsin)

        st0 = vj.initial_guess_from_boundary(static, bdy, indata, vmec_project=False)
        g0 = vj.eval_geom(st0, static)
        signgs = vj.signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)
        flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
        pressure = jnp.zeros_like(jnp.asarray(static.s))

        # Small explicit solve (differentiable-through-iterations).
        res = vj.solve_fixed_boundary_gd(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=float(indata.get_float("GAMMA", 0.0)),
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            jacobian_penalty=1e3,
            jit_grad=False,
            verbose=False,
        )

        theta = vj.closed_theta_grid(128)
        phi = jnp.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        B = vj.bmag_from_state_physical(res.state, static, indata=indata, theta=theta, phi=phi, s_index=int(cfg.ns) - 1)
        return jnp.mean(B)

    params0 = jnp.zeros((3,), dtype=jnp.float64)
    g = jax.grad(objective)(params0)
    print("params = [dRBC(0,0), dRBC(0,1), dZBS(0,1)]")
    print("grad(mean(|B|_LCFS)) =", np.asarray(g))


if __name__ == "__main__":
    main()

