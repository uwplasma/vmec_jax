#!/usr/bin/env python
"""Tiny autodiff demo.

Compute a simple scalar functional of the boundary surface and differentiate it w.r.t.
boundary Fourier coefficients. This is *not* a physics objective, just a proof that the
basis evaluation path is differentiable in JAX.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from the examples/ directory without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vmec_jax._compat import has_jax
from vmec_jax.config import load_config
from vmec_jax.modes import vmec_mode_table
from vmec_jax.grids import make_angle_grid
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.fourier import build_helical_basis, eval_fourier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    args = ap.parse_args()

    if not has_jax():
        print("JAX not importable in this environment. Install with: pip install -e .[jax]")
        return

    import jax
    import jax.numpy as jnp

    cfg, indata = load_config(args.input)
    modes = vmec_mode_table(cfg.mpol, cfg.ntor)
    grid = make_angle_grid(cfg.ntheta, cfg.nzeta, cfg.nfp)
    bdy = boundary_from_indata(indata, modes)
    basis = build_helical_basis(modes, grid)

    # pick a single coefficient to differentiate: rbc(n=0,m=1) is usually the major radius cosine term
    # our ModeTable uses (m,n) so look for (m=1,n=0)
    k_target = int(np.where((modes.m == 1) & (modes.n == 0))[0][0])

    R_cos0 = jnp.asarray(bdy.R_cos)
    R_sin0 = jnp.asarray(bdy.R_sin)

    def objective(delta):
        R_cos = R_cos0.at[k_target].add(delta)
        R = eval_fourier(R_cos, R_sin0, basis)
        # simple differentiable scalar: mean(R^2) on the grid
        return jnp.mean(R * R)

    g = jax.grad(objective)(0.0)
    print("d/d(delta) mean(R^2) at delta=0 for coefficient (m=1,n=0):", float(g))


if __name__ == "__main__":
    main()
