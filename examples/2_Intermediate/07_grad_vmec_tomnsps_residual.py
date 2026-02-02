from __future__ import annotations

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
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout_reference_fields, vmec_residual_internal_from_kernels
from vmec_jax.vmec_tomnsp import vmec_angle_grid
from vmec_jax.wout import read_wout, state_from_wout


def main():
    enable_x64()
    if not has_jax():
        raise RuntimeError("This example requires JAX installed.")
    import jax
    import jax.numpy as jnp

    root = REPO_ROOT
    input_path = root / "examples/input.circular_tokamak"
    wout_path = root / "examples/wout_circular_tokamak_reference.nc"

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)

    # Use VMEC's internal theta grid for the transform.
    grid = vmec_angle_grid(ntheta=cfg.ntheta, nzeta=cfg.nzeta, nfp=cfg.nfp, lasym=cfg.lasym)
    cfg_vm = replace(cfg, ntheta=int(grid.theta.size), nzeta=int(grid.zeta.size))
    static = build_static(cfg_vm, grid=grid)

    st0_np = state_from_wout(wout)
    st0 = replace(
        st0_np,
        Rcos=jnp.asarray(st0_np.Rcos),
        Rsin=jnp.asarray(st0_np.Rsin),
        Zcos=jnp.asarray(st0_np.Zcos),
        Zsin=jnp.asarray(st0_np.Zsin),
        Lcos=jnp.asarray(st0_np.Lcos),
        Lsin=jnp.asarray(st0_np.Lsin),
    )

    # Define a differentiable scalar: RMS of the (0,0) frcc mode over s.
    def loss_fn(delta):
        st = replace(st0, Rcos=st0.Rcos.at[2, 0].add(delta))  # perturb one Fourier coefficient
        k = vmec_forces_rz_from_wout_reference_fields(state=st, static=static, wout=wout)
        res = vmec_residual_internal_from_kernels(k, cfg_ntheta=cfg.ntheta, cfg_nzeta=cfg.nzeta, wout=wout)
        fr00 = res.frcc[:, 0, 0]
        return jnp.sqrt(jnp.mean(fr00[1:] ** 2))

    g = jax.grad(loss_fn)(jnp.asarray(1e-6))
    val = loss_fn(jnp.asarray(1e-6))
    print(f"loss={float(val):.6e}  dloss/d(delta)={float(g):.6e}")

    out = root / "examples/outputs"
    out.mkdir(exist_ok=True)
    np.savez(out / "grad_vmec_tomnsps_residual.npz", loss=float(val), grad=float(g))
    print(f"Wrote {out / 'grad_vmec_tomnsps_residual.npz'}")


if __name__ == "__main__":
    main()
