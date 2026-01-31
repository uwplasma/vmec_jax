"""Step-1 autodiff demo over the full (s,theta,zeta) geometry kernel.

This is a slightly more "end-to-end" gradient than examples/01_grad_demo.py:
we differentiate a scalar objective that depends on *all* radial surfaces.

Usage
-----
    python 03_grad_full_coords.py <input_file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="VMEC input namelist file")
    ap.add_argument("--verbose", action="store_true", help="Print extra debug information")
    ap.add_argument("--topk", type=int, default=10, help="How many largest gradient entries to print")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        ap.error(f"Input file not found: {args.input}")

    from vmec_jax._compat import has_jax, enable_x64

    if not has_jax():
        print("[03_grad_full_coords] JAX is not installed. Try: pip install -e .[jax]")
        return

    enable_x64(True)

    import jax
    import jax.numpy as jnp

    from vmec_jax.config import load_config
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.static import build_static
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.coords import eval_coords
    from vmec_jax.state import pack_state, unpack_state

    cfg, indata = load_config(str(inp))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    state0 = initial_guess_from_boundary(static, bdy, indata)

    x0 = pack_state(state0)
    layout = state0.layout

    @jax.jit
    def objective(x):
        st = unpack_state(x, layout)
        c = eval_coords(st, static.basis)
        # A toy scalar objective: mean(R^2) over all s,theta,zeta.
        return jnp.mean(c.R * c.R)

    g = jax.grad(objective)(x0)
    print("\n==== vmec_jax step-1 full-geometry grad ====")
    print("objective(x0) =", float(objective(x0)))
    print("|grad|_2      =", float(jnp.linalg.norm(g)))
    print("|grad|_inf    =", float(jnp.max(jnp.abs(g))))

    if args.verbose:
        import numpy as np

        g_np = np.asarray(g)
        ns, K = int(layout.ns), int(layout.K)
        blk = ns * K
        fields = ["Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin"]
        idx = np.argsort(-np.abs(g_np))[: max(1, int(args.topk))]
        print("\n-- largest |grad| entries (mapped back to (field, s, m, n)) --")
        print("  rank  field   s-index  (m,n)      grad_value")
        for r, ii in enumerate(idx, start=1):
            which = int(ii // blk)
            rem = int(ii % blk)
            isurf = int(rem // K)
            ik = int(rem % K)
            m = int(np.asarray(static.modes.m)[ik])
            n = int(np.asarray(static.modes.n)[ik])
            print(f"  {r:>4d}  {fields[which]:>4s}   {isurf:>4d}   ({m:>2d},{n:>3d})  {g_np[ii]:>+12.5e}")

if __name__ == "__main__":
    main()
