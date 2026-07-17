#!/usr/bin/env python
"""Single-stage free-boundary optimization: fit coil currents to confine a plasma.

The *two-stage* stellarator workflow finds a good plasma boundary first, then
searches for coils that reproduce it.  vmex can do the coil half in one
gradient-based shot, because the free-boundary condition ``B_out . n = 0`` on the
target plasma boundary is a smooth, exactly-differentiable objective of the coil
degrees of freedom (the plasma's own field is removed by the **virtual-casing
principle**; see :mod:`vmex.core.freeboundary_diff`).  ``jax.grad`` of that
objective is exact and finite-difference-validated to ~1e-9, so a standard
least-squares driver drives it directly — no finite-difference coil scan.

This script takes the bundled CTH-like free-boundary equilibrium, *perturbs* its
coil-group currents away from the confining values, and recovers them by
minimizing the boundary field error ``<(B.n)^2>`` with exact gradients — the coil
half of a single-stage optimization.  (The plasma half — co-optimizing the
boundary for quasisymmetry through the implicit-differentiation adjoint — uses
the same ``least_squares`` machinery on the fixed-boundary side; combining both
objective families is the full single-stage loop.)

Requires the optional ``virtual_casing_jax`` dependency and the fetched
``wout``/``mgrid`` reference assets (``python tools/fetch_assets.py``).
"""

import os
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import scipy.optimize

from vmex.core import freeboundary_diff as FBD
from vmex.core.mgrid import MgridField, read_mgrid
from vmex.core.wout import read_wout

# --------------------------- parameters ------------------------------------
DATA = Path(__file__).resolve().parent / "data"
WOUT = DATA / "single_grid" / "wout_cth_like_free_bdy.nc"
MGRID = DATA / "mgrid_cth_like.nc"
EXTCUR0 = np.array([4700.0, 1000.0])   # the confining coil-group currents
PERTURB = np.array([0.80, 1.35])       # start 20% / +35% off — B.n is then large
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
NPHI = NTHETA = 20 if CI else 32


def main() -> None:
    if not FBD.have_virtual_casing_jax():
        raise SystemExit("needs virtual_casing_jax (pip install -e /path/to/virtual_casing_jax)")
    for path in (WOUT, MGRID):
        if not path.exists():
            raise SystemExit(f"missing {path.name}; run tools/fetch_assets.py")

    # Target plasma boundary + its virtual-casing free-boundary problem.
    wout = read_wout(WOUT)
    prob = FBD.FreeBoundaryDiffProblem.from_wout(wout, nphi=NPHI, ntheta=NTHETA, digits=4)
    base = MgridField.from_mgrid_data(read_mgrid(MGRID), extcur=jnp.asarray(EXTCUR0))

    def bn_objective(extcur):
        """Boundary field error <(B_out . n)^2> for coil-group currents ``extcur``."""
        mf = MgridField(br=base.br, bp=base.bp, bz=base.bz, extcur=extcur,
                        rmin=base.rmin, rmax=base.rmax, zmin=base.zmin,
                        zmax=base.zmax, nfp=base.nfp)
        return prob.bnormal_objective(mf)

    obj_and_grad = jax.jit(jax.value_and_grad(bn_objective))

    def scipy_fun(x):
        v, g = obj_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    x_start = EXTCUR0 * PERTURB
    print(f"confining currents : {EXTCUR0.tolist()}  ->  <(B.n)^2> = {scipy_fun(EXTCUR0)[0]:.3e}")
    print(f"perturbed start    : {x_start.tolist()}  ->  <(B.n)^2> = {scipy_fun(x_start)[0]:.3e}")

    # Gradient-based recovery (BFGS with the exact virtual-casing gradient).
    res = scipy.optimize.minimize(scipy_fun, x_start, jac=True, method="L-BFGS-B",
                                  options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-10})
    print(f"\nrecovered currents : {res.x.tolist()}")
    print(f"  <(B.n)^2> {scipy_fun(x_start)[0]:.3e} -> {res.fun:.3e} "
          f"in {res.nit} iterations")
    err = np.abs(res.x / EXTCUR0 - 1.0)
    print(f"  currents recovered to {100 * err.max():.2f}% of the confining values "
          "(exact virtual-casing gradient — no finite-difference coil scan)")


if __name__ == "__main__":
    main()
