#!/usr/bin/env python
"""True single-stage: optimize the plasma boundary AND the coils *simultaneously*.

The companion :mod:`single_stage_free_boundary_opt` fits coil currents to a
*fixed* target boundary.  This script does the full single-stage problem: one
least-squares descent over BOTH the plasma-boundary Fourier coefficients and the
coil-group currents, driven by a single exact gradient that threads through

  * the **implicit-differentiation adjoint** for the fixed-boundary equilibrium
    (boundary dofs -> converged VMEC state -> physics targets), and
  * the **virtual-casing** free-boundary residual (coils + the *moving* plasma
    boundary -> ``B.n`` on that boundary),

at the same time.  ``jax.value_and_grad`` of the combined objective is exact and
finite-difference validated (see ``tests/test_single_stage_simultaneous.py``):
the boundary half comes out of the adjoint, the coil half out of virtual casing,
and the coupling (the coil residual is evaluated on the boundary the plasma solve
just produced) is differentiated too.

Making the virtual-casing plasma field differentiable in the *boundary* (not just
the coils) needs its adaptive quadrature/patch precision frozen to static values
first -- :func:`~vmex.core.freeboundary_diff.plan_vc_precision` selects it
once from the starting boundary; see that module and ``virtual_casing_jax``'s
``PrecisionPlan``.

Objective (a genuine single-stage functional)::

    J(boundary, extcur) = w_bn * < (B_ext . n)^2 >         # coil<->plasma consistency
                        + w_iota * (iota_edge - iota_*)^2   # a plasma physics target

Starting from the bundled CTH-like free-boundary equilibrium (self-consistent, so
``<(B.n)^2>`` is already tiny), we ask for a slightly different edge rotational
transform: the optimizer must reshape the boundary to move ``iota`` *and*
re-tune the coils so they still produce that boundary.

Requires the optional ``virtual_casing_jax`` dependency and the fetched
``mgrid`` reference asset (``python tools/fetch_assets.py``).
"""

import dataclasses
import os
from pathlib import Path

import numpy as np
import scipy.optimize

import jax
import jax.numpy as jnp

import vmex as vj
from vmex.core import freeboundary_diff as FBD
from vmex.core import implicit as im
from vmex.core.mgrid import MgridField, read_mgrid

DATA = Path(__file__).resolve().parent / "data"
INPUT = DATA / "input.cth_like_free_bdy"
MGRID = DATA / "mgrid_cth_like.nc"
EXTCUR0 = np.array([4700.0, 1000.0])          # confining coil-group currents
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
NPHI = NTHETA = 16 if CI else 24
SOLVE = dict(ftol=1e-10, max_iterations=2000)
W_BN, W_IOTA = 1.0, 1.0
DIOTA = 0.03                                    # asked-for edge-iota change

# Boundary dofs we let the optimizer move: a few low-order shaping modes
# (n = 0, m in {1, 2}) of R_cos and Z_sin.  extcur are the two coil currents.
BOUNDARY_MODES = [(1, 0), (2, 0)]               # (m, n)


def main() -> None:
    if not FBD.have_virtual_casing_jax():
        raise SystemExit("needs virtual_casing_jax (pip install -e /path/to/virtual_casing_jax)")
    for p in (INPUT, MGRID):
        if not p.exists():
            raise SystemExit(f"missing {p.name}; run tools/fetch_assets.py")

    inp = vj.VmecInput.from_file(str(INPUT))
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)
    base = MgridField.from_mgrid_data(read_mgrid(MGRID), extcur=jnp.asarray(EXTCUR0))

    # DOF packing:  x = [ R_cos(m,0), Z_sin(m,0) for m in modes,  extcur ]
    r0 = np.array([float(np.asarray(p0.rbc)[ntor, m]) for m, _ in BOUNDARY_MODES])
    z0 = np.array([float(np.asarray(p0.zbs)[ntor, m]) for m, _ in BOUNDARY_MODES])
    x0 = np.concatenate([r0, z0, EXTCUR0])
    nb = len(BOUNDARY_MODES)

    def unpack(x):
        rbc = p0.rbc
        zbs = p0.zbs
        for i, (m, _) in enumerate(BOUNDARY_MODES):
            rbc = rbc.at[ntor, m].set(x[i])
            zbs = zbs.at[ntor, m].set(x[nb + i])
        params = dataclasses.replace(p0, rbc=rbc, zbs=zbs)
        extcur = jnp.asarray(x[2 * nb:])
        return params, extcur

    # Precision plan: choose the virtual-casing quadrature/patch ONCE from the
    # starting (concrete) boundary, then hold it fixed so J differentiates in x.
    sol0 = im.run(inp, p0, **SOLVE)
    sd0 = FBD.surface_field_data_from_state(inp, sol0.state, nphi=NPHI, ntheta=NTHETA)
    plan = FBD.plan_vc_precision(sd0, digits=4)
    iota_target = float(sol0.iota_edge) + DIOTA

    def objective(x):
        params, extcur = unpack(x)
        sol = im.run(inp, params, **SOLVE)
        sd = FBD.surface_field_data_from_state(inp, sol.state, nphi=NPHI, ntheta=NTHETA)
        prob = FBD.FreeBoundaryDiffProblem.from_surface_data(sd, digits=4, precision=plan)
        mf = MgridField(br=base.br, bp=base.bp, bz=base.bz, extcur=extcur,
                        rmin=base.rmin, rmax=base.rmax, zmin=base.zmin,
                        zmax=base.zmax, nfp=base.nfp)
        j_bn = prob.bnormal_objective(mf)
        j_iota = (sol.iota_edge - iota_target) ** 2
        return W_BN * j_bn + W_IOTA * j_iota

    value_and_grad = jax.value_and_grad(objective)

    # Per-dof step scales: boundary Fourier modes are O(0.01-1) while coil
    # currents are O(1e3), so optimize in scaled coordinates u (x = x0 + D*u)
    # with a bounded step -- otherwise L-BFGS takes wild steps and the trial
    # boundary self-intersects (VmecJacobianError).
    D = np.array([0.02] * nb + [0.02] * nb + [200.0, 200.0])
    bounds = [(-3.0, 3.0)] * len(x0)              # boundary +/-0.06, coils +/-600

    def scipy_fun(u):
        x = x0 + D * u
        v, g = value_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float) * D    # chain rule dJ/du

    def report(tag, x):
        params, extcur = unpack(x)
        sol = im.run(inp, params, **SOLVE)
        sd = FBD.surface_field_data_from_state(inp, sol.state, nphi=NPHI, ntheta=NTHETA)
        prob = FBD.FreeBoundaryDiffProblem.from_surface_data(sd, digits=4, precision=plan)
        mf = MgridField(br=base.br, bp=base.bp, bz=base.bz, extcur=extcur,
                        rmin=base.rmin, rmax=base.rmax, zmin=base.zmin,
                        zmax=base.zmax, nfp=base.nfp)
        print(f"  {tag:9s}: <(B.n)^2>={float(prob.bnormal_objective(mf)):.3e}  "
              f"iota_edge={float(sol.iota_edge):.4f}  extcur={np.asarray(extcur).tolist()}")

    print(f"target: iota_edge {float(sol0.iota_edge):.4f} -> {iota_target:.4f}, "
          f"keeping <(B.n)^2> small.  dofs: {2*nb} boundary + 2 coil currents.")
    report("start", x0)

    res = scipy.optimize.minimize(
        scipy_fun, np.zeros_like(x0), jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 8 if CI else 40, "ftol": 1e-12, "gtol": 1e-10})

    j0 = scipy_fun(np.zeros_like(x0))[0]
    report("optimum", x0 + D * res.x)
    print(f"\nJ {j0:.3e} -> {res.fun:.3e} in {res.nit} iterations "
          f"(exact simultaneous boundary+coil gradient).")


if __name__ == "__main__":
    main()
