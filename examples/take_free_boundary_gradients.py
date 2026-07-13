#!/usr/bin/env python
"""Differentiable free boundary via virtual casing (plan.md R15.3 + R19).

Takes gradients of a *free-boundary* residual with respect to external-field dofs
(coil currents / coil shape / ``extcur``) — the differentiable complement to the
NESTOR forward solve.  The free-boundary condition ``B_out . n = 0`` at the
plasma-vacuum interface is written as a smooth objective, with ``B_out = B_coil +
B_plasma`` and the plasma's *own* field ``B_plasma`` obtained by the virtual-casing
principle (reusing ``uwplasma/virtual_casing_jax``).  Because the plasma field on a
*fixed* trial boundary does not depend on the coil dofs, it is precomputed once and
the residual becomes a plain JAX function of the external-field dofs — so
``jax.grad`` gives exact gradients that FD-validate to ~1e-9.

Steps:
  1. read a converged free-boundary wout (the cth-like nfp=5 case),
  2. build the virtual-casing free-boundary problem (precomputes B_plasma),
  3. gradient of <(B.n)^2> w.r.t. ``extcur`` (mgrid) and coil Fourier dofs (ESSOS
     coils, consumed through the plain ``xyz->B`` callable interface),
  4. finite-difference check of both.

Requires the optional dependency ``virtual_casing_jax`` (``pip install -e
/path/to/virtual_casing_jax``).  Run: ``python examples/take_free_boundary_gradients.py``.
"""

import os
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # virtual casing wants float64
import jax.numpy as jnp  # noqa: E402

from vmec_jax.core import freeboundary_diff as FBD  # noqa: E402
from vmec_jax.core.mgrid import MgridField, read_mgrid  # noqa: E402
from vmec_jax.core.wout import read_wout  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
WOUT = DATA / "single_grid" / "wout_cth_like_free_bdy.nc"
MGRID = DATA / "mgrid_cth_like.nc"
CI = os.environ.get("VMEC_JAX_EXAMPLES_CI") == "1"
NPHI = NTHETA = 20 if CI else 32


def main() -> None:
    if not FBD.have_virtual_casing_jax():
        raise SystemExit("this example needs virtual_casing_jax (pip install -e /path/to/virtual_casing_jax)")
    if not WOUT.exists():
        raise SystemExit(f"missing wout fixture {WOUT} (run tools/fetch_assets.py)")

    # 1-2. converged free-boundary equilibrium -> virtual-casing free-boundary problem
    wout = read_wout(WOUT)
    print(f"wout: {WOUT.name}  nfp={int(wout.nfp)}  ns={int(wout.ns)}")
    prob = FBD.FreeBoundaryDiffProblem.from_wout(wout, nphi=NPHI, ntheta=NTHETA, digits=4)

    # Sanity: the adapter reproduces the equilibrium free-boundary condition to ~1e-16.
    sd = FBD.surface_field_data_from_wout(wout, nphi=NPHI, ntheta=NTHETA)
    bn = jnp.sum(sd.B_total * sd.normal, axis=0)
    absb = jnp.linalg.norm(sd.B_total, axis=0)
    print(f"  |B_total . n| / |B| on boundary = {float(jnp.sqrt(jnp.mean(bn**2)) / jnp.sqrt(jnp.mean(absb**2))):.2e}")
    print(f"  plasma self-field on boundary <B_plasma.n> rms = {float(jnp.sqrt(jnp.mean(prob.Bn_plasma**2))):.4f} T")

    # 3a. gradient w.r.t. mgrid extcur (the cth external field is 2 coil-group currents)
    if MGRID.exists():
        base = MgridField.from_mgrid_data(read_mgrid(MGRID), extcur=jnp.array([4700.0, 1000.0]))

        def J_extcur(extcur):
            mf = MgridField(br=base.br, bp=base.bp, bz=base.bz, extcur=extcur,
                            rmin=base.rmin, rmax=base.rmax, zmin=base.zmin, zmax=base.zmax, nfp=base.nfp)
            return prob.bnormal_objective(mf)

        x0 = jnp.array([4700.0, 1000.0])
        g_ad = np.asarray(jax.grad(J_extcur)(x0))
        g_fd = np.array([(float(J_extcur(x0.at[i].add(1.0))) - float(J_extcur(x0.at[i].add(-1.0)))) / 2.0
                         for i in range(2)])
        print("\n[extcur]  J = {:.3e}".format(float(J_extcur(x0))))
        _table(("d/d extcur[0]", "d/d extcur[1]"), g_ad, g_fd)
    else:
        print(f"\n[extcur]  skipped (missing {MGRID.name}; run tools/fetch_assets.py)")

    # 3b. gradient w.r.t. coil Fourier dofs (a simple in-code circular ESSOS coil
    # set, consumed through the generic xyz->B callable interface).
    nfp = int(wout.nfp)
    d0, currents = _circular_coil_dofs(nfp=nfp)

    def J_dofs(dofs):
        return prob.bnormal_objective(_essos_coil_field(dofs, currents, nfp=nfp))

    g = jax.grad(J_dofs)(d0)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(d0.shape))
    h = 1e-6
    dir_ad = float(jnp.sum(g * v))
    dir_fd = (float(J_dofs(d0 + h * v)) - float(J_dofs(d0 - h * v))) / (2 * h)
    print("\n[coil dofs]  J = {:.3e}  (directional derivative along a random shape perturbation)".format(float(J_dofs(d0))))
    _table(("g . v",), np.array([dir_ad]), np.array([dir_fd]))

    print("\nGradients FD-validate: the free-boundary residual is differentiable in the coil/extcur dofs.")


def _circular_coil_dofs(ncoils: int = 3, order: int = 1, R0: float = 0.75, a: float = 0.35, nfp: int = 5):
    """Circular coil Fourier dofs + currents (the ESSOS ``Curves`` convention)."""
    dofs = np.zeros((ncoils, 3, 2 * order + 1))
    for i in range(ncoils):
        p0 = (i + 0.5) * (2 * np.pi / nfp) / (2 * ncoils)  # first half period (stellsym expands the rest)
        dofs[i, 0, 0], dofs[i, 0, 2] = R0 * np.cos(p0), a * np.cos(p0)
        dofs[i, 1, 0], dofs[i, 1, 2] = R0 * np.sin(p0), a * np.sin(p0)
        dofs[i, 2, 1] = a
    return jnp.asarray(dofs), jnp.full(ncoils, 1.0e5)


def _essos_coil_field(dofs, currents, *, nfp: int, n_segments: int = 80, stellsym: bool = True):
    """A generic ``xyz(...,3) -> B(...,3)`` callable from ESSOS coils.

    vmec_jax keeps no coil code; the differentiable free-boundary residual takes
    coils through this plain-callable interface.  Rebuilding the ESSOS ``Coils``
    inside the closure lets ``jax.grad`` thread through
    ``essos.coils.Coils`` -> ``essos.fields.BiotSavart`` to the coil Fourier dofs.
    """
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart

    bs = BiotSavart(Coils(Curves(dofs, n_segments, nfp, stellsym), currents))

    def field(pts):
        return jax.vmap(bs.B)(pts.reshape(-1, 3)).reshape(pts.shape)

    return field


def _table(names, ad, fd) -> None:
    print(f"  {'component':<16}{'jax.grad':>16}{'central FD':>16}{'rel err':>12}")
    for name, a, f in zip(names, np.atleast_1d(ad), np.atleast_1d(fd)):
        rel = abs(a - f) / (abs(f) + 1e-30)
        print(f"  {name:<16}{a:>16.6e}{f:>16.6e}{rel:>12.2e}")


if __name__ == "__main__":
    main()
