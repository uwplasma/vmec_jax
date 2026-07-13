"""Differentiable free boundary via virtual casing (plan.md R15.3 + R19).

Validates :mod:`vmec_jax.core.freeboundary_diff` — the DIFFERENTIABLE
free-boundary path that complements the NESTOR forward solve (R15.1/R15.2, which
these tests never touch).  The free-boundary condition ``B_out . n = 0`` is
written as a smooth objective with ``B_out = B_coil + B_plasma`` and the plasma's
own field ``B_plasma`` from the virtual-casing principle (reusing
``uwplasma/virtual_casing_jax``).

Lanes
-----
- ``test_surface_data_reproduces_equilibrium_bnormal`` — the wout->surface-data
  adapter reproduces the VMEC free-boundary condition ``B_total . n / |B|`` ~ 1e-16
  on a converged equilibrium (validates the geometry+field synthesis).
- ``test_synthetic_surface_gradient_fd_validates`` — asset-free: on a synthetic
  torus, ``jax.grad`` of the residual w.r.t. coil Fourier dofs matches central FD
  (the core deliverable's machinery, small grid).
- ``test_cth_gradient_fd_validates`` (``full``) — the real cth-like case: gradients
  w.r.t. ``extcur`` (mgrid) and coil dofs vs central FD.

The whole module is gated behind ``importorskip('virtual_casing_jax')`` — the
optional uwplasma dependency (``pip install -e /path/to/virtual_casing_jax``);
CI stays green without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("virtual_casing_jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from virtual_casing_jax import VmecSurfaceFieldData  # noqa: E402

from vmec_jax.core import freeboundary_diff as FBD  # noqa: E402
from vmec_jax.core.mgrid import MgridField, read_mgrid  # noqa: E402
from vmec_jax.core.wout import read_wout  # noqa: E402

# jit-enable the whole module: virtual casing is far too slow interpreted.
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

REPO = Path(__file__).resolve().parents[1]
WOUT = REPO / "examples" / "data" / "single_grid" / "wout_cth_like_free_bdy.nc"
MGRID = REPO / "examples" / "data" / "mgrid_cth_like.nc"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _circular_coil_dofs(ncoils=3, order=1, R0=0.75, a=0.35, nfp=5):
    """Circular coil Fourier dofs (the ESSOS ``Curves`` Fourier convention)."""
    dofs = np.zeros((ncoils, 3, 2 * order + 1))
    for i in range(ncoils):
        p0 = (i + 0.5) * (2 * np.pi / nfp) / (2 * ncoils)
        dofs[i, 0, 0], dofs[i, 0, 2] = R0 * np.cos(p0), a * np.cos(p0)
        dofs[i, 1, 0], dofs[i, 1, 2] = R0 * np.sin(p0), a * np.sin(p0)
        dofs[i, 2, 1] = a
    return jnp.asarray(dofs), jnp.full(ncoils, 1.0e5)


def _essos_coil_field(dofs, currents, *, nfp=5, n_segments=64, stellsym=True):
    """A generic ``xyz(...,3) -> B(...,3)`` callable from ESSOS coils.

    vmec_jax carries no coil code; the differentiable free-boundary residual
    consumes coils through the plain-callable interface, differentiating in the
    ESSOS coils' Fourier dofs (rebuilt inside the closure so ``jax.grad`` threads
    through ``essos.coils.Coils`` -> ``essos.fields.BiotSavart``).
    """
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart

    bs = BiotSavart(Coils(Curves(dofs, n_segments, nfp, stellsym), currents))

    def field(pts):
        return jax.vmap(bs.B)(pts.reshape(-1, 3)).reshape(pts.shape)

    return field


def _synthetic_surface(nphi=12, ntheta=12, nfp=3, R0=1.0, a=0.3, B0=1.0):
    """A circular torus with a purely-toroidal (tangent) field: B_total . n = 0."""
    theta = jnp.linspace(0.0, 2 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0.0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    ph, th = jnp.meshgrid(phi, theta, indexing="ij")  # both (nphi, ntheta)
    R = R0 + a * jnp.cos(th)
    Z = a * jnp.sin(th)
    cph, sph = jnp.cos(ph), jnp.sin(ph)
    gamma = jnp.stack([R * cph, R * sph, Z], axis=0)
    e_th = jnp.stack([-a * jnp.sin(th) * cph, -a * jnp.sin(th) * sph, a * jnp.cos(th)], axis=0)
    e_ph = jnp.stack([-R * sph, R * cph, jnp.zeros_like(R)], axis=0)
    area = jnp.cross(e_th, e_ph, axis=0)
    normal = area / jnp.linalg.norm(area, axis=0)
    Btor = B0 * R0 / R
    B_total = jnp.stack([-Btor * sph, Btor * cph, jnp.zeros_like(Btor)], axis=0)  # toroidal, tangent
    return VmecSurfaceFieldData(gamma=gamma, B_total=B_total, normal=normal, area_vector=area,
                                theta=theta, phi=phi, nfp=nfp, stellsym=False, signgs=1,
                                source_convention="synthetic")


def _directional_fd(fun, x0, v, h):
    return (float(fun(x0 + h * v)) - float(fun(x0 - h * v))) / (2.0 * h)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_surface_data_reproduces_equilibrium_bnormal():
    """wout->surface-data adapter reproduces the VMEC free-boundary condition."""
    if not WOUT.exists():
        pytest.skip(f"wout fixture unavailable: {WOUT}")
    wout = read_wout(WOUT)
    sd = FBD.surface_field_data_from_wout(wout, nphi=24, ntheta=24)
    assert sd.gamma.shape == (3, 24, 24)
    Bn = jnp.sum(sd.B_total * sd.normal, axis=0)
    absB = jnp.linalg.norm(sd.B_total, axis=0)
    rms = float(jnp.sqrt(jnp.mean(Bn**2)) / jnp.sqrt(jnp.mean(absB**2)))
    assert rms < 1e-10, f"B_total . n / |B| = {rms:.2e} (expected ~machine epsilon)"


def test_synthetic_surface_gradient_fd_validates():
    """Asset-free: grad of the free-boundary residual w.r.t. coil dofs vs central FD."""
    sd = _synthetic_surface()
    # The synthetic toroidal field is tangent to the torus by construction.
    Bn = jnp.sum(sd.B_total * sd.normal, axis=0)
    assert float(jnp.max(jnp.abs(Bn))) < 1e-12

    prob = FBD.FreeBoundaryDiffProblem.from_surface_data(sd, digits=3)
    assert prob.Bn_plasma.shape == (12, 12)
    assert bool(jnp.all(jnp.isfinite(prob.Bn_plasma)))

    pytest.importorskip("essos")
    d0, currents = _circular_coil_dofs(nfp=3, R0=1.0, a=0.5)

    def J(dofs):
        return prob.bnormal_objective(_essos_coil_field(dofs, currents, nfp=3))

    g = jax.grad(J)(d0)
    assert bool(jnp.all(jnp.isfinite(g)))
    v = jnp.asarray(np.random.default_rng(0).standard_normal(d0.shape))
    dir_ad = float(jnp.sum(g * v))
    dir_fd = _directional_fd(J, d0, v, 1e-6)
    assert abs(dir_ad - dir_fd) <= 1e-5 * abs(dir_fd) + 1e-9, f"AD {dir_ad:.6e} vs FD {dir_fd:.6e}"


@pytest.mark.full
def test_cth_gradient_fd_validates():
    """cth-like case: free-boundary residual gradients (extcur + coil dofs) vs central FD."""
    if not WOUT.exists():
        pytest.skip(f"wout fixture unavailable: {WOUT}")
    wout = read_wout(WOUT)
    prob = FBD.FreeBoundaryDiffProblem.from_wout(wout, nphi=24, ntheta=24, digits=4)

    # (a) extcur via the cth mgrid (2 coil-group currents) — exact full gradient.
    if MGRID.exists():
        base = MgridField.from_mgrid_data(read_mgrid(MGRID), extcur=jnp.array([4700.0, 1000.0]))

        def J_extcur(extcur):
            mf = MgridField(br=base.br, bp=base.bp, bz=base.bz, extcur=extcur,
                            rmin=base.rmin, rmax=base.rmax, zmin=base.zmin, zmax=base.zmax, nfp=base.nfp)
            return prob.bnormal_objective(mf)

        x0 = jnp.array([4700.0, 1000.0])
        g_ad = np.asarray(jax.grad(J_extcur)(x0))
        g_fd = np.array([_directional_fd(J_extcur, x0, jnp.asarray(np.eye(2)[i]), 1.0) for i in range(2)])
        rel = np.abs(g_ad - g_fd) / (np.abs(g_fd) + 1e-30)
        assert np.all(rel < 1e-4), f"extcur grad rel err {rel} (AD {g_ad}, FD {g_fd})"

    # (b) coil Fourier dofs via Biot-Savart (ESSOS coils, callable interface) —
    # directional derivative.
    pytest.importorskip("essos")
    d0, currents = _circular_coil_dofs(nfp=int(wout.nfp))
    nfp = int(wout.nfp)

    def J_dofs(dofs):
        return prob.bnormal_objective(_essos_coil_field(dofs, currents, nfp=nfp))

    g = jax.grad(J_dofs)(d0)
    v = jnp.asarray(np.random.default_rng(1).standard_normal(d0.shape))
    dir_ad = float(jnp.sum(g * v))
    dir_fd = _directional_fd(J_dofs, d0, v, 1e-6)
    assert abs(dir_ad - dir_fd) <= 1e-5 * abs(dir_fd) + 1e-9, f"coil-dof AD {dir_ad:.6e} vs FD {dir_fd:.6e}"

    # (c) pressure-balance residual is finite and differentiable too.
    jp = jax.grad(lambda d: prob.pressure_balance_objective(_essos_coil_field(d, currents, nfp=nfp)))(d0)
    assert bool(jnp.all(jnp.isfinite(jp)))
