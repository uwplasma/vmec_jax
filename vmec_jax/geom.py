"""Geometry and metric/Jacobian utilities.

Step-2 extends the step-1 coordinate kernel (R,Z,lambda on an (s,theta,zeta)
grid) with:

* radial derivatives (via finite differences on coefficient arrays)
* a cylindrical -> Cartesian embedding using the *physical* toroidal angle
* covariant basis vectors and metric tensor g_ij
* Jacobian sqrt(g) = e_s · (e_theta × e_phi)

This module is intentionally **minimal**: it contains only what we need to
start validating the geometric pieces against VMEC2000 and to support a
future force/residual kernel.

Notes on angles
---------------
We follow the conventions already used in :mod:`vmec_jax.fourier`:

* ``zeta`` spans a single field period, i.e. zeta ∈ [0, 2π)
* the physical toroidal angle is ``phi = zeta / NFP``
* the Fourier phase is m*theta - n*zeta, equivalent to m*theta - (n*NFP)*phi

Accordingly:
* ``eval_fourier_dzeta_phys`` returns ∂/∂phi.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._compat import jnp, jit, has_jax
from .coords import eval_coords
from .fourier import HelicalBasis, eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys
from .radial import d_ds_coeffs


# Idempotent PyTree registration helper (same pattern as in fourier.py)
if has_jax():
    try:
        from jax.tree_util import register_pytree_node_class as _register_pytree_node_class  # type: ignore

        def register_pytree_node_class(cls):  # type: ignore
            try:
                return _register_pytree_node_class(cls)
            except ValueError as e:
                if "Duplicate custom PyTreeDef type registration" in str(e):
                    return cls
                raise
    except Exception:  # pragma: no cover
        import jax.tree_util as _tu

        def register_pytree_node_class(cls):  # type: ignore
            try:
                _tu.register_pytree_node(
                    cls,
                    lambda x: x.tree_flatten(),
                    lambda aux, children: cls.tree_unflatten(aux, children),
                )
            except ValueError as e:
                if "Duplicate custom PyTreeDef type registration" not in str(e):
                    raise
            return cls
else:

    def register_pytree_node_class(cls):  # type: ignore
        return cls


@register_pytree_node_class
@dataclass(frozen=True)
class Geom:
    """Geometry, derivatives, metric, and Jacobian on the 3D grid."""

    # Coordinates on (s,theta,zeta)
    R: any
    Z: any
    L: any

    # Derivatives in physical coordinates: s, theta, phi
    Rs: any
    Zs: any
    Ls: any
    Rt: any
    Zt: any
    Lt: any
    Rp: any
    Zp: any
    Lp: any

    # Jacobian and metric (covariant)
    sqrtg: any
    g_ss: any
    g_st: any
    g_sp: any
    g_tt: any
    g_tp: any
    g_pp: any

    def tree_flatten(self):
        children = (
            self.R,
            self.Z,
            self.L,
            self.Rs,
            self.Zs,
            self.Ls,
            self.Rt,
            self.Zt,
            self.Lt,
            self.Rp,
            self.Zp,
            self.Lp,
            self.sqrtg,
            self.g_ss,
            self.g_st,
            self.g_sp,
            self.g_tt,
            self.g_tp,
            self.g_pp,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            R,
            Z,
            L,
            Rs,
            Zs,
            Ls,
            Rt,
            Zt,
            Lt,
            Rp,
            Zp,
            Lp,
            sqrtg,
            g_ss,
            g_st,
            g_sp,
            g_tt,
            g_tp,
            g_pp,
        ) = children
        return cls(
            R=R,
            Z=Z,
            L=L,
            Rs=Rs,
            Zs=Zs,
            Ls=Ls,
            Rt=Rt,
            Zt=Zt,
            Lt=Lt,
            Rp=Rp,
            Zp=Zp,
            Lp=Lp,
            sqrtg=sqrtg,
            g_ss=g_ss,
            g_st=g_st,
            g_sp=g_sp,
            g_tt=g_tt,
            g_tp=g_tp,
            g_pp=g_pp,
        )

    # --- Convenience aliases (VMEC-style names) ---
    # These make scripts/readers happier and keep naming consistent across steps.

    @property
    def R_s(self):
        return self.Rs

    @property
    def Z_s(self):
        return self.Zs

    @property
    def L_s(self):
        return self.Ls

    @property
    def R_theta(self):
        return self.Rt

    @property
    def Z_theta(self):
        return self.Zt

    @property
    def L_theta(self):
        return self.Lt

    @property
    def R_phi(self):
        return self.Rp

    @property
    def Z_phi(self):
        return self.Zp

    @property
    def L_phi(self):
        return self.Lp


def _cross(a, b):
    """Cross product along the last axis."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return jnp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def _dot(a, b):
    return jnp.sum(a * b, axis=-1)


@jit
def _eval_geom_jit(state, basis: HelicalBasis, s_grid, zeta_grid):
    """JIT-friendly geometry/metric computation.

    Parameters
    ----------
    state:
        VMECState (PyTree).
    basis:
        HelicalBasis on the (theta,zeta) grid.
    s_grid:
        (ns,) radial coordinate.
    zeta_grid:
        (nzeta,) field-period toroidal coordinate in [0, 2π).
    """
    # Coordinates (no radial derivatives)
    c = eval_coords(state, basis)

    # Angular derivatives (already in coords kernel)
    Rt = c.R_theta
    Zt = c.Z_theta
    Lt = c.L_theta
    Rp = c.R_phi
    Zp = c.Z_phi
    Lp = c.L_phi

    # Radial derivatives: differentiate coefficients then evaluate.
    Rcos_s = d_ds_coeffs(state.Rcos, s_grid)
    Rsin_s = d_ds_coeffs(state.Rsin, s_grid)
    Zcos_s = d_ds_coeffs(state.Zcos, s_grid)
    Zsin_s = d_ds_coeffs(state.Zsin, s_grid)
    Lcos_s = d_ds_coeffs(state.Lcos, s_grid)
    Lsin_s = d_ds_coeffs(state.Lsin, s_grid)

    Rs = eval_fourier(Rcos_s, Rsin_s, basis)
    Zs = eval_fourier(Zcos_s, Zsin_s, basis)
    Ls = eval_fourier(Lcos_s, Lsin_s, basis)

    # Physical toroidal angle phi = zeta / NFP
    phi = zeta_grid / basis.nfp
    cosphi = jnp.cos(phi)[None, None, :]
    sinphi = jnp.sin(phi)[None, None, :]

    R = c.R
    Z = c.Z

    # Covariant basis vectors in Cartesian space.
    e_s = jnp.stack([Rs * cosphi, Rs * sinphi, Zs], axis=-1)
    e_t = jnp.stack([Rt * cosphi, Rt * sinphi, Zt], axis=-1)
    e_p = jnp.stack([Rp * cosphi - R * sinphi, Rp * sinphi + R * cosphi, Zp], axis=-1)

    # Metric tensor components g_ij = e_i · e_j
    g_ss = _dot(e_s, e_s)
    g_st = _dot(e_s, e_t)
    g_sp = _dot(e_s, e_p)
    g_tt = _dot(e_t, e_t)
    g_tp = _dot(e_t, e_p)
    g_pp = _dot(e_p, e_p)

    # Jacobian sqrt(g) = e_s · (e_t × e_p)
    sqrtg = _dot(e_s, _cross(e_t, e_p))

    return Geom(
        R=R,
        Z=Z,
        L=c.L,
        Rs=Rs,
        Zs=Zs,
        Ls=Ls,
        Rt=Rt,
        Zt=Zt,
        Lt=Lt,
        Rp=Rp,
        Zp=Zp,
        Lp=Lp,
        sqrtg=sqrtg,
        g_ss=g_ss,
        g_st=g_st,
        g_sp=g_sp,
        g_tt=g_tt,
        g_tp=g_tp,
        g_pp=g_pp,
    )


def eval_geom(state, static):
    """Compute geometry/metric/Jacobian on the full 3D grid.

    This is a light wrapper that keeps the user-facing signature nice.
    """
    return _eval_geom_jit(state, static.basis, static.s, static.grid.zeta)
