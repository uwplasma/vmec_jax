"""VMEC-style half-mesh metric + B-covariant ingredients (Step-10).

This module ports the *core* algebra from VMEC2000's ``bcovar`` for the
fixed-boundary, no-preconditioner parity stage:

- Build half-mesh metric elements ``g_uu, g_uv, g_vv`` using VMEC's even/odd-m
  decomposition and half-mesh staggering.
- Build half-mesh Jacobian-related fields via :mod:`vmec_jax.vmec_jacobian`.
- Compute VMEC contravariant field components ``(B^u, B^v)`` and the covariant
  components ``(B_u, B_v)`` on the radial half mesh.
- Provide force-kernel inputs used by VMEC's ``forces`` routine.

The implementation here is intentionally limited to what's needed for validated
step-wise parity work and does *not* attempt to reproduce VMEC's symmetry-reduced
angle grids (ntheta2/ntheta3 endpoint weighting) yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .field import bsup_from_sqrtg_lambda, lamscale_from_phips
from .vmec_jacobian import VmecHalfMeshJacobian, jacobian_half_mesh_from_parity
from .vmec_parity import internal_odd_from_physical, split_rzl_even_odd_m


@dataclass(frozen=True)
class VmecHalfMeshBcovar:
    """Half-mesh quantities used downstream by VMEC force/residue kernels."""

    jac: VmecHalfMeshJacobian

    # Half-mesh metric elements in cylindrical coordinates.
    guu: Any  # (ns, ntheta, nzeta)
    guv: Any  # (ns, ntheta, nzeta)
    gvv: Any  # (ns, ntheta, nzeta)

    # Half-mesh magnetic field components.
    bsupu: Any  # (ns, ntheta, nzeta)
    bsupv: Any  # (ns, ntheta, nzeta)
    bsubu: Any  # (ns, ntheta, nzeta)
    bsubv: Any  # (ns, ntheta, nzeta)

    # bsq = |B|^2/2 + p on half mesh (VMEC convention).
    bsq: Any  # (ns, ntheta, nzeta)

    # Force-kernel inputs (VMEC `bcovar` post-processing).
    gij_b_uu: Any  # (ns, ntheta, nzeta) = (B^u B^u) * sqrt(g)
    gij_b_uv: Any  # (ns, ntheta, nzeta) = (B^u B^v) * sqrt(g)
    gij_b_vv: Any  # (ns, ntheta, nzeta) = (B^v B^v) * sqrt(g)
    lu_e: Any  # (ns, ntheta, nzeta) = R * bsq
    lv_e: Any  # (ns, ntheta, nzeta) = (sqrt(g)/R) * bsq  (tau * bsq)

    # Lambda derivatives on half mesh (scaled-lambda).
    lam_u: Any  # (ns, ntheta, nzeta)
    lam_v: Any  # (ns, ntheta, nzeta)

    # Scalar lambda scaling factor.
    lamscale: Any


def _pshalf_from_s(s: Any) -> Any:
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    sh = 0.5 * (s[1:] + s[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, 0.0))


def _half_mesh_from_even_odd(even, odd_int, *, s):
    """VMEC half-mesh staggering for fields of form X = X_even + sqrt(s) X_odd."""
    even = jnp.asarray(even)
    odd_int = jnp.asarray(odd_int)
    s = jnp.asarray(s)

    ns = int(s.shape[0])
    if ns < 2:
        return even

    pshalf = _pshalf_from_s(s)[:, None, None]
    out = jnp.zeros_like(even)
    out = out.at[1:].set(0.5 * (even[1:] + even[:-1] + pshalf[1:] * (odd_int[1:] + odd_int[:-1])))
    out = out.at[0].set(out[1])
    return out


def _metric_even_odd(*, a0, a1, b0, b1, s):
    """Even/odd decomposition of (a0 + sqrt(s)a1)^2 + (b0 + sqrt(s)b1)^2."""
    s = jnp.asarray(s)
    ss = s[:, None, None]
    even = a0 * a0 + b0 * b0 + ss * (a1 * a1 + b1 * b1)
    odd = 2.0 * (a0 * a1 + b0 * b1)
    return even, odd


def _metric_cross_even_odd(*, a0, a1, b0, b1, s):
    """Even/odd decomposition of (a0 + sqrt(s)a1)(b0 + sqrt(s)b1)."""
    s = jnp.asarray(s)
    ss = s[:, None, None]
    even = a0 * b0 + ss * (a1 * b1)
    odd = a0 * b1 + a1 * b0
    return even, odd


def vmec_bcovar_half_mesh_from_wout(
    *,
    state,
    static,
    wout,
    pres: Any | None = None,
) -> VmecHalfMeshBcovar:
    """Compute VMEC-style half-mesh metric and B components for parity tests.

    Parameters
    ----------
    state:
        :class:`~vmec_jax.state.VMECState` (typically from :func:`~vmec_jax.wout.state_from_wout`).
    static:
        Static precomputations from :func:`~vmec_jax.static.build_static`.
    wout:
        :class:`~vmec_jax.wout.WoutData` providing ``phipf``, ``chipf``, ``phips``, ``signgs``.
    pres:
        Optional pressure profile on the *half mesh* in VMEC internal units (mu0*Pa).
        If omitted, uses ``wout.pres``.
    """
    s = jnp.asarray(static.s)

    # Split real-space fields into even/odd-m subsets, then convert odd physical
    # contribution to VMEC's internal odd field by dividing by sqrt(s).
    parity = split_rzl_even_odd_m(state, static.basis, static.modes.m)

    R1 = internal_odd_from_physical(parity.R_odd, s)
    Z1 = internal_odd_from_physical(parity.Z_odd, s)
    Ru1 = internal_odd_from_physical(parity.Rt_odd, s)
    Zu1 = internal_odd_from_physical(parity.Zt_odd, s)
    Rv1 = internal_odd_from_physical(parity.Rp_odd, s)
    Zv1 = internal_odd_from_physical(parity.Zp_odd, s)

    Lu1 = internal_odd_from_physical(parity.Lt_odd, s)
    Lv1 = internal_odd_from_physical(parity.Lp_odd, s)

    # Half-mesh Jacobian quantities from VMEC's discrete formula.
    jac = jacobian_half_mesh_from_parity(
        pr1_even=parity.R_even,
        pr1_odd=R1,
        pz1_even=parity.Z_even,
        pz1_odd=Z1,
        pru_even=parity.Rt_even,
        pru_odd=Ru1,
        pzu_even=parity.Zt_even,
        pzu_odd=Zu1,
        s=s,
    )

    # Metric elements on full mesh split into even/odd (internal) pieces, then
    # staggered to the half mesh using VMEC's pshalf convention.
    guu_e, guu_o = _metric_even_odd(a0=parity.Rt_even, a1=Ru1, b0=parity.Zt_even, b1=Zu1, s=s)
    guv_e, guv_o = _metric_cross_even_odd(a0=parity.Rt_even, a1=Ru1, b0=parity.Rp_even, b1=Rv1, s=s)
    guv_e2, guv_o2 = _metric_cross_even_odd(a0=parity.Zt_even, a1=Zu1, b0=parity.Zp_even, b1=Zv1, s=s)
    guv_e = guv_e + guv_e2
    guv_o = guv_o + guv_o2
    gvv_e, gvv_o = _metric_even_odd(a0=parity.Rp_even, a1=Rv1, b0=parity.Zp_even, b1=Zv1, s=s)

    # R^2 term in cylindrical metric: gvv <- gvv + R^2
    ss = s[:, None, None]
    R2_e = parity.R_even * parity.R_even + ss * (R1 * R1)
    R2_o = 2.0 * parity.R_even * R1

    guu = _half_mesh_from_even_odd(guu_e, guu_o, s=s)
    guv = _half_mesh_from_even_odd(guv_e, guv_o, s=s)
    gvv = _half_mesh_from_even_odd(gvv_e, gvv_o, s=s) + _half_mesh_from_even_odd(R2_e, R2_o, s=s)

    # Lambda derivatives on half mesh (scaled lambda).
    lam_u = _half_mesh_from_even_odd(parity.Lt_even, Lu1, s=s)
    lam_v = _half_mesh_from_even_odd(parity.Lp_even, Lv1, s=s)

    # VMEC uses -lam_v internally (see totzsps), but the public formula for B^u
    # in terms of lambda uses lam_v directly:
    #   bsupu = overg * (chipf - lamscale * lam_v)
    lamscale = lamscale_from_phips(wout.phips, s)
    bsupu, bsupv = bsup_from_sqrtg_lambda(
        sqrtg=jac.sqrtg,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=wout.phipf,
        chipf=wout.chipf,
        signgs=int(wout.signgs),
        lamscale=lamscale,
    )

    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + gvv * bsupv

    b2 = bsupu * bsubu + bsupv * bsubv
    pres_h = jnp.asarray(wout.pres if pres is None else pres)[:, None, None]
    bsq = 0.5 * b2 + pres_h

    # Force-kernel inputs matching what `forces.f` expects after `bcovar`.
    gij_b_uu = (bsupu * bsupu) * jac.sqrtg
    gij_b_uv = (bsupu * bsupv) * jac.sqrtg
    gij_b_vv = (bsupv * bsupv) * jac.sqrtg
    lu_e = bsq * jac.r12
    lv_e = bsq * jac.tau

    return VmecHalfMeshBcovar(
        jac=jac,
        guu=guu,
        guv=guv,
        gvv=gvv,
        bsupu=bsupu,
        bsupv=bsupv,
        bsubu=bsubu,
        bsubv=bsubv,
        bsq=bsq,
        gij_b_uu=gij_b_uu,
        gij_b_uv=gij_b_uv,
        gij_b_vv=gij_b_vv,
        lu_e=lu_e,
        lv_e=lv_e,
        lam_u=lam_u,
        lam_v=lam_v,
        lamscale=lamscale,
    )

