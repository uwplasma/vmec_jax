"""VMEC force/residue kernels (Step-10 parity work).

This module implements a direct, array-based port of VMEC2000's ``forces`` core
for the **R/Z** equations, operating on:

- VMEC even/odd-m real-space decomposition (odd stored in 1/sqrt(s) form),
- half-mesh quantities from :mod:`vmec_jax.vmec_bcovar`.

Scope
-----
This is a parity/debug kernel used to validate the algebra and staggering.
It is *not* yet the full VMEC solver pipeline (no constraints, no vacuum/free
boundary, no 2D preconditioner, and no full lambda residue parity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .fourier import project_to_modes
from .fourier import build_helical_basis, eval_fourier
from .grids import AngleGrid
from .modes import ModeTable
from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from .vmec_tomnsp import VmecTrigTables, tomnsps_rzl, vmec_angle_grid, vmec_trig_tables
from .vmec_parity import internal_odd_from_physical, split_rzl_even_odd_m


@dataclass(frozen=True)
class VmecRZForceKernels:
    """Force kernels on the full angular grid, split by m-parity."""

    # R kernels
    armn_e: Any  # (ns, ntheta, nzeta)
    armn_o: Any  # (ns, ntheta, nzeta)
    brmn_e: Any  # (ns, ntheta, nzeta)
    brmn_o: Any  # (ns, ntheta, nzeta)
    crmn_e: Any  # (ns, ntheta, nzeta)
    crmn_o: Any  # (ns, ntheta, nzeta)

    # Z kernels
    azmn_e: Any  # (ns, ntheta, nzeta)
    azmn_o: Any  # (ns, ntheta, nzeta)
    bzmn_e: Any  # (ns, ntheta, nzeta)
    bzmn_o: Any  # (ns, ntheta, nzeta)
    czmn_e: Any  # (ns, ntheta, nzeta)
    czmn_o: Any  # (ns, ntheta, nzeta)

    # Carry bcovar outputs for downstream scalings.
    bc: Any


def _pshalf_from_s(s: Any) -> Any:
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    sh = 0.5 * (s[1:] + s[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, 0.0))


def _with_axis_zero(a):
    a = jnp.asarray(a)
    if a.shape[0] == 0:
        return a
    return a.at[0].set(jnp.zeros_like(a[0]))


def _avg_forward_half_to_int(a):
    """VMEC's forward-average from half mesh to integer mesh along s."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    out = a
    out = out.at[:-1].set(0.5 * (a[:-1] + a[1:]))
    out = out.at[-1].set(0.5 * a[-1])
    return out


def _sum_forward_half(a):
    """VMEC's forward-sum (a(js) <- a(js) + a(js+1)) along s."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    out = a
    out = out.at[:-1].set(a[:-1] + a[1:])
    return out


def _diff_forward_half(a, b):
    """VMEC's forward difference a(js) <- a(js+1) - a(js) plus a 2-term average b."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.shape[0] < 2:
        return a
    out = a
    out = out.at[:-1].set(a[1:] - a[:-1] + 0.5 * (b[:-1] + b[1:]))
    out = out.at[-1].set(-a[-1] + 0.5 * b[-1])
    return out


def _diff_forward_half_noavg(a):
    """VMEC's forward difference a(js) <- a(js+1) - a(js)."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    out = a
    out = out.at[:-1].set(a[1:] - a[:-1])
    out = out.at[-1].set(-a[-1])
    return out


def _avg_forward_half(a):
    """VMEC's forward average a(js) <- 0.5*(a(js)+a(js+1))."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    out = a
    out = out.at[:-1].set(0.5 * (a[:-1] + a[1:]))
    out = out.at[-1].set(0.5 * a[-1])
    return out


def vmec_forces_rz_from_wout(*, state, static, wout) -> VmecRZForceKernels:
    """Compute VMEC R/Z force kernels (armn/brmn/...) from a `wout` equilibrium."""
    s = jnp.asarray(static.s)
    ohs = jnp.asarray(1.0 / (s[1] - s[0])) if s.shape[0] >= 2 else jnp.asarray(0.0)
    dshalfds = jnp.asarray(0.25, dtype=s.dtype)

    bc = vmec_bcovar_half_mesh_from_wout(state=state, static=static, wout=wout)

    # Real-space parity fields for R/Z and angular derivatives.
    parity = split_rzl_even_odd_m(state, static.basis, static.modes.m)
    R1 = internal_odd_from_physical(parity.R_odd, s)
    Z1 = internal_odd_from_physical(parity.Z_odd, s)
    Ru1 = internal_odd_from_physical(parity.Rt_odd, s)
    Zu1 = internal_odd_from_physical(parity.Zt_odd, s)
    Rv1 = internal_odd_from_physical(parity.Rp_odd, s)
    Zv1 = internal_odd_from_physical(parity.Zp_odd, s)

    pr1_0, pr1_1 = jnp.asarray(parity.R_even), jnp.asarray(R1)
    pz1_0, pz1_1 = jnp.asarray(parity.Z_even), jnp.asarray(Z1)
    pru_0, pru_1 = jnp.asarray(parity.Rt_even), jnp.asarray(Ru1)
    pzu_0, pzu_1 = jnp.asarray(parity.Zt_even), jnp.asarray(Zu1)
    prv_0, prv_1 = jnp.asarray(parity.Rp_even), jnp.asarray(Rv1)
    pzv_0, pzv_1 = jnp.asarray(parity.Zp_even), jnp.asarray(Zv1)

    # Half-mesh sqrt(s) and full-mesh sqrt(s).
    pshalf = _pshalf_from_s(s)[:, None, None]
    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]

    # Inputs `forces.f` expects after `bcovar` (half mesh).
    lu_e = _with_axis_zero(bc.lu_e)
    lv_e = _with_axis_zero(bc.lv_e)
    guu = _with_axis_zero(bc.gij_b_uu)
    guv = _with_axis_zero(bc.gij_b_uv)
    gvv = _with_axis_zero(bc.gij_b_vv)

    # Jacobian-related half-mesh fields.
    ru12 = bc.jac.ru12
    zu12 = bc.jac.zu12
    rs = bc.jac.rs
    zs = bc.jac.zs

    # Scratch arrays (VMEC names: guus, guvs, gvvs, bsqr).
    guus = guu * pshalf
    guvs = guv * pshalf
    gvvs = gvv * pshalf

    armn_e = ohs * zu12 * lu_e
    azmn_e = -ohs * ru12 * lu_e
    brmn_e = zs * lu_e
    bzmn_e = -rs * lu_e
    bsqr = dshalfds * lu_e / jnp.where(pshalf != 0, pshalf, 1.0)

    armn_o = armn_e * pshalf
    azmn_o = azmn_e * pshalf
    brmn_o = brmn_e * pshalf
    bzmn_o = bzmn_e * pshalf

    # Forward-average half-mesh GIJ and shalf-weighted GIJ to integer mesh.
    guu_i = _avg_forward_half_to_int(guu)
    gvv_i = _avg_forward_half_to_int(gvv)
    guus_i = _avg_forward_half_to_int(guus)
    gvvs_i = _avg_forward_half_to_int(gvvs)
    guv_i = _avg_forward_half_to_int(guv)
    guvs_i = _avg_forward_half_to_int(guvs)

    # bsqr is forward-summed (not averaged) in VMEC.
    bsqr_s = _sum_forward_half(bsqr)

    # Differences/averages to build even-parity kernels.
    armn_e = _diff_forward_half(armn_e, lv_e)
    azmn_e = _diff_forward_half_noavg(azmn_e)
    brmn_e = _avg_forward_half(brmn_e)
    bzmn_e = _avg_forward_half(bzmn_e)

    armn_e = armn_e - (gvvs_i * pr1_1 + gvv_i * pr1_0)
    brmn_e = brmn_e + bsqr_s * pz1_1 - (guus_i * pru_1 + guu_i * pru_0)
    bzmn_e = bzmn_e - (bsqr_s * pr1_1 + guus_i * pzu_1 + guu_i * pzu_0)

    lv_es = lv_e * pshalf
    lu_o = dshalfds * lu_e

    # Odd-parity kernels.
    if armn_o.shape[0] >= 2:
        armn_o = armn_o.at[:-1].set(
            armn_o[1:] - armn_o[:-1] - pzu_0[:-1] * bsqr_s[:-1] + 0.5 * (lv_es[:-1] + lv_es[1:])
        )
        azmn_o = azmn_o.at[:-1].set(azmn_o[1:] - azmn_o[:-1] + pru_0[:-1] * bsqr_s[:-1])
        brmn_o = brmn_o.at[:-1].set(0.5 * (brmn_o[:-1] + brmn_o[1:]))
        bzmn_o = bzmn_o.at[:-1].set(0.5 * (bzmn_o[:-1] + bzmn_o[1:]))
        lu_o = lu_o.at[:-1].set(lu_o[:-1] + lu_o[1:])

    armn_o = armn_o.at[-1].set(-armn_o[-1] - pzu_0[-1] * bsqr_s[-1] + 0.5 * lv_es[-1])
    azmn_o = azmn_o.at[-1].set(-azmn_o[-1] + pru_0[-1] * bsqr_s[-1])
    brmn_o = brmn_o.at[-1].set(0.5 * brmn_o[-1])
    bzmn_o = bzmn_o.at[-1].set(0.5 * bzmn_o[-1])

    # Scale GIJ for odd-kernel contributions by s (VMEC: sqrts^2).
    ss = (psqrts * psqrts).astype(guu_i.dtype)
    guu_s = guu_i * ss
    gvv_s = gvv_i * ss

    armn_o = armn_o - (pzu_1 * lu_o + gvv_s * pr1_1 + gvvs_i * pr1_0)
    azmn_o = azmn_o + pru_1 * lu_o
    brmn_o = brmn_o + pz1_1 * lu_o - (guu_s * pru_1 + guus_i * pru_0)
    bzmn_o = bzmn_o - (pr1_1 * lu_o + guu_s * pzu_1 + guus_i * pzu_0)

    # 3D kernels (C terms); handle axisymmetric cases by returning zeros.
    lthreed = bool(np.any(np.asarray(static.modes.n) != 0))
    if lthreed:
        brmn_e = brmn_e - (guv_i * prv_0 + guvs_i * prv_1)
        bzmn_e = bzmn_e - (guv_i * pzv_0 + guvs_i * pzv_1)

        crmn_e = guv_i * pru_0 + gvv_i * prv_0 + gvvs_i * prv_1 + guvs_i * pru_1
        czmn_e = guv_i * pzu_0 + gvv_i * pzv_0 + gvvs_i * pzv_1 + guvs_i * pzu_1

        guv_s = guv_i * ss
        brmn_o = brmn_o - (guvs_i * prv_0 + guv_s * prv_1)
        bzmn_o = bzmn_o - (guvs_i * pzv_0 + guv_s * pzv_1)

        crmn_o = guvs_i * pru_0 + gvvs_i * prv_0 + gvv_s * prv_1 + guv_s * pru_1
        czmn_o = guvs_i * pzu_0 + gvvs_i * pzv_0 + gvv_s * pzv_1 + guv_s * pzu_1
    else:
        z = jnp.zeros_like(armn_e)
        crmn_e = z
        crmn_o = z
        czmn_e = z
        czmn_o = z

    return VmecRZForceKernels(
        armn_e=armn_e,
        armn_o=armn_o,
        brmn_e=brmn_e,
        brmn_o=brmn_o,
        crmn_e=crmn_e,
        crmn_o=crmn_o,
        azmn_e=azmn_e,
        azmn_o=azmn_o,
        bzmn_e=bzmn_e,
        bzmn_o=bzmn_o,
        czmn_e=czmn_e,
        czmn_o=czmn_o,
        bc=bc,
    )


def vmec_forces_rz_from_wout_reference_fields(*, state, static, wout) -> VmecRZForceKernels:
    """Compute VMEC R/Z force kernels using `wout`'s stored (sqrtg, bsup, ``|B|``).

    This is a parity/debug variant that reduces the number of derived quantities
    computed by vmec_jax, making it easier to validate the *forces* algebra in
    isolation.
    """
    s = jnp.asarray(static.s)
    ohs = jnp.asarray(1.0 / (s[1] - s[0])) if s.shape[0] >= 2 else jnp.asarray(0.0)
    dshalfds = jnp.asarray(0.25, dtype=s.dtype)

    # Geometry parity arrays.
    parity = split_rzl_even_odd_m(state, static.basis, static.modes.m)
    R1 = internal_odd_from_physical(parity.R_odd, s)
    Z1 = internal_odd_from_physical(parity.Z_odd, s)
    Ru1 = internal_odd_from_physical(parity.Rt_odd, s)
    Zu1 = internal_odd_from_physical(parity.Zt_odd, s)
    Rv1 = internal_odd_from_physical(parity.Rp_odd, s)
    Zv1 = internal_odd_from_physical(parity.Zp_odd, s)

    pr1_0, pr1_1 = jnp.asarray(parity.R_even), jnp.asarray(R1)
    pz1_0, pz1_1 = jnp.asarray(parity.Z_even), jnp.asarray(Z1)
    pru_0, pru_1 = jnp.asarray(parity.Rt_even), jnp.asarray(Ru1)
    pzu_0, pzu_1 = jnp.asarray(parity.Zt_even), jnp.asarray(Zu1)
    prv_0, prv_1 = jnp.asarray(parity.Rp_even), jnp.asarray(Rv1)
    pzv_0, pzv_1 = jnp.asarray(parity.Zp_even), jnp.asarray(Zv1)

    # Half-mesh Jacobian-like quantities (r12/rs/zs/ru12/zu12) from our parity kernel.
    from .vmec_jacobian import jacobian_half_mesh_from_parity

    jac = jacobian_half_mesh_from_parity(
        pr1_even=pr1_0,
        pr1_odd=pr1_1,
        pz1_even=pz1_0,
        pz1_odd=pz1_1,
        pru_even=pru_0,
        pru_odd=pru_1,
        pzu_even=pzu_0,
        pzu_odd=pzu_1,
        s=s,
    )

    # Evaluate stored wout Nyquist fields on our angular grid.
    grid = AngleGrid(theta=np.asarray(static.grid.theta), zeta=np.asarray(static.grid.zeta), nfp=int(wout.nfp))
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    basis_nyq = build_helical_basis(modes_nyq, grid)

    sqrtg = jnp.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))
    bsupu = jnp.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv = jnp.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    bmag = jnp.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))

    # bsq = |B|^2/2 + p (half mesh).
    pres_h = jnp.asarray(wout.pres)[:, None, None]
    bsq = 0.5 * (bmag * bmag) + pres_h

    # Use sqrtg from wout to define tau; r12 from our parity half-mesh construction.
    r12 = jnp.asarray(jac.r12)
    tau = jnp.where(r12 != 0, sqrtg / r12, 0.0)

    lu_e = _with_axis_zero(bsq * r12)
    lv_e = _with_axis_zero(bsq * tau)

    guu = _with_axis_zero((bsupu * bsupu) * sqrtg)
    guv = _with_axis_zero((bsupu * bsupv) * sqrtg)
    gvv = _with_axis_zero((bsupv * bsupv) * sqrtg)

    # Half-mesh sqrt(s) and full-mesh sqrt(s).
    pshalf = _pshalf_from_s(s)[:, None, None]
    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]

    # Scratch arrays (VMEC names: guus, guvs, gvvs, bsqr).
    guus = guu * pshalf
    guvs = guv * pshalf
    gvvs = gvv * pshalf

    armn_e = ohs * jac.zu12 * lu_e
    azmn_e = -ohs * jac.ru12 * lu_e
    brmn_e = jac.zs * lu_e
    bzmn_e = -jac.rs * lu_e
    bsqr0 = dshalfds * lu_e / jnp.where(pshalf != 0, pshalf, 1.0)

    armn_o = armn_e * pshalf
    azmn_o = azmn_e * pshalf
    brmn_o = brmn_e * pshalf
    bzmn_o = bzmn_e * pshalf

    # Forward-average half-mesh GIJ and shalf-weighted GIJ to integer mesh.
    guu_i = _avg_forward_half_to_int(guu)
    gvv_i = _avg_forward_half_to_int(gvv)
    guus_i = _avg_forward_half_to_int(guus)
    gvvs_i = _avg_forward_half_to_int(gvvs)
    guv_i = _avg_forward_half_to_int(guv)
    guvs_i = _avg_forward_half_to_int(guvs)

    bsqr_s = _sum_forward_half(bsqr0)

    # Even-parity kernels.
    armn_e = _diff_forward_half(armn_e, lv_e)
    azmn_e = _diff_forward_half_noavg(azmn_e)
    brmn_e = _avg_forward_half(brmn_e)
    bzmn_e = _avg_forward_half(bzmn_e)

    armn_e = armn_e - (gvvs_i * pr1_1 + gvv_i * pr1_0)
    brmn_e = brmn_e + bsqr_s * pz1_1 - (guus_i * pru_1 + guu_i * pru_0)
    bzmn_e = bzmn_e - (bsqr_s * pr1_1 + guus_i * pzu_1 + guu_i * pzu_0)

    lv_es = lv_e * pshalf
    lu_o = dshalfds * lu_e

    # Odd-parity kernels.
    if armn_o.shape[0] >= 2:
        armn_o = armn_o.at[:-1].set(
            armn_o[1:] - armn_o[:-1] - pzu_0[:-1] * bsqr_s[:-1] + 0.5 * (lv_es[:-1] + lv_es[1:])
        )
        azmn_o = azmn_o.at[:-1].set(azmn_o[1:] - azmn_o[:-1] + pru_0[:-1] * bsqr_s[:-1])
        brmn_o = brmn_o.at[:-1].set(0.5 * (brmn_o[:-1] + brmn_o[1:]))
        bzmn_o = bzmn_o.at[:-1].set(0.5 * (bzmn_o[:-1] + bzmn_o[1:]))
        lu_o = lu_o.at[:-1].set(lu_o[:-1] + lu_o[1:])

    armn_o = armn_o.at[-1].set(-armn_o[-1] - pzu_0[-1] * bsqr_s[-1] + 0.5 * lv_es[-1])
    azmn_o = azmn_o.at[-1].set(-azmn_o[-1] + pru_0[-1] * bsqr_s[-1])
    brmn_o = brmn_o.at[-1].set(0.5 * brmn_o[-1])
    bzmn_o = bzmn_o.at[-1].set(0.5 * bzmn_o[-1])

    ss = (psqrts * psqrts).astype(guu_i.dtype)
    guu_s = guu_i * ss
    gvv_s = gvv_i * ss

    armn_o = armn_o - (pzu_1 * lu_o + gvv_s * pr1_1 + gvvs_i * pr1_0)
    azmn_o = azmn_o + pru_1 * lu_o
    brmn_o = brmn_o + pz1_1 * lu_o - (guu_s * pru_1 + guus_i * pru_0)
    bzmn_o = bzmn_o - (pr1_1 * lu_o + guu_s * pzu_1 + guus_i * pzu_0)

    lthreed = bool(np.any(np.asarray(static.modes.n) != 0))
    if lthreed:
        brmn_e = brmn_e - (guv_i * prv_0 + guvs_i * prv_1)
        bzmn_e = bzmn_e - (guv_i * pzv_0 + guvs_i * pzv_1)
        crmn_e = guv_i * pru_0 + gvv_i * prv_0 + gvvs_i * prv_1 + guvs_i * pru_1
        czmn_e = guv_i * pzu_0 + gvv_i * pzv_0 + gvvs_i * pzv_1 + guvs_i * pzu_1
        guv_s = guv_i * ss
        brmn_o = brmn_o - (guvs_i * prv_0 + guv_s * prv_1)
        bzmn_o = bzmn_o - (guvs_i * pzv_0 + guv_s * pzv_1)
        crmn_o = guvs_i * pru_0 + gvvs_i * prv_0 + gvv_s * prv_1 + guv_s * pru_1
        czmn_o = guvs_i * pzu_0 + gvvs_i * pzv_0 + gvv_s * pzv_1 + guv_s * pzu_1
    else:
        z = jnp.zeros_like(armn_e)
        crmn_e = z
        crmn_o = z
        czmn_e = z
        czmn_o = z

    # `bc` object is used only for downstream scaling helpers; provide the pieces we need.
    class _BC:
        pass

    bc_obj = _BC()
    bc_obj.jac = jac
    bc_obj.gij_b_uu = guu
    bc_obj.gij_b_uv = guv
    bc_obj.gij_b_vv = gvv

    return VmecRZForceKernels(
        armn_e=armn_e,
        armn_o=armn_o,
        brmn_e=brmn_e,
        brmn_o=brmn_o,
        crmn_e=crmn_e,
        crmn_o=crmn_o,
        azmn_e=azmn_e,
        azmn_o=azmn_o,
        bzmn_e=bzmn_e,
        bzmn_o=bzmn_o,
        czmn_e=czmn_e,
        czmn_o=czmn_o,
        bc=bc_obj,
    )


@dataclass(frozen=True)
class VmecRZResidualCoeffs:
    gcr_cos: Any  # (ns, K)
    gcr_sin: Any  # (ns, K)
    gcz_cos: Any  # (ns, K)
    gcz_sin: Any  # (ns, K)


def _select_parity_coeffs(*, coeff_even, coeff_odd, m):
    mask_even = (m % 2) == 0
    return jnp.where(mask_even[None, :], coeff_even, coeff_odd)


def rz_residual_coeffs_from_kernels(k: VmecRZForceKernels, *, static) -> VmecRZResidualCoeffs:
    """Compute Fourier-space residual coefficients gcr/gcz from force kernels.

    This mirrors VMEC's ``tomnsps`` combination:
        FR = A - dB/du + dC/dv
    using coefficient-space differentiation (no finite differences).
    """
    m = jnp.asarray(static.modes.m, dtype=jnp.asarray(k.armn_e).dtype)
    n = jnp.asarray(static.modes.n, dtype=m.dtype)
    n_phys = n * int(static.grid.nfp)

    # Project each parity field to helical coefficients.
    aR_e_c, aR_e_s = project_to_modes(k.armn_e, static.basis)
    aR_o_c, aR_o_s = project_to_modes(k.armn_o, static.basis)
    bR_e_c, bR_e_s = project_to_modes(k.brmn_e, static.basis)
    bR_o_c, bR_o_s = project_to_modes(k.brmn_o, static.basis)
    cR_e_c, cR_e_s = project_to_modes(k.crmn_e, static.basis)
    cR_o_c, cR_o_s = project_to_modes(k.crmn_o, static.basis)

    aZ_e_c, aZ_e_s = project_to_modes(k.azmn_e, static.basis)
    aZ_o_c, aZ_o_s = project_to_modes(k.azmn_o, static.basis)
    bZ_e_c, bZ_e_s = project_to_modes(k.bzmn_e, static.basis)
    bZ_o_c, bZ_o_s = project_to_modes(k.bzmn_o, static.basis)
    cZ_e_c, cZ_e_s = project_to_modes(k.czmn_e, static.basis)
    cZ_o_c, cZ_o_s = project_to_modes(k.czmn_o, static.basis)

    aR_c = _select_parity_coeffs(coeff_even=aR_e_c, coeff_odd=aR_o_c, m=m)
    aR_s = _select_parity_coeffs(coeff_even=aR_e_s, coeff_odd=aR_o_s, m=m)
    bR_c = _select_parity_coeffs(coeff_even=bR_e_c, coeff_odd=bR_o_c, m=m)
    bR_s = _select_parity_coeffs(coeff_even=bR_e_s, coeff_odd=bR_o_s, m=m)
    cR_c = _select_parity_coeffs(coeff_even=cR_e_c, coeff_odd=cR_o_c, m=m)
    cR_s = _select_parity_coeffs(coeff_even=cR_e_s, coeff_odd=cR_o_s, m=m)

    aZ_c = _select_parity_coeffs(coeff_even=aZ_e_c, coeff_odd=aZ_o_c, m=m)
    aZ_s = _select_parity_coeffs(coeff_even=aZ_e_s, coeff_odd=aZ_o_s, m=m)
    bZ_c = _select_parity_coeffs(coeff_even=bZ_e_c, coeff_odd=bZ_o_c, m=m)
    bZ_s = _select_parity_coeffs(coeff_even=bZ_e_s, coeff_odd=bZ_o_s, m=m)
    cZ_c = _select_parity_coeffs(coeff_even=cZ_e_c, coeff_odd=cZ_o_c, m=m)
    cZ_s = _select_parity_coeffs(coeff_even=cZ_e_s, coeff_odd=cZ_o_s, m=m)

    # Derivatives in coefficient space.
    dBdu_R_c = m[None, :] * bR_s
    dBdu_R_s = -m[None, :] * bR_c
    dCdv_R_c = -(n_phys[None, :]) * cR_s
    dCdv_R_s = (n_phys[None, :]) * cR_c

    dBdu_Z_c = m[None, :] * bZ_s
    dBdu_Z_s = -m[None, :] * bZ_c
    dCdv_Z_c = -(n_phys[None, :]) * cZ_s
    dCdv_Z_s = (n_phys[None, :]) * cZ_c

    gcr_cos = aR_c - dBdu_R_c + dCdv_R_c
    gcr_sin = aR_s - dBdu_R_s + dCdv_R_s
    gcz_cos = aZ_c - dBdu_Z_c + dCdv_Z_c
    gcz_sin = aZ_s - dBdu_Z_s + dCdv_Z_s

    return VmecRZResidualCoeffs(gcr_cos=gcr_cos, gcr_sin=gcr_sin, gcz_cos=gcz_cos, gcz_sin=gcz_sin)


@dataclass(frozen=True)
class VmecInternalResidualRZL:
    """Internal VMEC-style residual arrays produced by `tomnsps`."""

    frcc: Any
    frss: Any | None
    fzsc: Any
    fzcs: Any | None
    flsc: Any
    flcs: Any | None


def vmec_residual_internal_from_kernels(
    k: VmecRZForceKernels,
    *,
    cfg_ntheta: int,
    cfg_nzeta: int,
    wout,
    trig: VmecTrigTables | None = None,
) -> VmecInternalResidualRZL:
    """Compute internal residual coefficient arrays using VMEC's `tomnsps` conventions."""
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(cfg_ntheta),
            nzeta=int(cfg_nzeta),
            nfp=int(wout.nfp),
            mmax=int(wout.mpol) - 1,
            nmax=int(wout.ntor),
            lasym=bool(wout.lasym),
        )

    # Lambda kernels are optional for early parity work.
    z = jnp.zeros_like(k.armn_e)
    blmn_even = getattr(k.bc, "blmn_even", z)
    blmn_odd = getattr(k.bc, "blmn_odd", z)
    clmn_even = getattr(k.bc, "clmn_even", z)
    clmn_odd = getattr(k.bc, "clmn_odd", z)

    out = tomnsps_rzl(
        armn_even=k.armn_e,
        armn_odd=k.armn_o,
        brmn_even=k.brmn_e,
        brmn_odd=k.brmn_o,
        crmn_even=k.crmn_e,
        crmn_odd=k.crmn_o,
        azmn_even=k.azmn_e,
        azmn_odd=k.azmn_o,
        bzmn_even=k.bzmn_e,
        bzmn_odd=k.bzmn_o,
        czmn_even=k.czmn_e,
        czmn_odd=k.czmn_o,
        blmn_even=blmn_even,
        blmn_odd=blmn_odd,
        clmn_even=clmn_even,
        clmn_odd=clmn_odd,
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        trig=trig,
    )
    return VmecInternalResidualRZL(
        frcc=out.frcc,
        frss=out.frss,
        fzsc=out.fzsc,
        fzcs=out.fzcs,
        flsc=out.flsc,
        flcs=out.flcs,
    )


@dataclass(frozen=True)
class VmecRZResidualScalars:
    fsqr_like: float
    fsqz_like: float


def rz_residual_scalars_like_vmec(
    coeffs: VmecRZResidualCoeffs,
    *,
    bc,
    wout,
    s,
) -> VmecRZResidualScalars:
    """Compute VMEC-like invariant scalars for the R/Z residuals.

    This uses VMEC's documented structure:
        fsqr = gnorm * sum(gcr^2),  with gnorm = r1*fnorm and r1 = 1/(2*r0scale)^2 = 1/4.

    We approximate the missing VMEC angular weighting with a uniform tensor grid.
    """
    s = np.asarray(s)
    if s.size < 2:
        return VmecRZResidualScalars(fsqr_like=float("nan"), fsqz_like=float("nan"))

    # VMEC's r0scale from fixaray defaults to 1 (mscale(0)*nscale(0)).
    r1 = 0.25

    # VMEC uses volume and energies normalized by (2π)^2 in its internal scaling.
    vol_norm = float(wout.volume_p / (4.0 * np.pi**2))
    e_norm = float(max(wout.wb, wout.wp))
    r2 = e_norm / vol_norm if vol_norm != 0.0 else float("inf")

    # Approximate <guu * R^2> with a uniform angular average of the half-mesh field.
    r12 = np.asarray(bc.jac.r12)
    guu = np.asarray(bc.gij_b_uu)
    guu_r2 = guu * (r12 * r12)
    avg_guu_r2 = float(np.mean(guu_r2[1:]))  # exclude axis surface

    fnorm = 1.0 / (avg_guu_r2 * (r2 * r2)) if avg_guu_r2 != 0.0 else float("inf")
    gnorm = r1 * fnorm

    gcr2 = float(np.sum(np.asarray(coeffs.gcr_cos)[1:] ** 2 + np.asarray(coeffs.gcr_sin)[1:] ** 2))
    gcz2 = float(np.sum(np.asarray(coeffs.gcz_cos)[1:] ** 2 + np.asarray(coeffs.gcz_sin)[1:] ** 2))
    return VmecRZResidualScalars(fsqr_like=gnorm * gcr2, fsqz_like=gnorm * gcz2)
