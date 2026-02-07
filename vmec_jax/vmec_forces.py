"""VMEC force/residue kernels (Step-10 parity work).

This module implements a direct, array-based port of VMEC2000's ``forces`` core
for the **R/Z** equations, operating on:

- VMEC even/odd-m real-space decomposition (odd stored in 1/sqrt(s) form),
- half-mesh quantities from :mod:`vmec_jax.vmec_bcovar`.

Scope
-----
This is a parity/debug kernel used to validate the algebra and staggering.
It is *not* yet the full VMEC solver pipeline (no vacuum/free boundary, no 2D
preconditioner, and no full lambda residue parity), but it *does* include the
VMEC constraint-force pipeline (`tcon` + `alias`) for fixed-boundary parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .fourier import project_to_modes
from .fourier import build_helical_basis, eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys
from .field import lamscale_from_phips
from .grids import AngleGrid
from .modes import ModeTable
from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from .vmec_constraints import alias_gcon, tcon_from_bcovar_precondn_diag, tcon_from_tcon0_heuristic
from .vmec_tomnsp import TomnspsRZL, VmecTrigTables, tomnsps_rzl, tomnspa_rzl, vmec_angle_grid, vmec_trig_tables
from .vmec_parity import internal_odd_from_physical_vmec_m1, split_rzl_even_odd_m


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

    # Constraint force kernels (passed to `tomnsps` as ARCON/AZCON).
    arcon_e: Any  # (ns, ntheta, nzeta)
    arcon_o: Any  # (ns, ntheta, nzeta)
    azcon_e: Any  # (ns, ntheta, nzeta)
    azcon_o: Any  # (ns, ntheta, nzeta)
    gcon: Any  # (ns, ntheta, nzeta)

    # Geometry parity fields (VMEC internal decomposition):
    #   X(s,θ,ζ) = X_even(s,θ,ζ) + sqrt(s) * X_odd_internal(s,θ,ζ)
    #
    # These are used to reproduce additional VMEC conventions (e.g. `lforbal`).
    pr1_even: Any  # (ns, ntheta, nzeta)
    pr1_odd: Any  # (ns, ntheta, nzeta)
    pz1_even: Any  # (ns, ntheta, nzeta)
    pz1_odd: Any  # (ns, ntheta, nzeta)
    pru_even: Any  # (ns, ntheta, nzeta)
    pru_odd: Any  # (ns, ntheta, nzeta)
    pzu_even: Any  # (ns, ntheta, nzeta)
    pzu_odd: Any  # (ns, ntheta, nzeta)

    # Optional diagnostic (set by constraint pipeline).
    tcon: Any | None = None  # (ns,)


@dataclass(frozen=True)
class VmecConstraintKernels:
    """Constraint-force kernels produced by the `alias` pipeline."""

    rcon_force: Any  # (ns, ntheta, nzeta)
    zcon_force: Any  # (ns, ntheta, nzeta)
    arcon_e: Any  # (ns, ntheta, nzeta)
    arcon_o: Any  # (ns, ntheta, nzeta)
    azcon_e: Any  # (ns, ntheta, nzeta)
    azcon_o: Any  # (ns, ntheta, nzeta)
    gcon: Any  # (ns, ntheta, nzeta)
    tcon: Any  # (ns,)


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


def _constraint_kernels_from_state(
    *,
    state,
    static,
    wout,
    bc,
    pru_0,
    pru_1,
    pzu_0,
    pzu_1,
    constraint_tcon0: float | None,
    trig: VmecTrigTables | None = None,
) -> VmecConstraintKernels:
    """Compute VMEC constraint-force kernels from state/parity fields.

    This follows the fixed-boundary pipeline in `funct3d` -> `alias` -> `forces`.
    """
    s = jnp.asarray(static.s)
    ns = int(s.shape[0])
    dtype = jnp.asarray(state.Rcos).dtype

    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
        z = jnp.zeros_like(pru_0)
        tcon = jnp.zeros((ns,), dtype=dtype)
        return VmecConstraintKernels(
            rcon_force=z,
            zcon_force=z,
            arcon_e=z,
            arcon_o=z,
            azcon_e=z,
            azcon_o=z,
            gcon=z,
            tcon=tcon,
        )

    # xmpq(m,1) = m*(m-1).
    m_modes = np.asarray(static.modes.m, dtype=int)
    m_k = jnp.asarray(static.modes.m, dtype=dtype)
    xmpq1 = m_k * (m_k - 1.0)

    mask_even = jnp.asarray((m_modes % 2) == 0, dtype=dtype)
    mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
    mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)

    rcon_even = eval_fourier(state.Rcos * xmpq1 * mask_even, state.Rsin * xmpq1 * mask_even, static.basis)
    zcon_even = eval_fourier(state.Zcos * xmpq1 * mask_even, state.Zsin * xmpq1 * mask_even, static.basis)

    rcon_odd_m1 = eval_fourier(state.Rcos * xmpq1 * mask_m1, state.Rsin * xmpq1 * mask_m1, static.basis)
    rcon_odd_rest = eval_fourier(state.Rcos * xmpq1 * mask_odd_rest, state.Rsin * xmpq1 * mask_odd_rest, static.basis)
    zcon_odd_m1 = eval_fourier(state.Zcos * xmpq1 * mask_m1, state.Zsin * xmpq1 * mask_m1, static.basis)
    zcon_odd_rest = eval_fourier(state.Zcos * xmpq1 * mask_odd_rest, state.Zsin * xmpq1 * mask_odd_rest, static.basis)

    rcon_odd_int = internal_odd_from_physical_vmec_m1(odd_m1_phys=rcon_odd_m1, odd_mge2_phys=rcon_odd_rest, s=s)
    zcon_odd_int = internal_odd_from_physical_vmec_m1(odd_m1_phys=zcon_odd_m1, odd_mge2_phys=zcon_odd_rest, s=s)

    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    rcon_phys = jnp.asarray(rcon_even) + psqrts * jnp.asarray(rcon_odd_int)
    zcon_phys = jnp.asarray(zcon_even) + psqrts * jnp.asarray(zcon_odd_int)

    # Fixed-boundary scaling for rcon0/zcon0 (funct3d.f).
    rcon0 = (s[:, None, None] * jnp.asarray(rcon_phys[-1])[None, :, :]).astype(jnp.asarray(rcon_phys).dtype)
    zcon0 = (s[:, None, None] * jnp.asarray(zcon_phys[-1])[None, :, :]).astype(jnp.asarray(zcon_phys).dtype)

    # Physical ru0/zu0 for ztemp formation.
    ru0 = jnp.asarray(pru_0) + psqrts * jnp.asarray(pru_1)
    zu0 = jnp.asarray(pzu_0) + psqrts * jnp.asarray(pzu_1)

    ztemp = (rcon_phys - rcon0) * ru0 + (zcon_phys - zcon0) * zu0

    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout.nfp),
            mmax=int(wout.mpol) - 1,
            nmax=int(wout.ntor),
            lasym=bool(wout.lasym),
            dtype=jnp.asarray(ztemp).dtype,
        )

    # VMEC computes the constraint strength `tcon(js)` in `bcovar.f` using
    # diagonal preconditioner pieces and flux-surface norms.
    tcon = tcon_from_bcovar_precondn_diag(
        tcon0=float(constraint_tcon0),
        trig=trig,
        s=s,
        signgs=int(wout.signgs),
        lasym=bool(wout.lasym),
        bsq=bc.bsq,
        r12=bc.jac.r12,
        sqrtg=bc.jac.sqrtg,
        ru12=bc.jac.ru12,
        zu12=bc.jac.zu12,
        ru0=ru0,
        zu0=zu0,
    )
    # Fallback to a conservative constant profile if ill-conditioned.
    finite = jnp.all(jnp.isfinite(tcon))
    tcon_heur = tcon_from_tcon0_heuristic(
        tcon0=float(constraint_tcon0),
        s=s,
        trig=trig,
        lasym=bool(wout.lasym),
    )
    tcon = jnp.where(finite, tcon, tcon_heur)

    gcon = alias_gcon(
        ztemp=ztemp,
        trig=trig,
        ntor=int(wout.ntor),
        mpol=int(wout.mpol),
        signgs=int(wout.signgs),
        tcon=tcon,
        lasym=bool(wout.lasym),
    )

    rcon_force = (rcon_phys - rcon0) * gcon
    zcon_force = (zcon_phys - zcon0) * gcon

    arcon_e = ru0 * gcon
    azcon_e = zu0 * gcon
    arcon_o = arcon_e * psqrts
    azcon_o = azcon_e * psqrts

    return VmecConstraintKernels(
        rcon_force=rcon_force,
        zcon_force=zcon_force,
        arcon_e=arcon_e,
        arcon_o=arcon_o,
        azcon_e=azcon_e,
        azcon_o=azcon_o,
        gcon=gcon,
        tcon=tcon,
    )


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


def vmec_forces_rz_from_wout(
    *,
    state,
    static,
    wout,
    indata=None,
    constraint_tcon0: float | None = None,
    use_wout_bsup: bool = False,
    use_vmec_synthesis: bool = False,
    trig: VmecTrigTables | None = None,
) -> VmecRZForceKernels:
    """Compute VMEC R/Z force kernels (armn/brmn/...) from a `wout` equilibrium.

    Parameters
    ----------
    use_wout_bsup:
        If True, use the Nyquist `bsup*` fields stored in the `wout` file when
        forming the B-product tensors inside `bcovar`. This isolates the forces
        algebra from small differences in the derived contravariant field. In
        this parity mode, lambda-force kernels (`blmn/clmn`) are also formed
        from averaged `wout` `bsub*` fields.
    """
    s = jnp.asarray(static.s)
    ohs = jnp.asarray(1.0 / (s[1] - s[0])) if s.shape[0] >= 2 else jnp.asarray(0.0)
    dshalfds = jnp.asarray(0.25, dtype=s.dtype)

    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        use_wout_bsup=use_wout_bsup,
        use_wout_bsub_for_lambda=use_wout_bsup,
        use_wout_bmag_for_bsq=use_wout_bsup,
        use_vmec_synthesis=use_vmec_synthesis,
        trig=trig,
    )

    # Real-space parity fields for R/Z and angular derivatives.
    parity = split_rzl_even_odd_m(state, static.basis, static.modes.m)

    # VMEC axis convention (vmec_params.f: jmin1):
    # - m=1 odd-m internal fields are extrapolated to the axis (copy js=2),
    # - odd-m with m>=3 are zero on the axis.
    m_modes = np.asarray(static.modes.m, dtype=int)
    dtype = jnp.asarray(state.Rcos).dtype
    mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
    mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)
    mask_even = jnp.asarray((m_modes % 2) == 0, dtype=dtype)

    def _odd_internal_vmec(*, coeff_cos, coeff_sin, eval_fn):
        phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1, static.basis)
        phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest, static.basis)
        return internal_odd_from_physical_vmec_m1(odd_m1_phys=phys_m1, odd_mge2_phys=phys_rest, s=s)

    R1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier)
    Z1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier)
    Ru1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier_dtheta)
    Zu1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier_dtheta)
    Rv1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier_dzeta_phys)
    Zv1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier_dzeta_phys)

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
    #
    # Important: by the time `forces.f` runs, VMEC has overwritten `guu/guv/gvv`
    # with the B-product tensors:
    #   GIJ = (B^i B^j) * sqrt(g)   for i,j ∈ {u,v}
    # (see `bcovar.f` "STORE LU * LV COMBINATIONS USED IN FORCES").
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

    # ---------------------------------------------------------------------
    # Constraint force pipeline: compute gcon from ztemp via alias and apply
    # the constraint force kernels to B-terms (forces.f "CONSTRAINT FORCE").
    # ---------------------------------------------------------------------
    if indata is not None:
        constraint_tcon0 = float(indata.get_float("TCON0", 0.0))
    con = _constraint_kernels_from_state(
        state=state,
        static=static,
        wout=wout,
        bc=bc,
        pru_0=pru_0,
        pru_1=pru_1,
        pzu_0=pzu_0,
        pzu_1=pzu_1,
        constraint_tcon0=constraint_tcon0,
        trig=trig,
    )

    brmn_e = brmn_e + con.rcon_force
    bzmn_e = bzmn_e + con.zcon_force
    brmn_o = brmn_o + con.rcon_force * psqrts
    bzmn_o = bzmn_o + con.zcon_force * psqrts

    arcon_e = con.arcon_e
    arcon_o = con.arcon_o
    azcon_e = con.azcon_e
    azcon_o = con.azcon_o
    gcon = con.gcon

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
        arcon_e=arcon_e,
        arcon_o=arcon_o,
        azcon_e=azcon_e,
        azcon_o=azcon_o,
        gcon=gcon,
        tcon=con.tcon,
        pr1_even=pr1_0,
        pr1_odd=pr1_1,
        pz1_even=pz1_0,
        pz1_odd=pz1_1,
        pru_even=pru_0,
        pru_odd=pru_1,
        pzu_even=pzu_0,
        pzu_odd=pzu_1,
    )


def vmec_forces_rz_from_wout_reference_fields(
    *,
    state,
    static,
    wout,
    indata=None,
    constraint_tcon0: float | None = None,
) -> VmecRZForceKernels:
    """Compute VMEC R/Z force kernels using `wout`'s stored (sqrtg, bsup, ``|B|``).

    This is a parity/debug variant that reduces the number of derived quantities
    computed by vmec_jax, making it easier to validate the *forces* algebra in
    isolation. If `constraint_tcon0` (or `indata.TCON0`) is provided, the VMEC
    constraint-force pipeline is also applied.
    """
    s = jnp.asarray(static.s)
    ohs = jnp.asarray(1.0 / (s[1] - s[0])) if s.shape[0] >= 2 else jnp.asarray(0.0)
    dshalfds = jnp.asarray(0.25, dtype=s.dtype)

    # Geometry parity arrays.
    parity = split_rzl_even_odd_m(state, static.basis, static.modes.m)

    m_modes = np.asarray(static.modes.m, dtype=int)
    dtype = jnp.asarray(state.Rcos).dtype
    mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
    mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)
    mask_even = jnp.asarray((m_modes % 2) == 0, dtype=dtype)

    def _odd_internal_vmec(*, coeff_cos, coeff_sin, eval_fn):
        phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1, static.basis)
        phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest, static.basis)
        return internal_odd_from_physical_vmec_m1(odd_m1_phys=phys_m1, odd_mge2_phys=phys_rest, s=s)

    R1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier)
    Z1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier)
    Ru1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier_dtheta)
    Zu1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier_dtheta)
    Rv1 = _odd_internal_vmec(coeff_cos=state.Rcos, coeff_sin=state.Rsin, eval_fn=eval_fourier_dzeta_phys)
    Zv1 = _odd_internal_vmec(coeff_cos=state.Zcos, coeff_sin=state.Zsin, eval_fn=eval_fourier_dzeta_phys)

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
    bsubu = jnp.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv = jnp.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))
    bmag = jnp.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))

    # bsq = |B|^2/2 + p (half mesh).
    pres_h = jnp.asarray(wout.pres)[:, None, None]
    bsq = 0.5 * (bmag * bmag) + pres_h

    # Use sqrtg from wout to define tau; r12 from our parity half-mesh construction.
    r12 = jnp.asarray(jac.r12)
    tau = jnp.where(r12 != 0, sqrtg / r12, 0.0)

    lu_e = _with_axis_zero(bsq * r12)
    lv_e = _with_axis_zero(bsq * tau)

    # Metric elements on the half mesh (bcovar.f convention). We keep these for
    # scaling diagnostics, but the *forces* kernel below uses GIJ (B-products).
    def _half_mesh_from_even_odd(even, odd_int, *, s):
        even = jnp.asarray(even)
        odd_int = jnp.asarray(odd_int)
        s = jnp.asarray(s)
        ns_ = int(s.shape[0])
        if ns_ < 2:
            return even
        psh = _pshalf_from_s(s)[:, None, None]
        out = jnp.zeros_like(even)
        out = out.at[1:].set(0.5 * (even[1:] + even[:-1] + psh[1:] * (odd_int[1:] + odd_int[:-1])))
        out = out.at[0].set(out[1])
        return out

    ss0 = s[:, None, None]
    guu_e = pru_0 * pru_0 + pzu_0 * pzu_0 + ss0 * (pru_1 * pru_1 + pzu_1 * pzu_1)
    guu_o = 2.0 * (pru_0 * pru_1 + pzu_0 * pzu_1)
    guv_e = prv_0 * pru_0 + pzv_0 * pzu_0 + ss0 * (prv_1 * pru_1 + pzv_1 * pzu_1)
    guv_o = prv_0 * pru_1 + prv_1 * pru_0 + pzv_0 * pzu_1 + pzv_1 * pzu_0
    gvv_e = prv_0 * prv_0 + pzv_0 * pzv_0 + ss0 * (prv_1 * prv_1 + pzv_1 * pzv_1)
    gvv_o = 2.0 * (prv_0 * prv_1 + pzv_0 * pzv_1)

    # Add R^2 term to gvv in cylindrical coordinates.
    r2_e = pr1_0 * pr1_0 + ss0 * (pr1_1 * pr1_1)
    r2_o = 2.0 * (pr1_0 * pr1_1)

    guu_metric = _with_axis_zero(_half_mesh_from_even_odd(guu_e, guu_o, s=s))
    guv_metric = _with_axis_zero(_half_mesh_from_even_odd(guv_e, guv_o, s=s))
    gvv_metric = _with_axis_zero(
        _half_mesh_from_even_odd(gvv_e, gvv_o, s=s) + _half_mesh_from_even_odd(r2_e, r2_o, s=s)
    )

    # GIJ = (B^i B^j)*sqrt(g) used in forces.f.
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

    # Build lambda-force kernels (blmn/clmn) using the VMEC formulas but with
    # reference-field inputs.
    lamscale = lamscale_from_phips(wout.phips, s)

    # For reference-field parity we form the lambda-force kernels from the
    # stored wout bsubu/bsubv fields by averaging to the full mesh. This avoids
    # re-deriving bsubv_e from lambda derivatives, which can amplify small
    # discrepancies in the reference path.
    ns = int(s.shape[0])
    bsubu_e = jnp.zeros_like(bsubu)
    bsubv_e = jnp.zeros_like(bsubv)
    if ns >= 2:
        bsubu_e = bsubu_e.at[:-1].set(0.5 * (bsubu[:-1] + bsubu[1:]))
        bsubu_e = bsubu_e.at[-1].set(0.5 * bsubu[-1])
        bsubv_e = bsubv_e.at[:-1].set(0.5 * (bsubv[:-1] + bsubv[1:]))
        bsubv_e = bsubv_e.at[-1].set(0.5 * bsubv[-1])

    # Scale for tomnsps (skip axis surface).
    clmn_even = jnp.zeros_like(bsubu_e)
    blmn_even = jnp.zeros_like(bsubv_e)
    if ns >= 2:
        clmn_even = clmn_even.at[1:].set(-lamscale * bsubu_e[1:])
        blmn_even = blmn_even.at[1:].set(-lamscale * bsubv_e[1:])
    clmn_odd = psqrts * clmn_even
    blmn_odd = psqrts * blmn_even

    # `bc` object is used only for downstream scaling helpers; provide the pieces we need.
    class _BC:
        pass

    bc_obj = _BC()
    from .vmec_jacobian import VmecHalfMeshJacobian

    bc_obj.jac = VmecHalfMeshJacobian(
        r12=jac.r12,
        rs=jac.rs,
        zs=jac.zs,
        ru12=jac.ru12,
        zu12=jac.zu12,
        tau=tau,
        sqrtg=sqrtg,
    )
    bc_obj.guu = guu_metric
    bc_obj.guv = guv_metric
    bc_obj.gvv = gvv_metric
    bc_obj.bsubu = bsubu
    bc_obj.bsubv = bsubv
    bc_obj.lamscale = lamscale
    bc_obj.bsq = bsq
    bc_obj.clmn_even = clmn_even
    bc_obj.clmn_odd = clmn_odd
    bc_obj.blmn_even = blmn_even
    bc_obj.blmn_odd = blmn_odd

    if indata is not None:
        constraint_tcon0 = float(indata.get_float("TCON0", 0.0))
    con = _constraint_kernels_from_state(
        state=state,
        static=static,
        wout=wout,
        bc=bc_obj,
        pru_0=pru_0,
        pru_1=pru_1,
        pzu_0=pzu_0,
        pzu_1=pzu_1,
        constraint_tcon0=constraint_tcon0,
    )

    brmn_e = brmn_e + con.rcon_force
    bzmn_e = bzmn_e + con.zcon_force
    brmn_o = brmn_o + con.rcon_force * psqrts
    bzmn_o = bzmn_o + con.zcon_force * psqrts

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
        arcon_e=con.arcon_e,
        arcon_o=con.arcon_o,
        azcon_e=con.azcon_e,
        azcon_o=con.azcon_o,
        gcon=con.gcon,
        tcon=con.tcon,
        pr1_even=pr1_0,
        pr1_odd=pr1_1,
        pz1_even=pz1_0,
        pz1_odd=pz1_1,
        pru_even=pru_0,
        pru_odd=pru_1,
        pzu_even=pzu_0,
        pzu_odd=pzu_1,
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
    """Internal VMEC-style residual arrays produced by `tomnsps` (+ `tomnspa` when `lasym=True`)."""

    frcc: Any
    frss: Any | None
    fzsc: Any
    fzcs: Any | None
    flsc: Any
    flcs: Any | None

    # Asymmetric components from `tomnspa` (lasym=True only).
    frsc: Any | None = None
    frcs: Any | None = None
    fzcc: Any | None = None
    fzss: Any | None = None
    flcc: Any | None = None
    flss: Any | None = None


def vmec_residual_internal_from_kernels(
    k: VmecRZForceKernels,
    *,
    cfg_ntheta: int,
    cfg_nzeta: int,
    wout,
    trig: VmecTrigTables | None = None,
    apply_lforbal: bool = False,
    include_edge: bool = False,
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

    lasym = bool(wout.lasym)

    def _symforce_split_one(
        a,
        *,
        trig: VmecTrigTables,
        kind: str,
    ):
        """Split a field into VMEC symmetric/antisymmetric parts for lasym transforms.

        VMEC's `tomnsps`/`tomnspa` always integrate on the restricted interval
        u∈[0,π] (i=1..ntheta2). For `lasym=True`, VMEC first decomposes each
        kernel into a "symmetric" piece (paired with cos(mu±nv)) and an
        "antisymmetric" piece (paired with sin(mu±nv)) using `symforce.f`.

        The mapping is not uniform across kernels (some have reversed dominant
        symmetry); see `VMEC2000/Sources/General/symforce.f`.
        """
        a = jnp.asarray(a)
        ns, ntheta3, nzeta = a.shape
        nt2 = int(trig.ntheta2)
        nt1 = int(trig.ntheta1)
        if int(trig.ntheta3) != int(ntheta3):
            raise ValueError("symforce: theta size mismatch")
        if nt2 <= 0 or nt2 > ntheta3:
            raise ValueError("symforce: invalid ntheta2")

        # Reflection map (0-based) for i=1..ntheta2:
        #   ir = ntheta1 + 2 - i, with i==1 -> ir=1   (Fortran, 1-based)
        i0 = jnp.arange(nt2, dtype=jnp.int32)
        ir0 = jnp.where(i0 == 0, 0, nt1 - i0)
        kk = (nzeta - jnp.arange(nzeta, dtype=jnp.int32)) % nzeta

        a_half = a[:, :nt2, :]
        a_ref = a[:, ir0, :][:, :, kk]

        if kind in ("ars", "bzs", "bls", "rcs", "czs", "cls"):
            a_sym_half = 0.5 * (a_half + a_ref)
            a_asym_half = 0.5 * (a_half - a_ref)
        elif kind in ("brs", "azs", "zcs", "crs"):
            # Reversed dominant symmetry (see `symforce.f`).
            a_sym_half = 0.5 * (a_half - a_ref)
            a_asym_half = 0.5 * (a_half + a_ref)
        else:  # pragma: no cover
            raise ValueError(f"symforce: unknown kind {kind!r}")

        pad = jnp.zeros((ns, ntheta3 - nt2, nzeta), dtype=a.dtype)
        return (
            jnp.concatenate([a_sym_half, pad], axis=1),
            jnp.concatenate([a_asym_half, pad], axis=1),
        )

    if lasym:
        # Decompose each kernel before calling tomnsps/tomnspa.
        armn_e_s, armn_e_a = _symforce_split_one(k.armn_e, trig=trig, kind="ars")
        armn_o_s, armn_o_a = _symforce_split_one(k.armn_o, trig=trig, kind="ars")
        brmn_e_s, brmn_e_a = _symforce_split_one(k.brmn_e, trig=trig, kind="brs")
        brmn_o_s, brmn_o_a = _symforce_split_one(k.brmn_o, trig=trig, kind="brs")
        crmn_e_s, crmn_e_a = _symforce_split_one(k.crmn_e, trig=trig, kind="crs")
        crmn_o_s, crmn_o_a = _symforce_split_one(k.crmn_o, trig=trig, kind="crs")

        azmn_e_s, azmn_e_a = _symforce_split_one(k.azmn_e, trig=trig, kind="azs")
        azmn_o_s, azmn_o_a = _symforce_split_one(k.azmn_o, trig=trig, kind="azs")
        bzmn_e_s, bzmn_e_a = _symforce_split_one(k.bzmn_e, trig=trig, kind="bzs")
        bzmn_o_s, bzmn_o_a = _symforce_split_one(k.bzmn_o, trig=trig, kind="bzs")
        czmn_e_s, czmn_e_a = _symforce_split_one(k.czmn_e, trig=trig, kind="czs")
        czmn_o_s, czmn_o_a = _symforce_split_one(k.czmn_o, trig=trig, kind="czs")

        blmn_e_s, blmn_e_a = _symforce_split_one(blmn_even, trig=trig, kind="bls")
        blmn_o_s, blmn_o_a = _symforce_split_one(blmn_odd, trig=trig, kind="bls")
        clmn_e_s, clmn_e_a = _symforce_split_one(clmn_even, trig=trig, kind="cls")
        clmn_o_s, clmn_o_a = _symforce_split_one(clmn_odd, trig=trig, kind="cls")

        arcon_e_s, arcon_e_a = _symforce_split_one(k.arcon_e, trig=trig, kind="rcs")
        arcon_o_s, arcon_o_a = _symforce_split_one(k.arcon_o, trig=trig, kind="rcs")
        azcon_e_s, azcon_e_a = _symforce_split_one(k.azcon_e, trig=trig, kind="zcs")
        azcon_o_s, azcon_o_a = _symforce_split_one(k.azcon_o, trig=trig, kind="zcs")

        out_sym = tomnsps_rzl(
            armn_even=armn_e_s,
            armn_odd=armn_o_s,
            brmn_even=brmn_e_s,
            brmn_odd=brmn_o_s,
            crmn_even=crmn_e_s,
            crmn_odd=crmn_o_s,
            azmn_even=azmn_e_s,
            azmn_odd=azmn_o_s,
            bzmn_even=bzmn_e_s,
            bzmn_odd=bzmn_o_s,
            czmn_even=czmn_e_s,
            czmn_odd=czmn_o_s,
            blmn_even=blmn_e_s,
            blmn_odd=blmn_o_s,
            clmn_even=clmn_e_s,
            clmn_odd=clmn_o_s,
            arcon_even=arcon_e_s,
            arcon_odd=arcon_o_s,
            azcon_even=azcon_e_s,
            azcon_odd=azcon_o_s,
            mpol=int(wout.mpol),
            ntor=int(wout.ntor),
            nfp=int(wout.nfp),
            lasym=True,
            trig=trig,
            include_edge=bool(include_edge),
        )

        out_asym = tomnspa_rzl(
            armn_even=armn_e_a,
            armn_odd=armn_o_a,
            brmn_even=brmn_e_a,
            brmn_odd=brmn_o_a,
            crmn_even=crmn_e_a,
            crmn_odd=crmn_o_a,
            azmn_even=azmn_e_a,
            azmn_odd=azmn_o_a,
            bzmn_even=bzmn_e_a,
            bzmn_odd=bzmn_o_a,
            czmn_even=czmn_e_a,
            czmn_odd=czmn_o_a,
            blmn_even=blmn_e_a,
            blmn_odd=blmn_o_a,
            clmn_even=clmn_e_a,
            clmn_odd=clmn_o_a,
            arcon_even=arcon_e_a,
            arcon_odd=arcon_o_a,
            azcon_even=azcon_e_a,
            azcon_odd=azcon_o_a,
            mpol=int(wout.mpol),
            ntor=int(wout.ntor),
            nfp=int(wout.nfp),
            lasym=True,
            trig=trig,
        )
    else:
        out_sym = tomnsps_rzl(
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
            arcon_even=k.arcon_e,
            arcon_odd=k.arcon_o,
            azcon_even=k.azcon_e,
            azcon_odd=k.azcon_o,
            mpol=int(wout.mpol),
            ntor=int(wout.ntor),
            nfp=int(wout.nfp),
            lasym=False,
            trig=trig,
            include_edge=bool(include_edge),
        )
        out_asym = None

    # VMEC `lforbal` modifies the (m=1,n=0) symmetric forces to satisfy the
    # flux-surface-averaged force balance exactly. This primarily affects
    # the Step-10 scalars `fsqr/fsqz`. See `VMEC2000/Sources/General/tomnsp_mod.f`.
    if bool(apply_lforbal):
        from .vmec_lforbal import apply_lforbal_to_tomnsps, lforbal_factors_from_state

        ns = int(jnp.asarray(out_sym.frcc).shape[0])
        s_grid = jnp.linspace(0.0, 1.0, ns, dtype=jnp.asarray(out_sym.frcc).dtype)
        factors = lforbal_factors_from_state(
            bc=k.bc,
            trig=trig,
            wout=wout,
            s=s_grid,
            pru_even=k.pru_even,
            pru_odd=k.pru_odd,
            pzu_even=k.pzu_even,
            pzu_odd=k.pzu_odd,
            pr1_odd=k.pr1_odd,
            pz1_odd=k.pz1_odd,
        )
        frcc2, fzsc2 = apply_lforbal_to_tomnsps(frcc=out_sym.frcc, fzsc=out_sym.fzsc, factors=factors, trig=trig)
        out_sym = TomnspsRZL(
            frcc=frcc2,
            frss=out_sym.frss,
            fzsc=fzsc2,
            fzcs=out_sym.fzcs,
            flsc=out_sym.flsc,
            flcs=out_sym.flcs,
        )

    return VmecInternalResidualRZL(
        frcc=out_sym.frcc,
        frss=out_sym.frss,
        fzsc=out_sym.fzsc,
        fzcs=out_sym.fzcs,
        flsc=out_sym.flsc,
        flcs=out_sym.flcs,
        frsc=None if out_asym is None else out_asym.frsc,
        frcs=None if out_asym is None else out_asym.frcs,
        fzcc=None if out_asym is None else out_asym.fzcc,
        fzss=None if out_asym is None else out_asym.fzss,
        flcc=None if out_asym is None else out_asym.flcc,
        flss=None if out_asym is None else out_asym.flss,
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
