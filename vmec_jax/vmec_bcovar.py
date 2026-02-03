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
from .field import TWOPI
from .field import lamscale_from_phips
from .field import chips_from_chipf
from .vmec_jacobian import VmecHalfMeshJacobian, jacobian_half_mesh_from_parity
from .fourier import eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys
from .vmec_parity import internal_odd_from_physical_vmec_m1, split_rzl_even_odd_m


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

    # VMEC lambda force kernels (Fourier-space transform inputs) on full mesh.
    # These correspond to `bsubu_e/bsubv_e` in `bcovar.f` after the `-lamscale`
    # scaling, and are used as `(CLMN, BLMN)` in `tomnsps`.
    clmn_even: Any  # (ns, ntheta, nzeta)
    clmn_odd: Any  # (ns, ntheta, nzeta)
    blmn_even: Any  # (ns, ntheta, nzeta)
    blmn_odd: Any  # (ns, ntheta, nzeta)

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
    ns = int(s.shape[0])

    # Split real-space fields into even/odd-m subsets, then convert odd physical
    # contribution to VMEC's internal odd field by dividing by sqrt(s).
    parity = split_rzl_even_odd_m(state, static.basis, static.modes.m)

    # VMEC axis convention (vmec_params.f: jmin1):
    # - m=1 odd-m internal fields are extrapolated to the axis (copy js=2),
    # - odd-m with m>=3 are zero on the axis.
    m_modes = np.asarray(static.modes.m, dtype=int)
    dtype = jnp.asarray(state.Rcos).dtype
    mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
    mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)

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

    Lu1 = _odd_internal_vmec(coeff_cos=state.Lcos, coeff_sin=state.Lsin, eval_fn=eval_fourier_dtheta)
    Lv1 = _odd_internal_vmec(coeff_cos=state.Lcos, coeff_sin=state.Lsin, eval_fn=eval_fourier_dzeta_phys)

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

    # ---------------------------------------------------------------------
    # Contravariant B components (bsupu, bsupv) on the half mesh.
    # ---------------------------------------------------------------------
    # VMEC does not form bsupu/bsupv from pointwise (sqrtg, lam_u, lam_v) alone.
    # Instead, it:
    #   1) builds full-mesh LU = d(lambda)/du and LV = -d(lambda)/dv (even/odd-m),
    #   2) scales LU/LV by lamscale and adds phipf to LU_even,
    #   3) averages (LU,LV) from full -> half radial mesh using pshalf,
    #   4) adds the full-mesh flux function chips(js) via `add_fluxes`.
    #
    # See `VMEC2000/Sources/General/bcovar.f` and `add_fluxes.f90`.
    lamscale = lamscale_from_phips(wout.phips, s)

    # VMEC adds the **full-mesh** flux function `chips(js)` to bsupu in
    # `add_fluxes`, while `wout` commonly stores the half-mesh array `chipf`.
    chipf_out = getattr(wout, "chipf", None)
    if chipf_out is not None:
        chips_eff = chips_from_chipf(chipf_out)
    else:
        chips_eff = jnp.asarray(getattr(wout, "iotaf", getattr(wout, "iotas", 0.0))) * jnp.asarray(wout.phipf)

    # Keep the existing vmec_jax normalization for overg to preserve the tested
    # energy/volume scaling used elsewhere in the parity suite.
    denom = int(wout.signgs) * jac.sqrtg * jnp.asarray(TWOPI, dtype=jac.sqrtg.dtype)
    overg = jnp.where(denom != 0, 1.0 / denom, 0.0)

    # Full-mesh LU = d(lambda)/du and LV = -d(lambda)/dv in VMEC conventions.
    lu0_full = jnp.asarray(parity.Lt_even)
    lu1_full = jnp.asarray(Lu1)
    lv0_full = -jnp.asarray(parity.Lp_even)
    lv1_full = -jnp.asarray(Lv1)

    # Scale by lamscale and add phipf to LU_even.
    lu0_full = (lamscale * lu0_full) + jnp.asarray(wout.phipf)[:, None, None]
    lu1_full = lamscale * lu1_full
    lv0_full = lamscale * lv0_full
    lv1_full = lamscale * lv1_full

    pshalf = _pshalf_from_s(s)[:, None, None]

    # Radial full->half average (Fortran: for l=2..ns).
    bsupu = jnp.zeros_like(jac.sqrtg)
    bsupv = jnp.zeros_like(jac.sqrtg)
    if ns >= 2:
        avg_lu0 = lu0_full[1:] + lu0_full[:-1]
        avg_lu1 = lu1_full[1:] + lu1_full[:-1]
        avg_lv0 = lv0_full[1:] + lv0_full[:-1]
        avg_lv1 = lv1_full[1:] + lv1_full[:-1]

        bsupv = bsupv.at[1:].set(0.5 * overg[1:] * (avg_lu0 + pshalf[1:] * avg_lu1))
        bsupu = bsupu.at[1:].set(0.5 * overg[1:] * (avg_lv0 + pshalf[1:] * avg_lv1))

    # `add_fluxes`: bsupu += chips*overg (chips is a 1D full-mesh flux function).
    bsupu = bsupu + jnp.asarray(chips_eff)[:, None, None] * overg

    # VMEC enforces axis bsup*=0 explicitly.
    if ns >= 1:
        bsupu = bsupu.at[0].set(jnp.zeros_like(bsupu[0]))
        bsupv = bsupv.at[0].set(jnp.zeros_like(bsupv[0]))

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

    # ---------------------------------------------------------------------
    # Lambda force kernels (bcovar.f "lambda full mesh forces" block)
    # ---------------------------------------------------------------------
    # This reproduces the structure in `bcovar.f`:
    #   - compute an intermediate bsubv_e on the full radial mesh from LU and metrics
    #   - average (bsubuh,bsubvh) from the half mesh onto the full mesh
    #   - blend bsubv_e with averaged bsubvh using bdamp(s) for near-axis stability
    #
    # Inputs:
    # - LU (full mesh, parity-split): LU = phipf + lamscale*dλ/du
    # - lvv (half mesh): lvv = (g_vv / (signgs*sqrtg*2π))
    # - bsubu/bsubv (half mesh): covariant B components

    # Full-mesh LU parity pieces (odd is VMEC-internal 1/sqrt(s) representation).
    lu0 = (lamscale * parity.Lt_even) + jnp.asarray(wout.phipf)[:, None, None]
    lu1 = lamscale * Lu1

    # lvv on half mesh: phipog * gvv (bcovar.f uses phipog==overg-like array).
    lvv = overg * gvv

    # Intermediate full-mesh bsubv_e (before blending), following bcovar.f.
    bsubv_e = jnp.zeros_like(bsubv)
    if ns >= 2:
        bsubv_e = bsubv_e.at[:-1].set(0.5 * (lvv[:-1] + lvv[1:]) * lu0[:-1])
        bsubv_e = bsubv_e.at[-1].set(0.5 * lvv[-1] * lu0[-1])

    lvv_sh = lvv * pshalf
    bsubu_tmp = guv * bsupu  # bcovar: pguv*bsupu (sigma_an=1 isotropic)
    if ns >= 2:
        bsubv_e = bsubv_e.at[:-1].add(
            0.5 * ((lvv_sh[:-1] + lvv_sh[1:]) * lu1[:-1] + bsubu_tmp[:-1] + bsubu_tmp[1:])
        )
        bsubv_e = bsubv_e.at[-1].add(0.5 * (lvv_sh[-1] * lu1[-1] + bsubu_tmp[-1]))

    # Average lambda forces onto full radial mesh (bsubu_e from bsubu half mesh).
    bsubu_e = jnp.zeros_like(bsubu)
    if ns >= 2:
        bsubu_e = bsubu_e.at[:-1].set(0.5 * (bsubu[:-1] + bsubu[1:]))
        bsubu_e = bsubu_e.at[-1].set(0.5 * bsubu[-1])

    # Blend bsubv_e with half-mesh bsubv average using bdamp(s) (VMEC: bdamp=2*pdamp*(1-s)).
    pdamp = 0.05
    bdamp = (2.0 * pdamp * (1.0 - s)).astype(jnp.asarray(bsubv_e).dtype)[:, None, None]
    if ns >= 2:
        bsubv_avg = jnp.zeros_like(bsubv_e)
        bsubv_avg = bsubv_avg.at[:-1].set(0.5 * (bsubv[:-1] + bsubv[1:]))
        bsubv_avg = bsubv_avg.at[-1].set(0.5 * bsubv[-1])
        bsubv_e = bdamp * bsubv_e + (1.0 - bdamp) * bsubv_avg
    else:
        bsubv_e = bdamp * bsubv_e + (1.0 - bdamp) * bsubv_e

    # Final scaling for tomnsps:
    # VMEC applies the "-lamscale" factor only for js>=2 (1-based). The axis (js=1)
    # is excluded so the lambda-force kernels do not introduce spurious constant
    # contributions from the copied/extrapolated half-mesh axis values.
    #
    # VMEC also exposes odd-m pieces as sqrt(s)*bsub*_e.
    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    clmn_even = jnp.zeros_like(bsubu_e)
    blmn_even = jnp.zeros_like(bsubv_e)
    if ns >= 2:
        clmn_even = clmn_even.at[1:].set(-lamscale * bsubu_e[1:])
        blmn_even = blmn_even.at[1:].set(-lamscale * bsubv_e[1:])
    clmn_odd = psqrts * clmn_even
    blmn_odd = psqrts * blmn_even

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
        clmn_even=clmn_even,
        clmn_odd=clmn_odd,
        blmn_even=blmn_even,
        blmn_odd=blmn_odd,
    )
