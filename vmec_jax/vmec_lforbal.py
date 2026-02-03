"""VMEC `lforbal` (m=1,n=0 force-balance) correction for Step-10 parity work.

VMEC2000 optionally enforces the flux-surface-averaged force balance equation
*exactly* for the (m=1,n=0) Fourier components of the symmetric R/Z forces
after `tomnsps` (see `VMEC2000/Sources/General/tomnsp_mod.f`):

    frcc(n=0,m=1,js) <- rzu_fac(js) * (t1*equif(js) + work)
    fzsc(n=0,m=1,js) <- rru_fac(js) * (t1*equif(js) - work)

with:

    work = frcc_fac(js)*frcc + fzsc_fac(js)*fzsc

The factors (rzu_fac, rru_fac, frcc_fac, fzsc_fac) are computed in `bcovar.f`
from the 1D preconditioner helper `precondn.f` and are functions of the current
equilibrium state. This module ports just enough of that logic to apply the
same correction in vmec_jax parity checks.

Scope
-----
This is currently used for Step-10 scalar parity (fsqr/fsqz) against bundled
VMEC2000 `wout_*.nc` files. It is not yet used in a full time-stepper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .vmec_tomnsp import VmecTrigTables

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class VmecLforbalFactors:
    """Factors used by VMEC's `lforbal` correction (per radial surface)."""

    rzu_fac: Any  # (ns,)
    rru_fac: Any  # (ns,)
    frcc_fac: Any  # (ns,)
    fzsc_fac: Any  # (ns,)
    equif: Any  # (ns,)


def _pshalf_from_s(s: Any) -> Any:
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    sh = 0.5 * (s[1:] + s[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, 0.0))


def _sm_sp_from_s(s: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute VMEC `sm(i)` and `sp(i)` arrays (profil1d.f), in 1-based indexing."""
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        z = jnp.zeros((ns + 1,), dtype=s.dtype)
        return z, z

    hs = s[1] - s[0]
    # Fortran-style arrays of length ns+1 (indices 0..ns, with 1..ns used).
    i = jnp.arange(ns + 1, dtype=s.dtype)
    psqrts = jnp.where(i >= 1, jnp.sqrt(jnp.maximum(hs * (i - 1.0), 0.0)), 0.0)
    # Avoid roundoff at the edge like VMEC: psqrts(:,ns) = 1.
    psqrts = psqrts.at[ns].set(jnp.asarray(1.0, dtype=psqrts.dtype))
    pshalf = jnp.where(i >= 1, jnp.sqrt(jnp.maximum(hs * jnp.abs(i - 1.5), 0.0)), 0.0)

    sm = jnp.zeros((ns + 1,), dtype=s.dtype)
    sp = jnp.zeros((ns + 1,), dtype=s.dtype)

    # sm(i) = pshalf(i)/psqrts(i), i>=2
    idx = jnp.arange(2, ns + 1)
    sm = sm.at[idx].set(jnp.where(psqrts[idx] != 0, pshalf[idx] / psqrts[idx], 0.0))
    sm = sm.at[1].set(0.0)

    # sp(i) = pshalf(i+1)/psqrts(i) for i<ns, else 1/psqrts(ns)
    idx2 = jnp.arange(2, ns)
    sp = sp.at[idx2].set(jnp.where(psqrts[idx2] != 0, pshalf[idx2 + 1] / psqrts[idx2], 0.0))
    sp = sp.at[ns].set(jnp.where(psqrts[ns] != 0, 1.0 / psqrts[ns], 0.0))
    sp = sp.at[0].set(0.0)
    sp = sp.at[1].set(sm[2] if ns >= 2 else jnp.asarray(0.0, dtype=s.dtype))
    return sm, sp


def _pwint_from_trig(trig: VmecTrigTables, *, nzeta: int, dtype) -> jnp.ndarray:
    """Return VMEC `pwint_ns` as a (ntheta3,nzeta) array (profil3d.f)."""
    w_theta = jnp.asarray(trig.cosmui3[:, 0], dtype=dtype) / jnp.asarray(trig.mscale[0], dtype=dtype)
    return w_theta[:, None] * jnp.ones((int(nzeta),), dtype=dtype)[None, :]


def equif_from_bcovar(*, bc, trig: VmecTrigTables, wout, s: Any) -> jnp.ndarray:
    """Compute VMEC's `equif(js)` profile from bsubu/bsubv averages (fbal.f)."""
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        return jnp.zeros((ns,), dtype=jnp.float64)

    hs = s[1] - s[0]
    ohs = jnp.where(hs != 0, 1.0 / hs, 0.0).astype(jnp.float64)
    signgs = float(getattr(wout, "signgs", 1))

    bsubu = jnp.asarray(bc.bsubu, dtype=jnp.float64)
    bsubv = jnp.asarray(bc.bsubv, dtype=jnp.float64)
    _, ntheta, nzeta = bsubu.shape
    pwint = _pwint_from_trig(trig, nzeta=nzeta, dtype=jnp.float64)  # (ntheta,nzeta)

    # VMEC sets pwint(:,1)=0 and only fills js>=2.
    js_fortran = jnp.arange(ns, dtype=jnp.int32) + 1
    w_s = (js_fortran >= 2).astype(jnp.float64)[:, None, None]
    pwint3 = w_s * pwint[None, :, :]

    buco = jnp.sum(bsubu * pwint3, axis=(1, 2))
    bvco = jnp.sum(bsubv * pwint3, axis=(1, 2))

    # Ampere-law discrete currents.
    buco_fwd = jnp.concatenate([buco[1:], jnp.zeros((1,), dtype=buco.dtype)], axis=0)
    bvco_fwd = jnp.concatenate([bvco[1:], jnp.zeros((1,), dtype=bvco.dtype)], axis=0)
    jcurv = signgs * ohs * (buco_fwd - buco)
    jcuru = -signgs * ohs * (bvco_fwd - bvco)

    vp = jnp.asarray(getattr(wout, "vp", np.zeros((ns,), dtype=float)), dtype=jnp.float64)
    vpphi = 0.5 * (jnp.concatenate([vp[1:], vp[-1:]], axis=0) + vp)
    vpphi = jnp.where(vpphi != 0, vpphi, jnp.ones_like(vpphi))

    pres = jnp.asarray(getattr(wout, "pres", np.zeros((ns,), dtype=float)), dtype=jnp.float64)
    pres_fwd = jnp.concatenate([pres[1:], pres[-1:]], axis=0)
    presgrad = (pres_fwd - pres) * ohs

    # IMPORTANT: `wout_*.nc` stores `phipf/chipf` with an extra `2π*signgs` factor
    # (see `wrout.f`: `twopi*signgs*phipf`, `twopi*signgs*chipf`). Internally
    # (e.g. `fbal.f`) VMEC uses the unscaled arrays. Undo that scaling here.
    phipf_wout = jnp.asarray(getattr(wout, "phipf", np.zeros((ns,), dtype=float)), dtype=jnp.float64)
    chipf_wout = jnp.asarray(getattr(wout, "chipf", np.zeros((ns,), dtype=float)), dtype=jnp.float64)
    sgn = float(getattr(wout, "signgs", 1))
    denom = TWOPI * sgn
    denom = denom if denom != 0.0 else 1.0
    phipf = phipf_wout / denom
    chipf = chipf_wout / denom

    equif = ((-phipf * jcuru + chipf * jcurv) / vpphi) + presgrad
    equif = equif.at[0].set(0.0)
    equif = equif.at[-1].set(0.0)
    return equif


def _eqfactor_from_precondn_like_vmec(
    *,
    bsq,
    sqrtg,
    r12,
    xu12,
    xue,
    xuo,
    trigmult,
    trig: VmecTrigTables,
    wout,
    s: Any,
) -> jnp.ndarray:
    """Compute the `eqfactor(js)` output of VMEC `precondn.f` (reduced to what's needed).

    This follows `precondn_par` but computes only the terms required for
    `eqfactor` (returned as a length-ns array).
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        return jnp.zeros((ns,), dtype=jnp.float64)

    hs = s[1] - s[0]
    ohs = jnp.where(hs != 0, 1.0 / hs, 0.0).astype(jnp.float64)

    # pfactor = -4*r0scale^2 (precondn.f). Our trig tables use r0scale=1.
    pfactor = -4.0 * float(trig.r0scale) ** 2
    cp25 = 0.25

    bsq = jnp.asarray(bsq, dtype=jnp.float64)
    sqrtg = jnp.asarray(sqrtg, dtype=jnp.float64)
    r12 = jnp.asarray(r12, dtype=jnp.float64)
    xu12 = jnp.asarray(xu12, dtype=jnp.float64)
    xue = jnp.asarray(xue, dtype=jnp.float64)
    xuo = jnp.asarray(xuo, dtype=jnp.float64)

    _, ntheta, nzeta = bsq.shape
    pwint = _pwint_from_trig(trig, nzeta=nzeta, dtype=jnp.float64)  # (ntheta,nzeta)

    # VMEC sets pwint(:,1)=0 (axis) and fills js>=2.
    js_fortran = jnp.arange(ns, dtype=jnp.int32) + 1
    w_s = (js_fortran >= 2).astype(jnp.float64)[:, None, None]
    pwint3 = w_s * pwint[None, :, :]

    pshalf = _pshalf_from_s(s).astype(jnp.float64)
    # Precondn uses pshalf(js) (1D) but as a (ntheta,nzeta) broadcast.
    pshalf3 = pshalf[:, None, None]

    # ptau = pfactor * r12^2 * bsq * pwint / sqrtg
    gs = jnp.where(sqrtg != 0, sqrtg, jnp.ones_like(sqrtg))
    t1w = (pfactor * r12 * bsq) * pwint3
    ptau = (r12 * t1w) / gs

    # temp(js) = sum( pfactor*r12*bsq*pwint * trigmult * xu12 )
    trigmult = jnp.asarray(trigmult, dtype=jnp.float64)
    if trigmult.shape != (ntheta, nzeta):
        raise ValueError("trigmult must have shape (ntheta3,nzeta)")
    temp = jnp.sum(t1w * trigmult[None, :, :] * xu12, axis=(1, 2))

    # Normalize by vp(js) and then apply the forward sum with signgs.
    vp = jnp.asarray(getattr(wout, "vp", np.zeros((ns,), dtype=float)), dtype=jnp.float64)
    vp_safe = jnp.where(vp != 0, vp, jnp.ones_like(vp))
    temp = temp / vp_safe
    temp_fwd = jnp.concatenate([temp[1:], jnp.zeros((1,), dtype=temp.dtype)], axis=0)
    signgs = float(getattr(wout, "signgs", 1))
    temp = signgs * (temp + temp_fwd)

    # ax terms needed for axd(:,2).
    t1 = xu12 * ohs
    # t2(js) uses xue(js)/pshalf(js) + xuo(js), all divided by pshalf(js)
    t2 = cp25 * (xue / pshalf3 + xuo) / pshalf3
    xue_prev = jnp.concatenate([xue[:1], xue[:-1]], axis=0)
    xuo_prev = jnp.concatenate([xuo[:1], xuo[:-1]], axis=0)
    t3 = cp25 * (xue_prev / pshalf3 + xuo_prev) / pshalf3

    a3 = (t1 + t2) * (t1 + t2)
    a4 = (-t1 + t3) * (-t1 + t3)
    ax3 = jnp.sum(ptau * a3, axis=(1, 2))
    ax4 = jnp.sum(ptau * a4, axis=(1, 2))

    sm, sp = _sm_sp_from_s(s)  # (ns+1,), 1-based indexing
    # axd(js,2) = ax(js,3)*sm(js)^2 + ax(js+1,4)*sp(js)^2
    # Build 1-based views of ax3/ax4 for js=1..ns by padding at front.
    ax3_f = jnp.concatenate([jnp.zeros((1,), dtype=ax3.dtype), ax3], axis=0)
    ax4_f = jnp.concatenate([jnp.zeros((1,), dtype=ax4.dtype), ax4], axis=0)
    ax4_next = jnp.concatenate([ax4_f[2:], jnp.zeros((2,), dtype=ax4_f.dtype)], axis=0)  # index shift by +1
    axd2 = ax3_f * (sm * sm) + ax4_next * (sp * sp)

    # eqfactor(js) = axd(js,2)*hs^2/temp(js), defined for js=2..ns-1.
    eq = jnp.zeros((ns + 1,), dtype=jnp.float64)
    js = jnp.arange(ns + 1, dtype=jnp.int32)
    mask = (js >= 2) & (js <= (ns - 1))
    temp_f = jnp.concatenate([jnp.zeros((1,), dtype=temp.dtype), temp], axis=0)
    denom = jnp.where(temp_f != 0, temp_f, jnp.ones_like(temp_f))
    core = axd2 * (hs * hs) / denom
    eq = jnp.where(mask.astype(eq.dtype), core, eq)
    # Return 0-based (ns,) from 1-based eq(1..ns).
    return eq[1:]


def lforbal_factors_from_state(
    *,
    bc,
    trig: VmecTrigTables,
    wout,
    s: Any,
    pru_even,
    pru_odd,
    pzu_even,
    pzu_odd,
    pr1_odd,
    pz1_odd,
) -> VmecLforbalFactors:
    """Compute VMEC lforbal factors from bcovar + parity-split R/Z derivatives."""
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2 or int(trig.ntheta3) < 2 or int(getattr(wout, "mpol", 0)) <= 1:
        z = jnp.zeros((ns,), dtype=jnp.float64)
        return VmecLforbalFactors(rzu_fac=z, rru_fac=z, frcc_fac=z, fzsc_fac=z, equif=z)

    # Build trigmult arrays from fixaray.f: cos01/sin01 correspond to m=1 with mscale.
    cos01_theta = jnp.asarray(trig.cosmu[:, 1], dtype=jnp.float64)  # (ntheta3,)
    sin01_theta = jnp.asarray(trig.sinmum[:, 1], dtype=jnp.float64)  # (ntheta3,)
    nzeta = int(trig.cosnv.shape[0])
    cos01 = cos01_theta[:, None] * jnp.ones((nzeta,), dtype=jnp.float64)[None, :]
    sin01 = sin01_theta[:, None] * jnp.ones((nzeta,), dtype=jnp.float64)[None, :]

    # IMPORTANT: `wout_*.nc` stores a *normalized* version of VMEC's equif
    # (see `VMEC2000/Sources/Input_Output/eqfor.f`), not the raw quantity used
    # by `lforbal` inside `tomnsp_mod.f`. For correct parity we must compute
    # the raw profile from `bsubu/bsubv` (see `VMEC2000/Sources/General/fbal.f`).
    eq = equif_from_bcovar(bc=bc, trig=trig, wout=wout, s=s)

    # Compute eqfactor outputs from precondn (reduced).
    rzu_eq = _eqfactor_from_precondn_like_vmec(
        bsq=bc.bsq,
        sqrtg=bc.jac.sqrtg,
        r12=bc.jac.r12,
        xu12=bc.jac.zu12,
        xue=pzu_even,
        xuo=pzu_odd,
        trigmult=cos01,
        trig=trig,
        wout=wout,
        s=s,
    )
    rru_eq = _eqfactor_from_precondn_like_vmec(
        bsq=bc.bsq,
        sqrtg=bc.jac.sqrtg,
        r12=bc.jac.r12,
        xu12=bc.jac.ru12,
        xue=pru_even,
        xuo=pru_odd,
        trigmult=sin01,
        trig=trig,
        wout=wout,
        s=s,
    )

    # bcovar.f scales eqfactor by psqrts and then halves it, while also defining
    # frcc_fac=1/rzu_fac and fzsc_fac=-1/rru_fac *before* halving.
    psqrts = jnp.sqrt(jnp.maximum(s, 0.0)).astype(jnp.float64)
    psqrts = psqrts.at[-1].set(1.0)

    rzu_full = psqrts * rzu_eq
    rru_full = psqrts * rru_eq
    # Avoid divide-by-zero.
    frcc_fac = jnp.where(rzu_full != 0, 1.0 / rzu_full, 0.0)
    fzsc_fac = jnp.where(rru_full != 0, -1.0 / rru_full, 0.0)
    rzu_fac = 0.5 * rzu_full
    rru_fac = 0.5 * rru_full

    return VmecLforbalFactors(rzu_fac=rzu_fac, rru_fac=rru_fac, frcc_fac=frcc_fac, fzsc_fac=fzsc_fac, equif=eq)


def apply_lforbal_to_tomnsps(
    *,
    frcc,
    fzsc,
    factors: VmecLforbalFactors,
    trig: VmecTrigTables,
):
    """Apply the VMEC lforbal correction to (frcc,fzsc) arrays in-place (returns new arrays)."""
    frcc = jnp.asarray(frcc)
    fzsc = jnp.asarray(fzsc)
    ns, mpol, nt = frcc.shape
    if ns < 2 or mpol <= 1 or nt < 1:
        return frcc, fzsc

    # Only affects n=0, m=1 on js=2..ns-1 (fixed-boundary convention: js=ns excluded).
    m1 = 1
    n0 = 0
    js = jnp.arange(ns, dtype=jnp.int32) + 1  # 1..ns
    mask = (js >= 2) & (js <= (ns - 1))
    mask_f = mask.astype(frcc.dtype)

    fr = frcc[:, m1, n0]
    fz = fzsc[:, m1, n0]

    work = jnp.asarray(factors.frcc_fac, dtype=frcc.dtype) * fr + jnp.asarray(factors.fzsc_fac, dtype=frcc.dtype) * fz

    # tomnsp_mod.f: t1 = nscale(0)*r0scale. With our trig tables nscale(0)=1.
    t1 = jnp.asarray(float(trig.r0scale), dtype=frcc.dtype)
    equif = jnp.asarray(factors.equif, dtype=frcc.dtype)
    rzu = jnp.asarray(factors.rzu_fac, dtype=frcc.dtype)
    rru = jnp.asarray(factors.rru_fac, dtype=frcc.dtype)

    fr_new = rzu * (t1 * equif + work)
    fz_new = rru * (t1 * equif - work)

    fr_new = fr * (1.0 - mask_f) + fr_new * mask_f
    fz_new = fz * (1.0 - mask_f) + fz_new * mask_f

    frcc = jnp.asarray(frcc).at[:, m1, n0].set(fr_new)
    fzsc = jnp.asarray(fzsc).at[:, m1, n0].set(fz_new)
    return frcc, fzsc
