"""VMEC `residue/getfsq`-style scalar residuals (Step-10 parity work).

VMEC2000 reports scalar force residual measures:

  - ``fsqr`` : R-equation residual norm
  - ``fsqz`` : Z-equation residual norm
  - ``fsql`` : lambda-equation residual norm

Internally these are computed from Fourier-space force arrays produced by
``tomnsps`` and normalized by the force norms ``fnorm`` and ``fnormL``
computed in ``bcovar``.

This module implements the *scalar* pieces needed for Step-10 output-parity
tests against bundled VMEC2000 ``wout`` files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .vmec_tomnsp import TomnspsRZL, VmecTrigTables


@dataclass(frozen=True)
class VmecForceNorms:
    """VMEC normalization constants used by `getfsq` and `residue`."""

    fnorm: float
    fnormL: float
    r1: float  # 1/(2*r0scale)^2


@dataclass(frozen=True)
class VmecForceNormsDynamic:
    """VMEC normalization constants computed directly from bcovar fields (JAX-traceable).

    This is the state-based counterpart of :class:`VmecForceNorms`, which relies on
    `wout` scalars (`vp`, `wb`, `wp`). VMEC uses these quantities to normalize the
    reported scalar residuals (`fsqr/fsqz/fsql`) in `residue/getfsq`.
    """

    fnorm: Any
    fnormL: Any
    r1: Any  # 1/(2*r0scale)^2
    r2: Any
    volume: Any
    wb: Any
    wp: Any
    vp: Any  # (ns,)


@dataclass(frozen=True)
class VmecFsqScalars:
    fsqr: float
    fsqz: float
    fsql: float


@dataclass(frozen=True)
class VmecFsqScalarsDynamic:
    """JAX-traceable scalar residuals (fsqr, fsqz, fsql)."""

    fsqr: Any
    fsqz: Any
    fsql: Any


@dataclass(frozen=True)
class VmecFsqSums:
    """Unnormalized sum-of-squares pieces used by `getfsq`-style scalars."""

    gcr2: float
    gcz2: float
    gcl2: float
    gcr2_blocks: dict[str, float]
    gcz2_blocks: dict[str, float]
    gcl2_blocks: dict[str, float]


def _constrain_m1_pair(*, gcr: Any, gcz: Any, lconm1: bool) -> tuple[Any, Any]:
    """VMEC's `constrain_m1` transform for the m=1 polar constraint.

    In `VMEC2000/Sources/General/residue.f90:constrain_m1_par`, VMEC optionally
    applies a polar constraint by rotating the (R,Z) force pair into
    (gcr+gcz, gcr-gcz)/sqrt(2) and then setting the second component to zero
    once close to convergence.

    For Step-10 parity work we apply the rotation when `lconm1=True` and always
    zero the constrained component (this matches the converged-equilibrium
    regime used for regression against bundled `wout_*.nc` files).
    """
    gcr = jnp.asarray(gcr)
    gcz = jnp.asarray(gcz)
    if bool(lconm1):
        osqrt2 = jnp.asarray(1.0 / np.sqrt(2.0), dtype=gcr.dtype)
        tmp = gcr
        gcr = osqrt2 * (gcr + gcz)
        gcz = osqrt2 * (tmp - gcz)
    gcz = jnp.zeros_like(gcz)
    return gcr, gcz


def vmec_apply_m1_constraints(
    *,
    frzl: TomnspsRZL,
    lconm1: bool = True,
) -> TomnspsRZL:
    """Apply VMEC's converged-iteration m=1 polar constraints to Fourier forces.

    VMEC calls `constrain_m1_par` on two block pairs prior to computing `fsqr/fsqz`:

    - 3D symmetric constraint: (R_ss, Z_cs)  -> enforce Z_cs(m=1) ≈ 0
    - asymmetric constraint:   (R_sc, Z_cc)  -> enforce Z_cc(m=1) ≈ 0

    See `VMEC2000/Sources/General/residue.f90`.
    """
    mpol = int(jnp.asarray(frzl.frcc).shape[1])
    if mpol <= 1:
        return frzl

    frss = frzl.frss
    fzcs = frzl.fzcs
    frsc = getattr(frzl, "frsc", None)
    fzcc = getattr(frzl, "fzcc", None)

    # 3D: constrain (rss,zcs) at m=1.
    if frss is not None and fzcs is not None:
        gcr, gcz = _constrain_m1_pair(gcr=frss[:, 1, :], gcz=fzcs[:, 1, :], lconm1=lconm1)
        frss = jnp.asarray(frss).at[:, 1, :].set(gcr)
        fzcs = jnp.asarray(fzcs).at[:, 1, :].set(gcz)

    # lasym: constrain (rsc,zcc) at m=1.
    if frsc is not None and fzcc is not None:
        gcr, gcz = _constrain_m1_pair(gcr=frsc[:, 1, :], gcz=fzcc[:, 1, :], lconm1=lconm1)
        frsc = jnp.asarray(frsc).at[:, 1, :].set(gcr)
        fzcc = jnp.asarray(fzcc).at[:, 1, :].set(gcz)

    return TomnspsRZL(
        frcc=frzl.frcc,
        frss=frss,
        fzsc=frzl.fzsc,
        fzcs=fzcs,
        flsc=frzl.flsc,
        flcs=frzl.flcs,
        frsc=frsc,
        frcs=getattr(frzl, "frcs", None),
        fzcc=fzcc,
        fzss=getattr(frzl, "fzss", None),
        flcc=getattr(frzl, "flcc", None),
        flss=getattr(frzl, "flss", None),
    )


def vmec_zero_m1_zforce(
    *,
    frzl: TomnspsRZL,
    enabled: bool = True,
) -> TomnspsRZL:
    """Zero the m=1 Z-force coefficients (VMEC++ early-iteration safeguard)."""
    enabled = jnp.asarray(enabled)
    mask = enabled.astype(jnp.asarray(frzl.fzsc).dtype)
    mpol = int(jnp.asarray(frzl.fzsc).shape[1])
    if mpol <= 1:
        return frzl

    fzsc = frzl.fzsc
    fzcs = frzl.fzcs
    fzcc = getattr(frzl, "fzcc", None)

    fzsc = jnp.asarray(fzsc)
    fzsc_new = fzsc[:, 1, :] * (1.0 - mask)
    if hasattr(fzsc, "at"):
        fzsc = fzsc.at[:, 1, :].set(fzsc_new)
    else:  # numpy fallback
        fzsc = fzsc.copy()
        fzsc[:, 1, :] = np.asarray(fzsc_new)
    if fzcs is not None:
        fzcs = jnp.asarray(fzcs)
        fzcs_new = fzcs[:, 1, :] * (1.0 - mask)
        if hasattr(fzcs, "at"):
            fzcs = fzcs.at[:, 1, :].set(fzcs_new)
        else:
            fzcs = fzcs.copy()
            fzcs[:, 1, :] = np.asarray(fzcs_new)
    if fzcc is not None:
        fzcc = jnp.asarray(fzcc)
        fzcc_new = fzcc[:, 1, :] * (1.0 - mask)
        if hasattr(fzcc, "at"):
            fzcc = fzcc.at[:, 1, :].set(fzcc_new)
        else:
            fzcc = fzcc.copy()
            fzcc[:, 1, :] = np.asarray(fzcc_new)

    return TomnspsRZL(
        frcc=frzl.frcc,
        frss=frzl.frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=frzl.flsc,
        flcs=frzl.flcs,
        frsc=getattr(frzl, "frsc", None),
        frcs=getattr(frzl, "frcs", None),
        fzcc=fzcc,
        fzss=getattr(frzl, "fzss", None),
        flcc=getattr(frzl, "flcc", None),
        flss=getattr(frzl, "flss", None),
    )


def vmec_rz_norm_from_state(
    *,
    state,
    static,
    s: Any | None = None,
    apply_scalxc: bool = True,
    ns_min: int | None = None,
    ns_max: int | None = None,
) -> Any:
    """Compute VMEC++-style rzNorm from Fourier coefficients (n>=0 storage).

    VMEC++ defines fNorm1 as 1 / rzNorm, where rzNorm is the sum of squares of
    R/Z Fourier coefficients stored with n>=0. This helper mirrors that storage
    convention by masking out n<0 modes from vmec_jax's signed mode table.
    """
    mpol = int(static.cfg.mpol)
    ntor = int(static.cfg.ntor)
    nrange = ntor + 1
    ncoeff = int(jnp.asarray(state.Rcos).shape[1])

    m = jnp.asarray(static.modes.m)
    n = jnp.asarray(static.modes.n)
    idx_pos = -jnp.ones((mpol, nrange), dtype=jnp.int32)
    idx_neg = -jnp.ones((mpol, nrange), dtype=jnp.int32)
    for k in range(ncoeff):
        m_k = int(m[k])
        n_k = int(n[k])
        if n_k >= 0:
            idx_pos = idx_pos.at[m_k, n_k].set(k)
        else:
            idx_neg = idx_neg.at[m_k, -n_k].set(k)

    def _signed_cos_to_mn(a):
        a = jnp.asarray(a)
        rcc = jnp.zeros((a.shape[0], mpol, nrange), dtype=a.dtype)
        rss = jnp.zeros_like(rcc)
        for m_i in range(mpol):
            for n_i in range(nrange):
                kp = int(idx_pos[m_i, n_i])
                if kp < 0:
                    continue
                pos = a[:, kp]
                kn = int(idx_neg[m_i, n_i])
                if kn >= 0:
                    neg = a[:, kn]
                else:
                    neg = jnp.zeros_like(pos)
                rcc = rcc.at[:, m_i, n_i].set(pos + neg)
                rss = rss.at[:, m_i, n_i].set(pos - neg)
        return rcc, rss

    def _signed_sin_to_mn(a):
        a = jnp.asarray(a)
        sc = jnp.zeros((a.shape[0], mpol, nrange), dtype=a.dtype)
        cs = jnp.zeros_like(sc)
        for m_i in range(mpol):
            for n_i in range(nrange):
                kp = int(idx_pos[m_i, n_i])
                if kp < 0:
                    continue
                pos = a[:, kp]
                kn = int(idx_neg[m_i, n_i])
                if kn >= 0:
                    neg = a[:, kn]
                else:
                    neg = jnp.zeros_like(pos)
                sc = sc.at[:, m_i, n_i].set(pos + neg)
                cs = cs.at[:, m_i, n_i].set(neg - pos)
        return sc, cs

    rcc, rss = _signed_cos_to_mn(state.Rcos)
    zsc, zcs = _signed_sin_to_mn(state.Zsin)

    if bool(apply_scalxc):
        if s is None:
            s = jnp.asarray(static.s)
        scalxc = vmec_scalxc_from_s(s=s, mpol=mpol).astype(rcc.dtype)[:, :, None]
        rcc = rcc * scalxc
        rss = rss * scalxc
        zsc = zsc * scalxc
        zcs = zcs * scalxc

    if ns_min is None:
        ns_min = 0
    if ns_max is None:
        ns_max = int(jnp.asarray(zsc).shape[0])
    sl = slice(int(ns_min), int(ns_max))

    rz_norm = jnp.sum(zsc[sl] * zsc[sl])
    m_idx = jnp.arange(mpol)[None, :, None]
    n_idx = jnp.arange(nrange)[None, None, :]
    include_rcc = (m_idx > 0) | (n_idx > 0)
    rz_norm = rz_norm + jnp.sum(jnp.where(include_rcc, rcc[sl] * rcc[sl], 0.0))

    if bool(getattr(static.cfg, "lthreed", True)):
        rz_norm = rz_norm + jnp.sum(rss[sl] * rss[sl]) + jnp.sum(zcs[sl] * zcs[sl])

    if bool(getattr(static.cfg, "lasym", False)):
        rsc, rcs = _signed_sin_to_mn(state.Rsin)
        zcc, zss = _signed_cos_to_mn(state.Zcos)
        if bool(apply_scalxc):
            if s is None:
                s = jnp.asarray(static.s)
            scalxc = vmec_scalxc_from_s(s=s, mpol=mpol).astype(rcc.dtype)[:, :, None]
            rsc = rsc * scalxc
            rcs = rcs * scalxc
            zcc = zcc * scalxc
            zss = zss * scalxc
        rz_norm = rz_norm + jnp.sum(rsc[sl] * rsc[sl])
        rz_norm = rz_norm + jnp.sum(jnp.where(include_rcc, zcc[sl] * zcc[sl], 0.0))
        if bool(getattr(static.cfg, "lthreed", True)):
            rz_norm = rz_norm + jnp.sum(rcs[sl] * rcs[sl]) + jnp.sum(zss[sl] * zss[sl])

    return rz_norm


def vmec_wint_from_trig(trig: VmecTrigTables, *, nzeta: int) -> jnp.ndarray:
    """Return VMEC's angular integration weights as a (ntheta3,nzeta) array.

    This corresponds to VMEC's per-angle weights `pwint_ns` (see `profil3d.f`):

      pwint_ns(lk) = cosmui3(lt,0)/mscale(0)

    replicated across zeta. Surface-dependent masking (e.g. `pwint(:,1)=0` on
    the axis) is handled by :func:`vmec_pwint_from_trig`.
    """
    w_theta = jnp.asarray(trig.cosmui3[:, 0]) / jnp.asarray(trig.mscale[0])
    return w_theta[:, None] * jnp.ones((int(nzeta),), dtype=w_theta.dtype)[None, :]


def vmec_pwint_from_trig(trig: VmecTrigTables, *, ns: int, nzeta: int) -> jnp.ndarray:
    """Return VMEC's `pwint` weights as a (ns,ntheta3,nzeta) array.

    VMEC defines `pwint(:,1)=0` on the magnetic axis, and for js>=2 uses the
    same angular weights `pwint_ns`. See
    `VMEC2000/Sources/Initialization_Cleanup/profil3d.f`.
    """
    ns = int(ns)
    if ns < 1:
        raise ValueError("ns must be >= 1")
    w_ang = vmec_wint_from_trig(trig, nzeta=int(nzeta))  # (ntheta3,nzeta)
    pwint = jnp.broadcast_to(w_ang[None, :, :], (ns,) + w_ang.shape)
    pwint = pwint.at[0].set(jnp.zeros_like(w_ang))
    return pwint


def vmec_force_norms_from_bcovar(*, bc, trig: VmecTrigTables, wout, s) -> VmecForceNorms:
    """Compute (fnorm, fnormL) using VMEC's bcovar normalization formulas.

    Notes
    -----
    VMEC uses:
      volume = hs * sum(vp(2:ns))
      r2     = max(wb, wp) / volume
      fnorm  = 1 / (sum(guu*r12^2*wint) * r2^2)
      fnormL = 1 / (sum((bsubu^2+bsubv^2)*wint) * lamscale^2)
    """
    s = np.asarray(s)
    if s.size < 2:
        return VmecForceNorms(fnorm=float("nan"), fnormL=float("nan"), r1=float("nan"))

    hs = float(s[1] - s[0])
    vp = np.asarray(wout.vp, dtype=float)
    volume = hs * float(np.sum(vp[1:]))  # vp(2:ns)

    wb = float(wout.wb)
    wp = float(wout.wp)
    r2 = max(wb, wp) / volume if volume != 0.0 else float("inf")

    guu = jnp.asarray(bc.guu)
    pwint = vmec_pwint_from_trig(trig, ns=int(guu.shape[0]), nzeta=int(guu.shape[2]))

    # R/Z force norm: use the half-mesh metric element guu and R12 from the Jacobian.
    # (VMEC `bcovar.f` multiplies `guu` by `r12**2` just before forming `fnorm`.)
    r12 = jnp.asarray(bc.jac.r12)
    guu_r12sq = (guu * (r12 * r12)).astype(jnp.float64)

    # VMEC's `profil3d` sets pwint(js=1)=0, so the axis does not contribute to
    # the force norms (even though arrays often satisfy X(:,1)=X(:,2)).
    denom_f = float(jnp.sum((guu_r12sq * pwint).astype(jnp.float64)))
    fnorm = 1.0 / (denom_f * (r2 * r2)) if denom_f != 0.0 else float("inf")

    bsubu = jnp.asarray(bc.bsubu)
    bsubv = jnp.asarray(bc.bsubv)
    lamscale = float(np.asarray(bc.lamscale))
    denom_L = float(jnp.sum(((bsubu * bsubu) + (bsubv * bsubv)) * pwint))
    fnormL = 1.0 / (denom_L * (lamscale * lamscale)) if denom_L != 0.0 else float("inf")

    r0scale = float(trig.r0scale)
    r1 = 1.0 / (2.0 * r0scale) ** 2
    return VmecForceNorms(fnorm=float(fnorm), fnormL=float(fnormL), r1=float(r1))


def vmec_force_norms_from_bcovar_dynamic(
    *,
    bc,
    trig: VmecTrigTables,
    s: Any,
    signgs: int,
) -> VmecForceNormsDynamic:
    """Compute (fnorm, fnormL) using VMEC's bcovar normalization formulas *without* `wout`.

    This routine reconstructs the normalization scalars directly from bcovar's
    real-space fields, mirroring how the corresponding `wout` scalars are built:

    - ``vp(js) = ⟨ signgs*sqrtg ⟩`` (surface integral with VMEC weights)
    - ``volume = hs * sum(vp(2:ns))``
    - ``wb = hs * sum( ⟨ signgs*sqrtg * (|B|^2/2) ⟩(2:ns) )``
    - ``wp = hs * sum( vp(js) * pres(js) , js=2..ns )``

    and then:

    - ``r2 = max(wb, wp) / volume``
    - ``fnorm  = 1 / (sum(guu*r12^2*wint) * r2^2)``
    - ``fnormL = 1 / (sum((bsubu^2+bsubv^2)*wint) * lamscale^2)``

    Returns JAX scalars/arrays suitable for use inside jitted solver objectives.
    """
    signgs = int(signgs)
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        z = jnp.asarray(float("nan"))
        return VmecForceNormsDynamic(fnorm=z, fnormL=z, r1=z, r2=z, volume=z, wb=z, wp=z, vp=jnp.zeros((ns,)))

    hs = jnp.asarray(s[1] - s[0], dtype=jnp.asarray(s).dtype)

    sqrtg = jnp.asarray(bc.jac.sqrtg)
    if sqrtg.ndim != 3:
        raise ValueError(f"bc.jac.sqrtg must be (ns,ntheta3,nzeta), got shape {sqrtg.shape}")
    nzeta = int(sqrtg.shape[2])

    w_ang = vmec_wint_from_trig(trig, nzeta=nzeta)  # (ntheta3,nzeta)

    # VMEC's `pwint` has pwint(js=1)=0 (axis masked). Implement that here via a
    # radial mask to keep this routine JIT-friendly.
    mask_js = (jnp.arange(ns, dtype=jnp.int32) > 0).astype(sqrtg.dtype)[:, None, None]

    # Volume derivative per surface (normalized by (2π)^2, VMEC convention).
    jac = jnp.asarray(float(signgs), dtype=sqrtg.dtype) * sqrtg
    jac = jac * mask_js
    vp = jnp.sum(w_ang[None, :, :] * jac, axis=(1, 2))
    volume = hs * jnp.sum(vp[1:])  # vp(2:ns)

    # Magnetic energy scalar wb (same normalization as wout.wb).
    bsupu = jnp.asarray(bc.bsupu)
    bsupv = jnp.asarray(bc.bsupv)
    bsubu = jnp.asarray(bc.bsubu)
    bsubv = jnp.asarray(bc.bsubv)
    b2 = (bsupu * bsubu) + (bsupv * bsubv)
    wblocal = jnp.sum(w_ang[None, :, :] * jac * (0.5 * b2), axis=(1, 2))
    wb = hs * jnp.sum(wblocal[1:])  # js=2..ns

    # Pressure scalar wp. Pressure is flux-surface function, but bcovar stores
    # bsq = |B|^2/2 + pres on the half mesh, so we can reconstruct pres robustly.
    bsq = jnp.asarray(bc.bsq)
    pres = bsq - (0.5 * b2)
    pres_1d = pres[:, 0, 0]
    wp = hs * jnp.sum(vp[1:] * pres_1d[1:])

    r2 = jnp.where(volume != 0.0, jnp.maximum(wb, wp) / volume, jnp.asarray(float("inf"), dtype=wb.dtype))

    # Force norms.
    r12 = jnp.asarray(bc.jac.r12)
    guu = jnp.asarray(bc.guu).astype(jnp.float64)
    guu_r12sq = (guu * (r12 * r12)).astype(jnp.float64)

    pwint = (w_ang[None, :, :] * mask_js).astype(jnp.float64)
    denom_f = jnp.sum(guu_r12sq * pwint)
    fnorm = jnp.where(denom_f != 0.0, 1.0 / (denom_f * (r2 * r2)), jnp.asarray(float("inf"), dtype=denom_f.dtype))

    denom_L = jnp.sum((((bsubu * bsubu) + (bsubv * bsubv)) * pwint).astype(jnp.float64))
    lamscale = jnp.asarray(bc.lamscale, dtype=denom_L.dtype)
    fnormL = jnp.where(
        denom_L != 0.0,
        1.0 / (denom_L * (lamscale * lamscale)),
        jnp.asarray(float("inf"), dtype=denom_L.dtype),
    )

    r0scale = jnp.asarray(float(trig.r0scale), dtype=denom_f.dtype)
    r1 = 1.0 / (2.0 * r0scale) ** 2
    return VmecForceNormsDynamic(fnorm=fnorm, fnormL=fnormL, r1=r1, r2=r2, volume=volume, wb=wb, wp=wp, vp=vp)


def vmec_scalxc_from_s(*, s: Any, mpol: int) -> jnp.ndarray:
    """Reproduce VMEC's `scalxc(js,m)` factors used to scale forces before `residue`.

    In `profil3d.f`, VMEC defines `scalxc` to convert odd-m Fourier coefficients
    into a `1/sqrt(s)` internal representation:

      scalxc(js, m odd) = 1 / max(sqrt(s_js), sqrt(s_2))

    with `scalxc=1` for even m. After `tomnsps`, VMEC multiplies the force
    coefficient array by the *same* `scalxc` again (`funct3d.f: gc = gc*scalxc`)
    before calling `residue/getfsq`. This scaling is therefore part of the
    definition of the reported scalar residuals `fsqr/fsqz/fsql` in `wout`.

    Parameters
    ----------
    s:
        Radial grid (ns,) in [0,1]. VMEC uses a uniform toroidal-flux grid.
    mpol:
        Number of poloidal modes (m = 0..mpol-1).
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    mpol = int(mpol)
    if ns == 0 or mpol <= 0:
        return jnp.zeros((ns, max(mpol, 0)), dtype=jnp.asarray(s).dtype)

    sqrts = jnp.sqrt(jnp.maximum(s, 0.0))
    # VMEC (profil3d.f) explicitly sets sqrts(ns)=1 to avoid edge roundoff.
    if ns >= 1:
        sqrts = sqrts.at[-1].set(jnp.asarray(1.0, dtype=sqrts.dtype))
    sq2 = sqrts[1] if ns >= 2 else jnp.asarray(1.0, dtype=sqrts.dtype)
    scal_odd = 1.0 / jnp.maximum(sqrts, sq2)

    m = jnp.arange(mpol, dtype=jnp.int32)
    is_odd = (m % 2) == 1
    out = jnp.where(is_odd[None, :], scal_odd[:, None], jnp.ones((ns, mpol), dtype=sqrts.dtype))
    return out


def vmec_apply_scalxc_to_tomnsps(*, frzl: TomnspsRZL, s: Any) -> TomnspsRZL:
    """Apply VMEC's post-tomnsps `scalxc` scaling to force coefficient blocks."""
    frcc = jnp.asarray(frzl.frcc)
    ns, mpol, _ = frcc.shape
    scalxc = vmec_scalxc_from_s(s=s, mpol=int(mpol)).astype(frcc.dtype)[:, :, None]

    def _maybe_scale(x):
        if x is None:
            return None
        return jnp.asarray(x) * scalxc

    return TomnspsRZL(
        frcc=frcc * scalxc,
        frss=_maybe_scale(frzl.frss),
        fzsc=jnp.asarray(frzl.fzsc) * scalxc,
        fzcs=_maybe_scale(frzl.fzcs),
        flsc=jnp.asarray(frzl.flsc) * scalxc,
        flcs=_maybe_scale(frzl.flcs),
        frsc=_maybe_scale(getattr(frzl, "frsc", None)),
        frcs=_maybe_scale(getattr(frzl, "frcs", None)),
        fzcc=_maybe_scale(getattr(frzl, "fzcc", None)),
        fzss=_maybe_scale(getattr(frzl, "fzss", None)),
        flcc=_maybe_scale(getattr(frzl, "flcc", None)),
        flss=_maybe_scale(getattr(frzl, "flss", None)),
    )


def vmec_gcx2_from_tomnsps(
    *,
    frzl: TomnspsRZL,
    lconm1: bool = True,
    apply_m1_constraints: bool = True,
    include_edge: bool = False,
    apply_scalxc: bool = True,
    s: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Return VMEC-style (gcr2, gcz2, gcl2) sum-of-squares as JAX scalars.

    This is the JAX-traceable core used by both diagnostics and solver objectives.
    Unlike :func:`vmec_fsq_sums_from_tomnsps`, it does not build per-block Python
    dictionaries and does not cast to Python floats.
    """
    if bool(apply_m1_constraints):
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(lconm1))

    if bool(apply_scalxc):
        ns = int(jnp.asarray(frzl.frcc).shape[0])
        if s is None:
            # VMEC uses a uniform toroidal-flux grid. `static.s` is also uniform.
            s = jnp.linspace(0.0, 1.0, ns, dtype=jnp.asarray(frzl.frcc).dtype)
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)

    ns = int(jnp.asarray(frzl.frcc).shape[0])
    if ns <= 1:
        jsmax = ns
    else:
        jsmax = ns if bool(include_edge) else (ns - 1)

    gcr2 = jnp.sum(jnp.asarray(frzl.frcc)[:jsmax] ** 2)
    gcz2 = jnp.sum(jnp.asarray(frzl.fzsc)[:jsmax] ** 2)
    gcl2 = jnp.sum(jnp.asarray(frzl.flsc) ** 2)
    if frzl.frss is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frss)[:jsmax] ** 2)
    if frzl.fzcs is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcs)[:jsmax] ** 2)
    if frzl.flcs is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcs) ** 2)

    if getattr(frzl, "frsc", None) is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frsc)[:jsmax] ** 2)
    if getattr(frzl, "fzcc", None) is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcc)[:jsmax] ** 2)
    if getattr(frzl, "flcc", None) is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcc) ** 2)

    if getattr(frzl, "frcs", None) is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frcs)[:jsmax] ** 2)
    if getattr(frzl, "fzss", None) is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzss)[:jsmax] ** 2)
    if getattr(frzl, "flss", None) is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flss) ** 2)

    return gcr2, gcz2, gcl2


def vmec_fsq_from_tomnsps(
    *,
    frzl: TomnspsRZL,
    norms: VmecForceNorms,
    lconm1: bool = True,
    apply_m1_constraints: bool = True,
    include_edge: bool = False,
    apply_scalxc: bool = True,
    s: Any | None = None,
) -> VmecFsqScalars:
    """Compute (fsqr,fsqz,fsql) from VMEC-style tomnsps outputs.

    When `lasym=True`, VMEC also computes and includes the asymmetric blocks
    produced by `tomnspa`. In this repo those blocks (if present) are carried on
    the same dataclass as optional fields.
    """
    gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
        frzl=frzl,
        lconm1=bool(lconm1),
        apply_m1_constraints=bool(apply_m1_constraints),
        include_edge=bool(include_edge),
        apply_scalxc=bool(apply_scalxc),
        s=s,
    )

    fsqr = norms.r1 * norms.fnorm * float(gcr2)
    fsqz = norms.r1 * norms.fnorm * float(gcz2)
    fsql = norms.fnormL * float(gcl2)
    return VmecFsqScalars(fsqr=float(fsqr), fsqz=float(fsqz), fsql=float(fsql))


def vmec_fsq_from_tomnsps_dynamic(
    *,
    frzl: TomnspsRZL,
    norms: VmecForceNormsDynamic,
    lconm1: bool = True,
    apply_m1_constraints: bool = True,
    include_edge: bool = False,
    apply_scalxc: bool = True,
    s: Any | None = None,
) -> VmecFsqScalarsDynamic:
    """Compute (fsqr,fsqz,fsql) as JAX scalars from VMEC-style tomnsps outputs."""
    gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
        frzl=frzl,
        lconm1=bool(lconm1),
        apply_m1_constraints=bool(apply_m1_constraints),
        include_edge=bool(include_edge),
        apply_scalxc=bool(apply_scalxc),
        s=s,
    )
    fsqr = jnp.asarray(norms.r1) * jnp.asarray(norms.fnorm) * gcr2
    fsqz = jnp.asarray(norms.r1) * jnp.asarray(norms.fnorm) * gcz2
    fsql = jnp.asarray(norms.fnormL) * gcl2
    return VmecFsqScalarsDynamic(fsqr=fsqr, fsqz=fsqz, fsql=fsql)


def vmec_fsq_sums_from_tomnsps(
    *,
    frzl: TomnspsRZL,
    lconm1: bool = True,
    apply_m1_constraints: bool = True,
    include_edge: bool = False,
    apply_scalxc: bool = True,
    s: Any | None = None,
) -> VmecFsqSums:
    """Return the sum-of-squares components used in `vmec_fsq_from_tomnsps`.

    This is a diagnostic helper to attribute `fsqr/fsqz/fsql` differences to
    particular tomnsps/tomnspa blocks during the parity push.
    """
    if bool(apply_m1_constraints):
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(lconm1))

    if bool(apply_scalxc):
        ns = int(jnp.asarray(frzl.frcc).shape[0])
        if s is None:
            s = jnp.linspace(0.0, 1.0, ns, dtype=jnp.asarray(frzl.frcc).dtype)
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)

    ns = int(jnp.asarray(frzl.frcc).shape[0])
    if ns <= 1:
        jsmax = ns
    else:
        jsmax = ns if bool(include_edge) else (ns - 1)

    gcr2_blocks: dict[str, float] = {}
    gcz2_blocks: dict[str, float] = {}
    gcl2_blocks: dict[str, float] = {}

    def _add_block(d: dict[str, float], name: str, a: Any | None, *, slice_js: bool):
        if a is None:
            return
        arr = jnp.asarray(a)
        if slice_js:
            arr = arr[:jsmax]
        d[name] = float(jnp.sum(arr ** 2))

    # Symmetric blocks (tomnsps).
    _add_block(gcr2_blocks, "frcc", frzl.frcc, slice_js=True)
    _add_block(gcr2_blocks, "frss", frzl.frss, slice_js=True)
    _add_block(gcz2_blocks, "fzsc", frzl.fzsc, slice_js=True)
    _add_block(gcz2_blocks, "fzcs", frzl.fzcs, slice_js=True)
    _add_block(gcl2_blocks, "flsc", frzl.flsc, slice_js=False)
    _add_block(gcl2_blocks, "flcs", frzl.flcs, slice_js=False)

    # Asymmetric blocks (tomnspa, lasym=True).
    _add_block(gcr2_blocks, "frsc", getattr(frzl, "frsc", None), slice_js=True)
    _add_block(gcr2_blocks, "frcs", getattr(frzl, "frcs", None), slice_js=True)
    _add_block(gcz2_blocks, "fzcc", getattr(frzl, "fzcc", None), slice_js=True)
    _add_block(gcz2_blocks, "fzss", getattr(frzl, "fzss", None), slice_js=True)
    _add_block(gcl2_blocks, "flcc", getattr(frzl, "flcc", None), slice_js=False)
    _add_block(gcl2_blocks, "flss", getattr(frzl, "flss", None), slice_js=False)

    gcr2 = float(sum(gcr2_blocks.values()))
    gcz2 = float(sum(gcz2_blocks.values()))
    gcl2 = float(sum(gcl2_blocks.values()))
    return VmecFsqSums(
        gcr2=gcr2,
        gcz2=gcz2,
        gcl2=gcl2,
        gcr2_blocks=gcr2_blocks,
        gcz2_blocks=gcz2_blocks,
        gcl2_blocks=gcl2_blocks,
    )
