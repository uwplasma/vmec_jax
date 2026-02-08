"""VMEC++ preconditioner helpers (axisymmetric parity path)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .vmec_residue import vmec_wint_from_trig
from .vmec_tomnsp import TomnspsRZL


def _sqrt_profiles_from_s(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=float)
    sqrt_sf = np.sqrt(np.maximum(s, 0.0))
    if s.size < 2:
        return sqrt_sf, sqrt_sf[:0]
    sh = 0.5 * (s[1:] + s[:-1])
    sqrt_sh = np.sqrt(np.maximum(sh, 0.0))
    return sqrt_sf, sqrt_sh


def _sm_sp_from_s(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=float)
    ns = int(s.shape[0])
    if ns < 2:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    sqrt_sf, sqrt_sh = _sqrt_profiles_from_s(s)
    sm = np.zeros((ns - 1,), dtype=float)
    sp = np.zeros((ns - 1,), dtype=float)
    for jh in range(ns - 1):
        jfi = jh
        jfo = jh + 1
        denom_outer = sqrt_sf[jfo] if sqrt_sf[jfo] != 0.0 else 1.0
        sm[jh] = sqrt_sh[jh] / denom_outer
        if jh > 0:
            denom_inner = sqrt_sf[jfi] if sqrt_sf[jfi] != 0.0 else 1.0
            sp[jh] = sqrt_sh[jh] / denom_inner
    if ns > 1:
        sp[0] = sm[0]
    return sm, sp


def vmecpp_lambda_preconditioner(
    *,
    bc,
    trig,
    s: np.ndarray,
    cfg,
    damping_factor: float = 1.0,
) -> np.ndarray:
    """Compute VMEC++ lambda preconditioner (n>=0 storage)."""
    guu = np.asarray(bc.guu, dtype=float)
    guv = np.asarray(bc.guv, dtype=float)
    gvv = np.asarray(bc.gvv, dtype=float)
    gsqrt = np.asarray(bc.jac.sqrtg, dtype=float)
    ns = int(guu.shape[0])
    ntheta = int(guu.shape[1])
    nzeta = int(guu.shape[2])
    w_ang = np.asarray(vmec_wint_from_trig(trig, nzeta=nzeta), dtype=float)
    w_int = np.asarray(w_ang[:, 0], dtype=float)

    # half-grid accumulation (shifted by +1)
    b_lambda = np.zeros((ns + 1,), dtype=float)
    d_lambda = np.zeros((ns + 1,), dtype=float)
    c_lambda = np.zeros((ns + 1,), dtype=float)
    gsqrt_safe = np.where(gsqrt != 0.0, gsqrt, 1.0)
    for jh in range(ns - 1):
        for kl in range(ntheta * nzeta):
            l = kl % ntheta
            k = kl // ntheta
            b_lambda[jh + 1] += guu[jh, l, k] / gsqrt_safe[jh, l, k] * w_int[l]
            c_lambda[jh + 1] += gvv[jh, l, k] / gsqrt_safe[jh, l, k] * w_int[l]
            if bool(cfg.lthreed):
                d_lambda[jh + 1] += guv[jh, l, k] / gsqrt_safe[jh, l, k] * w_int[l]

    # constant extrapolation toward axis
    b_lambda[0] = b_lambda[1]
    d_lambda[0] = d_lambda[1]
    c_lambda[0] = c_lambda[1]
    b_lambda[ns] = b_lambda[ns - 1]
    d_lambda[ns] = d_lambda[ns - 1]
    c_lambda[ns] = c_lambda[ns - 1]

    # average onto full grid
    b_full = np.zeros((ns,), dtype=float)
    d_full = np.zeros((ns,), dtype=float)
    c_full = np.zeros((ns,), dtype=float)
    for jf in range(1, ns):
        b_full[jf] = 0.5 * (b_lambda[jf + 1] + b_lambda[jf])
        d_full[jf] = 0.5 * (d_lambda[jf + 1] + d_lambda[jf])
        c_full[jf] = 0.5 * (c_lambda[jf + 1] + c_lambda[jf])

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    p_factor = float(damping_factor) / (4.0 * float(bc.lamscale) * float(bc.lamscale))
    sqrt_sf = np.sqrt(np.maximum(np.asarray(s, dtype=float), 0.0))
    if sqrt_sf.size > 0:
        sqrt_sf[-1] = 1.0

    lam_prec = np.zeros((ns, mpol, nrange), dtype=float)
    for jf in range(1, ns):
        for n in range(nrange):
            tnn = (n * cfg.nfp) ** 2
            for m in range(mpol):
                if m == 0 and n == 0:
                    continue
                tmm = m * m
                pwr = min(tmm / (16.0 * 16.0), 8.0)
                tmn = 2.0 * m * n * cfg.nfp
                faclam = tnn * b_full[jf] + tmn * np.copysign(d_full[jf], b_full[jf]) + tmm * c_full[jf]
                if faclam == 0.0:
                    faclam = -1.0e-10
                lam_prec[jf, m, n] = p_factor / faclam * (sqrt_sf[jf] ** pwr)
    return lam_prec


def _compute_preconditioning_matrix(
    *,
    xs: np.ndarray,
    xu12: np.ndarray,
    xu_e: np.ndarray,
    xu_o: np.ndarray,
    x1_o: np.ndarray,
    r12: np.ndarray,
    total_pressure: np.ndarray,
    tau: np.ndarray,
    bsupv: np.ndarray,
    sqrtg: np.ndarray,
    w_int: np.ndarray,
    sqrt_sh: np.ndarray,
    sm: np.ndarray,
    sp: np.ndarray,
    delta_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ns = int(xs.shape[0])
    ntheta = int(xs.shape[1])
    nzeta = int(xs.shape[2])
    ax = np.zeros((ns - 1, 4), dtype=float)
    bx = np.zeros((ns - 1, 3), dtype=float)
    cx = np.zeros((ns - 1,), dtype=float)
    pfactor = -4.0
    tau_safe = np.where(tau != 0.0, tau, 1.0)
    sqrt_sh_safe = np.where(sqrt_sh != 0.0, sqrt_sh, 1.0)
    for jh in range(ns - 1):
        for kl in range(ntheta * nzeta):
            l = kl % ntheta
            k = kl // ntheta
            p_tau = pfactor * r12[jh, l, k] * total_pressure[jh, l, k] / tau_safe[jh, l, k] * w_int[l]
            t1a = xu12[jh, l, k] / delta_s
            t2a = 0.25 * (xu_e[jh + 1, l, k] / sqrt_sh_safe[jh] + xu_o[jh + 1, l, k]) / sqrt_sh_safe[jh]
            t3a = 0.25 * (xu_e[jh, l, k] / sqrt_sh_safe[jh] + xu_o[jh, l, k]) / sqrt_sh_safe[jh]
            ax[jh, 0] += p_tau * t1a * t1a
            ax[jh, 1] += p_tau * (t1a + t2a) * (-t1a + t3a)
            ax[jh, 2] += p_tau * (t1a + t2a) * (t1a + t2a)
            ax[jh, 3] += p_tau * (-t1a + t3a) * (-t1a + t3a)
            t1b = 0.5 * (xs[jh, l, k] + 0.5 / sqrt_sh_safe[jh] * x1_o[jh + 1, l, k])
            t2b = 0.5 * (xs[jh, l, k] + 0.5 / sqrt_sh_safe[jh] * x1_o[jh, l, k])
            bx[jh, 0] += p_tau * t1b * t2b
            bx[jh, 1] += p_tau * t1b * t1b
            bx[jh, 2] += p_tau * t2b * t2b
            cx[jh] += 0.25 * pfactor * (bsupv[jh, l, k] ** 2) * sqrtg[jh, l, k] * w_int[l]
    axm = np.zeros((ns - 1, 2), dtype=float)
    bxm = np.zeros((ns - 1, 2), dtype=float)
    axm[:, 0] = -ax[:, 0]
    axm[:, 1] = ax[:, 1] * sm * sp
    bxm[:, 0] = bx[:, 0]
    bxm[:, 1] = bx[:, 0] * sm * sp
    axd = np.zeros((ns, 2), dtype=float)
    bxd = np.zeros((ns, 2), dtype=float)
    cxd = np.zeros((ns,), dtype=float)
    for jf in range(ns):
        jhi = jf - 1
        jho = jf
        if jf > 0:
            axd[jf, 0] += ax[jhi, 0]
            axd[jf, 1] += ax[jhi, 2] * sm[jhi] * sm[jhi]
            bxd[jf, 0] += bx[jhi, 1]
            bxd[jf, 1] += bx[jhi, 1] * sm[jhi] * sm[jhi]
            cxd[jf] += cx[jhi]
        if jf < ns - 1:
            axd[jf, 0] += ax[jho, 0]
            axd[jf, 1] += ax[jho, 3] * sp[jho] * sp[jho]
            bxd[jf, 0] += bx[jho, 2]
            bxd[jf, 1] += bx[jho, 2] * sp[jho] * sp[jho]
            cxd[jf] += cx[jho]
    return axm, axd, bxm, bxd, cxd


def _tridiagonal_solve_vmecpp(a: np.ndarray, d: np.ndarray, b: np.ndarray, rhs: np.ndarray, jmin: int, jmax: int) -> np.ndarray:
    """VMEC++-style Thomas solve with jmin/jmax bounds."""
    out = rhs.copy()
    n = int(rhs.shape[0])
    if jmax <= jmin or n == 0:
        return out
    jmin = max(0, int(jmin))
    jmax = min(int(jmax), n)
    a = a.copy()
    d = d.copy()
    b = b.copy()
    # zero out before jmin
    for j in range(0, jmin):
        a[j] = 0.0
        d[j] = 1.0
        b[j] = 0.0
        out[j] = 0.0
    if d[jmin] == 0.0:
        d[jmin] = 1.0e-12
    a[jmin] /= d[jmin]
    for j in range(jmin + 1, jmax - 1):
        denom = d[j] - a[j - 1] * b[j]
        if denom == 0.0:
            denom = 1.0e-12
        a[j] /= denom
    out[jmin] /= d[jmin]
    for j in range(jmin + 1, jmax):
        denom = d[j] - a[j - 1] * b[j]
        if denom == 0.0:
            denom = 1.0e-12
        out[j] = (out[j] - out[j - 1] * b[j]) / denom
    for j in range(jmax - 2, jmin - 1, -1):
        out[j] = out[j] - a[j] * out[j + 1]
    return out


def vmecpp_rz_preconditioner(
    *,
    frzl_in: TomnspsRZL,
    bc,
    k,
    trig,
    s: np.ndarray,
    cfg,
) -> TomnspsRZL:
    """Apply VMEC++ R/Z radial preconditioner (axisymmetric only)."""
    if bool(cfg.lthreed) or bool(cfg.lasym):
        return frzl_in
    s_arr = np.asarray(s, dtype=float)
    ns = int(s_arr.shape[0])
    ntheta = int(bc.guu.shape[1])
    nzeta = int(bc.guu.shape[2])
    w_ang = np.asarray(vmec_wint_from_trig(trig, nzeta=nzeta), dtype=float)
    w_int = np.asarray(w_ang[:, 0], dtype=float)
    r12 = np.asarray(bc.jac.r12, dtype=float)
    tau = np.asarray(bc.jac.tau, dtype=float)
    total_pressure = np.asarray(bc.bsq, dtype=float)
    bsupv = np.asarray(bc.bsupv, dtype=float)
    sqrt_sf, sqrt_sh = _sqrt_profiles_from_s(s_arr)
    sm, sp = _sm_sp_from_s(s_arr)
    delta_s = float(s_arr[1] - s_arr[0]) if ns >= 2 else 1.0

    arm, ard, brm, brd, cxd = _compute_preconditioning_matrix(
        xs=np.asarray(bc.jac.zs, dtype=float),
        xu12=np.asarray(bc.jac.zu12, dtype=float),
        xu_e=np.asarray(k.pzu_even, dtype=float),
        xu_o=np.asarray(k.pzu_odd, dtype=float),
        x1_o=np.asarray(k.pz1_odd, dtype=float),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
    )
    azm, azd, bzm, bzd, _ = _compute_preconditioning_matrix(
        xs=np.asarray(bc.jac.rs, dtype=float),
        xu12=np.asarray(bc.jac.ru12, dtype=float),
        xu_e=np.asarray(k.pru_even, dtype=float),
        xu_o=np.asarray(k.pru_odd, dtype=float),
        x1_o=np.asarray(k.pr1_odd, dtype=float),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
    )

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    ar = np.zeros((ns, mpol, nrange), dtype=float)
    br = np.zeros_like(ar)
    dr = np.zeros_like(ar)
    az = np.zeros_like(ar)
    bz = np.zeros_like(ar)
    dz = np.zeros_like(ar)
    for m in range(mpol):
        m_par = m % 2
        jmin = 1 if m > 0 else 0
        for n in range(nrange):
            for jf in range(ns):
                if jf < jmin:
                    continue
                if jf < ns - 1:
                    ar[jf, m, n] = -(arm[jf, m_par] + brm[jf, m_par] * m * m)
                    az[jf, m, n] = -(azm[jf, m_par] + bzm[jf, m_par] * m * m)
                dr[jf, m, n] = -(ard[jf, m_par] + brd[jf, m_par] * m * m + cxd[jf] * (n * cfg.nfp) ** 2)
                dz[jf, m, n] = -(azd[jf, m_par] + bzd[jf, m_par] * m * m + cxd[jf] * (n * cfg.nfp) ** 2)
                if jf > 0:
                    br[jf, m, n] = -(arm[jf - 1, m_par] + brm[jf - 1, m_par] * m * m)
                    bz[jf, m, n] = -(azm[jf - 1, m_par] + bzm[jf - 1, m_par] * m * m)
                if jf == 1 and m == 1:
                    dr[jf, m, n] += br[jf, m, n]
                    dz[jf, m, n] += bz[jf, m, n]

    frcc_u = np.array(frzl_in.frcc, dtype=float, copy=True)
    fzsc_u = np.array(frzl_in.fzsc, dtype=float, copy=True)
    jmax = ns - 1
    for m in range(mpol):
        jmin = 1 if m > 0 else 0
        for n in range(nrange):
            frcc_u[:, m, n] = _tridiagonal_solve_vmecpp(ar[:, m, n], dr[:, m, n], br[:, m, n], frcc_u[:, m, n], jmin, jmax)
            fzsc_u[:, m, n] = _tridiagonal_solve_vmecpp(az[:, m, n], dz[:, m, n], bz[:, m, n], fzsc_u[:, m, n], jmin, jmax)

    return TomnspsRZL(
        frcc=frcc_u,
        frss=frzl_in.frss,
        fzsc=fzsc_u,
        fzcs=frzl_in.fzcs,
        flsc=frzl_in.flsc,
        flcs=frzl_in.flcs,
        frsc=getattr(frzl_in, "frsc", None),
        frcs=getattr(frzl_in, "frcs", None),
        fzcc=getattr(frzl_in, "fzcc", None),
        fzss=getattr(frzl_in, "fzss", None),
        flcc=getattr(frzl_in, "flcc", None),
        flss=getattr(frzl_in, "flss", None),
    )
