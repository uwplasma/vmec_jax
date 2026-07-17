"""NESTOR vacuum solve (Merkel Green's-function method), pure JAX.

Computes the scalar magnetic potential on the plasma boundary from the
normal component of the external field, following P. Merkel
[J. Comp. Phys. 66, 83 (1986)] as implemented by VMEC2000's NESTOR
(``Sources/NESTOR_vacuum/{precal,surface,bextern,analyt,greenf,fourp,
fouri,scalpot,vacuum}.f``).

The math is ported from the legacy parity-proven implementation
(``vmex.solvers.free_boundary.jax_nestor_operator`` — host table
builders — and ``...free_boundary.adjoint.vmec_nestor`` — the JAX
assembly), cleaned into one module:

- :func:`vacuum_basis` — all geometry-independent tables (VMEC ``precal.f``:
  mode tables, weighted sin/cos projection bases, ``cmns`` analytic-integral
  coefficients, tan tables, per-period trig tables, ``fourp`` index maps).
  Host NumPy, cached per resolution.
- :func:`make_vacuum_solver` — jit-compiled closures over a basis:

  * ``full(boundary, bexni)``: the complete NESTOR update
    (``ivacskip == 0``): non-singular Green-function source/kernel
    (``greenf`` + ``fourp``), analytic singular terms (``analyt``), mode
    projection (``fouri``) and the dense ``mnpd2 x mnpd2`` solve
    (``solver``) — returns ``potvac`` plus the cached pieces
    (``mode_matrix`` = ``amatsav``, ``bvec_nonsing`` = ``bvecsav``).
  * ``skip(boundary, bexni, bvec_nonsing, mode_matrix)``: the incremental
    update (``ivacskip != 0``): only the analytic source is recomputed and
    the cached matrix is reused (``scalpot.f`` skip branch).

- :func:`vacuum_channels` — surface field from ``potvac``:
  ``B_u = bexu + d(pot)/du`` etc., contravariant components through the
  boundary metric, and ``bsqvac = |B|^2/2`` (``vacuum.f`` tail).

Conventions (identical to VMEC NESTOR):

- Angular grid: ``theta_j = 2*pi*j/nu_full`` for ``j < ntheta3`` (full range
  when ``lasym``), ``zeta_k = 2*pi*k/nzeta`` per field period.
- ``Rv/Zv`` and the second derivatives are *geometric-phi* derivatives
  (``xn = n*nfp`` in ``surface.f``); the ``onp = 1/nfp`` factors below fold
  them into per-period metric quantities exactly as ``surface.f`` does.
- ``bexni = -(B.n) * wint * (2*pi)**2`` with the *non-unit* normal
  ``n = signgs*(R*Zu, Ru*Zv - Rv*Zu, -R*Ru)`` (``bextern.f``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

__all__ = [
    "VacuumBasis",
    "VacuumBoundary",
    "build_cmns",
    "make_vacuum_solver",
    "precal_tan_tables",
    "vacuum_basis",
    "vacuum_channels",
]

Array = Any


# ---------------------------------------------------------------------------
# Host precomputation (precal.f)
# ---------------------------------------------------------------------------


def build_cmns(*, mf: int, nf: int, onp: float) -> np.ndarray:
    """VMEC ``precal.f`` ``cmns(l, m, n)`` analytic-integral coefficients."""
    mf = max(0, int(mf))
    nf = max(0, int(nf))
    lmax = mf + nf
    cmn = np.zeros((lmax + 1, mf + 1, nf + 1), dtype=float)
    for m in range(mf + 1):
        for n in range(nf + 1):
            jmn = m + n
            imn = m - n
            kmn = abs(imn)
            smn = (jmn + kmn) // 2
            f1 = 1.0
            f2 = 1.0
            f3 = 1.0
            for i in range(1, kmn + 1):
                f1 *= float(smn + 1 - i)
                f2 *= float(i)
            for ell in range(kmn, jmn + 1, 2):
                cmn[ell, m, n] = (f1 / (f2 * f3)) * ((-1.0) ** ((ell - imn) // 2))
                f1 = f1 * 0.25 * float((jmn + ell + 2) * (jmn - ell))
                f2 = f2 * 0.5 * float(ell + 2 + kmn)
                f3 = f3 * 0.5 * float(ell + 2 - kmn)

    alp = 2.0 * np.pi * float(onp)
    cmns = np.zeros_like(cmn)
    if mf >= 1 and nf >= 1:
        cmns[:, 1:, 1:] = 0.5 * alp * (
            cmn[:, 1:, 1:] + cmn[:, :mf, 1:] + cmn[:, 1:, :nf] + cmn[:, :mf, :nf]
        )
    if mf >= 1:
        cmns[:, 1:, 0] = 0.5 * alp * (cmn[:, 1:, 0] + cmn[:, :mf, 0])
    if nf >= 1:
        cmns[:, 0, 1:] = 0.5 * alp * (cmn[:, 0, 1:] + cmn[:, 0, :nf])
    cmns[:, 0, 0] = 0.5 * alp * (cmn[:, 0, 0] + cmn[:, 0, 0])
    return cmns


def precal_tan_tables(*, nu: int, nv: int, nvper: int) -> tuple[np.ndarray, np.ndarray]:
    """VMEC ``precal.f`` ``tanu/tanv`` tables consumed by ``greenf.f``."""
    nu = max(1, int(nu))
    nv = max(1, int(nv))
    nvper = max(1, int(nvper))
    kp_count = int(nvper) if int(nv) == 1 else 1
    nuv_tan = int(2 * nu * nv * kp_count)
    tanu = np.zeros((nuv_tan,), dtype=float)
    tanv = np.zeros((nuv_tan,), dtype=float)
    alu = 2.0 * np.pi / float(nu)
    alv = 2.0 * np.pi / float(nv)
    alp_per = 2.0 * np.pi / float(nvper)
    epstan = np.finfo(float).eps
    bigno = 1.0e50
    i = 0
    for kp in range(1, kp_count + 1):
        argp = 0.5 * alp_per * float(kp - 1)
        for ku in range(1, 2 * nu + 1):
            argu = 0.5 * alu * float(ku - 1)
            near_quarter = (
                abs(argu - 0.25 * 2.0 * np.pi) < epstan
                or abs(argu - 0.75 * 2.0 * np.pi) < epstan
            )
            for kv in range(1, nv + 1):
                argv = 0.5 * alv * float(kv - 1) + argp
                tanu[i] = bigno if near_quarter else 2.0 * np.tan(argu)
                tanv[i] = bigno if abs(argv - 0.25 * 2.0 * np.pi) < epstan else 2.0 * np.tan(argv)
                i += 1
    return tanu, tanv


@dataclass(frozen=True, eq=False)
class VacuumBasis:
    """Geometry-independent NESTOR tables (host NumPy; ``precal.f``).

    Shapes: ``sin_phase/cos_phase/sinmni/cosmni`` are ``(nuv3, mnpd)``;
    ``theta/zeta/wint`` are flat ``(nuv3,)`` over the reduced ``(ntheta3,
    nzeta)`` grid; ``cmns`` is ``(mf+nf+1, mf+1, nf+1)``.
    """

    mf: int
    nf: int
    nfp: int
    nvper: int
    lasym: bool
    ntheta3: int
    nzeta: int
    nu_full: int
    nuv3: int
    nuv_full: int
    mnpd: int
    mnpd2: int
    mn0: int
    onp: float
    theta: np.ndarray
    zeta: np.ndarray
    wint: np.ndarray
    xmpot: np.ndarray
    n_raw: np.ndarray
    sin_phase: np.ndarray
    cos_phase: np.ndarray
    sinmni: np.ndarray
    cosmni: np.ndarray
    imirr: np.ndarray
    imirr_full: np.ndarray
    cmns: np.ndarray
    # -- greenf/fourp tables -------------------------------------------------
    idx_all: np.ndarray
    tanu: np.ndarray
    tanv: np.ndarray
    cosuv: np.ndarray
    sinuv: np.ndarray
    cosper: np.ndarray
    sinper: np.ndarray
    cosv_tab: np.ndarray
    sinv_tab: np.ndarray
    cosui: np.ndarray
    sinui: np.ndarray


def vacuum_basis(
    *,
    mf: int,
    nf: int,
    ntheta3: int,
    nzeta: int,
    nfp: int,
    lasym: bool,
    wint: np.ndarray,
) -> VacuumBasis:
    """Build every geometry-independent NESTOR table for one resolution.

    ``mf = mpol + 1`` and ``nf = ntor`` (VMEC ``vacmod0``); ``wint`` are the
    VMEC angular integration weights on the ``(ntheta3, nzeta)`` grid.
    Ported from the legacy ``build_vmec_mode_basis`` +
    ``ensure_vmec_nonsingular_kernel_tables``.
    """
    ntheta3 = int(ntheta3)
    nzeta = int(nzeta)
    nfp = max(1, int(nfp))
    mf = max(0, int(mf))
    nf = max(0, int(nf))
    lasym = bool(lasym)
    onp = 1.0 / float(nfp)
    nvper = 64 if nzeta == 1 else nfp

    pi2 = 2.0 * np.pi
    nu_full = int(ntheta3) if lasym else max(int(ntheta3), 2 * (int(ntheta3) - 1))
    theta = (pi2 / float(max(1, nu_full))) * np.arange(ntheta3, dtype=float)
    zeta = (pi2 / float(max(1, nzeta))) * np.arange(nzeta, dtype=float)
    th = np.broadcast_to(theta[:, None], (ntheta3, nzeta)).reshape(-1)
    ze = np.broadcast_to(zeta[None, :], (ntheta3, nzeta)).reshape(-1)

    w = np.asarray(wint, dtype=float).reshape(-1)
    if w.size != th.size:
        raise ValueError(f"wint size {w.size} != nuv3 {th.size}")

    # Mode table: n outer (-nf..nf), m inner (0..mf) — VMEC fouri.f order.
    mvals: list[int] = []
    nvals: list[int] = []
    for n in range(-nf, nf + 1):
        for m in range(0, mf + 1):
            mvals.append(m)
            nvals.append(n)
    xmpot = np.asarray(mvals, dtype=np.int64)
    n_raw = np.asarray(nvals, dtype=np.int64)
    mnpd = int(xmpot.size)
    mnpd2 = int(mnpd * (2 if lasym else 1))
    mn0 = int(np.flatnonzero((xmpot == 0) & (n_raw == 0))[0])

    phase = (xmpot[None, :] * th[:, None]) - (n_raw[None, :] * ze[:, None])
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    weight = ((pi2 * pi2) * w)[:, None]

    idx = np.arange(th.size, dtype=np.int64)
    lt = idx // max(1, nzeta)
    lz = idx % max(1, nzeta)
    if lasym or (nu_full == ntheta3):
        lt_m = (ntheta3 - lt) % max(1, ntheta3)
    else:
        lt_m_full = (nu_full - lt) % max(1, nu_full)
        lt_m = np.minimum(lt_m_full, (nu_full - lt_m_full) % max(1, nu_full))
    lz_m = (nzeta - lz) % max(1, nzeta)
    imirr = (lt_m * nzeta + lz_m).astype(np.int64)
    nuv_full = int(max(1, nu_full) * max(1, nzeta))
    idx_full = np.arange(nuv_full, dtype=np.int64)
    ku_m_full = (nu_full - idx_full // max(1, nzeta)) % max(1, nu_full)
    kv_m_full = (nzeta - idx_full % max(1, nzeta)) % max(1, nzeta)
    imirr_full = (ku_m_full * nzeta + kv_m_full).astype(np.int64)

    # -- greenf/fourp trig tables (ensure_vmec_nonsingular_kernel_tables) ---
    nu = nu_full
    nv = nzeta
    tanu, tanv = precal_tan_tables(nu=nu, nv=nv, nvper=nvper)
    alv = pi2 / float(max(1, nv))
    alvp = onp * alv
    kv = np.arange(nv, dtype=np.int64)
    cosuv = np.broadcast_to(np.cos(alvp * kv)[None, :], (nu, nv)).reshape(-1)
    sinuv = np.broadcast_to(np.sin(alvp * kv)[None, :], (nu, nv)).reshape(-1)
    alp_per = pi2 / float(max(1, nvper))
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))

    kv_idx = np.arange(nv, dtype=float)
    n_idx = np.arange(nf + 1, dtype=float)[:, None]
    cosv_tab = np.cos(alv * n_idx * kv_idx[None, :])
    sinv_tab = np.sin(alv * n_idx * kv_idx[None, :])

    alu = pi2 / float(max(1, nu))
    nu_fourp = int(nu // 2 + 1)
    ku_idx = np.arange(nu_fourp, dtype=float)
    m_idx = np.arange(mf + 1, dtype=float)[:, None]
    cosui = np.cos(alu * m_idx * ku_idx[None, :]) * (alu * alv * 2.0)
    sinui = np.sin(alu * m_idx * ku_idx[None, :]) * (alu * alv * 2.0)
    cosui[:, 0] *= 0.5
    cosui[:, -1] *= 0.5

    return VacuumBasis(
        mf=mf, nf=nf, nfp=nfp, nvper=nvper, lasym=lasym,
        ntheta3=ntheta3, nzeta=nzeta, nu_full=nu_full,
        nuv3=int(th.size), nuv_full=nuv_full, mnpd=mnpd, mnpd2=mnpd2,
        mn0=mn0, onp=onp,
        theta=th, zeta=ze, wint=w, xmpot=xmpot, n_raw=n_raw,
        sin_phase=sin_phase, cos_phase=cos_phase,
        sinmni=weight * sin_phase, cosmni=weight * cos_phase,
        imirr=imirr, imirr_full=imirr_full,
        cmns=build_cmns(mf=mf, nf=nf, onp=onp),
        idx_all=np.arange(nuv_full, dtype=np.int64),
        tanu=tanu, tanv=tanv, cosuv=cosuv, sinuv=sinuv,
        cosper=cosper, sinper=sinper,
        cosv_tab=cosv_tab, sinv_tab=sinv_tab, cosui=cosui, sinui=sinui,
    )


# ---------------------------------------------------------------------------
# Boundary bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VacuumBoundary:
    """Boundary geometry on the reduced ``(ntheta3, nzeta)`` grid.

    ``Ru/Zu``: theta derivatives; ``Rv/Zv`` (and all second derivatives):
    *geometric-phi* derivatives (``xn = n*nfp``), exactly as ``surface.f``.
    """

    R: Array
    Z: Array
    Ru: Array
    Zu: Array
    Rv: Array
    Zv: Array
    ruu: Array
    ruv: Array
    rvv: Array
    zuu: Array
    zuv: Array
    zvv: Array


jax.tree_util.register_dataclass(
    VacuumBoundary,
    data_fields=["R", "Z", "Ru", "Zu", "Rv", "Zv", "ruu", "ruv", "rvv", "zuu", "zuv", "zvv"],
    meta_fields=[],
)


# ---------------------------------------------------------------------------
# JAX assembly (greenf/fourp, analyt, fouri)
# ---------------------------------------------------------------------------


def _full_grid_from_active(b: VacuumBoundary, basis: VacuumBasis) -> tuple[Array, ...]:
    """Extend stellarator-symmetric active-grid geometry to the full grid.

    Symmetric runs sample only ``ntheta3 = ntheta1/2 + 1`` rows; the
    non-singular Green block needs all ``nu_full`` rows, rebuilt through
    stellarator symmetry ``(u, v) -> (-u, -v)``: ``R(-u,-v) = R(u,v)``,
    ``Z(-u,-v) = -Z(u,v)`` etc.  For ``lasym`` runs the active grid *is* the
    full grid.  (Second derivatives are only consumed on the primed —
    active — rows, so their mirrored rows stay zero, as in the legacy port.)
    """
    ntheta3, nv = int(basis.ntheta3), int(basis.nzeta)
    nu_full = int(basis.nu_full)
    arrays = (b.R, b.Z, b.Ru, b.Zu, b.Rv, b.Zv, b.ruu, b.ruv, b.rvv, b.zuu, b.zuv, b.zvv)
    if basis.lasym or nu_full == ntheta3:
        return tuple(jnp.asarray(a) for a in arrays)
    shape_full = (nu_full, nv)
    signs = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0)  # R, Z, Ru, Zu, Rv, Zv
    out: list[Array] = []
    kv_m = (nv - np.arange(nv, dtype=np.int64)) % max(1, nv)
    rows = [
        (ku, (nu_full - ku) % max(1, nu_full))
        for ku in range(1, max(1, ntheta3 - 1))
        if ((nu_full - ku) % max(1, nu_full)) >= ntheta3
    ]
    for i, arr in enumerate(arrays):
        a = jnp.asarray(arr)
        full = jnp.zeros(shape_full, dtype=a.dtype).at[:ntheta3, :].set(a)
        if i < 6:
            for ku, km in rows:
                full = full.at[km, :].set(signs[i] * a[ku, kv_m])
        out.append(full)
    return tuple(out)


def _fourp_tables(basis: VacuumBasis) -> dict[str, np.ndarray]:
    """``fourp.f`` mode-projection index/coefficient tables (host)."""
    nv = int(basis.nzeta)
    nf = int(basis.nf)
    mf = int(basis.mf)
    nu_fourp = int(basis.cosui.shape[1])
    iuv_grid = (
        np.arange(nu_fourp, dtype=np.int64)[:, None] * nv
        + np.arange(nv, dtype=np.int64)[None, :]
    )
    iref_grid = basis.imirr_full[iuv_grid]
    mf1 = mf + 1
    idx_p_rows: list[int] = []
    idx_m_rows: list[int] = []
    negative_positions: list[int] = []
    flat_pos = 0
    for m in range(mf + 1):
        for n in range(nf + 1):
            idx_p_rows.append(m + (n + nf) * mf1)
            if n != 0 and m != 0:
                idx_m_rows.append(m + ((-n) + nf) * mf1)
                negative_positions.append(flat_pos)
            flat_pos += 1
    return {
        "iuv_grid": iuv_grid.astype(np.int32),
        "iref_grid": np.asarray(iref_grid, dtype=np.int32),
        "cosv_modes": 0.5 * basis.onp * basis.cosv_tab[: nf + 1, :],
        "sinv_modes": 0.5 * basis.onp * basis.sinv_tab[: nf + 1, :],
        "idx_p_flat": np.asarray(idx_p_rows, dtype=np.int32),
        "idx_m_negative": np.asarray(idx_m_rows, dtype=np.int32),
        "negative_positions": np.asarray(negative_positions, dtype=np.int32),
        "sinm_sym": basis.sinui[: mf + 1, :],
        "cosm_sym": -basis.cosui[: mf + 1, :],
        "sinm_asym": basis.cosui[: mf + 1, :],
        "cosm_asym": basis.sinui[: mf + 1, :],
    }


def _nonsingular_terms(
    full: tuple[Array, ...], bexni: Array, basis: VacuumBasis, signgs: int
) -> tuple[Array, Array]:
    """``greenf.f`` + ``fourp.f``: non-singular source ``gstore`` and kernel ``grpmn``.

    Ported from the legacy ``vmec_nonsingular_terms_from_bexni_jax``
    (a ``lax.scan`` over primed points ``ip``).
    """
    R2, Z2, Ru2, Zu2, Rv2, Zv2, ruu2, ruv2, rvv2, zuu2, zuv2, zvv2 = full
    nu = int(basis.nu_full)
    nv = int(basis.nzeta)
    nuv_full = int(nu * nv)
    nuv3 = int(basis.nuv3)
    mnpd = int(basis.mnpd)
    mnpd2 = int(basis.mnpd2)
    onp = float(basis.onp)
    sign = float(int(signgs))
    nvper = int(basis.nvper)
    lasym = bool(basis.lasym)

    Rf = jnp.reshape(R2, (-1,))
    Zf = jnp.reshape(Z2, (-1,))
    R_uf = jnp.reshape(Ru2, (-1,))
    Z_uf = jnp.reshape(Zu2, (-1,))
    R_vf = jnp.reshape(Rv2, (-1,))
    Z_vf = jnp.reshape(Zv2, (-1,))
    ruuf = jnp.reshape(ruu2, (-1,))
    ruvf = jnp.reshape(ruv2, (-1,))
    rvvf = jnp.reshape(rvv2, (-1,))
    zuuf = jnp.reshape(zuu2, (-1,))
    zuvf = jnp.reshape(zuv2, (-1,))
    zvvf = jnp.reshape(zvv2, (-1,))

    snr = sign * Rf * Z_uf
    snv = sign * (R_uf * Z_vf - R_vf * Z_uf)
    snz = -sign * Rf * R_uf
    drv = -(Rf * snr + Zf * snz)
    guu_b = R_uf * R_uf + Z_uf * Z_uf
    guv_b = (R_uf * R_vf + Z_uf * Z_vf) * onp * 2.0
    gvv_b = (R_vf * R_vf + Z_vf * Z_vf + Rf * Rf) * (onp * onp)
    auu = 0.5 * (snr * ruuf + snz * zuuf)
    auv = (snr * ruvf + snv * R_uf + snz * zuvf) * onp
    avv = (snv * R_vf + 0.5 * (snr * (rvvf - Rf) + snz * zvvf)) * (onp * onp)
    rzb2 = Rf * Rf + Zf * Zf

    idx_all = jnp.asarray(basis.idx_all, dtype=jnp.int32)
    tanu = jnp.asarray(basis.tanu)
    tanv = jnp.asarray(basis.tanv)
    cosper = jnp.asarray(basis.cosper)
    sinper = jnp.asarray(basis.sinper)
    rcosuv = Rf * jnp.asarray(basis.cosuv)
    rsinuv = Rf * jnp.asarray(basis.sinuv)
    bex = jnp.reshape(jnp.asarray(bexni), (-1,))[:nuv3]

    ft = _fourp_tables(basis)
    iuv_grid = jnp.asarray(ft["iuv_grid"])
    iref_grid = jnp.asarray(ft["iref_grid"])
    cosv_modes = jnp.asarray(ft["cosv_modes"])
    sinv_modes = jnp.asarray(ft["sinv_modes"])
    idx_p_flat = jnp.asarray(ft["idx_p_flat"])
    idx_m_negative = jnp.asarray(ft["idx_m_negative"])
    negative_positions = jnp.asarray(ft["negative_positions"])
    sinm_sym = jnp.asarray(ft["sinm_sym"])
    cosm_sym = jnp.asarray(ft["cosm_sym"])
    sinm_asym = jnp.asarray(ft["sinm_asym"])
    cosm_asym = jnp.asarray(ft["cosm_asym"])

    gstore0 = jnp.zeros((nuv_full,), dtype=Rf.dtype)
    grpmn0 = jnp.zeros((mnpd2, nuv3), dtype=Rf.dtype)

    def _ip_body(carry: tuple[Array, Array], ip: Array) -> tuple[tuple[Array, Array], None]:
        gstore_acc, grpmn_acc = carry
        ip = jnp.asarray(ip, dtype=jnp.int32)
        xip = rcosuv[ip]
        yip = rsinuv[ip]
        ivoff = jnp.asarray(nuv_full, dtype=jnp.int32) - ip
        iskip = ip // jnp.asarray(max(1, nv), dtype=jnp.int32)
        iuoff = jnp.asarray(nuv_full, dtype=jnp.int32) - jnp.asarray(nv, dtype=jnp.int32) * iskip
        gsave = rzb2[ip] + rzb2 - 2.0 * Zf[ip] * Zf
        dsave = drv[ip] + Zf * snz[ip]
        delgr = jnp.zeros((nuv_full,), dtype=Rf.dtype)
        delgrp = jnp.zeros((nuv_full,), dtype=Rf.dtype)

        for kp in range(nvper):
            xper = xip * cosper[kp] - yip * sinper[kp]
            yper = yip * cosper[kp] + xip * sinper[kp]
            sxsave = (snr[ip] * xper - snv[ip] * yper) / Rf[ip]
            sysave = (snr[ip] * yper + snv[ip] * xper) / Rf[ip]
            base = gsave - 2.0 * (xper * rcosuv + yper * rsinuv)
            deriv_num = rcosuv * sxsave + rsinuv * sysave + dsave

            if kp == 0 or nv == 1:
                tidx_u = idx_all + iuoff
                ivoff_k = ivoff + jnp.asarray(2 * nu * kp if nv == 1 else 0, dtype=jnp.int32)
                tidx_v = idx_all + ivoff_k
                tanu_use = tanu[tidx_u]
                tanv_use = tanv[tidx_v]
                ga1 = tanu_use * (guu_b[ip] * tanu_use + guv_b[ip] * tanv_use) \
                    + gvv_b[ip] * tanv_use * tanv_use
                ga2 = (tanu_use * (auu[ip] * tanu_use + auv[ip] * tanv_use)
                       + avv[ip] * tanv_use * tanv_use) / ga1
                ga1s = 1.0 / jnp.sqrt(ga1)
                mask = idx_all != ip if kp == 0 else jnp.ones((nuv_full,), dtype=bool)
                safe_base = jnp.where(mask, base, 1.0)
                ftemp = 1.0 / safe_base
                htemp = jnp.sqrt(ftemp)
                deriv = ftemp * htemp * deriv_num
                delgr = delgr + jnp.where(mask, htemp - ga1s, 0.0)
                delgrp = delgrp + jnp.where(mask, deriv - ga2 * ga1s, 0.0)
            else:
                ftemp = 1.0 / base
                htemp = jnp.sqrt(ftemp)
                delgr = delgr + htemp
                delgrp = delgrp + ftemp * htemp * deriv_num

        if nv == 1 and nvper > 1:
            scale = 1.0 / float(nvper)
            delgr = delgr * scale
            delgrp = delgrp * scale

        gstore_next = gstore_acc + bex[ip] * delgr
        del_iuv = delgrp[iuv_grid]
        del_ref = delgrp[iref_grid]
        ka_grid = del_iuv - del_ref
        g1_sym = jnp.einsum("uv,fv->uf", ka_grid, cosv_modes)
        g2_sym = jnp.einsum("uv,fv->uf", ka_grid, sinv_modes)
        gcos = jnp.einsum("mu,uf->mf", sinm_sym, g1_sym)
        gsin = jnp.einsum("mu,uf->mf", cosm_sym, g2_sym)
        total_plus = jnp.reshape(gcos + gsin, (-1,))
        total_minus = jnp.reshape(gcos - gsin, (-1,))
        cols_p = jnp.full_like(idx_p_flat, ip)
        cols_m = jnp.full_like(idx_m_negative, ip)
        grpmn_next = grpmn_acc.at[(idx_p_flat, cols_p)].add(total_plus)
        grpmn_next = grpmn_next.at[(idx_m_negative, cols_m)].add(total_minus[negative_positions])

        if lasym:
            ks_grid = del_iuv + del_ref
            g1_a = jnp.einsum("uv,fv->uf", ks_grid, cosv_modes)
            g2_a = jnp.einsum("uv,fv->uf", ks_grid, sinv_modes)
            gcos_a = jnp.einsum("mu,uf->mf", sinm_asym, g1_a)
            gsin_a = jnp.einsum("mu,uf->mf", cosm_asym, g2_a)
            plus_a = jnp.reshape(gcos_a + gsin_a, (-1,))
            minus_a = jnp.reshape(gcos_a - gsin_a, (-1,))
            grpmn_next = grpmn_next.at[(mnpd + idx_p_flat, cols_p)].add(plus_a)
            grpmn_next = grpmn_next.at[(mnpd + idx_m_negative, cols_m)].add(
                minus_a[negative_positions]
            )

        return (gstore_next, grpmn_next), None

    (gstore, grpmn), _ = jax.lax.scan(
        _ip_body, (gstore0, grpmn0), jnp.arange(nuv3, dtype=jnp.int32)
    )
    return gstore, grpmn


def _analytic_terms(
    b: VacuumBoundary, bexni: Array, basis: VacuumBasis, signgs: int,
    *, include_kernel: bool = True,
) -> tuple[Array, Array | None]:
    """``analyt.f``: singular source ``bvec`` and (optionally) kernel ``grpmn``.

    Ported from the legacy ``vmec_analytic_terms_from_geometry_jax``,
    including the Fortran ``analysesum2`` swapped-argument quirk for the
    ``m != 0 and n != 0`` branch.  ``include_kernel=False`` mirrors
    ``analyt(ivacskip != 0)``, which recomputes the source only.
    """
    lasym = bool(basis.lasym)
    mf = int(basis.mf)
    nf = int(basis.nf)
    onp = float(basis.onp)
    sign = float(int(signgs))
    npts = int(basis.nuv3)
    theta = jnp.asarray(basis.theta)
    zeta = jnp.asarray(basis.zeta)
    bex = jnp.reshape(jnp.asarray(bexni), (-1,))[:npts]

    Rf = jnp.reshape(jnp.asarray(b.R), (-1,))
    Ruf = jnp.reshape(jnp.asarray(b.Ru), (-1,))
    Rvf = jnp.reshape(jnp.asarray(b.Rv), (-1,))
    Zuf = jnp.reshape(jnp.asarray(b.Zu), (-1,))
    Zvf = jnp.reshape(jnp.asarray(b.Zv), (-1,))
    ruuf = jnp.reshape(jnp.asarray(b.ruu), (-1,))
    ruvf = jnp.reshape(jnp.asarray(b.ruv), (-1,))
    rvvf = jnp.reshape(jnp.asarray(b.rvv), (-1,))
    zuuf = jnp.reshape(jnp.asarray(b.zuu), (-1,))
    zuvf = jnp.reshape(jnp.asarray(b.zuv), (-1,))
    zvvf = jnp.reshape(jnp.asarray(b.zvv), (-1,))

    guu_b = Ruf * Ruf + Zuf * Zuf
    guv_b = (Ruf * Rvf + Zuf * Zvf) * (2.0 * onp)
    gvv_b = (Rvf * Rvf + Zvf * Zvf + Rf * Rf) * (onp * onp)
    adp = guu_b + guv_b + gvv_b
    adm = guu_b - guv_b + gvv_b
    cma = gvv_b - guu_b
    sqrtc = 2.0 * jnp.sqrt(gvv_b)
    sqrta = 2.0 * jnp.sqrt(guu_b)
    sqad1 = jnp.sqrt(adp)
    sqad2 = jnp.sqrt(adm)
    tlp = (1.0 / sqad1) * jnp.log((sqad1 * sqrtc + adp + cma) / (sqad1 * sqrta - adp + cma))
    tlm = (1.0 / sqad2) * jnp.log((sqad2 * sqrtc + adm + cma) / (sqad2 * sqrta - adm + cma))
    tlp_prev = jnp.zeros_like(tlp)
    tlm_prev = jnp.zeros_like(tlm)
    tlpm = tlp + tlm

    snr = sign * Rf * Zuf
    snv = sign * (Ruf * Zvf - Rvf * Zuf)
    snz = -sign * Rf * Ruf
    auu = 0.5 * (snr * ruuf + snz * zuuf)
    auv = (snr * ruvf + snv * Ruf + snz * zuvf) * onp
    avv = (snv * Rvf + 0.5 * (snr * (rvvf - Rf) + snz * zvvf)) * (onp * onp)
    azp1u = auu + auv + avv
    azm1u = auu - auv + avv
    cma11u = avv - auu
    delt1u = adp * adm - cma * cma
    r1p = (azp1u * (delt1u - cma * cma) / adp - azm1u * adp + 2.0 * cma11u * cma) / delt1u
    r1m = (azm1u * (delt1u - cma * cma) / adm - azp1u * adm + 2.0 * cma11u * cma) / delt1u
    r0p = (-azp1u * adm * cma / adp - azm1u * cma + 2.0 * cma11u * adm) / delt1u
    r0m = (-azm1u * adp * cma / adm - azp1u * cma + 2.0 * cma11u * adp) / delt1u
    ra1p = azp1u / adp
    ra1m = azm1u / adm

    bsin = jnp.zeros((mf + 1, 2 * nf + 1), dtype=Rf.dtype)
    bcos = jnp.zeros((mf + 1, 2 * nf + 1), dtype=Rf.dtype)
    gsin = jnp.zeros((mf + 1, 2 * nf + 1, npts), dtype=Rf.dtype) if include_kernel else None
    gcos = jnp.zeros((mf + 1, 2 * nf + 1, npts), dtype=Rf.dtype) if include_kernel else None
    cmns = np.asarray(basis.cmns)  # static: skip exact-zero coefficients

    sign1 = 1.0
    fl1 = 0.0
    for ell in range(0, mf + nf + 1):
        fl = fl1
        slp = slm = slpm = None
        if include_kernel:
            slp = (r1p * fl + ra1p) * tlp + r0p * fl * tlp_prev \
                - (r1p + r0p) / sqrtc + sign1 * (r0p - r1p) / sqrta
            slm = (r1m * fl + ra1m) * tlm + r0m * fl * tlm_prev \
                - (r1m + r0m) / sqrtc + sign1 * (r0m - r1m) / sqrta
            slpm = slp + slm
        for nabs in range(0, nf + 1):
            zv = float(nabs) * zeta
            cosv = jnp.cos(zv)
            sinv = jnp.sin(zv)
            for m in range(0, mf + 1):
                cm = float(cmns[ell, m, nabs])
                if cm == 0.0:
                    continue
                mu = float(m) * theta
                sinu = jnp.sin(mu)
                cosu = jnp.cos(mu)
                col_p = int(nabs + nf)
                col_m = int((-nabs) + nf)
                if nabs == 0 or m == 0:
                    sinp = (sinu * cosv - sinv * cosu) * cm
                    bsin = bsin.at[m, col_p].add(jnp.sum(tlpm * bex * sinp))
                    if include_kernel:
                        gsin = gsin.at[m, col_p, :].add(slpm * sinp)
                    if lasym:
                        cosp = (cosu * cosv + sinv * sinu) * cm
                        bcos = bcos.at[m, col_p].add(jnp.sum(tlpm * bex * cosp))
                        if include_kernel:
                            gcos = gcos.at[m, col_p, :].add(slpm * cosp)
                else:
                    sinp0 = sinu * cosv * cm
                    temp = -cosu * sinv * cm
                    sinm = sinp0 - temp
                    sinp = sinp0 + temp
                    # analyt.f calls analysesum2 with swapped (slm, tlm,
                    # slp, tlp) order; preserved for exact parity.
                    bsin = bsin.at[m, col_p].add(jnp.sum(tlm * bex * sinp))
                    bsin = bsin.at[m, col_m].add(jnp.sum(tlp * bex * sinm))
                    if include_kernel:
                        gsin = gsin.at[m, col_p, :].add(slm * sinp)
                        gsin = gsin.at[m, col_m, :].add(slp * sinm)
                    if lasym:
                        cosp0 = cosu * cosv * cm
                        temp2 = sinu * sinv * cm
                        cosm = cosp0 - temp2
                        cosp = cosp0 + temp2
                        bcos = bcos.at[m, col_p].add(jnp.sum(tlm * bex * cosp))
                        bcos = bcos.at[m, col_m].add(jnp.sum(tlp * bex * cosm))
                        if include_kernel:
                            gcos = gcos.at[m, col_p, :].add(slm * cosp)
                            gcos = gcos.at[m, col_m, :].add(slp * cosm)

        fl1 = fl1 + 1.0
        fl2 = 2.0 * fl1 - 1.0
        sign1 = -sign1
        tlp_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlp - fl * adm * tlp_prev) / (adp * fl1)
        tlm_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlm - fl * adp * tlm_prev) / (adm * fl1)
        tlp_prev = tlp
        tlm_prev = tlm
        tlp = tlp_next
        tlm = tlm_next
        tlpm = tlp + tlm

    m_j = np.asarray(basis.xmpot, dtype=np.int64)
    col_j = np.asarray(basis.n_raw + nf, dtype=np.int64)
    out_s = bsin[m_j, col_j]
    gr_s = gsin[m_j, col_j, :] if include_kernel else None
    if lasym:
        out_c = bcos[m_j, col_j]
        bvec = jnp.concatenate([out_s, out_c], axis=0)
        grp = (
            jnp.concatenate([gr_s, gcos[m_j, col_j, :]], axis=0) if include_kernel else None
        )
        return bvec, grp
    return out_s, gr_s


def _mode_rhs_from_gsource(gsource: Array, basis: VacuumBasis) -> Array:
    """``fouri.f`` source symmetrization + mode projection (``bvec``)."""
    gsrc = jnp.reshape(jnp.asarray(gsource), (-1,))
    nuv3 = int(basis.nuv3)
    onp = float(basis.onp)
    if basis.lasym:
        src = onp * gsrc[:nuv3]
    else:
        mirror = jnp.asarray(basis.imirr_full, dtype=jnp.int32)[:nuv3]
        src = 0.5 * onp * (gsrc[:nuv3] - gsrc[mirror])
    sin = jnp.asarray(basis.sinmni)
    bsin = sin.T @ src
    skip = jnp.asarray((basis.xmpot == 0) & (basis.n_raw < 0))
    bsin = jnp.where(skip, 0.0, bsin)
    if not basis.lasym:
        return bsin
    bcos = jnp.asarray(basis.cosmni).T @ src
    return jnp.concatenate([bsin, jnp.where(skip, 0.0, bcos)], axis=0)


def _mode_matrix_from_grpmn(grpmn: Array, basis: VacuumBasis) -> Array:
    """``fouri.f`` mode-space matrix assembly (``amatrix``/``amatsav``)."""
    g = jnp.asarray(grpmn)
    sin = jnp.asarray(basis.sinmni)
    mnpd = int(basis.mnpd)
    skip_col = jnp.asarray((basis.xmpot == 0) & (basis.n_raw < 0))
    pi3 = float(4.0 * (np.pi**3))

    gsin = g[:mnpd, :]
    a11 = gsin @ sin
    a11 = jnp.where(skip_col[None, :], 0.0, a11)
    a11 = a11 + pi3 * jnp.eye(mnpd, dtype=a11.dtype)
    if not basis.lasym:
        return a11

    cos = jnp.asarray(basis.cosmni)
    gcos = g[mnpd: 2 * mnpd, :]
    a12 = jnp.where(skip_col[None, :], 0.0, gsin @ cos)
    a21 = jnp.where(skip_col[None, :], 0.0, gcos @ sin)
    a22 = jnp.where(skip_col[None, :], 0.0, gcos @ cos)
    a22 = a22 + pi3 * jnp.eye(mnpd, dtype=a22.dtype)
    mn0 = int(basis.mn0)
    a22 = a22.at[mn0, mn0].add(pi3)
    return jnp.concatenate(
        [jnp.concatenate([a11, a12], axis=1), jnp.concatenate([a21, a22], axis=1)], axis=0
    )


# ---------------------------------------------------------------------------
# Solver closures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class VacuumSolver:
    """Jitted NESTOR update closures over one :class:`VacuumBasis`.

    ``full``: complete update — returns ``(potvac, mode_matrix,
    bvec_nonsing, rhs, gsource, grpmn)``.  ``skip``: incremental update with
    the cached ``(bvec_nonsing, mode_matrix)`` — returns ``(potvac, rhs)``.
    """

    basis: VacuumBasis
    signgs: int
    full: Any
    skip: Any


def make_vacuum_solver(basis: VacuumBasis, *, signgs: int = -1) -> VacuumSolver:
    """Build the jit-compiled full/skip NESTOR updates for one basis."""

    def _full(boundary: VacuumBoundary, bexni: Array):
        full_grid = _full_grid_from_active(boundary, basis)
        gsource, grpmn_nonsing = _nonsingular_terms(full_grid, bexni, basis, signgs)
        bvec_nonsing = _mode_rhs_from_gsource(gsource, basis)
        bvec_analytic, grpmn_analytic = _analytic_terms(
            boundary, bexni, basis, signgs, include_kernel=True
        )
        rhs = bvec_nonsing + bvec_analytic
        grpmn = grpmn_nonsing + grpmn_analytic
        mode_matrix = _mode_matrix_from_grpmn(grpmn, basis)
        potvac = jnp.linalg.solve(mode_matrix, rhs)
        return potvac, mode_matrix, bvec_nonsing, rhs, gsource, grpmn

    def _skip(boundary: VacuumBoundary, bexni: Array, bvec_nonsing: Array, mode_matrix: Array):
        bvec_analytic, _ = _analytic_terms(
            boundary, bexni, basis, signgs, include_kernel=False
        )
        rhs = jnp.asarray(bvec_nonsing) + bvec_analytic
        potvac = jnp.linalg.solve(jnp.asarray(mode_matrix), rhs)
        return potvac, rhs

    return VacuumSolver(
        basis=basis, signgs=int(signgs), full=jax.jit(_full), skip=jax.jit(_skip)
    )


# ---------------------------------------------------------------------------
# Surface field from potvac (vacuum.f tail)
# ---------------------------------------------------------------------------


def vacuum_channels(
    *,
    basis: VacuumBasis,
    potvac: Array,
    bexu: Array,
    bexv: Array,
    guu: Array,
    guv: Array,
    gvv: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """``(bsqvac, bsubu, bsubv, bsupu, bsupv)`` on the boundary grid.

    ``bexu/bexv`` are the covariant external-field components with the
    geometric-phi convention (``bexv = Rv*br + R*bp + Zv*bz``); ``guu/guv/
    gvv`` the matching physical surface metric.  Equivalent to the
    ``vacuum.f`` tail (whose ``huv = 0.5*nfp*guv_b`` and ``hvv = nfp^2*
    gvv_b`` reduce to exactly this physical metric).  ``bsqvac = |B|^2/2``.
    """
    pot = jnp.reshape(jnp.asarray(potvac), (-1,))
    mnpd = int(basis.mnpd)
    potsin = pot[:mnpd]
    xm = jnp.asarray(basis.xmpot, dtype=jnp.float64)
    xn = jnp.asarray(basis.n_raw, dtype=jnp.float64) * float(basis.nfp)
    cos_phase = jnp.asarray(basis.cos_phase)
    sin_phase = jnp.asarray(basis.sin_phase)

    potu = cos_phase @ (xm * potsin)
    potv = cos_phase @ (-xn * potsin)
    if basis.lasym:
        potcos = pot[mnpd: 2 * mnpd]
        potu = potu - sin_phase @ (xm * potcos)
        potv = potv - sin_phase @ (-xn * potcos)

    shape = jnp.asarray(bexu).shape
    bsubu = jnp.asarray(bexu) + jnp.reshape(potu, shape)
    bsubv = jnp.asarray(bexv) + jnp.reshape(potv, shape)
    det = jnp.asarray(guu) * jnp.asarray(gvv) - jnp.asarray(guv) * jnp.asarray(guv)
    bsupu = (jnp.asarray(gvv) * bsubu - jnp.asarray(guv) * bsubv) / det
    bsupv = (jnp.asarray(guu) * bsubv - jnp.asarray(guv) * bsubu) / det
    bsqvac = 0.5 * (bsubu * bsupu + bsubv * bsupv)
    return bsqvac, bsubu, bsubv, bsupu, bsupv
