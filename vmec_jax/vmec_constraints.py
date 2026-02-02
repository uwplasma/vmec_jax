"""VMEC constraint pipeline pieces (`alias` / `gcon`) for Step-10 parity work.

VMEC uses a "constraint force" to maintain polar regularity and enforce internal
relationships near the magnetic axis. In the VMEC2000 fixed-boundary pipeline:

1. `totzsps` produces real-space constraint-like arrays `rcon`, `zcon`
2. `funct3d` forms a scalar field

   ztemp = (rcon - rcon0)*ru0 + (zcon - zcon0)*zu0

3. `alias` computes `gcon` from `ztemp` via a de-aliased spectral operator
4. `forces` uses `gcon` to add constraint force kernels and to form the
   `arcon/azcon` arrays passed to `tomnsps`.

This module ports the *core* of VMEC's `alias.f` needed for parity diagnostics.

Notes
-----
The overall constraint strength depends on `tcon(js)`, which VMEC computes in
`bcovar.f` from preconditioner-related quantities. For parity work we provide a
`tcon_from_indata_heuristic` that matches VMEC's scaling structure but does not
yet reproduce the full `bcovar` computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .vmec_tomnsp import VmecTrigTables


@dataclass(frozen=True)
class VmecConstraintTables:
    faccon: Any  # (mpol,)  m index is 0..mpol-1 (mpol1 in VMEC)
    tcon: Any  # (ns,)


def faccon_from_signgs(*, mpol: int, signgs: int, dtype=jnp.float64) -> Any:
    """Compute VMEC's `faccon(m)` array (fixaray.f).

    VMEC (fixaray.f):
      faccon(0) = 0
      faccon(mpol1) = 0
      faccon(1:mpol1-1) = -0.25*signgs / xmpq(2:mpol1,1)^2

    where xmpq(m,1) = m*(m-1). Note the index shift by one.
    """
    mpol = int(mpol)
    if mpol <= 0:
        raise ValueError("mpol must be positive")
    signgs = int(signgs)

    m = np.arange(mpol, dtype=float)
    fac = np.zeros((mpol,), dtype=float)
    if mpol >= 3:
        denom = ((m[1:-1] + 1.0) * (m[1:-1])) ** 2  # xmpq(m+1,1)^2 for m=1..mpol-2
        fac[1:-1] = (-0.25 * float(signgs)) / denom
    return jnp.asarray(fac, dtype=dtype)


def tcon_from_indata_heuristic(*, indata, s, trig: VmecTrigTables, lasym: bool) -> Any:
    """Heuristic `tcon(js)` profile (placeholder for VMEC's bcovar-derived `tcon`).

    VMEC's actual computation depends on preconditioner quantities (`ard/azd`)
    and flux-surface norms of `ru0/zu0`. Until those kernels are ported, we use
    a conservative, VMEC-shaped scaling with a constant radial factor.
    """
    s = np.asarray(s, dtype=float)
    if s.size < 2:
        return jnp.zeros_like(jnp.asarray(s, dtype=jnp.float64))

    tcon0 = float(indata.get_float("TCON0", 0.0))
    tcon0 = min(abs(tcon0), 1.0)
    ns = int(s.size)
    hs = float(s[1] - s[0])
    r0scale = float(trig.r0scale)

    # bcovar.f:
    #   tcon_mul = tcon0*(1 + ns*(1/60 + ns/(200*120)))
    #   tcon_mul = tcon_mul / (4*r0scale^2)^2
    #   tcon(js) = (...) * tcon_mul * (32*hs)^2
    tcon_mul = tcon0 * (1.0 + ns * (1.0 / 60.0 + ns / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (r0scale**2)) ** 2)
    tcon_val = tcon_mul * (32.0 * hs) ** 2

    tcon = np.full((ns,), tcon_val, dtype=float)
    tcon[0] = 0.0
    if ns >= 3:
        tcon[-1] = 0.5 * tcon[-2]
    if lasym:
        tcon *= 0.5
    return jnp.asarray(tcon, dtype=jnp.asarray(trig.cosmu).dtype)


def tcon_from_bcovar_precondn_diag(
    *,
    indata,
    trig: VmecTrigTables,
    s,
    signgs: int,
    lasym: bool,
    bsq,
    r12,
    sqrtg,
    ru12,
    zu12,
    ru0,
    zu0,
) -> Any:
    """Compute VMEC-like `tcon(js)` using the diagonal `precondn` pieces from `bcovar.f`.

    This implements the `tcon(js)` block in `bcovar.f` for the common fixed-boundary
    path (no HBANGLE), but in a reduced form that computes only the quantities
    needed for the constraint scaling:

      tcon(js) = min(|ard(js,1)|/arnorm(js), |azd(js,1)|/aznorm(js)) * tcon_mul * (32*hs)^2

    where `ard(js,1)` and `azd(js,1)` are the (1,1) diagonal elements produced by
    `precondn` for the Z-like and R-like calls, and:

      arnorm(js) = sum(wint * ru0^2),  aznorm(js) = sum(wint * zu0^2).

    Notes
    -----
    - This does **not** attempt to reproduce the full 1D/2D preconditioner; it
      only reproduces the diagonal element used by `tcon`.
    - The expressions match `precondn.f` for the `axd(js,1)` contribution:
        ax(js,1) = sum(ptau * (xu12*ohs)^2)
        axd(js,1) = ax(js,1) + ax(js+1,1)
      with ptau = pfactor * r12^2 * bsq * wint / gsqrt.
    """
    s = np.asarray(s, dtype=float)
    ns = int(s.size)
    if ns < 2:
        return jnp.zeros((ns,), dtype=jnp.asarray(trig.cosmu).dtype)

    hs = float(s[1] - s[0])
    ohs = 1.0 / hs if hs != 0.0 else 0.0

    # VMEC precondn: pfactor = -4*r0scale^2 (v8.51+).
    pfactor = -4.0 * float(trig.r0scale) ** 2

    # Angular integration weights (wint) on the VMEC internal grid.
    w_theta = jnp.asarray(trig.cosmui3[:, 0]) / jnp.asarray(trig.mscale[0])
    wint = w_theta[:, None] * jnp.ones((int(trig.cosnv.shape[0]),), dtype=w_theta.dtype)[None, :]
    wint3 = wint[None, :, :]

    bsq = jnp.asarray(bsq)
    r12 = jnp.asarray(r12)
    sqrtg = jnp.asarray(sqrtg)
    ru12 = jnp.asarray(ru12)
    zu12 = jnp.asarray(zu12)
    ru0 = jnp.asarray(ru0)
    zu0 = jnp.asarray(zu0)

    # Avoid division by zero in ptau.
    gs = jnp.where(sqrtg != 0, sqrtg, jnp.ones_like(sqrtg))
    ptau = (pfactor * (r12 * r12) * bsq * wint3) / gs

    # ax(js,1) for each surface js (precondn.f). We compute it for js>=2 and
    # set js=1 (axis) to 0 as in VMEC.
    ax_r = jnp.sum(ptau * ((zu12 * ohs) ** 2), axis=(1, 2))  # corresponds to ard(js,1)
    ax_z = jnp.sum(ptau * ((ru12 * ohs) ** 2), axis=(1, 2))  # corresponds to azd(js,1)
    ax_r = ax_r.at[0].set(0.0)
    ax_z = ax_z.at[0].set(0.0)

    # axd(js,1) = ax(js,1) + ax(js+1,1), with ax(ns+1)=0.
    ard1 = ax_r + jnp.concatenate([ax_r[1:], jnp.zeros((1,), dtype=ax_r.dtype)], axis=0)
    azd1 = ax_z + jnp.concatenate([ax_z[1:], jnp.zeros((1,), dtype=ax_z.dtype)], axis=0)

    # Flux-surface norms of (ru0, zu0).
    arnorm = jnp.sum((ru0 * ru0) * wint3, axis=(1, 2))
    aznorm = jnp.sum((zu0 * zu0) * wint3, axis=(1, 2))
    # Avoid zero division.
    arnorm = jnp.where(arnorm != 0, arnorm, jnp.ones_like(arnorm))
    aznorm = jnp.where(aznorm != 0, aznorm, jnp.ones_like(aznorm))

    # bcovar.f scaling for tcon_mul (with clamped tcon0).
    tcon0 = float(indata.get_float("TCON0", 0.0))
    tcon0 = min(abs(tcon0), 1.0)
    ns_f = float(ns)
    tcon_mul = tcon0 * (1.0 + ns_f * (1.0 / 60.0 + ns_f / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)

    tcon = jnp.zeros((ns,), dtype=jnp.asarray(trig.cosmu).dtype)
    if ns >= 3:
        js = jnp.arange(ns, dtype=jnp.int32) + 1  # Fortran-like 1..ns
        mask = (js >= 2) & (js <= (ns - 1))
        ratio_r = jnp.abs(ard1) / arnorm
        ratio_z = jnp.abs(azd1) / aznorm
        core = jnp.minimum(ratio_r, ratio_z) * (tcon_mul * (32.0 * hs) ** 2)
        tcon = jnp.where(mask.astype(core.dtype), core, tcon)
        # tcon(ns) = 0.5*tcon(ns-1)
        tcon = tcon.at[-1].set(0.5 * tcon[-2])
    if lasym:
        tcon = 0.5 * tcon
    return tcon


def alias_gcon(
    *,
    ztemp: Any,
    trig: VmecTrigTables,
    ntor: int,
    mpol: int,
    signgs: int,
    tcon: Any,
    lasym: bool,
) -> Any:
    """Compute VMEC's `gcon` field from `ztemp` (alias.f).

    Parameters
    ----------
    ztemp:
        Shape (ns, ntheta3, nzeta), defined on the VMEC internal theta grid.
    trig:
        Trig tables from :func:`vmec_jax.vmec_tomnsp.vmec_trig_tables`.
    ntor, mpol:
        Spectral truncation. Uses n=0..ntor and m=0..mpol-1.
    signgs:
        VMEC orientation sign (+1/-1), used in `faccon`.
    tcon:
        Per-surface constraint scale factor (ns,).
    lasym:
        If True, uses the full lasym de-aliasing path in `alias.f`.
    """
    ztemp = jnp.asarray(ztemp)
    ns, ntheta3, nzeta = ztemp.shape
    if int(ntheta3) != int(trig.ntheta3):
        raise ValueError("ztemp theta size must match trig.ntheta3")
    if int(nzeta) != int(trig.cosnv.shape[0]):
        raise ValueError("ztemp nzeta must match trig.cosnv")

    mpol = int(mpol)
    ntor = int(ntor)
    faccon = faccon_from_signgs(mpol=mpol, signgs=signgs, dtype=jnp.asarray(trig.cosmu).dtype)  # (mpol,)
    tcon = jnp.asarray(tcon, dtype=jnp.asarray(trig.cosmu).dtype)  # (ns,)

    nt2 = int(trig.ntheta2)
    zhalf = ztemp[:, :nt2, :]  # (ns,nt2,nzeta)

    cosmui = trig.cosmui[:nt2, :mpol]  # (nt2,mpol)
    sinmui = trig.sinmui[:nt2, :mpol]
    cosmu = trig.cosmu[:nt2, :mpol]
    sinmu = trig.sinmu[:nt2, :mpol]

    cosnv = trig.cosnv[:, : (ntor + 1)]  # (nzeta,ntor+1)
    sinnv = trig.sinnv[:, : (ntor + 1)]

    # Theta integration on half interval.
    w1 = jnp.einsum("sik,im->smk", zhalf, cosmui)
    w2 = jnp.einsum("sik,im->smk", zhalf, sinmui)

    if not lasym:
        # Forward zeta transform (de-aliased):
        gcs = (tcon[:, None, None] * jnp.einsum("smk,kn->smn", w1, sinnv))
        gsc = (tcon[:, None, None] * jnp.einsum("smk,kn->smn", w2, cosnv))

        # Inverse zeta transform:
        work3 = jnp.einsum("smn,kn->smk", gcs, sinnv)
        work4 = jnp.einsum("smn,kn->smk", gsc, cosnv)

        cosmu_fac = cosmu * faccon[None, :]
        sinmu_fac = sinmu * faccon[None, :]
        gcon_half = jnp.einsum("smk,im->sik", work3, cosmu_fac) + jnp.einsum("smk,im->sik", work4, sinmu_fac)

        gcon = jnp.zeros((ns, ntheta3, nzeta), dtype=jnp.asarray(trig.cosmu).dtype)
        gcon = gcon.at[:, :nt2, :].set(gcon_half)
        return gcon

    # lasym=True path (alias.f: alias_par)
    ntheta1 = int(trig.ntheta1)
    if ntheta3 != ntheta1:
        raise ValueError("lasym=True requires trig.ntheta3 == trig.ntheta1")

    # Build reflected ztemp(kk,ir) restricted to i=1..ntheta2 (weights use original i).
    i0 = np.arange(nt2, dtype=int)
    ir = np.where(i0 == 0, 0, ntheta1 - i0)  # (nt2,)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta  # (nzeta,)
    zref = ztemp[:, ir, :][:, :, kk]  # (ns, nt2, nzeta)

    w3 = jnp.einsum("sik,im->smk", zref, cosmui)
    w4 = jnp.einsum("sik,im->smk", zref, sinmui)

    half = 0.5 * tcon[:, None, None]
    gcs = half * jnp.einsum("smk,kn->smn", (w1 - w3), sinnv)
    gsc = half * jnp.einsum("smk,kn->smn", (w2 - w4), cosnv)
    gss = half * jnp.einsum("smk,kn->smn", (w2 + w4), sinnv)
    gcc = half * jnp.einsum("smk,kn->smn", (w1 + w3), cosnv)

    work3 = jnp.einsum("smn,kn->smk", gcs, sinnv)
    work4 = jnp.einsum("smn,kn->smk", gsc, cosnv)
    work1 = jnp.einsum("smn,kn->smk", gcc, cosnv)
    work2 = jnp.einsum("smn,kn->smk", gss, sinnv)

    cosmu_fac = cosmu * faccon[None, :]
    sinmu_fac = sinmu * faccon[None, :]
    gcons_half = jnp.einsum("smk,im->sik", work3, cosmu_fac) + jnp.einsum("smk,im->sik", work4, sinmu_fac)
    gcona_half = jnp.einsum("smk,im->sik", work1, cosmu_fac) + jnp.einsum("smk,im->sik", work2, sinmu_fac)

    gcon = jnp.zeros((ns, ntheta3, nzeta), dtype=jnp.asarray(trig.cosmu).dtype)
    gcon = gcon.at[:, :nt2, :].set(gcons_half + gcona_half)

    # Extend theta into [pi,2pi) using VMEC's reflection + zeta reversal.
    n_second = ntheta1 - nt2
    if n_second > 0:
        i2 = np.arange(nt2, ntheta1, dtype=int)
        ir2 = (ntheta1 - i2).astype(int)  # maps into [1..nt2-1]
        gcons_ref = gcons_half[:, ir2, :][:, :, kk]
        gcona_ref = gcona_half[:, ir2, :][:, :, kk]
        gcon = gcon.at[:, nt2:, :].set(-gcons_ref + gcona_ref)

    return gcon
