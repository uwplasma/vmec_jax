"""VMEC Fourier transform conventions (`fixaray` + `tomnsps`) for parity work.

VMEC does not use a plain unweighted DFT for its force/residual transforms.
Instead it uses:

- symmetry-aware theta grids (`ntheta1/2/3`) and endpoint weights,
- mode normalization scalings (`mscale`, `nscale`),
- precomputed trig tables (`cosmu/sinmu`, `cosnv/sinnv`) and their derivative
  companions (`cosmum/sinmum`, `cosnvn/sinnvn`).

This module implements the *core* pieces needed for Step-10 parity:

- `vmec_trig_tables(...)`  : build VMEC-style trig and weight tables
- `tomnsps_rzl(...)`       : real-space -> Fourier-space force transform

Scope
-----
This is intended for diagnostics/regressions against VMEC2000 `wout` outputs.
It currently implements the `lasym=False` and `lasym=True` tables, but the
transform itself is primarily exercised for the parity kernel work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .grids import AngleGrid


@dataclass(frozen=True)
class VmecTrigTables:
    """Trig/weight tables consistent with VMEC `fixaray.f`."""

    # VMEC theta grid sizes.
    ntheta1: int
    ntheta2: int
    ntheta3: int

    # Integration weight normalization used by VMEC in fixaray (dnorm).
    dnorm: float
    # Surface-integration normalization (dnorm3 in fixaray).
    dnorm3: float

    # Mode normalization scalings.
    mscale: Any  # (mmax+1,)
    nscale: Any  # (nmax+1,)
    r0scale: float

    # Theta tables. Shapes (ntheta3, mmax+1).
    cosmu: Any
    sinmu: Any
    cosmum: Any
    sinmum: Any
    cosmui: Any
    sinmui: Any
    cosmumi: Any
    sinmumi: Any
    cosmui3: Any
    cosmumi3: Any

    # Zeta tables. Shapes (nzeta, nmax+1).
    cosnv: Any
    sinnv: Any
    cosnvn: Any
    sinnvn: Any


def vmec_theta_sizes(ntheta: int, *, lasym: bool) -> tuple[int, int, int]:
    """Reproduce VMEC `read_indata.f` theta sizes."""
    ntheta = int(ntheta)
    ntheta1 = 2 * (ntheta // 2)
    ntheta2 = 1 + ntheta1 // 2  # includes u=pi point
    ntheta3 = ntheta1 if bool(lasym) else ntheta2
    return int(ntheta1), int(ntheta2), int(ntheta3)


def vmec_trig_tables(
    *,
    ntheta: int,
    nzeta: int,
    nfp: int,
    mmax: int,
    nmax: int,
    lasym: bool,
    dtype=jnp.float64,
) -> VmecTrigTables:
    """Build VMEC-style trig and weight tables.

    This follows `fixaray.f` closely (with `mscale(0)=nscale(0)=1`).
    """
    ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(ntheta, lasym=lasym)
    nzeta = int(nzeta)
    nfp = int(nfp)
    mmax = int(mmax)
    nmax = int(nmax)
    if ntheta1 <= 0 or ntheta2 <= 0 or ntheta3 <= 0:
        raise ValueError("Invalid theta sizes")
    if nzeta <= 0:
        raise ValueError("nzeta must be positive")
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    if mmax < 0 or nmax < 0:
        raise ValueError("mmax/nmax must be nonnegative")

    # dnorm normalization in fixaray (always on reduced interval [0, pi]).
    dnorm = 1.0 / (nzeta * (ntheta2 - 1))
    # dnorm3 normalization for surface averages.
    if lasym:
        dnorm3 = 1.0 / (nzeta * ntheta1)
    else:
        dnorm3 = dnorm

    # VMEC uses `osqrt2 = 1/sqrt(2)` and sets:
    #   mscale(1:) = mscale(0)/osqrt2  => sqrt(2) for m>=1 (with mscale(0)=1)
    # Likewise for `nscale`. This ensures the trig tables implement the
    # orthonormal-like scaling VMEC assumes in `fixaray.f`.
    osqrt2 = 1.0 / np.sqrt(2.0)
    mscale = np.ones((mmax + 1,), dtype=float)
    nscale = np.ones((nmax + 1,), dtype=float)
    if mmax >= 1:
        mscale[1:] = mscale[0] / osqrt2
    if nmax >= 1:
        nscale[1:] = nscale[0] / osqrt2
    r0scale = float(mscale[0] * nscale[0])

    # Theta tables use argi = 2π*(i-1)/ntheta1, for i=1..ntheta3.
    i = np.arange(ntheta3, dtype=float)
    theta = (2.0 * np.pi) * i / float(ntheta1)  # (ntheta3,)
    m = np.arange(mmax + 1, dtype=float)  # (mmax+1,)
    arg = theta[:, None] * m[None, :]  # (ntheta3, mmax+1)
    cosmu = np.cos(arg) * mscale[None, :]
    sinmu = np.sin(arg) * mscale[None, :]

    cosmum = cosmu * m[None, :]
    sinmum = -sinmu * m[None, :]

    cosmui_base = dnorm * cosmu
    sinmui_base = dnorm * sinmu
    cosmui = cosmui_base.copy()
    sinmui = sinmui_base.copy()
    cosmui3 = (dnorm3 * cosmu).copy()

    # Endpoint half-weights in theta for i==1 or i==ntheta2.
    # Note: in VMEC, `ntheta2` is the u=pi index even when `lasym=True`.
    if ntheta2 >= 1 and ntheta2 <= ntheta3:
        cosmui[0, :] *= 0.5
        cosmui[ntheta2 - 1, :] *= 0.5
        if not lasym and ntheta2 == ntheta3:
            # When `ntheta3==ntheta2` (lasym=False), VMEC reuses the half-interval
            # integration weights for full-interval integrations too.
            cosmui3 = cosmui.copy()

    cosmumi = cosmui * m[None, :]
    sinmumi = -sinmui * m[None, :]
    cosmumi3 = cosmui3 * m[None, :]

    # Zeta tables use argj = 2π*(j-1)/nzeta, for j=1..nzeta.
    j = np.arange(nzeta, dtype=float)
    zeta = (2.0 * np.pi) * j / float(nzeta)
    n = np.arange(nmax + 1, dtype=float)
    argn = zeta[:, None] * n[None, :]
    cosnv = np.cos(argn) * nscale[None, :]
    sinnv = np.sin(argn) * nscale[None, :]

    cosnvn = cosnv * (n[None, :] * float(nfp))
    sinnvn = -sinnv * (n[None, :] * float(nfp))

    # Convert to backend arrays.
    return VmecTrigTables(
        ntheta1=ntheta1,
        ntheta2=ntheta2,
        ntheta3=ntheta3,
        dnorm=float(dnorm),
        dnorm3=float(dnorm3),
        mscale=jnp.asarray(mscale, dtype=dtype),
        nscale=jnp.asarray(nscale, dtype=dtype),
        r0scale=float(r0scale),
        cosmu=jnp.asarray(cosmu, dtype=dtype),
        sinmu=jnp.asarray(sinmu, dtype=dtype),
        cosmum=jnp.asarray(cosmum, dtype=dtype),
        sinmum=jnp.asarray(sinmum, dtype=dtype),
        cosmui=jnp.asarray(cosmui, dtype=dtype),
        sinmui=jnp.asarray(sinmui, dtype=dtype),
        cosmumi=jnp.asarray(cosmumi, dtype=dtype),
        sinmumi=jnp.asarray(sinmumi, dtype=dtype),
        cosmui3=jnp.asarray(cosmui3, dtype=dtype),
        cosmumi3=jnp.asarray(cosmumi3, dtype=dtype),
        cosnv=jnp.asarray(cosnv, dtype=dtype),
        sinnv=jnp.asarray(sinnv, dtype=dtype),
        cosnvn=jnp.asarray(cosnvn, dtype=dtype),
        sinnvn=jnp.asarray(sinnvn, dtype=dtype),
    )


def vmec_angle_grid(*, ntheta: int, nzeta: int, nfp: int, lasym: bool) -> AngleGrid:
    """Build the VMEC internal (theta,zeta) grid implied by `read_indata.f`.

    - If `lasym=False`, VMEC uses `ntheta3=ntheta2` points covering `[0,π]`
      including the endpoint `π`.
    - If `lasym=True`, VMEC uses `ntheta3=ntheta1` points covering `[0,2π)`
      endpoint-free.

    In both cases, `zeta` spans one field period `[0,2π)` endpoint-free.
    """
    ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(ntheta, lasym=lasym)
    nzeta = int(nzeta)
    if nzeta <= 0:
        nzeta = 1
    if lasym:
        i = np.arange(ntheta3, dtype=float)
        theta = (2.0 * np.pi) * i / float(ntheta1)
    else:
        i = np.arange(ntheta3, dtype=float)  # 0..ntheta2-1
        theta = (2.0 * np.pi) * i / float(ntheta1)  # includes pi at i=ntheta2-1
    j = np.arange(nzeta, dtype=float)
    zeta = (2.0 * np.pi) * j / float(nzeta)
    return AngleGrid(theta=theta.astype(float), zeta=zeta.astype(float), nfp=int(nfp))


@dataclass(frozen=True)
class TomnspsRZL:
    """Fourier-space force arrays in VMEC's internal (n,m) packing (subset)."""

    # R components: frcc is always present; frss only if lthreed.
    frcc: Any  # (ns, mpol, ntor+1)
    frss: Any | None  # (ns, mpol, ntor+1)

    # Z components: fzsc always present; fzcs only if lthreed.
    fzsc: Any  # (ns, mpol, ntor+1)
    fzcs: Any | None  # (ns, mpol, ntor+1)

    # Lambda components (for later): flsc always present; flcs only if lthreed.
    flsc: Any  # (ns, mpol, ntor+1)
    flcs: Any | None  # (ns, mpol, ntor+1)

    # Asymmetric components (only meaningful when lasym=True). These correspond
    # to `tomnspa_par` outputs:
    # - R: frsc (sin(mu)cos(nv)) and frcs (cos(mu)sin(nv), if lthreed)
    # - Z: fzcc (cos(mu)cos(nv)) and fzss (sin(mu)sin(nv), if lthreed)
    # - L: flcc (cos(mu)cos(nv)) and flss (sin(mu)sin(nv), if lthreed)
    frsc: Any | None = None  # (ns, mpol, ntor+1)
    frcs: Any | None = None  # (ns, mpol, ntor+1)
    fzcc: Any | None = None  # (ns, mpol, ntor+1)
    fzss: Any | None = None  # (ns, mpol, ntor+1)
    flcc: Any | None = None  # (ns, mpol, ntor+1)
    flss: Any | None = None  # (ns, mpol, ntor+1)


def _select_mparity(a_even, a_odd, m: np.ndarray):
    mask_even = jnp.asarray((m % 2) == 0, dtype=jnp.asarray(a_even).dtype)  # (mpol,)
    return mask_even[None, :, None] * a_even + (1.0 - mask_even[None, :, None]) * a_odd


def tomnsps_rzl(
    *,
    armn_even,
    armn_odd,
    brmn_even,
    brmn_odd,
    crmn_even,
    crmn_odd,
    azmn_even,
    azmn_odd,
    bzmn_even,
    bzmn_odd,
    czmn_even,
    czmn_odd,
    blmn_even=None,
    blmn_odd=None,
    clmn_even=None,
    clmn_odd=None,
    arcon_even=None,
    arcon_odd=None,
    azcon_even=None,
    azcon_odd=None,
    mpol: int,
    ntor: int,
    nfp: int,
    lasym: bool,
    trig: VmecTrigTables,
) -> TomnspsRZL:
    """VMEC real-space -> Fourier-space force transform (core of `tomnsps`).

    Parameters
    ----------
    *_even, *_odd:
        Real-space kernels on the VMEC angular grid, with shapes:
            (ns, ntheta3, nzeta)
        representing `mparity=0` and `mparity=1` pieces.
    mpol, ntor:
        VMEC spectral truncation (`m=0..mpol-1`, `n=0..ntor`).
    lasym:
        Included for signature completeness; the transform uses `trig.ntheta2`
        and `trig.ntheta3` as built by `vmec_trig_tables`.

    Notes
    -----
    This mirrors the structure of `tomnsp_mod.f:tomnsps_par` but returns only
    the primary R/Z/L blocks needed for residual norms and parity work.
    """
    # Shapes.
    armn_even = jnp.asarray(armn_even)
    armn_odd = jnp.asarray(armn_odd)
    ns, ntheta3, nzeta = armn_even.shape
    mpol = int(mpol)
    ntor = int(ntor)
    nfp = int(nfp)
    if mpol <= 0:
        raise ValueError("mpol must be positive")
    if ntor < 0:
        raise ValueError("ntor must be nonnegative")
    if trig.ntheta3 != int(ntheta3) or int(trig.cosnv.shape[0]) != int(nzeta):
        raise ValueError("Input grid does not match trig tables")

    # Restrict theta integration to VMEC's ntheta2 (u in [0,pi]).
    nt2 = int(trig.ntheta2)
    armn_even = armn_even[:, :nt2, :]
    armn_odd = armn_odd[:, :nt2, :]
    brmn_even = jnp.asarray(brmn_even)[:, :nt2, :]
    brmn_odd = jnp.asarray(brmn_odd)[:, :nt2, :]
    crmn_even = jnp.asarray(crmn_even)[:, :nt2, :]
    crmn_odd = jnp.asarray(crmn_odd)[:, :nt2, :]
    azmn_even = jnp.asarray(azmn_even)[:, :nt2, :]
    azmn_odd = jnp.asarray(azmn_odd)[:, :nt2, :]
    bzmn_even = jnp.asarray(bzmn_even)[:, :nt2, :]
    bzmn_odd = jnp.asarray(bzmn_odd)[:, :nt2, :]
    czmn_even = jnp.asarray(czmn_even)[:, :nt2, :]
    czmn_odd = jnp.asarray(czmn_odd)[:, :nt2, :]

    if arcon_even is None:
        arcon_even = jnp.zeros_like(armn_even)
    if arcon_odd is None:
        arcon_odd = jnp.zeros_like(armn_odd)
    if azcon_even is None:
        azcon_even = jnp.zeros_like(armn_even)
    if azcon_odd is None:
        azcon_odd = jnp.zeros_like(armn_odd)

    if blmn_even is None:
        blmn_even = jnp.zeros_like(armn_even)
    if blmn_odd is None:
        blmn_odd = jnp.zeros_like(armn_odd)
    if clmn_even is None:
        clmn_even = jnp.zeros_like(armn_even)
    if clmn_odd is None:
        clmn_odd = jnp.zeros_like(armn_odd)
    blmn_even = jnp.asarray(blmn_even)[:, :nt2, :]
    blmn_odd = jnp.asarray(blmn_odd)[:, :nt2, :]
    clmn_even = jnp.asarray(clmn_even)[:, :nt2, :]
    clmn_odd = jnp.asarray(clmn_odd)[:, :nt2, :]
    arcon_even = jnp.asarray(arcon_even)[:, :nt2, :]
    arcon_odd = jnp.asarray(arcon_odd)[:, :nt2, :]
    azcon_even = jnp.asarray(azcon_even)[:, :nt2, :]
    azcon_odd = jnp.asarray(azcon_odd)[:, :nt2, :]

    # Tables for m=0..mpol-1 and n=0..ntor.
    m = np.arange(mpol, dtype=int)
    cosmui = trig.cosmui[:nt2, :mpol]  # (nt2, mpol)
    sinmui = trig.sinmui[:nt2, :mpol]
    cosmumi = trig.cosmumi[:nt2, :mpol]
    sinmumi = trig.sinmumi[:nt2, :mpol]

    cosnv = trig.cosnv[:, : (ntor + 1)]  # (nzeta, ntor+1)
    sinnv = trig.sinnv[:, : (ntor + 1)]
    cosnvn = trig.cosnvn[:, : (ntor + 1)]
    sinnvn = trig.sinnvn[:, : (ntor + 1)]

    # VMEC constraint operator multiplier: xmpq(m,1)=m*(m-1).
    xmpq1 = (jnp.asarray(m, dtype=jnp.asarray(armn_even).dtype) * (jnp.asarray(m, dtype=jnp.asarray(armn_even).dtype) - 1.0))[
        None, :, None
    ]  # (1, mpol, 1)

    # Theta integration: compute work arrays for even-parity and odd-parity pieces.
    # Each is (ns, mpol, nzeta).
    # work1 indices follow tomnsp_mod.f numbering but we compute only needed combos.
    # R:
    w1_e = (
        jnp.einsum("sik,im->smk", armn_even, cosmui)
        + jnp.einsum("sik,im->smk", brmn_even, sinmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", arcon_even, cosmui)
    )
    w1_o = (
        jnp.einsum("sik,im->smk", armn_odd, cosmui)
        + jnp.einsum("sik,im->smk", brmn_odd, sinmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", arcon_odd, cosmui)
    )
    w2_e = -jnp.einsum("sik,im->smk", crmn_even, cosmui)
    w2_o = -jnp.einsum("sik,im->smk", crmn_odd, cosmui)
    w3_e = (
        jnp.einsum("sik,im->smk", armn_even, sinmui)
        + jnp.einsum("sik,im->smk", brmn_even, cosmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", arcon_even, sinmui)
    )
    w3_o = (
        jnp.einsum("sik,im->smk", armn_odd, sinmui)
        + jnp.einsum("sik,im->smk", brmn_odd, cosmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", arcon_odd, sinmui)
    )
    w4_e = -jnp.einsum("sik,im->smk", crmn_even, sinmui)
    w4_o = -jnp.einsum("sik,im->smk", crmn_odd, sinmui)

    # Z:
    w7_e = (
        jnp.einsum("sik,im->smk", azmn_even, sinmui)
        + jnp.einsum("sik,im->smk", bzmn_even, cosmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", azcon_even, sinmui)
    )
    w7_o = (
        jnp.einsum("sik,im->smk", azmn_odd, sinmui)
        + jnp.einsum("sik,im->smk", bzmn_odd, cosmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", azcon_odd, sinmui)
    )
    w8_e = -jnp.einsum("sik,im->smk", czmn_even, sinmui)
    w8_o = -jnp.einsum("sik,im->smk", czmn_odd, sinmui)
    w5_e = (
        jnp.einsum("sik,im->smk", azmn_even, cosmui)
        + jnp.einsum("sik,im->smk", bzmn_even, sinmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", azcon_even, cosmui)
    )
    w5_o = (
        jnp.einsum("sik,im->smk", azmn_odd, cosmui)
        + jnp.einsum("sik,im->smk", bzmn_odd, sinmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", azcon_odd, cosmui)
    )
    w6_e = -jnp.einsum("sik,im->smk", czmn_even, cosmui)
    w6_o = -jnp.einsum("sik,im->smk", czmn_odd, cosmui)

    # Lambda:
    w11_e = jnp.einsum("sik,im->smk", blmn_even, cosmumi)
    w11_o = jnp.einsum("sik,im->smk", blmn_odd, cosmumi)
    w12_e = -jnp.einsum("sik,im->smk", clmn_even, sinmui)
    w12_o = -jnp.einsum("sik,im->smk", clmn_odd, sinmui)
    w9_e = jnp.einsum("sik,im->smk", blmn_even, sinmumi)
    w9_o = jnp.einsum("sik,im->smk", blmn_odd, sinmumi)
    w10_e = -jnp.einsum("sik,im->smk", clmn_even, cosmui)
    w10_o = -jnp.einsum("sik,im->smk", clmn_odd, cosmui)

    # Select parity per m (mparity = mod(m,2)).
    w1 = _select_mparity(w1_e, w1_o, m)
    w2 = _select_mparity(w2_e, w2_o, m)
    w3 = _select_mparity(w3_e, w3_o, m)
    w4 = _select_mparity(w4_e, w4_o, m)
    w5 = _select_mparity(w5_e, w5_o, m)
    w6 = _select_mparity(w6_e, w6_o, m)
    w7 = _select_mparity(w7_e, w7_o, m)
    w8 = _select_mparity(w8_e, w8_o, m)
    w9 = _select_mparity(w9_e, w9_o, m)
    w10 = _select_mparity(w10_e, w10_o, m)
    w11 = _select_mparity(w11_e, w11_o, m)
    w12 = _select_mparity(w12_e, w12_o, m)

    lthreed = bool(ntor > 0)

    # Zeta integration. Result arrays are (ns, mpol, ntor+1).
    frcc = jnp.einsum("smk,kn->smn", w1, cosnv)
    fzsc = jnp.einsum("smk,kn->smn", w7, cosnv)
    flsc = jnp.einsum("smk,kn->smn", w11, cosnv)

    if lthreed:
        frcc = frcc + jnp.einsum("smk,kn->smn", w2, sinnvn)
        fzsc = fzsc + jnp.einsum("smk,kn->smn", w8, sinnvn)
        flsc = flsc + jnp.einsum("smk,kn->smn", w12, sinnvn)

        frss = jnp.einsum("smk,kn->smn", w3, sinnv) + jnp.einsum("smk,kn->smn", w4, cosnvn)
        fzcs = jnp.einsum("smk,kn->smn", w5, sinnv) + jnp.einsum("smk,kn->smn", w6, cosnvn)
        flcs = jnp.einsum("smk,kn->smn", w9, sinnv) + jnp.einsum("smk,kn->smn", w10, cosnvn)
    else:
        frss = None
        fzcs = None
        flcs = None

    # Apply VMEC's radial evolution masks (jmin2/jlam + fixed-boundary edge).
    # For parity work we use the default vmec_params values:
    #   jmin2(m=0)=1, jmin2(m>=1)=2; jlam(m)=2.
    js_fortran = jnp.arange(ns, dtype=jnp.int32) + 1  # 1..ns
    m_fortran = jnp.arange(mpol, dtype=jnp.int32)  # 0..mpol-1
    jmin2 = jnp.where(m_fortran == 0, 1, 2)[None, :]  # (1, mpol)
    jlam = jnp.full((1, mpol), 2, dtype=jmin2.dtype)

    # Fixed-boundary convention: R/Z not evolved on the boundary surface (js=ns).
    jsmax_rz = int(ns - 1)
    mask_rz = (js_fortran[:, None] >= jmin2) & (js_fortran[:, None] <= jsmax_rz)
    mask_l = js_fortran[:, None] >= jlam

    mask_rz = mask_rz.astype(jnp.asarray(frcc).dtype)[:, :, None]
    mask_l = mask_l.astype(jnp.asarray(flsc).dtype)[:, :, None]
    frcc = frcc * mask_rz
    fzsc = fzsc * mask_rz
    if frss is not None:
        frss = frss * mask_rz
    if fzcs is not None:
        fzcs = fzcs * mask_rz
    flsc = flsc * mask_l
    if flcs is not None:
        flcs = flcs * mask_l

    return TomnspsRZL(frcc=frcc, frss=frss, fzsc=fzsc, fzcs=fzcs, flsc=flsc, flcs=flcs)


def tomnspa_rzl(
    *,
    armn_even,
    armn_odd,
    brmn_even,
    brmn_odd,
    crmn_even,
    crmn_odd,
    azmn_even,
    azmn_odd,
    bzmn_even,
    bzmn_odd,
    czmn_even,
    czmn_odd,
    blmn_even=None,
    blmn_odd=None,
    clmn_even=None,
    clmn_odd=None,
    arcon_even=None,
    arcon_odd=None,
    azcon_even=None,
    azcon_odd=None,
    mpol: int,
    ntor: int,
    nfp: int,
    lasym: bool,
    trig: VmecTrigTables,
) -> TomnspsRZL:
    """VMEC `tomnspa` asymmetric force transform (real-space -> Fourier-space).

    This mirrors `tomnsp_mod.f:tomnspa_par`. It is only used when `lasym=True`,
    but is safe to call regardless; it returns the asymmetric blocks (frsc/fzcc/...)
    and leaves the symmetric blocks as zeros.
    """
    # Shapes.
    armn_even = jnp.asarray(armn_even)
    armn_odd = jnp.asarray(armn_odd)
    ns, ntheta3, nzeta = armn_even.shape
    mpol = int(mpol)
    ntor = int(ntor)
    nfp = int(nfp)
    if mpol <= 0:
        raise ValueError("mpol must be positive")
    if ntor < 0:
        raise ValueError("ntor must be nonnegative")
    if trig.ntheta3 != int(ntheta3) or int(trig.cosnv.shape[0]) != int(nzeta):
        raise ValueError("Input grid does not match trig tables")

    # Restrict theta integration to VMEC's ntheta2 (u in [0,pi]).
    nt2 = int(trig.ntheta2)
    armn_even = armn_even[:, :nt2, :]
    armn_odd = armn_odd[:, :nt2, :]
    brmn_even = jnp.asarray(brmn_even)[:, :nt2, :]
    brmn_odd = jnp.asarray(brmn_odd)[:, :nt2, :]
    crmn_even = jnp.asarray(crmn_even)[:, :nt2, :]
    crmn_odd = jnp.asarray(crmn_odd)[:, :nt2, :]
    azmn_even = jnp.asarray(azmn_even)[:, :nt2, :]
    azmn_odd = jnp.asarray(azmn_odd)[:, :nt2, :]
    bzmn_even = jnp.asarray(bzmn_even)[:, :nt2, :]
    bzmn_odd = jnp.asarray(bzmn_odd)[:, :nt2, :]
    czmn_even = jnp.asarray(czmn_even)[:, :nt2, :]
    czmn_odd = jnp.asarray(czmn_odd)[:, :nt2, :]

    if arcon_even is None:
        arcon_even = jnp.zeros_like(armn_even)
    if arcon_odd is None:
        arcon_odd = jnp.zeros_like(armn_odd)
    if azcon_even is None:
        azcon_even = jnp.zeros_like(armn_even)
    if azcon_odd is None:
        azcon_odd = jnp.zeros_like(armn_odd)

    if blmn_even is None:
        blmn_even = jnp.zeros_like(armn_even)
    if blmn_odd is None:
        blmn_odd = jnp.zeros_like(armn_odd)
    if clmn_even is None:
        clmn_even = jnp.zeros_like(armn_even)
    if clmn_odd is None:
        clmn_odd = jnp.zeros_like(armn_odd)
    blmn_even = jnp.asarray(blmn_even)[:, :nt2, :]
    blmn_odd = jnp.asarray(blmn_odd)[:, :nt2, :]
    clmn_even = jnp.asarray(clmn_even)[:, :nt2, :]
    clmn_odd = jnp.asarray(clmn_odd)[:, :nt2, :]
    arcon_even = jnp.asarray(arcon_even)[:, :nt2, :]
    arcon_odd = jnp.asarray(arcon_odd)[:, :nt2, :]
    azcon_even = jnp.asarray(azcon_even)[:, :nt2, :]
    azcon_odd = jnp.asarray(azcon_odd)[:, :nt2, :]

    m = np.arange(mpol, dtype=int)
    cosmui = trig.cosmui[:nt2, :mpol]  # (nt2, mpol)
    sinmui = trig.sinmui[:nt2, :mpol]
    cosmumi = trig.cosmumi[:nt2, :mpol]
    sinmumi = trig.sinmumi[:nt2, :mpol]

    cosnv = trig.cosnv[:, : (ntor + 1)]  # (nzeta, ntor+1)
    sinnv = trig.sinnv[:, : (ntor + 1)]
    cosnvn = trig.cosnvn[:, : (ntor + 1)]
    sinnvn = trig.sinnvn[:, : (ntor + 1)]

    xmpq1 = (jnp.asarray(m, dtype=jnp.asarray(armn_even).dtype) * (jnp.asarray(m, dtype=jnp.asarray(armn_even).dtype) - 1.0))[
        None, :, None
    ]  # (1, mpol, 1)

    # Theta integration work arrays (indices per tomnspa_par).
    # Base (present always):
    #   work1(3): frsc cosnv
    #   work1(5): fzcc cosnv
    #   work1(9): flcc cosnv
    w3_e = (
        jnp.einsum("sik,im->smk", armn_even, sinmui)
        + jnp.einsum("sik,im->smk", brmn_even, cosmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", arcon_even, sinmui)
    )
    w3_o = (
        jnp.einsum("sik,im->smk", armn_odd, sinmui)
        + jnp.einsum("sik,im->smk", brmn_odd, cosmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", arcon_odd, sinmui)
    )
    w5_e = (
        jnp.einsum("sik,im->smk", azmn_even, cosmui)
        + jnp.einsum("sik,im->smk", bzmn_even, sinmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", azcon_even, cosmui)
    )
    w5_o = (
        jnp.einsum("sik,im->smk", azmn_odd, cosmui)
        + jnp.einsum("sik,im->smk", bzmn_odd, sinmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", azcon_odd, cosmui)
    )
    w9_e = jnp.einsum("sik,im->smk", blmn_even, sinmumi)
    w9_o = jnp.einsum("sik,im->smk", blmn_odd, sinmumi)

    # 3D-only:
    #   work1(1/2): frcs (sinnv/cosnvn)
    #   work1(4): frsc sinnvn
    #   work1(6): fzcc sinnvn
    #   work1(7/8): fzss (sinnv/cosnvn)
    #   work1(10): flcc sinnvn
    #   work1(11/12): flss (sinnv/cosnvn)
    w1_e = jnp.einsum("sik,im->smk", armn_even, cosmui) + jnp.einsum("sik,im->smk", brmn_even, sinmumi) + xmpq1 * jnp.einsum(
        "sik,im->smk", arcon_even, cosmui
    )
    w1_o = jnp.einsum("sik,im->smk", armn_odd, cosmui) + jnp.einsum("sik,im->smk", brmn_odd, sinmumi) + xmpq1 * jnp.einsum(
        "sik,im->smk", arcon_odd, cosmui
    )
    w2_e = -jnp.einsum("sik,im->smk", crmn_even, cosmui)
    w2_o = -jnp.einsum("sik,im->smk", crmn_odd, cosmui)
    w4_e = -jnp.einsum("sik,im->smk", crmn_even, sinmui)
    w4_o = -jnp.einsum("sik,im->smk", crmn_odd, sinmui)
    w6_e = -jnp.einsum("sik,im->smk", czmn_even, cosmui)
    w6_o = -jnp.einsum("sik,im->smk", czmn_odd, cosmui)
    w7_e = (
        jnp.einsum("sik,im->smk", azmn_even, sinmui)
        + jnp.einsum("sik,im->smk", bzmn_even, cosmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", azcon_even, sinmui)
    )
    w7_o = (
        jnp.einsum("sik,im->smk", azmn_odd, sinmui)
        + jnp.einsum("sik,im->smk", bzmn_odd, cosmumi)
        + xmpq1 * jnp.einsum("sik,im->smk", azcon_odd, sinmui)
    )
    w8_e = -jnp.einsum("sik,im->smk", czmn_even, sinmui)
    w8_o = -jnp.einsum("sik,im->smk", czmn_odd, sinmui)
    w10_e = -jnp.einsum("sik,im->smk", clmn_even, cosmui)
    w10_o = -jnp.einsum("sik,im->smk", clmn_odd, cosmui)
    w11_e = jnp.einsum("sik,im->smk", blmn_even, cosmumi)
    w11_o = jnp.einsum("sik,im->smk", blmn_odd, cosmumi)
    w12_e = -jnp.einsum("sik,im->smk", clmn_even, sinmui)
    w12_o = -jnp.einsum("sik,im->smk", clmn_odd, sinmui)

    w1 = _select_mparity(w1_e, w1_o, m)
    w2 = _select_mparity(w2_e, w2_o, m)
    w3 = _select_mparity(w3_e, w3_o, m)
    w4 = _select_mparity(w4_e, w4_o, m)
    w5 = _select_mparity(w5_e, w5_o, m)
    w6 = _select_mparity(w6_e, w6_o, m)
    w7 = _select_mparity(w7_e, w7_o, m)
    w8 = _select_mparity(w8_e, w8_o, m)
    w9 = _select_mparity(w9_e, w9_o, m)
    w10 = _select_mparity(w10_e, w10_o, m)
    w11 = _select_mparity(w11_e, w11_o, m)
    w12 = _select_mparity(w12_e, w12_o, m)

    lthreed = bool(ntor > 0)

    # Zeta integration.
    frsc = jnp.einsum("smk,kn->smn", w3, cosnv)
    fzcc = jnp.einsum("smk,kn->smn", w5, cosnv)
    flcc = jnp.einsum("smk,kn->smn", w9, cosnv)

    if lthreed:
        frsc = frsc + jnp.einsum("smk,kn->smn", w4, sinnvn)
        fzcc = fzcc + jnp.einsum("smk,kn->smn", w6, sinnvn)
        flcc = flcc + jnp.einsum("smk,kn->smn", w10, sinnvn)

        frcs = jnp.einsum("smk,kn->smn", w1, sinnv) + jnp.einsum("smk,kn->smn", w2, cosnvn)
        fzss = jnp.einsum("smk,kn->smn", w7, sinnv) + jnp.einsum("smk,kn->smn", w8, cosnvn)
        flss = jnp.einsum("smk,kn->smn", w11, sinnv) + jnp.einsum("smk,kn->smn", w12, cosnvn)
    else:
        frcs = None
        fzss = None
        flss = None

    # VMEC `tomnspa` note (tomnsp_mod.f): the antisymmetric transform is
    # performed on a restricted theta interval after `symforce`. For the
    # 3D+lasym lambda blocks, VMEC's conventions imply an additional √2 scaling
    # compared to the symmetric (`tomnsps`) lambda blocks. This improves Step-10
    # `fsql` parity on lasym+3D reference equilibria.
    if bool(lthreed):
        s2 = jnp.asarray(np.sqrt(2.0), dtype=jnp.asarray(flcc).dtype)
        flcc = flcc * s2
        if flss is not None:
            flss = flss * s2

    # Apply radial evolution masks (same as tomnsps): fixed-boundary edge.
    js_fortran = jnp.arange(ns, dtype=jnp.int32) + 1  # 1..ns
    m_fortran = jnp.arange(mpol, dtype=jnp.int32)  # 0..mpol-1
    jmin2 = jnp.where(m_fortran == 0, 1, 2)[None, :]  # (1, mpol)
    jlam = jnp.full((1, mpol), 2, dtype=jmin2.dtype)

    jsmax_rz = int(ns - 1)
    mask_rz = (js_fortran[:, None] >= jmin2) & (js_fortran[:, None] <= jsmax_rz)
    mask_l = js_fortran[:, None] >= jlam
    mask_rz = mask_rz.astype(jnp.asarray(frsc).dtype)[:, :, None]
    mask_l = mask_l.astype(jnp.asarray(flcc).dtype)[:, :, None]

    frsc = frsc * mask_rz
    fzcc = fzcc * mask_rz
    if frcs is not None:
        frcs = frcs * mask_rz
    if fzss is not None:
        fzss = fzss * mask_rz

    flcc = flcc * mask_l
    if flss is not None:
        flss = flss * mask_l

    # Return with symmetric blocks zeroed; caller is expected to merge with tomnsps.
    z_sym = jnp.zeros_like(frsc)
    z3 = None if not lthreed else jnp.zeros_like(frsc)
    return TomnspsRZL(
        frcc=z_sym,
        frss=z3,
        fzsc=z_sym,
        fzcs=z3,
        flsc=z_sym,
        flcs=z3,
        frsc=frsc,
        frcs=frcs,
        fzcc=fzcc,
        fzss=fzss,
        flcc=flcc,
        flss=flss,
    )
