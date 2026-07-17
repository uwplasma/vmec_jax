"""Post-processed wout quantities ported from VMEC2000 output routines.

Every function in this module is a direct numpy port of the corresponding
VMEC2000 Fortran, validated against golden ``wout_*.nc`` files produced by
VMEC2000 (hiddenSymmetries build, ``version_ = 9.0``):

- ``compute_currents``   : ``LIBSTELL/read_wout_mod.f90::Compute_Currents``
                           (called from ``wrout.f``); writes ``currumnc`` /
                           ``currvmnc`` (+ ``*mns`` partners when ``lasym``).
- ``spectral_width``     : ``spectrum.f`` (with ``fixaray.f`` ``xmpq``,
                           ``pexp = 4``); writes ``specw``.
- ``poloidal_flux``      : ``eqfor.f`` (``chi = twopi*chi1`` accumulation);
                           writes ``chi`` [Wb].
- ``safety_factor``      : ``wrout.f`` (``qfact = 1/iotaf``, HUGE at zeros).
- ``beta_volume_profiles``: ``eqfor.f`` half-mesh volume-averaged beta and
                           ``<1/R>`` (``beta_vol``, ``over_r``, ``betaxis``).
- ``surface_extrema``    : ``eqfor.f`` (``rmax_surf``/``rmin_surf``/
                           ``zmax_surf`` grid extrema of the boundary).
- ``field_scalars``      : ``bcovar.f`` (``rbtor0``, ``rbtor`` from
                           ``fpsi = bvco``) and ``eqfor.f`` (``b0``,
                           ``volavgB``, ``IonLarmor``).

Unit conventions follow the netCDF file exactly (the same conventions used
by ``wrout.f`` *after* its write-time conversions): pressures in Pa,
currents in A (1/mu0 applied), ``phipf``/``chipf`` include the
``twopi*signgs`` factor.  All inputs are file-convention arrays with shape
``(ns, mn)`` for Fourier tables and ``(ns,)`` for radial profiles.
"""

from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi  # [N/A^2] permeability of free space (vparams.f)


# --------------------------------------------------------------------------
# angular grid / synthesis helpers (VMEC internal grid, fixaray.f weights)
# --------------------------------------------------------------------------

def internal_angle_grid(*, ntheta: int, nzeta: int, nfp: int, lasym: bool):
    """Return VMEC's internal angle grid and integration weights.

    Mirrors ``fixaray.f``: ``ntheta1 = 2*(ntheta/2)``, symmetric runs keep the
    half poloidal range ``[0, pi]`` (``ntheta2 = ntheta1/2 + 1`` points) with
    endpoint weights halved; ``lasym`` runs keep the full range (``ntheta1``
    points) with uniform weights.  ``zeta`` spans one field period.

    Returns ``(theta, zeta, wint)`` with ``wint`` shaped ``(ntheta3, nzeta)``
    normalized so that ``sum(wint) = 1``.
    """
    ntheta1 = 2 * (int(ntheta) // 2)
    ntheta2 = ntheta1 // 2 + 1
    ntheta3 = ntheta1 if lasym else ntheta2
    theta = 2.0 * np.pi * np.arange(ntheta3) / float(ntheta1)
    zeta = 2.0 * np.pi * np.arange(int(nzeta)) / float(int(nfp) * int(nzeta))
    wint = np.full((ntheta3, int(nzeta)), 2.0 / (ntheta1 * nzeta), dtype=float)
    if lasym:
        wint[:] = 1.0 / (ntheta1 * nzeta)
    else:
        wint[0, :] = 1.0 / (ntheta1 * nzeta)
        wint[-1, :] = 1.0 / (ntheta1 * nzeta)
    return theta, zeta, wint


def fourier_synthesis(cmn, smn, xm, xn, theta, zeta):
    """Synthesize ``f[js, j, k] = sum_mn cmn*cos(m u - n v) + smn*sin(...)``.

    ``xm``/``xn`` are the file-convention mode tables (``xn`` includes the
    ``nfp`` factor).  ``smn`` may be ``None`` for stellarator-symmetric
    cosine series.  Exactly reproduces grid values written by ``wrout.f``
    because VMEC's Nyquist coefficient tables (with the endpoint half-weight
    of ``wrout.f``) are a lossless representation of the internal grid.
    """
    xm = np.asarray(xm, dtype=float)
    xn = np.asarray(xn, dtype=float)
    ang = xm[:, None, None] * theta[None, :, None] - xn[:, None, None] * zeta[None, None, :]
    out = np.tensordot(np.asarray(cmn, dtype=float), np.cos(ang), axes=([1], [0]))
    if smn is not None:
        out = out + np.tensordot(np.asarray(smn, dtype=float), np.sin(ang), axes=([1], [0]))
    return out


def half_mesh_r12(rmnc, rmns, xm, xn, theta, zeta):
    """Half-mesh cylindrical ``R`` exactly as ``bcovar.f`` builds ``r12``.

    VMEC stores odd-``m`` coefficients internally divided by ``sqrt(s)`` and
    interpolates the even and (unscaled) odd parts separately:
    ``r12(js) = 0.5*(Re(js)+Re(js-1)) + shalf(js)*0.5*(Ro(js)+Ro(js-1))``.
    The odd part at the magnetic axis is not recoverable from the wout
    coefficients (they carry a ``sqrt(s)=0`` factor); it is linearly
    extrapolated, which only affects the first half-mesh surface at the
    ~1e-6 relative level (validated against golden VMEC2000 files).
    """
    rmnc = np.asarray(rmnc, dtype=float)
    ns = rmnc.shape[0]
    hs = 1.0 / (ns - 1)
    sqrt_s_full = np.sqrt(hs * np.arange(ns))
    sqrt_s_half = np.zeros(ns)
    sqrt_s_half[1:] = np.sqrt(hs * (np.arange(1, ns) - 0.5))

    m_odd = (np.asarray(xm).astype(int) % 2) == 1
    rc_even = np.where(~m_odd[None, :], rmnc, 0.0)
    rc_odd = np.where(m_odd[None, :], rmnc, 0.0)
    rs_even = rs_odd = None
    if rmns is not None:
        rmns = np.asarray(rmns, dtype=float)
        rs_even = np.where(~m_odd[None, :], rmns, 0.0)
        rs_odd = np.where(m_odd[None, :], rmns, 0.0)

    r_even = fourier_synthesis(rc_even, rs_even, xm, xn, theta, zeta)
    r_odd_scaled = fourier_synthesis(rc_odd, rs_odd, xm, xn, theta, zeta)
    r_odd = np.zeros_like(r_odd_scaled)
    r_odd[1:] = r_odd_scaled[1:] / sqrt_s_full[1:, None, None]
    r_odd[0] = 2.0 * r_odd[1] - r_odd[2]

    r12 = np.zeros_like(r_even)
    r12[1:] = 0.5 * (r_even[1:] + r_even[:-1]) + sqrt_s_half[1:, None, None] * 0.5 * (
        r_odd[1:] + r_odd[:-1]
    )
    return r12


def nyquist_mode_table(*, mnyq: int, nnyq: int, nfp: int):
    """VMEC2000 Nyquist mode tables ``(xm_nyq, xn_nyq)`` (``fixaray.f``).

    Ordering: ``m = 0`` with ``n = 0..nnyq``, then ``m = 1..mnyq`` with
    ``n = -nnyq..nnyq``; ``xn_nyq = n*nfp``.
    """
    ms, ns_ = [], []
    for m in range(int(mnyq) + 1):
        nmin = 0 if m == 0 else -int(nnyq)
        for n in range(nmin, int(nnyq) + 1):
            ms.append(m)
            ns_.append(n * int(nfp))
    return np.asarray(ms, dtype=float), np.asarray(ns_, dtype=float)


def expand_mode_columns(table, xm_old, xn_old, xm_new, xn_new):
    """Re-index a ``(ns, mn_old)`` Fourier table onto a larger mode set.

    Columns present in both sets are copied; new columns are zero.  Used
    when the solver ran with a reduced toroidal grid (``ntor = 0`` decks
    with ``NZETA > 1``): VMEC2000 still writes the full grid-Nyquist mode
    set, whose extra toroidal harmonics vanish identically.
    """
    table = np.asarray(table, dtype=float)
    old = {(int(m), int(n)): j for j, (m, n) in enumerate(zip(np.asarray(xm_old), np.asarray(xn_old)))}
    out = np.zeros((table.shape[0], np.asarray(xm_new).size), dtype=float)
    for j, (m, n) in enumerate(zip(np.asarray(xm_new), np.asarray(xn_new))):
        src = old.get((int(m), int(n)))
        if src is not None:
            out[:, j] = table[:, src]
    return out


# --------------------------------------------------------------------------
# currents (LIBSTELL read_wout_mod.f90 :: Compute_Currents)
# --------------------------------------------------------------------------

def _current_terms(bs, bu, bv, xm_nyq, xn_nyq, *, shalf, sfull, ohs, ns, s_weighted_bu0_index):
    """Shared js-loop of ``Compute_Currents`` for one parity block.

    ``s_weighted_bu0_index`` selects ``shalf(js)`` (symmetric block) or the
    Fortran source's ``shalf(js+1)`` (asymmetric block, replicated verbatim
    for bit-parity with VMEC2000, including its apparent index slip).
    """
    mn = bs.shape[1]
    t1 = np.zeros((ns, mn))
    t2 = np.zeros((ns, mn))
    t3 = np.zeros((ns, mn))
    odd = (np.asarray(xm_nyq).astype(int) % 2) == 1
    for j in range(1, ns - 1):  # Fortran js = 2, ns-1
        sh0 = shalf[j] if s_weighted_bu0_index == 0 else shalf[j + 1]
        t1[j] = np.where(
            odd,
            0.5 * (shalf[j + 1] * bs[j + 1] + shalf[j] * bs[j]) / sfull[j],
            0.5 * (bs[j + 1] + bs[j]),
        )
        bu0 = bu[j] / sh0
        bu1 = bu[j + 1] / shalf[j + 1]
        t2[j] = np.where(
            odd,
            ohs * (bu1 - bu0) * sfull[j] + 0.25 * (bu0 + bu1) / sfull[j],
            ohs * (bu[j + 1] - bu[j]),
        )
        bv0 = bv[j] / shalf[j]
        bv1 = bv[j + 1] / shalf[j + 1]
        t3[j] = np.where(
            odd,
            ohs * (bv1 - bv0) * sfull[j] + 0.25 * (bv0 + bv1) / sfull[j],
            ohs * (bv[j + 1] - bv[j]),
        )
    return t1, t2, t3


def _current_endpoints(curru, currv, xm_nyq, ns):
    """Axis/edge extrapolation from ``Compute_Currents`` (in place)."""
    low = np.asarray(xm_nyq) <= 1
    curru[0] = np.where(low, 2.0 * curru[1] - curru[2], 0.0)
    currv[0] = np.where(low, 2.0 * currv[1] - currv[2], 0.0)
    curru[ns - 1] = 2.0 * curru[ns - 2] - curru[ns - 3]
    currv[ns - 1] = 2.0 * currv[ns - 2] - currv[ns - 3]


def compute_currents(*, bsubsmns, bsubumnc, bsubvmnc, xm_nyq, xn_nyq,
                     bsubsmnc=None, bsubumns=None, bsubvmns=None, lasym=False):
    """Current-density harmonics ``currXmn = sqrt(g)*J^X`` (X = u, v) [A].

    Port of ``read_wout_mod.f90::Compute_Currents`` (called by ``wrout.f``).
    Inputs are the file-convention half-mesh ``bsub*`` tables shaped
    ``(ns, mnmax_nyq)``; outputs are full-mesh, already divided by ``mu0``.
    Returns ``(currumnc, currvmnc, currumns, currvmns)`` with the sine
    partners ``None`` unless ``lasym``.
    """
    bs = np.asarray(bsubsmns, dtype=float)
    bu = np.asarray(bsubumnc, dtype=float)
    bv = np.asarray(bsubvmnc, dtype=float)
    ns = bs.shape[0]
    ohs = float(ns - 1)
    hs = 1.0 / ohs
    shalf = np.zeros(ns)
    sfull = np.zeros(ns)
    shalf[1:] = np.sqrt(hs * (np.arange(2, ns + 1) - 1.5))
    sfull[1:] = np.sqrt(hs * (np.arange(2, ns + 1) - 1.0))
    xn = np.asarray(xn_nyq, dtype=float)
    xm = np.asarray(xm_nyq, dtype=float)

    t1, t2, t3 = _current_terms(bs, bu, bv, xm_nyq, xn_nyq, shalf=shalf,
                                sfull=sfull, ohs=ohs, ns=ns, s_weighted_bu0_index=0)
    currumnc = -xn[None, :] * t1 - t3
    currvmnc = -xm[None, :] * t1 + t2
    _current_endpoints(currumnc, currvmnc, xm_nyq, ns)
    currumnc /= MU0
    currvmnc /= MU0
    if not lasym:
        return currumnc, currvmnc, None, None

    bsc = np.asarray(bsubsmnc, dtype=float)
    bus = np.asarray(bsubumns, dtype=float)
    bvs = np.asarray(bsubvmns, dtype=float)
    t1, t2, t3 = _current_terms(bsc, bus, bvs, xm_nyq, xn_nyq, shalf=shalf,
                                sfull=sfull, ohs=ohs, ns=ns, s_weighted_bu0_index=1)
    currumns = xn[None, :] * t1 - t3
    currvmns = xm[None, :] * t1 + t2
    _current_endpoints(currumns, currvmns, xm_nyq, ns)
    return currumnc, currvmnc, currumns / MU0, currvmns / MU0


# --------------------------------------------------------------------------
# spectral width (spectrum.f)
# --------------------------------------------------------------------------

def spectral_width(*, rmnc, zmns, xm, xn, rmns=None, zmnc=None, pexp=4):
    """Spectral width ``<M>`` per surface (``spectrum.f``; ``specw`` in wout).

    ``specw = sum(t1*m**(pexp+1)) / sum(t1*m**pexp)`` with ``t1`` the summed
    squares of the *separable-basis* (cc/ss) coefficients.  In the combined
    helical basis of the wout tables this is ``rmnc**2 + zmns**2`` (plus the
    asymmetric partners) with an extra weight 2 for ``n != 0`` columns (the
    helical-to-separable change of basis doubles the sum of squares for
    paired ``+/-n`` modes).  ``specw(axis) = 1``.
    """
    rmnc = np.asarray(rmnc, dtype=float)
    zmns = np.asarray(zmns, dtype=float)
    t1 = rmnc**2 + zmns**2
    if rmns is not None and zmnc is not None:
        t1 = t1 + np.asarray(rmns, dtype=float) ** 2 + np.asarray(zmnc, dtype=float) ** 2
    xm = np.asarray(xm, dtype=float)
    w = np.where(np.asarray(xn, dtype=float) != 0.0, 2.0, 1.0)
    num = np.sum(t1 * (w * xm ** (pexp + 1))[None, :], axis=1)
    den = np.sum(t1 * (w * xm**pexp)[None, :], axis=1)
    specw = np.where(den != 0.0, num / np.where(den != 0.0, den, 1.0), 1.0)
    specw[0] = 1.0
    return specw


# --------------------------------------------------------------------------
# 1D profiles (eqfor.f / wrout.f)
# --------------------------------------------------------------------------

def poloidal_flux(*, phips, iotas):
    """Poloidal flux ``chi`` [Wb] on the full mesh (``eqfor.f``).

    ``chi = twopi * cumsum(hs * phips * iotas)`` with ``chi(axis) = 0``;
    ``phips`` and ``iotas`` are the half-mesh wout arrays.
    """
    phips = np.asarray(phips, dtype=float)
    iotas = np.asarray(iotas, dtype=float)
    ns = phips.shape[0]
    hs = 1.0 / (ns - 1)
    chi = np.zeros(ns)
    chi[1:] = 2.0 * np.pi * np.cumsum((hs * phips * iotas)[1:])
    return chi


def safety_factor(iotaf):
    """``q_factor = 1/iotaf`` with VMEC's ``HUGE`` placeholder at zeros."""
    iotaf = np.asarray(iotaf, dtype=float)
    nz = iotaf != 0.0
    return np.where(nz, 1.0 / np.where(nz, iotaf, 1.0), np.finfo(float).max)


def mass_profile(*, pres, vp, gamma):
    """Half-mesh ``mass`` in wout units (Pa): ``mass = pres * vp**gamma``.

    VMEC evolves ``pres = mass / vp**gamma`` (``bcovar.f``); the wout file
    stores both divided by ``mu0``.  For ``gamma = 0`` (all standard decks)
    ``mass == pres`` exactly, matching VMEC2000 output.
    """
    pres = np.asarray(pres, dtype=float)
    vp = np.asarray(vp, dtype=float)
    mass = pres * np.where(vp > 0.0, vp, 1.0) ** float(gamma)
    mass[0] = 0.0
    return mass


def beta_volume_profiles(*, bmnc, gmnc, xm_nyq, xn_nyq, pres, vp, signgs,
                         rmnc, xm, xn, ntheta, nzeta, nfp, lasym,
                         bmns=None, gmns=None, rmns=None):
    """Half-mesh ``beta_vol``, ``betaxis`` and ``over_r`` (``eqfor.f``).

    Synthesizes ``|B|`` and ``sqrt(g)`` on VMEC's internal angular grid from
    the Nyquist tables (lossless), then reproduces::

        tau       = signgs * wint * gsqrt
        s2        = sum(bsq*tau)/vp - pres      (bsq = B^2/2 + mu0*pres)
        beta_vol  = pres / s2
        over_r    = sum(tau/r12) / vp
        betaxis   = 1.5*beta_vol(2) - 0.5*beta_vol(3)

    ``pres``/``vp`` are the wout half-mesh arrays (``pres`` in Pa).
    """
    theta, zeta, wint = internal_angle_grid(ntheta=ntheta, nzeta=nzeta, nfp=nfp, lasym=lasym)
    B = fourier_synthesis(bmnc, bmns, xm_nyq, xn_nyq, theta, zeta)
    G = fourier_synthesis(gmnc, gmns, xm_nyq, xn_nyq, theta, zeta)
    tau = float(signgs) * wint[None, :, :] * G

    pres_int = MU0 * np.asarray(pres, dtype=float)  # internal units (mu0*Pa)
    vp = np.asarray(vp, dtype=float)
    ns = vp.shape[0]
    vp_safe = np.where(vp != 0.0, vp, 1.0)

    bsq = 0.5 * B**2 + pres_int[:, None, None]
    s2 = np.sum(bsq * tau, axis=(1, 2)) / vp_safe - pres_int
    beta_vol = np.where(s2 != 0.0, pres_int / np.where(s2 != 0.0, s2, 1.0), 0.0)
    beta_vol[0] = 0.0
    betaxis = 1.5 * beta_vol[1] - 0.5 * beta_vol[2] if ns >= 3 else 0.0

    r12 = half_mesh_r12(rmnc, rmns, xm, xn, theta, zeta)
    over_r = np.zeros(ns)
    over_r[1:] = np.sum(tau[1:] / r12[1:], axis=(1, 2)) / vp_safe[1:]
    return beta_vol, float(betaxis), over_r


def surface_extrema(*, rmnc, zmns, xm, xn, ntheta, nzeta, nfp, lasym,
                    rmns=None, zmnc=None):
    """Boundary extrema ``(rmax_surf, rmin_surf, zmax_surf)`` (``eqfor.f``).

    VMEC evaluates the extrema on its internal grid points only (no
    continuous optimization), so grid synthesis reproduces the file values
    exactly.
    """
    theta, zeta, _ = internal_angle_grid(ntheta=ntheta, nzeta=nzeta, nfp=nfp, lasym=lasym)
    rmnc_b = np.asarray(rmnc, dtype=float)[-1:, :]
    zmns_b = np.asarray(zmns, dtype=float)[-1:, :]
    rmns_b = np.asarray(rmns, dtype=float)[-1:, :] if rmns is not None else None
    zmnc_b = np.asarray(zmnc, dtype=float)[-1:, :] if zmnc is not None else None
    R = fourier_synthesis(rmnc_b, rmns_b, xm, xn, theta, zeta)[0]
    ang = np.asarray(xm, float)[:, None, None] * theta[None, :, None] - np.asarray(
        xn, float
    )[:, None, None] * zeta[None, None, :]
    Z = np.tensordot(zmns_b, np.sin(ang), axes=([1], [0]))[0]
    if zmnc_b is not None:
        Z = Z + np.tensordot(zmnc_b, np.cos(ang), axes=([1], [0]))[0]
    return float(np.max(R)), float(np.min(R)), float(np.max(np.abs(Z)))


def force_balance(*, bsubumnc, bsubvmnc, xm_nyq, xn_nyq, phipf, chipf,
                  pres, vp, signgs):
    """Flux-surface current averages and radial force balance (``fbal.f``).

    Recomputes, in file conventions, exactly what VMEC2000 writes:

    - ``buco``/``bvco``: half-mesh ``<B_u>``/``<B_v>`` - by definition the
      ``(m,n) = (0,0)`` harmonic of the (parity-proven) ``bsub[uv]mnc``
      tables (``calc_fbal``: ``buco = SUM(bsubu*wint)``);
    - ``jcuru``/``jcurv`` [A]: full-mesh surface-averaged current densities
      ``-+(signgs*ohs)*d<B_[vu]>`` with eqfor.f end extrapolations, /mu0;
    - ``equif``: normalized radial force balance (``calc_fbal`` +
      ``eqfor.f`` normalization, end points linearly extrapolated);
    - ``ctor`` [A]: ``signgs*twopi*(1.5*buco(ns) - 0.5*buco(ns1))/mu0``
      (``bcovar.f``).

    ``phipf``/``chipf`` are the file-convention arrays (``2*pi*signgs`` x
    internal); ``pres`` in Pa; ``vp`` as stored.  Validated bit-exact
    against golden VMEC2000 wout files (symmetric and lasym).
    """
    xm = np.asarray(xm_nyq)
    xn = np.asarray(xn_nyq)
    i00 = int(np.where((xm == 0) & (xn == 0))[0][0])
    buco = np.asarray(bsubumnc, dtype=float)[:, i00].copy()
    bvco = np.asarray(bsubvmnc, dtype=float)[:, i00].copy()
    buco[0] = 0.0
    bvco[0] = 0.0

    ns = buco.shape[0]
    ohs = float(ns - 1)
    sg = float(signgs)
    phipf_i = np.asarray(phipf, dtype=float) / (2.0 * np.pi * sg)
    chipf_i = np.asarray(chipf, dtype=float) / (2.0 * np.pi * sg)
    pres_i = MU0 * np.asarray(pres, dtype=float)
    vp = np.asarray(vp, dtype=float)

    jcuru = np.zeros(ns)
    jcurv = np.zeros(ns)
    equif = np.zeros(ns)
    js = np.arange(1, ns - 1)
    jcurv[js] = sg * ohs * (buco[js + 1] - buco[js])
    jcuru[js] = -sg * ohs * (bvco[js + 1] - bvco[js])
    vpphi = (vp[js + 1] + vp[js]) / 2.0
    presgrad = (pres_i[js + 1] - pres_i[js]) * ohs
    equif[js] = (-phipf_i[js] * jcuru[js] + chipf_i[js] * jcurv[js]) / vpphi + presgrad
    den = (np.abs(jcurv[js] * chipf_i[js]) + np.abs(jcuru[js] * phipf_i[js])
           + np.abs(presgrad * vpphi))
    equif[js] = equif[js] * vpphi / np.where(den != 0.0, den, 1.0)
    equif[0] = 2.0 * equif[1] - equif[2]
    equif[ns - 1] = 2.0 * equif[ns - 2] - equif[ns - 3]
    jcuru[0] = 2.0 * jcuru[1] - jcuru[2]
    jcurv[0] = 2.0 * jcurv[1] - jcurv[2]
    jcuru[ns - 1] = 2.0 * jcuru[ns - 2] - jcuru[ns - 3]
    jcurv[ns - 1] = 2.0 * jcurv[ns - 2] - jcurv[ns - 3]

    ctor = sg * 2.0 * np.pi * (1.5 * buco[-1] - 0.5 * buco[-2]) / MU0
    return buco, bvco, jcuru / MU0, jcurv / MU0, equif, float(ctor)


def eqfor_beta_scalars(*, pres, vp, bsq, r12, bsupv, sqrtg, wint, signgs):
    """``eqfor.f`` beta scalars ``(betapol, betator, betatotal)``.

    Inputs are the internal half-mesh field state: ``pres`` in mu0*Pa,
    ``bsq`` the total pressure ``|B|^2/2 + pres``, ``r12/bsupv/sqrtg`` the
    half-mesh grid arrays and ``wint`` the angular weights.  Reproduces::

        sump    = vnorm * sum(vp*pres)
        sumbtot = 2*(vnorm*sum(bsq*tau) - sump)
        sumbtor = vnorm*sum(tau*(r12*bsupv)**2)
        beta*   = 2*sump / sumb*
    """
    pres = np.asarray(pres, dtype=float)
    ns = int(pres.shape[0])
    if ns < 3:
        return 0.0, 0.0, 0.0
    hs = 1.0 / float(ns - 1)
    vnorm = (2.0 * np.pi) ** 2 * hs
    tau = float(signgs) * np.asarray(wint, dtype=float)[None, :, :] * np.asarray(sqrtg, dtype=float)
    vp = np.asarray(vp, dtype=float)
    bsq = np.asarray(bsq, dtype=float)
    sump = vnorm * float(np.sum(vp[1:] * pres[1:]))
    sumbtot = 2.0 * (vnorm * float(np.sum(bsq[1:] * tau[1:])) - sump)
    sumbtor = vnorm * float(np.sum(tau[1:] * (np.asarray(r12, dtype=float)[1:] * np.asarray(bsupv, dtype=float)[1:]) ** 2))
    sumbpol = sumbtot - sumbtor

    def _div(num, den):
        return num / den if den != 0.0 else 0.0

    return _div(2.0 * sump, sumbpol), _div(2.0 * sump, sumbtor), _div(2.0 * sump, sumbtot)


def aspect_ratio_scalars(*, r_boundary, zu_boundary, wint):
    """``aspectratio.f`` scalars ``(Aminor_p, Rmajor_p, aspect, volume_p)``.

    ``r_boundary``/``zu_boundary`` are the boundary-surface ``R`` and
    ``dZ/dtheta`` on the internal angular grid, ``wint`` the angular weights.
    """
    rb = np.asarray(r_boundary, dtype=float)
    zub = np.asarray(zu_boundary, dtype=float)
    wint = np.asarray(wint, dtype=float)
    t1 = rb * zub * wint
    volume_p = float(2.0 * np.pi * np.pi * abs(np.sum(rb * t1)))
    cross_area_p = float(2.0 * np.pi * abs(np.sum(t1)))
    if cross_area_p == 0.0:
        return 0.0, 0.0, 0.0, volume_p
    aminor = float(np.sqrt(cross_area_p / np.pi))
    rmajor = float(volume_p / (2.0 * np.pi * cross_area_p))
    return aminor, rmajor, (rmajor / aminor if aminor != 0.0 else 0.0), volume_p


def full_mesh_from_half(profile):
    """Full-mesh companion of a half-mesh profile (eqfor.f presf convention).

    ``f(1) = 1.5 f(2) - 0.5 f(3)``, interior averages, edge extrapolation.
    """
    p = np.asarray(profile, dtype=float)
    ns = int(p.shape[0])
    if ns < 2:
        return p.copy()
    out = np.zeros_like(p)
    out[0] = 1.5 * p[1] - 0.5 * p[2] if ns >= 3 else p[1]
    if ns > 2:
        out[1:-1] = 0.5 * (p[1:-1] + p[2:])
    out[-1] = 1.5 * p[-1] - 0.5 * p[-2]
    return out


def iotaf_from_iotas(iotas):
    """``add_fluxes.f90`` full-mesh iota from the half-mesh profile (no RFP)."""
    return full_mesh_from_half(iotas)


def chipf_from_chips(chips):
    """``add_fluxes.f90`` full-mesh ``chipf`` from half-mesh ``chips``."""
    return full_mesh_from_half(chips)


def toroidal_flux_profile(*, phipf_out, s):
    """``phi`` [Wb] on the full mesh: rectangle-rule integral of ``phipf``.

    ``phi[i] = sum_{j<=i} phipf_out[j]*(s[j]-s[j-1])`` with ``phi[0] = 0``
    (``phipf_out`` in file convention, i.e. including ``2*pi*signgs``).
    """
    y = np.asarray(phipf_out, dtype=float)
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(y)
    if y.shape[0] >= 2:
        out[1:] = np.cumsum(y[1:] * (s[1:] - s[:-1]))
    return out


def _lambda_half_mesh_weights(s):
    """``sm/sp`` weights of the wrout.f half-mesh lambda conversion (1-based)."""
    s_arr = np.asarray(s, dtype=float).reshape(-1)
    ns = int(s_arr.shape[0])
    sm_f = np.zeros((ns + 1,), dtype=float)
    sp_f = np.zeros((ns + 1,), dtype=float)
    if ns < 2:
        return sm_f, sp_f
    hs = float(s_arr[1] - s_arr[0])
    sqrts_f = np.zeros((ns + 1,), dtype=float)
    shalf_f = np.zeros((ns + 1,), dtype=float)
    for i in range(1, ns + 1):
        sqrts_f[i] = np.sqrt(max(hs * float(i - 1), 0.0))
        shalf_f[i] = np.sqrt(hs * abs(float(i) - 1.5))
    sqrts_f[ns] = 1.0
    for i in range(2, ns + 1):
        sm_f[i] = shalf_f[i] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        if i < ns:
            sp_f[i] = shalf_f[i + 1] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        else:
            sp_f[i] = 1.0 / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
    sm_f[1] = 0.0
    sp_f[0] = 0.0
    sp_f[1] = sm_f[2] if ns >= 2 else 0.0
    return sm_f, sp_f


def lambda_wout_from_full_mesh(*, lam_full, m_modes, s, phipf_internal, lamscale):
    """Internal full-mesh lambda coefficients -> the wout half-mesh ``lmns``.

    ``wrout.f``: the internal lambda (evolved as ``lamscale * lambda`` with
    the flux normalization ``phip``) is rescaled by ``lamscale/phipf`` and
    interpolated to the half mesh with parity-dependent ``sm/sp`` weights;
    the axis row is zeroed.
    """
    lam_full = np.asarray(lam_full, dtype=float)
    s_arr = np.asarray(s, dtype=float).reshape(-1)
    ns = int(s_arr.shape[0])
    m_modes = np.asarray(m_modes, dtype=int)
    phipf_internal = np.asarray(phipf_internal, dtype=float).reshape(-1)
    if float(lamscale) == 0.0:
        return np.zeros_like(lam_full)
    phipf_safe = np.where(phipf_internal == 0.0, 1.0, phipf_internal)
    lam_ext = lam_full * (float(lamscale) / phipf_safe[:, None])
    if ns < 2:
        return np.zeros_like(lam_ext)

    sm_f, sp_f = _lambda_half_mesh_weights(s_arr)
    lam_half = lam_ext.copy()
    mask_m_le1 = m_modes <= 1
    if np.any(mask_m_le1):
        lam_half[0, mask_m_le1] = lam_half[1, mask_m_le1]
    even_mask = (m_modes % 2) == 0
    odd_mask = ~even_mask
    for js_idx in range(ns - 1, 0, -1):
        if np.any(even_mask):
            lam_half[js_idx, even_mask] = 0.5 * (lam_half[js_idx, even_mask] + lam_half[js_idx - 1, even_mask])
        if np.any(odd_mask):
            lam_half[js_idx, odd_mask] = 0.5 * (
                sm_f[js_idx + 1] * lam_half[js_idx, odd_mask]
                + sp_f[js_idx] * lam_half[js_idx - 1, odd_mask]
            )
    lam_half[0, :] = 0.0
    return lam_half


def field_scalars(*, bvco, raxis_cc, wb, volume_p):
    """Edge/axis field scalars from ``bcovar.f`` / ``eqfor.f``.

    Returns ``(rbtor0, rbtor, b0, volavgB, ion_larmor)``:

    - ``rbtor0 = 1.5*bvco(2) - 0.5*bvco(3)``; ``rbtor`` same at the edge
      (``bcovar.f`` with ``fpsi = bvco``) [T*m].
    - ``b0 = rbtor0 / R_axis(v=0)`` with ``R_axis(v=0) = sum(raxis_cc)``
      (``eqfor.f`` ``b0 = fpsi0/r00``) [T].
    - ``volavgB = sqrt(2*wb*(2*pi)**2 / volume_p)`` [T]; identical to
      ``eqfor.f`` ``sqrt(|sumbtot/volume_p|)`` because
      ``wb = int(B^2/2 dV)/(2*pi)**2``.
    - ``IonLarmor = 0.0032/volavgB`` [m * sqrt(keV)].
    """
    bvco = np.asarray(bvco, dtype=float)
    rbtor0 = float(1.5 * bvco[1] - 0.5 * bvco[2])
    rbtor = float(1.5 * bvco[-1] - 0.5 * bvco[-2])
    r00 = float(np.sum(np.asarray(raxis_cc, dtype=float)))
    b0 = rbtor0 / r00 if r00 != 0.0 else 0.0
    volavgB = float(np.sqrt(np.abs(2.0 * float(wb) * (2.0 * np.pi) ** 2 / float(volume_p)))) if volume_p else 0.0
    ion_larmor = 0.0032 / volavgB if volavgB else 0.0
    return rbtor0, rbtor, b0, volavgB, ion_larmor
