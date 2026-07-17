"""Fourier-mode, angle-grid, and trigonometric-table bookkeeping for VMEC.

VMEC2000 counterparts
---------------------
- ``Sources/Initialization_Cleanup/fixaray.f``      — mode tables ``xm/xn``, the
  ``mscale/nscale`` normalization, and the trig tables ``cosmu/sinmu/cosnv/sinnv``
  (with derivative and integration-weighted companions).
- ``Sources/Input_Output/read_indata.f``            — the internal theta grid sizes
  ``ntheta1/ntheta2/ntheta3``.
- ``Sources/Initialization_Cleanup/profil3d.f``     — angular integration weights
  (``wint``).

Physics/numerics conventions (parity-critical)
----------------------------------------------
VMEC does *not* use a plain unweighted DFT.  Its transforms rely on:

1. **Reduced poloidal grid.**  For stellarator-symmetric runs (``lasym=False``)
   all angular integrals are evaluated on ``theta in [0, pi]`` (``ntheta2``
   points, endpoints included) with half-weights at ``theta=0`` and
   ``theta=pi``.  For ``lasym=True`` the full ``[0, 2*pi)`` grid (``ntheta1``
   points, endpoint-free) is stored, but the force transforms still integrate
   on the reduced interval after a symmetric/antisymmetric decomposition
   (``symforce.f``).

2. **Mode normalization.**  ``mscale(0)=nscale(0)=1`` and
   ``mscale(m>=1)=nscale(n>=1)=sqrt(2)``.  The trig tables carry these factors
   so that the two-stage projection (``tomnsps``) is the exact inverse of the
   two-stage synthesis (``totzsps``) on band-limited data (stellarator-symmetric
   grid), and VMEC's internal spectral coefficients are the physical (wout)
   coefficients divided by ``mscale(m)*nscale(|n|)``.

3. **Integration normalization** ``dnorm``: ``1/(nzeta*(ntheta2-1))`` for
   symmetric runs and ``1/(nzeta*ntheta3)`` for ``lasym=True`` (SPH012314).

All tables are built once per resolution with NumPy float64 (they are static,
trace-time constants for the jitted transforms in
:mod:`vmec_jax.core.transforms`; building them with NumPy avoids eager device
transfers and is exact regardless of the JAX x64 flag).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Resolution",
    "ModeTable",
    "TrigTables",
    "mode_table",
    "trig_tables",
]


@dataclass(frozen=True)
class Resolution:
    """Hashable static resolution/configuration of a VMEC spectral problem.

    VMEC2000: the INDATA resolution block (``mpol, ntor, ntheta, nzeta, nfp,
    lasym, ns_array``) plus the derived grid sizes from ``read_indata.f``.

    Parameters
    ----------
    mpol:
        Number of poloidal modes; ``m = 0 .. mpol-1``.
    ntor:
        Largest toroidal mode number; ``n = -ntor .. ntor`` (``n >= 0`` for m=0).
    ntheta:
        Requested number of poloidal grid points (VMEC rounds it; see
        ``ntheta1/ntheta2/ntheta3``).
    nzeta:
        Number of toroidal grid points per field period.
    nfp:
        Number of toroidal field periods.
    lasym:
        True for non-stellarator-symmetric equilibria.
    ns:
        Number of radial (flux) surfaces.
    """

    mpol: int
    ntor: int
    ntheta: int
    nzeta: int
    nfp: int
    lasym: bool
    ns: int

    def __post_init__(self) -> None:
        if self.mpol < 1:
            raise ValueError("mpol must be >= 1")
        if self.ntor < 0:
            raise ValueError("ntor must be >= 0")
        if self.ntheta < 2:
            raise ValueError("ntheta must be >= 2")
        if self.nzeta < 1:
            raise ValueError("nzeta must be >= 1")
        if self.nfp < 1:
            raise ValueError("nfp must be >= 1")
        if self.ns < 1:
            raise ValueError("ns must be >= 1")

    @property
    def ntheta1(self) -> int:
        """Full poloidal grid size, forced even (VMEC ``read_indata.f``)."""
        return 2 * (self.ntheta // 2)

    @property
    def ntheta2(self) -> int:
        """Reduced poloidal grid size covering ``[0, pi]`` incl. the endpoint."""
        return 1 + self.ntheta1 // 2

    @property
    def ntheta3(self) -> int:
        """Stored poloidal grid size: ``ntheta1`` if ``lasym`` else ``ntheta2``."""
        return self.ntheta1 if self.lasym else self.ntheta2

    @property
    def nznt(self) -> int:
        """Total angular grid size ``nzeta * ntheta3`` (VMEC ``nznt``)."""
        return self.nzeta * self.ntheta3

    @property
    def mnmax(self) -> int:
        """Number of (m, n) modes in VMEC ordering (``fixaray.f`` ``mnmax``)."""
        return (self.ntor + 1) + (self.mpol - 1) * (2 * self.ntor + 1)

    @property
    def lthreed(self) -> bool:
        """True when the configuration is three-dimensional (``ntor > 0``)."""
        return self.ntor > 0


@dataclass(frozen=True, eq=False)
class ModeTable:
    """VMEC (m, n) mode ordering plus the m-masks used downstream.

    VMEC2000: the ``xm/xn`` arrays built in ``fixaray.f`` (``nmin0`` logic):
    ``m = 0`` carries only ``n = 0 .. ntor``; ``m >= 1`` carries
    ``n = -ntor .. ntor``.  ``n`` is stored *before* multiplication by ``nfp``
    (VMEC's ``xn = n * nfp``).
    """

    m: np.ndarray  # (mnmax,) int
    n: np.ndarray  # (mnmax,) int, not multiplied by nfp

    @property
    def mnmax(self) -> int:
        """Number of (m, n) modes."""
        return int(self.m.size)

    @property
    def m_is_even(self) -> np.ndarray:
        """Even poloidal parity mask (VMEC ``mparity = 0``)."""
        return (self.m % 2) == 0

    @property
    def m_is_odd(self) -> np.ndarray:
        """Odd poloidal parity mask (VMEC ``mparity = 1``; carries sqrt(s))."""
        return (self.m % 2) == 1

    @property
    def m_is_zero(self) -> np.ndarray:
        """Mask for ``m == 0`` modes (axis/lambda special handling)."""
        return self.m == 0

    @property
    def m_is_one(self) -> np.ndarray:
        """Mask for ``m == 1`` modes (the ``residue.f90`` m=1 constraint)."""
        return self.m == 1


def mode_table(mpol: int, ntor: int) -> ModeTable:
    """Build the VMEC (m, n) mode table (VMEC2000: ``fixaray.f``, xm/xn).

    Ordering: ``m = 0`` has ``n = 0..ntor``; ``m = 1..mpol-1`` has
    ``n = -ntor..ntor``.  This matches ``vmec_jax.modes.vmec_mode_table`` and
    the ``xm/xn`` arrays written to ``wout`` files.
    """
    mpol = int(mpol)
    ntor = int(ntor)
    if mpol < 1:
        raise ValueError("mpol must be >= 1")
    if ntor < 0:
        raise ValueError("ntor must be >= 0")
    m_list: list[int] = []
    n_list: list[int] = []
    for m in range(mpol):
        n_min = 0 if m == 0 else -ntor
        for n in range(n_min, ntor + 1):
            m_list.append(m)
            n_list.append(n)
    return ModeTable(m=np.asarray(m_list, dtype=int), n=np.asarray(n_list, dtype=int))


@dataclass(frozen=True, eq=False)
class TrigTables:
    """Trigonometric/weight tables (VMEC2000: ``fixaray.f``).

    Theta tables have shape ``(ntheta3, mpol)``, zeta tables ``(nzeta, ntor+1)``.
    Suffix conventions follow ``fixaray.f``:

    - ``cosmu/sinmu``   : ``cos(m*theta)*mscale(m)`` / ``sin(m*theta)*mscale(m)``.
    - ``cosmum/sinmum`` : theta-derivative companions
      (``cosmum = m*cosmu``, ``sinmum = -m*sinmu``).
    - ``cosmui/sinmui`` : integration-weighted (``dnorm`` and the theta-endpoint
      half-weights on ``cosmui``; ``sinmu`` vanishes at 0 and pi so ``sinmui``
      needs no endpoint correction).
    - ``cosmumi/sinmumi``: integration-weighted derivative tables.
    - ``cosmui3/cosmumi3``: full-surface-average variants (``dnorm3``); for
      ``lasym=False`` these coincide with ``cosmui/cosmumi``.
    - ``cosnv/sinnv``   : ``cos(n*zeta)*nscale(n)`` / ``sin(n*zeta)*nscale(n)``.
    - ``cosnvn/sinnvn`` : zeta-derivative companions w.r.t. the *physical*
      toroidal angle (``cosnvn = n*nfp*cosnv``, ``sinnvn = -n*nfp*sinnv``).
    - ``wint``          : angular integration weights on the internal grid,
      shape ``(ntheta3, nzeta)`` (VMEC's ``wint`` from ``profil3d.f``, here
      without the radial replication).
    """

    ntheta1: int
    ntheta2: int
    ntheta3: int
    nfp: int
    lasym: bool

    dnorm: float
    dnorm3: float

    mscale: np.ndarray  # (mpol,)
    nscale: np.ndarray  # (ntor+1,)

    cosmu: np.ndarray
    sinmu: np.ndarray
    cosmum: np.ndarray
    sinmum: np.ndarray
    cosmui: np.ndarray
    sinmui: np.ndarray
    cosmumi: np.ndarray
    sinmumi: np.ndarray
    cosmui3: np.ndarray
    cosmumi3: np.ndarray

    cosnv: np.ndarray
    sinnv: np.ndarray
    cosnvn: np.ndarray
    sinnvn: np.ndarray

    wint: np.ndarray  # (ntheta3, nzeta)


def trig_tables(res: Resolution) -> TrigTables:
    """Build the VMEC trig/weight tables for a resolution.

    VMEC2000: ``Sources/Initialization_Cleanup/fixaray.f``.  The numerical
    conventions (``mscale/nscale`` = sqrt(2) for nonzero mode numbers,
    ``dnorm`` reduced-grid normalization, exact ``+/-mscale`` values at the
    ``theta = pi`` endpoint, endpoint half-weights on ``cosmui``) are ported
    verbatim from the parity-proven ``vmec_jax.kernels.tomnsp.vmec_trig_tables``.

    Poloidal columns cover ``m = 0 .. mpol-1``; toroidal columns ``n = 0 .. ntor``.
    """
    ntheta1 = res.ntheta1
    ntheta2 = res.ntheta2
    ntheta3 = res.ntheta3
    nzeta = res.nzeta
    nfp = res.nfp
    lasym = bool(res.lasym)
    m_max = res.mpol - 1  # largest poloidal mode number
    n_max = res.ntor  # largest toroidal mode number (before *nfp)

    # Integration normalization (fixaray.f).  ``dnorm`` weights the reduced
    # [0, pi] force projections (cosmui/sinmui) and is 1/(nzeta*(ntheta2-1))
    # for BOTH symmetry modes: lasym kernels are symmetrized first
    # (symforce.f), so the endpoint-half-weighted reduced integral with
    # weight 2/ntheta1 equals the full-grid average.  Only the full-surface
    # average normalization ``dnorm3`` (wint/cosmui3) is lasym-dependent:
    # 1/(nzeta*ntheta1) on the full grid.  (The previous revision used the
    # full-grid dnorm for cosmui too, halving every lasym force projection
    # and, through alias.f's gcon, the constraint-force weight relative to
    # the MHD force -- shifting the lasym fixed point away from VMEC2000.)
    dnorm = 1.0 / (nzeta * (ntheta2 - 1))
    dnorm3 = 1.0 / (nzeta * ntheta3) if lasym else dnorm

    # mscale(0)=nscale(0)=1; sqrt(2) for m,n >= 1 (fixaray.f, osqrt2).
    one_over_sqrt2 = 1.0 / np.sqrt(2.0)
    mscale = np.ones((m_max + 1,), dtype=float)
    nscale = np.ones((n_max + 1,), dtype=float)
    if m_max >= 1:
        mscale[1:] = mscale[0] / one_over_sqrt2
    if n_max >= 1:
        nscale[1:] = nscale[0] / one_over_sqrt2

    # Theta tables: arg_i = 2*pi*i/ntheta1 for i = 0 .. ntheta3-1.  For the
    # symmetric reduced grid, the theta = pi row (i = ntheta2-1) is set to the
    # exact alternating values +/- mscale (fixaray.f does the same) so that the
    # endpoint is free of cos(pi*m) rounding.
    m = np.arange(m_max + 1, dtype=float)
    cosmu = np.zeros((ntheta3, m_max + 1), dtype=float)
    sinmu = np.zeros_like(cosmu)
    for i in range(ntheta3):
        if (not lasym) and i == (ntheta2 - 1):
            signs = np.where((m.astype(int) % 2) == 0, 1.0, -1.0)
            cosmu[i, :] = signs * mscale
            sinmu[i, :] = 0.0
        else:
            arg = (2.0 * np.pi) * float(i) / float(ntheta1)
            cosmu[i, :] = np.cos(arg * m) * mscale
            sinmu[i, :] = np.sin(arg * m) * mscale

    cosmum = cosmu * m[None, :]
    sinmum = -sinmu * m[None, :]

    cosmui = (dnorm * cosmu).copy()
    sinmui = (dnorm * sinmu).copy()
    cosmui3 = (dnorm3 * cosmu).copy()

    # Endpoint half-weights at theta = 0 and theta = pi.  In VMEC, ntheta2 is
    # the theta = pi index even when lasym=True.
    cosmui[0, :] *= 0.5
    cosmui[ntheta2 - 1, :] *= 0.5
    if (not lasym) and ntheta2 == ntheta3:
        # Symmetric runs reuse the half-interval weights for full-surface
        # integrations too (fixaray.f).
        cosmui3 = cosmui.copy()

    cosmumi = cosmui * m[None, :]
    sinmumi = -sinmui * m[None, :]
    cosmumi3 = cosmui3 * m[None, :]

    # Zeta tables: arg_j = 2*pi*j/nzeta for j = 0 .. nzeta-1 (one field period).
    zeta = (2.0 * np.pi) * np.arange(nzeta, dtype=float) / float(nzeta)
    n = np.arange(n_max + 1, dtype=float)
    arg_n = zeta[:, None] * n[None, :]
    cosnv = np.cos(arg_n) * nscale[None, :]
    sinnv = np.sin(arg_n) * nscale[None, :]
    cosnvn = cosnv * (n[None, :] * float(nfp))
    sinnvn = -sinnv * (n[None, :] * float(nfp))

    # Angular integration weights on the internal grid (profil3d.f wint):
    # constant in zeta, endpoint-half-weighted in theta for symmetric runs.
    w_theta = cosmui3[:, 0] / float(mscale[0])
    wint = w_theta[:, None] * np.ones((nzeta,), dtype=float)[None, :]

    return TrigTables(
        ntheta1=ntheta1,
        ntheta2=ntheta2,
        ntheta3=ntheta3,
        nfp=nfp,
        lasym=lasym,
        dnorm=float(dnorm),
        dnorm3=float(dnorm3),
        mscale=mscale,
        nscale=nscale,
        cosmu=cosmu,
        sinmu=sinmu,
        cosmum=cosmum,
        sinmum=sinmum,
        cosmui=cosmui,
        sinmui=sinmui,
        cosmumi=cosmumi,
        sinmumi=sinmumi,
        cosmui3=cosmui3,
        cosmumi3=cosmumi3,
        cosnv=cosnv,
        sinnv=sinnv,
        cosnvn=cosnvn,
        sinnvn=sinnvn,
        wint=wint,
    )

