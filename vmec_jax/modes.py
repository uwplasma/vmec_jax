"""Mode tables matching VMEC's conventions.

VMEC uses a slightly special convention:
- m ranges from 0..mpol-1 (mpol1), but many arrays allocate up to mpol (mpol)
- n ranges from -ntor..ntor, except for m=0 where negative n are omitted
- VMEC internally uses the *field-period* toroidal angle zeta in [0,2pi) and stores xn = n*nfp

For the JAX port we keep explicit (m, n) pairs and apply nfp scaling when needed (e.g. physical toroidal derivatives).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ModeTable:
    m: np.ndarray  # (K,)
    n: np.ndarray  # (K,)  integer toroidal mode number (before *nfp)

    @property
    def K(self) -> int:
        return int(self.m.size)


def vmec_mode_table(mpol: int, ntor: int) -> ModeTable:
    """Create VMEC-like (m,n) pairs.

    Includes all (m,n) with m=0..mpol-1 and n in [-ntor..ntor], but for m=0 uses n>=0.

    This matches the xm/xn tables built in VMEC's `fixaray.f` (see nmin0 logic).
    """
    mpol = int(abs(mpol))
    ntor = int(abs(ntor))

    ms = []
    ns = []
    for m in range(0, mpol):
        nmin = 0 if m == 0 else -ntor
        for n in range(nmin, ntor + 1):
            ms.append(m)
            ns.append(n)
    return ModeTable(m=np.asarray(ms, dtype=int), n=np.asarray(ns, dtype=int))


def nyquist_mode_table(mpol: int, ntor: int) -> ModeTable:
    """Create a VMEC-style Nyquist mode table.

    VMEC's Nyquist grids extend the Fourier bandwidth by a small padding. Empirically
    (and consistent with `fixaray` defaults), the Nyquist mode limits are:

    - mmax_nyq = mpol + 3
    - nmax_nyq = ntor + 2  (but 0 when ntor==0)

    This matches the bundled VMEC2000 `wout_*.nc` files used for parity checks.
    """
    mpol = int(abs(mpol))
    ntor = int(abs(ntor))

    mmax = mpol + 3
    nmax = 0 if ntor == 0 else ntor + 2

    ms = []
    ns = []
    for m in range(0, mmax + 1):
        nmin = 0 if m == 0 else -nmax
        for n in range(nmin, nmax + 1):
            ms.append(m)
            ns.append(n)
    return ModeTable(m=np.asarray(ms, dtype=int), n=np.asarray(ns, dtype=int))


def default_grid_sizes(mpol: int, ntor: int, ntheta: int = 0, nzeta: int = 0) -> Tuple[int, int]:
    """Match VMEC defaults in initialize_vmec_arrays.f."""
    mpol = int(abs(mpol))
    ntor = int(abs(ntor))
    if ntheta <= 0:
        ntheta = 2 * mpol + 6
    # VMEC makes ntheta1 = even
    ntheta1 = 2 * (ntheta // 2)

    if ntor == 0 and nzeta == 0:
        nzeta = 1
    if nzeta <= 0:
        nzeta = 2 * ntor + 4

    return ntheta1, int(nzeta)
