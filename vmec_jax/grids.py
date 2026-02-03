"""Angle grids consistent with VMEC conventions.

VMEC typically computes on one field period. Internally the toroidal coordinate often
behaves like zeta = NFP * phi_phys, such that zeta in [0, 2pi) spans one field period.

We keep grids explicit and leave symmetry reductions (ntheta2 / ntheta3) for later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class AngleGrid:
    theta: np.ndarray  # (ntheta,)
    zeta: np.ndarray   # (nzeta,)
    nfp: int

    @property
    def ntheta(self) -> int:
        return int(self.theta.size)

    @property
    def nzeta(self) -> int:
        return int(self.zeta.size)


def make_angle_grid(ntheta: int, nzeta: int, nfp: int, endpoint: bool = False) -> AngleGrid:
    """Create theta, zeta grids.

    theta: [0, 2pi)
    zeta: [0, 2pi) for one field period (VMEC's internal convention)

    Physical toroidal angle would be phi_phys = zeta / nfp.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=endpoint)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=endpoint)
    return AngleGrid(theta=theta, zeta=zeta, nfp=int(nfp))


def angle_steps(*, ntheta: int, nzeta: int) -> tuple[float, float]:
    """Return uniform angle spacings (dtheta, dzeta) for periodic grids.

    VMEC uses periodic grids on [0, 2π) (typically without the endpoint). The
    uniform spacing is therefore `2π/N` even when `N==1`.
    """
    ntheta = int(ntheta)
    nzeta = int(nzeta)
    if ntheta < 1:
        raise ValueError("ntheta must be >= 1")
    if nzeta < 1:
        raise ValueError("nzeta must be >= 1")
    return TWOPI / float(ntheta), TWOPI / float(nzeta)
