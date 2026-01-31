"""Angle grids consistent with VMEC conventions.

VMEC typically computes on one field period. Internally the toroidal coordinate often
behaves like zeta = NFP * phi_phys, such that zeta in [0, 2pi) spans one field period.

We keep grids explicit and leave symmetry reductions (ntheta2 / ntheta3) for later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
