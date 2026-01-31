"""Boundary coefficient handling.

VMEC input files specify boundary Fourier coefficients in arrays:

- rbc(n,m), rbs(n,m)
- zbc(n,m), zbs(n,m)

where the boundary is expanded in helical harmonics

    R(θ,ζ) = Σ_{m,n} [ rbc(n,m) cos(mθ - nζ) + rbs(n,m) sin(mθ - nζ) ]
    Z(θ,ζ) = Σ_{m,n} [ zbc(n,m) cos(mθ - nζ) + zbs(n,m) sin(mθ - nζ) ]

This module provides helpers to map the sparse namelist assignments to dense coefficient
vectors aligned with a ModeTable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .namelist import InData
from .modes import ModeTable


@dataclass(frozen=True)
class BoundaryCoeffs:
    # Each is shape (K,), aligned with ModeTable
    R_cos: np.ndarray
    R_sin: np.ndarray
    Z_cos: np.ndarray
    Z_sin: np.ndarray


def _get_indexed(indata: InData, name: str) -> Dict[Tuple[int, ...], float]:
    d = indata.indexed.get(name.upper(), {})
    out: Dict[Tuple[int, ...], float] = {}
    for k, v in d.items():
        if isinstance(v, bool) or isinstance(v, str):
            continue
        out[tuple(k)] = float(v)
    return out


def boundary_from_indata(indata: InData, modes: ModeTable) -> BoundaryCoeffs:
    """Build dense boundary coefficient vectors aligned with `modes`.

    Notes:
    - Input indices are (n,m) in the namelist and in VMEC's vmec_input module.
    - Our ModeTable stores (m,n), so we swap.
    """
    rbc = _get_indexed(indata, "RBC")
    rbs = _get_indexed(indata, "RBS")
    zbc = _get_indexed(indata, "ZBC")
    zbs = _get_indexed(indata, "ZBS")

    K = modes.K
    R_cos = np.zeros((K,), dtype=float)
    R_sin = np.zeros((K,), dtype=float)
    Z_cos = np.zeros((K,), dtype=float)
    Z_sin = np.zeros((K,), dtype=float)

    # Build a quick lookup from (m,n) to k
    key_to_k = {(int(m), int(n)): k for k, (m, n) in enumerate(zip(modes.m, modes.n))}

    def assign_from(src: Dict[Tuple[int, ...], float], dest: np.ndarray, kind: str):
        for (n, m), val in src.items():
            k = key_to_k.get((int(m), int(n)))
            if k is None:
                continue
            dest[k] = val

    assign_from(rbc, R_cos, "rbc")
    assign_from(rbs, R_sin, "rbs")
    assign_from(zbc, Z_cos, "zbc")
    assign_from(zbs, Z_sin, "zbs")

    return BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)
