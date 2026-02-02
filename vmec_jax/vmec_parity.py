"""VMEC parity helper kernels (Step-10).

This module provides small kernels that reproduce specific VMEC discrete
conventions used in the reference ``wout_*.nc`` outputs, as needed for parity
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .fourier import eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys


@dataclass(frozen=True)
class ParityRZL:
    """Real-space R/Z/L fields split into VMEC even/odd-m parity pieces."""

    R_even: Any
    R_odd: Any
    Z_even: Any
    Z_odd: Any
    L_even: Any
    L_odd: Any
    Rt_even: Any
    Rt_odd: Any
    Zt_even: Any
    Zt_odd: Any
    Lt_even: Any
    Lt_odd: Any
    Rp_even: Any
    Rp_odd: Any
    Zp_even: Any
    Zp_odd: Any
    Lp_even: Any
    Lp_odd: Any


def split_rzl_even_odd_m(state, basis, modes_m: np.ndarray) -> ParityRZL:
    """Evaluate real-space fields for even-m and odd-m subsets separately."""
    m = np.asarray(modes_m, dtype=int)
    mask_even = jnp.asarray((m % 2) == 0).astype(jnp.asarray(state.Rcos).dtype)
    mask_odd = (1.0 - mask_even).astype(mask_even.dtype)

    def _eval_pair(cos, sin, mask):
        return eval_fourier(cos * mask, sin * mask, basis)

    def _eval_pair_dtheta(cos, sin, mask):
        return eval_fourier_dtheta(cos * mask, sin * mask, basis)

    def _eval_pair_dphi(cos, sin, mask):
        return eval_fourier_dzeta_phys(cos * mask, sin * mask, basis)

    R_even = _eval_pair(state.Rcos, state.Rsin, mask_even)
    R_odd = _eval_pair(state.Rcos, state.Rsin, mask_odd)
    Z_even = _eval_pair(state.Zcos, state.Zsin, mask_even)
    Z_odd = _eval_pair(state.Zcos, state.Zsin, mask_odd)
    L_even = _eval_pair(state.Lcos, state.Lsin, mask_even)
    L_odd = _eval_pair(state.Lcos, state.Lsin, mask_odd)

    Rt_even = _eval_pair_dtheta(state.Rcos, state.Rsin, mask_even)
    Rt_odd = _eval_pair_dtheta(state.Rcos, state.Rsin, mask_odd)
    Zt_even = _eval_pair_dtheta(state.Zcos, state.Zsin, mask_even)
    Zt_odd = _eval_pair_dtheta(state.Zcos, state.Zsin, mask_odd)
    Lt_even = _eval_pair_dtheta(state.Lcos, state.Lsin, mask_even)
    Lt_odd = _eval_pair_dtheta(state.Lcos, state.Lsin, mask_odd)

    Rp_even = _eval_pair_dphi(state.Rcos, state.Rsin, mask_even)
    Rp_odd = _eval_pair_dphi(state.Rcos, state.Rsin, mask_odd)
    Zp_even = _eval_pair_dphi(state.Zcos, state.Zsin, mask_even)
    Zp_odd = _eval_pair_dphi(state.Zcos, state.Zsin, mask_odd)
    Lp_even = _eval_pair_dphi(state.Lcos, state.Lsin, mask_even)
    Lp_odd = _eval_pair_dphi(state.Lcos, state.Lsin, mask_odd)

    return ParityRZL(
        R_even=R_even,
        R_odd=R_odd,
        Z_even=Z_even,
        Z_odd=Z_odd,
        L_even=L_even,
        L_odd=L_odd,
        Rt_even=Rt_even,
        Rt_odd=Rt_odd,
        Zt_even=Zt_even,
        Zt_odd=Zt_odd,
        Lt_even=Lt_even,
        Lt_odd=Lt_odd,
        Rp_even=Rp_even,
        Rp_odd=Rp_odd,
        Zp_even=Zp_even,
        Zp_odd=Zp_odd,
        Lp_even=Lp_even,
        Lp_odd=Lp_odd,
    )


def internal_odd_from_physical(phys_odd, s, *, eps: float = 1e-14):
    """Convert physical odd-m contribution to VMEC internal odd field.

    VMEC represents:

        X = X_even + sqrt(s) * X_odd_internal

    so:

        X_odd_internal = X_odd_physical / sqrt(s)
    """
    s = jnp.asarray(s)
    sh = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    mask = (sh > eps).astype(jnp.asarray(phys_odd).dtype)
    return jnp.asarray(phys_odd) * mask / jnp.where(sh > eps, sh, 1.0)
