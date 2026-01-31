"""Energy functionals (step-4).

This module provides a minimal, differentiable magnetic energy functional,
intended as the first objective for a fixed-boundary solver.

We match VMEC's reported `wb` normalization:

    wb = (1 / (2π)^2) * ∫ (B·B)/2 dV

where the integral is over the *full torus*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from ._compat import jnp
from .field import TWOPI, b2_from_bsup, bsup_from_geom, lamscale_from_phips
from .geom import eval_geom
from .namelist import InData
from .profiles import eval_profiles


@dataclass(frozen=True)
class FluxProfiles:
    """Simple 1D flux functions for step-4."""

    phipf: Any  # (ns,)
    chipf: Any  # (ns,)
    phips: Any  # (ns,)
    signgs: int
    lamscale: Any  # scalar


def _as_float_list(x: Any) -> list[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    return [float(x)]


def _poly_no_const(coeffs_1based, x):
    """Evaluate Σ_{i>=1} a_i x^i, where `coeffs_1based[i-1] == a_i`."""
    a = jnp.asarray(coeffs_1based)
    x = jnp.asarray(x)
    if a.shape[0] == 0:
        return jnp.zeros_like(x)
    # Σ_{i=1..N} a_i x^i = x * Σ_{k=0..N-1} a_{k+1} x^k
    y = jnp.zeros_like(x, dtype=a.dtype)
    for k in range(int(a.shape[0]) - 1, -1, -1):
        y = y * x + a[k]
    return x * y


def _poly_no_const_deriv(coeffs_1based, x):
    """Derivative of Σ_{i>=1} a_i x^i."""
    a = jnp.asarray(coeffs_1based)
    x = jnp.asarray(x)
    if a.shape[0] == 0:
        return jnp.zeros_like(x)
    # d/dx Σ_{i=1..N} a_i x^i = Σ_{i=1..N} i a_i x^{i-1}
    c = a * jnp.arange(1, int(a.shape[0]) + 1, dtype=a.dtype)
    y = jnp.zeros_like(x, dtype=a.dtype)
    for k in range(int(c.shape[0]) - 1, -1, -1):
        y = y * x + c[k]
    return y


def flux_profiles_from_indata(indata: InData, s, *, signgs: int) -> FluxProfiles:
    """Construct simple flux profiles (phipf/chipf) from &INDATA.

    This is a deliberately minimal port:
    - toroidal flux uses `PHIEDGE` and optional polynomial `APHI` (default: aphi=[1]).
    - poloidal flux derivative is derived from iota when available (ncurr=0 cases).
      For ncurr=1 (current-driven), `chipf` is not determined from the input alone;
      we currently set it to 0 unless iota is provided.
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])

    phiedge = float(indata.get_float("PHIEDGE", 1.0))

    aphi = _as_float_list(indata.get("APHI", []))
    if not aphi:
        aphi = [1.0]
    aphi_arr = jnp.asarray(aphi, dtype=s.dtype)
    norm = _poly_no_const(aphi_arr, jnp.asarray(1.0, dtype=s.dtype))
    norm = jnp.where(norm != 0, norm, jnp.asarray(1.0, dtype=s.dtype))

    torflux_deriv = _poly_no_const_deriv(aphi_arr, s) / norm
    phipf = phiedge * torflux_deriv * jnp.ones((ns,), dtype=s.dtype)

    prof = eval_profiles(indata, s)
    iota = prof.get("iota", jnp.zeros_like(s))
    chipf = iota * phipf

    phips = (signgs * phipf) / TWOPI
    lamscale = lamscale_from_phips(phips, s)
    return FluxProfiles(phipf=phipf, chipf=chipf, phips=phips, signgs=int(signgs), lamscale=lamscale)


def integrate_volume_density(density, sqrtg, s, theta, zeta, *, nfp: int, signgs: int):
    """Integrate `density` over the full torus using `sqrtg` and grid spacings."""
    density = jnp.asarray(density)
    sqrtg = jnp.asarray(sqrtg)
    s = jnp.asarray(s)
    theta = jnp.asarray(theta)
    zeta = jnp.asarray(zeta)
    nfp = int(nfp)
    signgs = int(signgs)

    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    if theta.shape[0] < 2 or zeta.shape[0] < 2:
        raise ValueError("theta and zeta must have at least 2 points")
    dtheta = theta[1] - theta[0]
    dzeta = zeta[1] - zeta[0]
    dphi = dzeta / nfp

    jac = signgs * sqrtg
    per_period = jnp.sum(density * jac) * ds * dtheta * dphi
    return per_period * nfp


def magnetic_wb_from_state(state, static, indata: InData, *, signgs: int) -> Tuple[Any, Dict[str, Any]]:
    """Compute VMEC-style `wb` and a small diagnostics dict."""
    g = eval_geom(state, static)
    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    bsupu, bsupv = bsup_from_geom(g, phipf=flux.phipf, chipf=flux.chipf, nfp=static.cfg.nfp, signgs=signgs, lamscale=flux.lamscale)
    B2 = b2_from_bsup(g, bsupu, bsupv)
    E = integrate_volume_density(0.5 * B2, g.sqrtg, static.s, static.grid.theta, static.grid.zeta, nfp=static.cfg.nfp, signgs=signgs)
    wb = E / (TWOPI * TWOPI)
    diag = {"energy_total": E, "lamscale": flux.lamscale}
    return wb, diag
