"""Initial guess construction for step-1.

VMEC has a fairly elaborate procedure to build an initial nested set of
surfaces from the boundary (and/or axis) Fourier coefficients.

For step-1 we implement a *regularity-aware* but intentionally simple guess that
is good enough to exercise the full (s,theta,zeta) geometry kernel:

- For m>0 harmonics, scale boundary coefficients like rho**m with rho = sqrt(s)
  to enforce regularity at the magnetic axis (matches VMEC/VMEC++).
- For m=0 harmonics, scale with s and, if axis coefficients are provided,
  linearly blend between the axis and the boundary.
- lambda coefficients are initialized to zero.

This guess is not intended to match VMEC's exact internal initial guess yet.
It is a stable, differentiable starting point that we can later improve while
keeping the geometry/transform kernels unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._compat import jnp, has_jax
from .boundary import BoundaryCoeffs
from .namelist import InData
from .state import StateLayout, VMECState
from .static import VMECStatic


def _read_axis_coeffs(indata: InData) -> dict[str, float | list[float]]:
    """Read axis arrays if present.

    VMEC supports axis series in a few naming conventions. For step-1 we only
    look for the common modern VMEC names:

    - RAXIS_CC, RAXIS_CS
    - ZAXIS_CC, ZAXIS_CS

    Each may be a scalar or a list. We return the raw values.
    """
    out: dict[str, float | list[float]] = {}
    for key in ("RAXIS_CC", "RAXIS_CS", "ZAXIS_CC", "ZAXIS_CS"):
        v = indata.get(key, None)
        if v is None:
            continue
        out[key] = v
    return out


def _axis_array(values: float | list[float] | None, ntor: int, *, dtype):
    if values is None:
        return None
    if isinstance(values, list):
        arr = [float(v) for v in values]
    else:
        arr = [float(values)]
    if len(arr) < ntor + 1:
        arr = arr + [0.0] * (ntor + 1 - len(arr))
    else:
        arr = arr[: ntor + 1]
    return jnp.asarray(arr, dtype=dtype)


def initial_guess_from_boundary(
    static: VMECStatic,
    boundary: BoundaryCoeffs,
    indata: InData | None = None,
    *,
    dtype=None,
) -> VMECState:
    """Build a VMECState initial guess from boundary coefficients.

    Parameters
    ----------
    static:
        Precomputed modes/grid/basis and radial coordinate.
    boundary:
        Boundary coefficients aligned with `static.modes`.
    indata:
        If provided, used to read optional axis specification. If absent or if
        the axis arrays are all zero, the axis is inferred from boundary m=0
        coefficients.
    dtype:
        Optional dtype for the returned arrays.
    """
    cfg = static.cfg
    K = static.modes.K
    layout = StateLayout(ns=cfg.ns, K=K, lasym=cfg.lasym)

    m = jnp.asarray(static.modes.m)
    s = jnp.asarray(static.s)
    if dtype is None:
        # Choose a dtype that avoids JAX warning spam.
        # VMEC expects float64; we default to float64 when x64 is enabled.
        if has_jax():
            try:
                import jax

                x64 = bool(jax.config.read("jax_enable_x64"))
            except Exception:
                x64 = True
            dtype = jnp.float64 if x64 else jnp.float32
        else:
            # numpy fallback: use float64 for VMEC parity
            import numpy as _np

            dtype = _np.float64

    # Base: broadcast boundary vectors to (ns,K)
    Rcos_b = jnp.asarray(boundary.R_cos, dtype=dtype)[None, :]
    Rsin_b = jnp.asarray(boundary.R_sin, dtype=dtype)[None, :]
    Zcos_b = jnp.asarray(boundary.Z_cos, dtype=dtype)[None, :]
    Zsin_b = jnp.asarray(boundary.Z_sin, dtype=dtype)[None, :]

    # Regularity scaling: use rho**m with rho = sqrt(s) for m>0 (VMEC/VMEC++).
    # For m=0, keep Rcos constant unless we blend with axis; other components
    # use s to ensure regularity at the axis.
    rho = jnp.sqrt(s)
    scale_r = jnp.where(m[None, :] > 0, rho[:, None] ** m[None, :], 1.0)
    scale_other = jnp.where(m[None, :] > 0, rho[:, None] ** m[None, :], s[:, None])
    Rcos = scale_r * Rcos_b
    Rsin = scale_other * Rsin_b
    Zcos = scale_other * Zcos_b
    Zsin = scale_other * Zsin_b

    # If user supplied a non-trivial axis spec, blend m=0 coefficients between
    # axis and boundary (linear in s), matching VMEC/VMEC++ conventions.
    if indata is not None:
        ax = _read_axis_coeffs(indata)
        raxis_cc = _axis_array(ax.get("RAXIS_CC", None), cfg.ntor, dtype=dtype)
        zaxis_cs = _axis_array(ax.get("ZAXIS_CS", None), cfg.ntor, dtype=dtype)

        # If axis arrays are all zero or missing, fall back to boundary-based axis.
        have_axis = False
        if raxis_cc is not None and np.any(np.asarray(raxis_cc) != 0.0):
            have_axis = True
        if zaxis_cs is not None and np.any(np.asarray(zaxis_cs) != 0.0):
            have_axis = True

        if have_axis:
            if raxis_cc is None:
                raxis_cc = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
            if zaxis_cs is None:
                zaxis_cs = jnp.zeros((cfg.ntor + 1,), dtype=dtype)

            # Blend only m=0 modes; we support all n>=0 entries in the mode table.
            m0_mask = static.modes.m == 0
            for n in range(cfg.ntor + 1):
                k_candidates = jnp.where(m0_mask & (static.modes.n == n))[0]
                if k_candidates.size == 0:
                    continue
                k = int(k_candidates[0])
                blend = s
                new_R = (1.0 - blend) * raxis_cc[n] + blend * Rcos_b[0, k]
                if has_jax():
                    Rcos = Rcos.at[:, k].set(new_R)
                else:
                    Rcos = jnp.array(Rcos)
                    Rcos[:, k] = new_R

    Lcos = jnp.zeros((cfg.ns, K), dtype=dtype)
    Lsin = jnp.zeros((cfg.ns, K), dtype=dtype)

    return VMECState(layout=layout, Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin, Lcos=Lcos, Lsin=Lsin)
