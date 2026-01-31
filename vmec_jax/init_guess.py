"""Initial guess construction for step-1.

VMEC has a fairly elaborate procedure to build an initial nested set of
surfaces from the boundary (and/or axis) Fourier coefficients.

For step-1 we implement a *regularity-aware* but intentionally simple guess that
is good enough to exercise the full (s,theta,zeta) geometry kernel:

- For m>0 harmonics, scale boundary coefficients like s**m to enforce regularity
  at the magnetic axis.
- For m=0 harmonics, keep the boundary coefficients constant in s unless the
  user provides an explicit axis specification.
- lambda coefficients are initialized to zero.

This guess is not intended to match VMEC's exact internal initial guess yet.
It is a stable, differentiable starting point that we can later improve while
keeping the geometry/transform kernels unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    # Regularity scaling for m>0: coefficient(s,k) = s**m * boundary(k).
    # For m=0 we keep constant unless we have explicit axis arrays.
    scale = jnp.where(m[None, :] > 0, s[:, None] ** m[None, :], 1.0)
    Rcos = scale * Rcos_b
    Rsin = scale * Rsin_b
    Zcos = scale * Zcos_b
    Zsin = scale * Zsin_b

    # If user supplied a non-trivial axis spec, override m=0 coefficients at s=0
    # (and blend linearly in s for m=0 only).
    if indata is not None:
        ax = _read_axis_coeffs(indata)
        # We only interpret the n=0 component for step-1 (m=0,n=0 mode).
        # This is enough to support the common "axis major radius" use case.
        # Full toroidal series will come later.
        raxis_cc = ax.get("RAXIS_CC", 0.0)
        zaxis_cs = ax.get("ZAXIS_CS", 0.0)

        def _as0(v):
            if isinstance(v, list):
                return float(v[0]) if v else 0.0
            return float(v)

        r0 = _as0(raxis_cc)
        z0 = _as0(zaxis_cs)
        # Detect if they intentionally left them at 0 to request boundary-based guess.
        if abs(r0) > 0.0 or abs(z0) > 0.0:
            # Find k for (m,n)=(0,0). Our mode table always includes this.
            k00 = int(jnp.where((static.modes.m == 0) & (static.modes.n == 0))[0][0])
            # Blend only for m=0 modes (we only set k00 for now).
            blend = s[:, None]
            new_R = (1.0 - blend[:, 0]) * r0 + blend[:, 0] * Rcos[:, k00]
            new_Z = (1.0 - blend[:, 0]) * z0 + blend[:, 0] * Zsin[:, k00]
            if has_jax():
                Rcos = Rcos.at[:, k00].set(new_R)
                Zsin = Zsin.at[:, k00].set(new_Z)
            else:
                # numpy fallback (non-differentiable): copy & assign
                Rcos = jnp.array(Rcos)
                Zsin = jnp.array(Zsin)
                Rcos[:, k00] = new_R
                Zsin[:, k00] = new_Z

    Lcos = jnp.zeros((cfg.ns, K), dtype=dtype)
    Lsin = jnp.zeros((cfg.ns, K), dtype=dtype)

    return VMECState(layout=layout, Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin, Lcos=Lcos, Lsin=Lsin)
