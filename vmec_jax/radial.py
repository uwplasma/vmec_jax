"""Radial operators for vmec_jax.

VMEC uses a normalized radial-like coordinate ``s`` in [0, 1].

For step-2 we need radial derivatives of the *Fourier coefficients* that are
stored on an ``(ns, K)`` grid. We deliberately start with a simple and robust
finite-difference operator that is:

* fast on a laptop
* differentiable end-to-end (JAX-friendly)
* easy to replace later (e.g. Chebyshev / global spectral radial basis)

The operator below is 2nd-order centered in the interior and 1st-order
one-sided at both ends.
"""

from __future__ import annotations

from ._compat import jnp, jit


@jit
def d_ds_coeffs(f, s):
    """Compute df/ds for arrays whose leading dimension is ``ns``.

    Parameters
    ----------
    f:
        Array of shape (ns, ...) (typically (ns, K)).
    s:
        Radial grid of shape (ns,). Currently assumed uniform.

    Returns
    -------
    df:
        Array with the same shape as ``f``.
    """
    ns = f.shape[0]
    if ns == 1:
        return jnp.zeros_like(f)

    # Assume uniform grid for now (VMEC-style). Keep s as argument so we can
    # swap in nonuniform/spectral later without changing call sites.
    ds = s[1] - s[0]

    left = (f[1] - f[0]) / ds
    right = (f[-1] - f[-2]) / ds

    if ns == 2:
        return jnp.stack([left, right], axis=0)

    interior = (f[2:] - f[:-2]) / (2.0 * ds)
    return jnp.concatenate([left[None, ...], interior, right[None, ...]], axis=0)
