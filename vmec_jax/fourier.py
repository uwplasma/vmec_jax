"""Helical Fourier basis utilities.

VMEC represents surfaces using Fourier series in the helical phase

    phase = m*theta - n*zeta,

where `zeta` is the *field-period* toroidal angle (one field period spans 0..2pi).

This module provides:
- basis matrices (cos/sin of phase) for a set of (m,n) modes
- evaluation of a scalar field from (cos, sin) Fourier coefficients
- evaluation of angular derivatives, including the physical toroidal derivative that
  includes the NFP scaling used in VMEC.

All functions are backend-agnostic: they use jax.numpy if available, otherwise numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ._compat import jnp, jit, has_jax
from .modes import ModeTable
from .grids import AngleGrid

# Make HelicalBasis a JAX PyTree so it can be passed through jitted functions.
# Registration is made idempotent to avoid duplicate-registration errors
# in notebook/reload workflows.
if has_jax():
    try:
        from jax.tree_util import register_pytree_node_class as _register_pytree_node_class  # type: ignore

        def register_pytree_node_class(cls):  # type: ignore
            try:
                return _register_pytree_node_class(cls)
            except ValueError as e:
                if "Duplicate custom PyTreeDef type registration" in str(e):
                    return cls
                raise
    except Exception:  # pragma: no cover
        import jax.tree_util as _tu

        def register_pytree_node_class(cls):  # type: ignore
            try:
                _tu.register_pytree_node(
                    cls,
                    lambda x: x.tree_flatten(),
                    lambda aux, children: cls.tree_unflatten(aux, children),
                )
            except ValueError as e:
                if "Duplicate custom PyTreeDef type registration" not in str(e):
                    raise
            return cls
else:

    def register_pytree_node_class(cls):  # type: ignore
        return cls


@register_pytree_node_class
@dataclass(frozen=True)
class HelicalBasis:
    """Precomputed basis on a theta×zeta grid for a given mode table."""

    cos_phase: any  # (K, ntheta, nzeta)
    sin_phase: any  # (K, ntheta, nzeta)
    m: any  # (K,)
    n: any  # (K,)
    nfp: int

    def tree_flatten(self):
        # Arrays are dynamic leaves; nfp is static auxiliary data.
        children = (self.cos_phase, self.sin_phase, self.m, self.n)
        aux = (int(self.nfp),)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (nfp,) = aux
        cos_phase, sin_phase, m, n = children
        return cls(cos_phase=cos_phase, sin_phase=sin_phase, m=m, n=n, nfp=int(nfp))


def build_helical_basis(modes: ModeTable, grid: AngleGrid) -> HelicalBasis:
    """Precompute cos/sin(phase) tensors.

    Notes:
    - This is O(K*ntheta*nzeta) memory; fine for VMEC defaults (low mpol, ntor) on a laptop.
    - Later we can switch to factored/FFT-based transforms once parity is validated.
    """
    m = jnp.asarray(modes.m)
    n = jnp.asarray(modes.n)
    theta = jnp.asarray(grid.theta)
    zeta = jnp.asarray(grid.zeta)

    # phase[K, ntheta, nzeta] via broadcasting
    phase = m[:, None, None] * theta[None, :, None] - n[:, None, None] * zeta[None, None, :]
    return HelicalBasis(
        cos_phase=jnp.cos(phase),
        sin_phase=jnp.sin(phase),
        m=m,
        n=n,
        nfp=grid.nfp,
    )


@jit
def eval_fourier(
    coeff_cos,
    coeff_sin,
    basis: HelicalBasis,
):
    """Evaluate f(theta,zeta) = Σ_k [c_k cos(phase_k) + s_k sin(phase_k)].

    coeff_cos, coeff_sin: shape (..., K)

    Returns: shape (..., ntheta, nzeta)
    """
    # Einsum works in both numpy and jax.numpy
    return jnp.einsum("...k,kij->...ij", coeff_cos, basis.cos_phase) + jnp.einsum(
        "...k,kij->...ij", coeff_sin, basis.sin_phase
    )


@jit
def eval_fourier_dtheta(coeff_cos, coeff_sin, basis: HelicalBasis):
    """∂/∂theta of the helical Fourier series."""
    m = basis.m
    # d/dtheta cos = -m sin; d/dtheta sin = +m cos
    return jnp.einsum("...k,kij->...ij", coeff_cos * (-m), basis.sin_phase) + jnp.einsum(
        "...k,kij->...ij", coeff_sin * m, basis.cos_phase
    )


@jit
def eval_fourier_dzeta_phys(coeff_cos, coeff_sin, basis: HelicalBasis):
    """∂/∂phi_phys where phi_phys is the physical toroidal angle.

    VMEC uses zeta on one field period, so phi_phys = zeta / NFP and
    ∂/∂phi_phys = NFP * ∂/∂zeta.

    In the helical phase m*theta - n*zeta, ∂/∂zeta introduces factor (+n).
    Therefore ∂/∂phi_phys introduces factor (+n*NFP).
    """
    n_phys = basis.n * basis.nfp
    # d/dzeta cos = +n sin; d/dzeta sin = -n cos  (because phase has -n*zeta)
    return jnp.einsum("...k,kij->...ij", coeff_cos * n_phys, basis.sin_phase) + jnp.einsum(
        "...k,kij->...ij", coeff_sin * (-n_phys), basis.cos_phase
    )


def project_to_modes(
    f: Array,
    basis: HelicalBasis,
    *,
    normalize: bool = True,
):
    """Project a real-space field `f(theta,zeta)` onto the helical Fourier basis.

    This is the inverse operation of `eval_fourier` *in the discrete sense* for a
    uniform tensor-product grid.

    Parameters
    ----------
    f:
        Real-space field on the angular grid. Shape (..., ntheta, nzeta).
    basis:
        Precomputed basis from `build_helical_basis`.
    normalize:
        If True, use uniform-grid orthogonality normalization (fast, no solves).
        If False, return raw inner products.

    Returns
    -------
    (coeff_cos, coeff_sin):
        Arrays of coefficients with shape (..., nmodes).

    Notes
    -----
    - This assumes the grid spans a full 2pi period in both theta and zeta.
    - For later parity-optimized grids (ntheta2/ntheta3) we'll provide a separate
      routine matching VMEC's normalization exactly.
    """
    # f: (..., i, j), basis: (k, i, j)
    inner_cos = jnp.einsum("...ij,kij->...k", f, basis.cos_phase)
    inner_sin = jnp.einsum("...ij,kij->...k", f, basis.sin_phase)

    if not normalize:
        return inner_cos, inner_sin

    ntheta = basis.cos_phase.shape[1]
    nzeta = basis.cos_phase.shape[2]
    norm = 2.0 / (ntheta * nzeta)

    coeff_cos = norm * inner_cos
    coeff_sin = norm * inner_sin

    # (m,n)=(0,0) mode should not get factor 2.
    # We detect it via (m==0 and n==0) in the basis metadata.
    if hasattr(basis, "m") and hasattr(basis, "n"):
        mask00 = jnp.logical_and(basis.m == 0, basis.n == 0)
        # Convert to float mask for broadcasting.
        mask00f = mask00.astype(coeff_cos.dtype)
        # Replace coefficients where mask00 = 1.
        coeff_cos = coeff_cos * (1.0 - mask00f) + (inner_cos / (ntheta * nzeta)) * mask00f
        coeff_sin = coeff_sin * (1.0 - mask00f) + 0.0 * mask00f

    return coeff_cos, coeff_sin