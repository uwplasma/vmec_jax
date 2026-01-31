"""Geometry evaluation on the (s,theta,zeta) grid.

Step-1 goal:
    Fourier coefficients -> R,Z,lambda on a full tensor grid
    plus their angular derivatives.

This corresponds to VMEC's `totzsp*` stages (Fourier synthesis) and provides the
raw arrays needed for downstream metric/Jacobian and force-balance calculations.

Design notes
------------
This function is intentionally a thin wrapper around the Fourier synthesis
kernels. It is the first building block toward a fast, end-to-end
differentiable VMEC port:

- Keep everything vectorized over the radial index `s`.
- Keep the basis precomputed and passed in as a PyTree so `jax.jit` works.
- Return plain JAX arrays (or numpy arrays if JAX isn't available).

Later steps will add metric/Jacobian and field quantities on top of these
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ._compat import jit, has_jax
from .fourier import (
    HelicalBasis,
    eval_fourier,
    eval_fourier_dtheta,
    eval_fourier_dzeta_phys,
)
from .state import VMECState


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
        # Very old JAX: emulate decorator.
        import jax.tree_util as _tu

        def register_pytree_node_class(cls):  # type: ignore
            try:
                _tu.register_pytree_node(
                    cls,
                    lambda x: (x.tree_flatten()[0], x.tree_flatten()[1]),
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
class Coords:
    """Real-space fields and angular derivatives.

    Shapes are (ns, ntheta, nzeta) for all arrays.
    """

    R: Any
    Z: Any
    L: Any
    R_theta: Any
    Z_theta: Any
    L_theta: Any
    R_phi: Any
    Z_phi: Any
    L_phi: Any

    # Make this a PyTree so `jit` can return it.
    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        children = (
            self.R,
            self.Z,
            self.L,
            self.R_theta,
            self.Z_theta,
            self.L_theta,
            self.R_phi,
            self.Z_phi,
            self.L_phi,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: Tuple[Any, ...]) -> "Coords":
        return cls(*children)


@jit
def eval_coords(state: VMECState, basis: HelicalBasis) -> Coords:
    """Evaluate R,Z,lambda and their angular derivatives.

    Parameters
    ----------
    state:
        Coefficient container with arrays of shape (ns, K).
    basis:
        Helical basis on an angle grid.

    Returns
    -------
    Coords
        All arrays are (ns, ntheta, nzeta) and are compatible with JAX autodiff.
    """
    R = eval_fourier(state.Rcos, state.Rsin, basis)
    Z = eval_fourier(state.Zcos, state.Zsin, basis)
    L = eval_fourier(state.Lcos, state.Lsin, basis)

    R_theta = eval_fourier_dtheta(state.Rcos, state.Rsin, basis)
    Z_theta = eval_fourier_dtheta(state.Zcos, state.Zsin, basis)
    L_theta = eval_fourier_dtheta(state.Lcos, state.Lsin, basis)

    R_phi = eval_fourier_dzeta_phys(state.Rcos, state.Rsin, basis)
    Z_phi = eval_fourier_dzeta_phys(state.Zcos, state.Zsin, basis)
    L_phi = eval_fourier_dzeta_phys(state.Lcos, state.Lsin, basis)

    return Coords(
        R=R,
        Z=Z,
        L=L,
        R_theta=R_theta,
        Z_theta=Z_theta,
        L_theta=L_theta,
        R_phi=R_phi,
        Z_phi=Z_phi,
        L_phi=L_phi,
    )
