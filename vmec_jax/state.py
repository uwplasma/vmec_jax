"""State layout helpers.

This module defines the coefficient container used throughout vmec_jax.

Important for JAX:
  * ``VMECState`` must be a **PyTree** so it can be passed into ``jax.jit``'d
    functions (e.g. ``eval_coords``) and differentiated with ``jax.grad``.
  * Registration must be **idempotent**: in interactive workflows, the module
    can be imported multiple times (or reloaded) and JAX will otherwise raise
    a duplicate registration error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ._compat import has_jax


def _is_jax_array(x: Any) -> bool:
    """Best-effort check for JAX arrays without hard dependency."""
    if not has_jax():
        return False
    try:
        import jax

        return isinstance(x, jax.Array)
    except Exception:
        # Older JAX versions: duck-typing fallback.
        return hasattr(x, "device_buffer") or (hasattr(x, "__array_priority__") and "jax" in type(x).__module__)


def _xp_from(*xs: Any):
    """Pick numpy or jax.numpy based on inputs."""
    if any(_is_jax_array(x) for x in xs if x is not None):
        import jax.numpy as jnp

        return jnp
    return np


# ----------------------------------------------------------------------------
# PyTree registration helper (idempotent)
# ----------------------------------------------------------------------------


def _register_pytree_node_class_safe(cls):
    """Register ``cls`` as a PyTree node class if JAX is available.

    If the class was already registered (e.g. due to reload), ignore the
    duplicate-registration error.
    """

    if not has_jax():
        return cls
    try:
        from jax.tree_util import register_pytree_node_class as _register

        try:
            return _register(cls)
        except ValueError as e:
            # JAX raises ValueError on duplicate registrations.
            msg = str(e)
            if "Duplicate custom PyTreeDef type registration" in msg:
                return cls
            raise
    except Exception:
        # Very old JAX: register manually.
        import jax.tree_util as _tu

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


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StateLayout:
    ns: int
    K: int
    lasym: bool

    @property
    def n_fields(self) -> int:
        # R, Z, L each have cos/sin blocks
        return 6

    @property
    def size(self) -> int:
        return self.ns * self.K * self.n_fields

    def split(self, x) -> Tuple[Any, ...]:
        """Split a flat vector into (Rcos, Rsin, Zcos, Zsin, Lcos, Lsin)."""
        xp = _xp_from(x)
        x = xp.asarray(x)
        if x.ndim != 1 or x.size != self.size:
            raise ValueError(f"Expected flat vector of length {self.size}, got shape {x.shape}")
        blk = self.ns * self.K
        out = []
        for i in range(self.n_fields):
            out.append(x[i * blk : (i + 1) * blk].reshape((self.ns, self.K)))
        return tuple(out)  # type: ignore

    def pack(self, Rcos, Rsin, Zcos, Zsin, Lcos, Lsin):
        """Pack coefficient blocks into a flat vector."""
        xp = _xp_from(Rcos, Rsin, Zcos, Zsin, Lcos, Lsin)
        parts = [xp.asarray(a).reshape((self.ns, self.K)) for a in (Rcos, Rsin, Zcos, Zsin, Lcos, Lsin)]
        return xp.concatenate([p.reshape(-1) for p in parts], axis=0)


@_register_pytree_node_class_safe
@dataclass(frozen=True)
class VMECState:
    """Fourier coefficient container.

    Arrays are shaped (ns, K). ``layout`` is treated as static aux-data for JIT.
    """

    layout: StateLayout
    Rcos: Any
    Rsin: Any
    Zcos: Any
    Zsin: Any
    Lcos: Any
    Lsin: Any

    # --- JAX PyTree protocol ---
    def tree_flatten(self):
        children = (self.Rcos, self.Rsin, self.Zcos, self.Zsin, self.Lcos, self.Lsin)
        aux = (int(self.layout.ns), int(self.layout.K), bool(self.layout.lasym))
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ns, K, lasym = aux
        layout = StateLayout(ns=int(ns), K=int(K), lasym=bool(lasym))
        Rcos, Rsin, Zcos, Zsin, Lcos, Lsin = children
        return cls(layout=layout, Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin, Lcos=Lcos, Lsin=Lsin)


def pack_state(state: VMECState):
    """Pack a VMECState into a flat vector (ns*K*6)."""
    return state.layout.pack(state.Rcos, state.Rsin, state.Zcos, state.Zsin, state.Lcos, state.Lsin)


def unpack_state(x, layout: StateLayout) -> VMECState:
    """Unpack a flat vector into a VMECState."""
    Rcos, Rsin, Zcos, Zsin, Lcos, Lsin = layout.split(x)
    return VMECState(layout=layout, Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin, Lcos=Lcos, Lsin=Lsin)


def zeros_state(layout: StateLayout, *, like=None) -> VMECState:
    """Create a zero-initialized state with the right shapes."""
    xp = _xp_from(like)
    z = xp.zeros((layout.ns, layout.K))
    return VMECState(layout=layout, Rcos=z, Rsin=z, Zcos=z, Zsin=z, Lcos=z, Lsin=z)
