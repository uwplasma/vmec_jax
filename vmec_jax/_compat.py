"""Small compatibility layer.

We want minimal dependencies, but also want the code to be importable in environments
without JAX (e.g. for parsing / numpy-only debugging). If JAX is available, we use it.

Notes on float64
----------------
VMEC historically relies on float64. JAX defaults to float32 unless x64 is enabled.
To keep results stable and reduce warning spam, we *default* to enabling x64 when
JAX is imported, unless the user has explicitly set ``JAX_ENABLE_X64``.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import os

import numpy as _np


def _try_import_jax() -> Tuple[Any, Any, Callable[[Callable[..., Any]], Callable[..., Any]]]:
    try:
        # Enable x64 by default for VMEC parity unless the user opted out.
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        import jax
        import jax.numpy as jnp

        # Also set via config (works even if env var was ignored because JAX was
        # already imported elsewhere, as long as it happens before first use).
        try:
            jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "0") == "1")
        except Exception:
            pass

        return jax, jnp, jax.jit
    except Exception:
        # numpy fallback: no autodiff, no jit
        return None, _np, (lambda f: f)


jax, jnp, jit = _try_import_jax()


def has_jax() -> bool:
    return jax is not None


def enable_x64(enable: bool = True) -> None:
    """Enable/disable float64 for JAX (no-op if JAX unavailable).

    VMEC historically relies on float64; we therefore enable x64 by default.
    This helper is useful in scripts/tests to be explicit.
    """
    if jax is None:
        return
    try:
        jax.config.update("jax_enable_x64", bool(enable))
    except Exception:
        # If JAX is already initialized in a way that disallows toggling,
        # we silently ignore (the existing dtype policy will apply).
        pass


def x64_enabled() -> bool:
    if jax is None:
        return True
    try:
        return bool(jax.config.read("jax_enable_x64"))
    except Exception:
        return bool(os.environ.get("JAX_ENABLE_X64", "0") == "1")


def asarray(x: Any, dtype: Any | None = None):
    """Create an array using the active backend (jax.numpy or numpy)."""
    return jnp.asarray(x, dtype=dtype)
