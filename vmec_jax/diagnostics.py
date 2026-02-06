"""Lightweight diagnostic helpers.

These utilities are intentionally dependency-free (NumPy-only) and meant to
print *useful* debugging information that you can copy/paste into chat.

We keep this module small and stable so it can be used from examples and tests
without pulling in plotting libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Summary:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    n_nan: int
    n_inf: int
    n_zero: int
    n_neg: int
    q: Tuple[float, float, float, float, float]


def _as_array(x: Any) -> np.ndarray:
    """Convert x to a NumPy array (safe for JAX arrays too)."""
    return np.asarray(x)


def summarize_array(name: str, x: Any, *, q: Sequence[float] = (0.0, 0.01, 0.5, 0.99, 1.0)) -> Summary:
    """Return basic stats + quantiles for an array-like."""
    a = _as_array(x)
    af = a.reshape(-1)
    # Handle empty arrays defensively
    if af.size == 0:
        return Summary(
            name=name,
            shape=tuple(a.shape),
            dtype=str(a.dtype),
            min=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            std=float("nan"),
            n_nan=0,
            n_inf=0,
            n_zero=0,
            n_neg=0,
            q=(float("nan"),) * 5,
        )

    n_nan = int(np.sum(np.isnan(af))) if np.issubdtype(af.dtype, np.floating) else 0
    n_inf = int(np.sum(np.isinf(af))) if np.issubdtype(af.dtype, np.floating) else 0
    finite = af
    if np.issubdtype(af.dtype, np.floating):
        finite = af[np.isfinite(af)]
        if finite.size == 0:
            finite = af

    qvals = tuple(float(np.quantile(finite, qq)) for qq in q)

    return Summary(
        name=name,
        shape=tuple(a.shape),
        dtype=str(a.dtype),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        n_nan=n_nan,
        n_inf=n_inf,
        n_zero=int(np.sum(finite == 0)),
        n_neg=int(np.sum(finite < 0)),
        q=qvals,
    )


def print_summary(s: Summary, *, indent: str = "") -> None:
    """Pretty-print a Summary."""
    q0, q1, q50, q99, q100 = s.q
    print(
        f"{indent}{s.name}: shape={s.shape} dtype={s.dtype} "
        f"min={s.min:.6g} max={s.max:.6g} mean={s.mean:.6g} std={s.std:.6g}"
    )
    print(
        f"{indent}  q[0%]={q0:.6g} q[1%]={q1:.6g} q[50%]={q50:.6g} q[99%]={q99:.6g} q[100%]={q100:.6g}"
    )
    if s.n_nan or s.n_inf or s.n_zero or s.n_neg:
        print(
            f"{indent}  counts: nan={s.n_nan} inf={s.n_inf} zero={s.n_zero} neg={s.n_neg}"
        )


def summarize_many(names_and_arrays: Iterable[Tuple[str, Any]], *, indent: str = "") -> None:
    """Summarize many arrays."""
    for name, arr in names_and_arrays:
        print_summary(summarize_array(name, arr), indent=indent)


def print_jacobian_stats(sqrtg: Any, *, indent: str = "") -> None:
    """Print useful statistics for the Jacobian sqrt(g)."""
    a = _as_array(sqrtg)
    print_summary(summarize_array("sqrtg", a), indent=indent)
    print_summary(summarize_array("|sqrtg|", np.abs(a)), indent=indent)


def slice_excluding_axis(a: Any, axis_dim: int = 0) -> np.ndarray:
    """Return a[1:] along the chosen axis (used to avoid s=0 degeneracy)."""
    x = _as_array(a)
    if x.ndim == 0 or x.shape[axis_dim] <= 1:
        return x
    slc = [slice(None)] * x.ndim
    slc[axis_dim] = slice(1, None)
    return x[tuple(slc)]
