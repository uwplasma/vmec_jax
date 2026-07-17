"""Backend (CPU/GPU) selection policy for the core solve lanes.

Measured basis: ``benchmarks/gpu_baseline.json`` (2026-07-09, 2x RTX A4000,
jax 0.6.2 cuda12) — see its ``meta.notes`` and commit ``a324f503``:

- Per-*iteration* throughput favours the GPU at every tested size in the
  legacy lane (0.83 ms vs 1.90 ms at ``ns=35, mpol=2, ntor=2``, up to 3x on
  NuhrenbergZille-class decks: 90 s vs 277 s wall).
- The GPU pays fixed per-solve overheads (~0.2-0.4 s dispatch/transfer floor
  plus compile/cache-load on cold processes), so *small* decks that converge
  in well under a second of CPU work finish faster on the CPU
  (``solovev``: 0.043 s CPU vs 0.289 s CUDA warm wall; ``cth_like_fixed_bdy``:
  0.198 s vs 0.383 s).

The rule implemented here uses the per-iteration work proxy
``ns * mnmax * nznt`` (radial surfaces x spectral modes x angular grid — the
cost driver of the ``totzsps/tomnsps`` batched matmuls that dominate one
``funct3d`` pass):

======================================  ==========  =================  ======
deck (first NS_ARRAY stage)             work proxy  warm wall CPU/GPU  winner
======================================  ==========  =================  ======
solovev (11*6*10)                              660  0.043 / 0.289 s    cpu
nfp4_QH_warm_start (35*8*48)                13,440  0.954 / 0.574 s    gpu*
cth_like_fixed_bdy (15*5*324)               24,300  0.198 / 0.383 s    cpu
LandremanPaul2021_QA (16*128*240)          491,520  14.54 / 4.19 s     gpu
NuhrenbergZille_1988_QHS (11*162*286)      509,652  276.9 / 90.1 s     gpu
======================================  ==========  =================  ======

Below :data:`GPU_MIN_ITERATION_WORK` the measured difference is < 0.5 s
either way (the ``*`` misclassification costs ~0.4 s); above it the GPU wins
by 2-3x and increasingly more with size.  The threshold ``100_000`` sits
between the two clusters (geometric mean of 24.3e3 and 491.5e3 ~ 109e3).

The policy is a *default* only: an explicit ``device=`` argument to
``solve``/``solve_multigrid`` always wins, and when the user pinned the JAX
platform themselves (``JAX_PLATFORMS``/``JAX_PLATFORM_NAME``) the automatic
policy stands down entirely (:func:`resolve_device` returns ``None``).
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import jax

__all__ = [
    "GPU_MIN_ITERATION_WORK",
    "iteration_work",
    "recommended_device",
    "resolve_device",
    "resolve_implicit_device",
    "device_context",
]

#: Minimum ``ns * mnmax * nznt`` per-iteration work for the GPU to be the
#: recommended default (see the measured table in the module docstring).
GPU_MIN_ITERATION_WORK = 100_000


def iteration_work(resolution: Any) -> int:
    """Per-iteration work proxy ``ns * mnmax * nznt`` of a ``Resolution``."""
    return int(resolution.ns) * int(resolution.mnmax) * int(resolution.nznt)


def recommended_device(resolution: Any) -> str:
    """``"cpu"`` or ``"gpu"``: the measured-rule recommendation for one stage.

    Purely resolution-based (``benchmarks/gpu_baseline.json`` thresholds; see
    the module docstring); does **not** check what hardware is present — use
    :func:`resolve_device` for the availability- and pin-aware decision.
    """
    return "cpu" if iteration_work(resolution) < GPU_MIN_ITERATION_WORK else "gpu"


def _user_pinned_platform() -> bool:
    """True when the user pinned the JAX platform via the environment."""
    return bool(
        os.environ.get("JAX_PLATFORMS", "").strip()
        or os.environ.get("JAX_PLATFORM_NAME", "").strip()
    )


def resolve_device(device: Any, resolution: Any):
    """Map a ``device=`` argument to a concrete ``jax.Device`` (or ``None``).

    ``None`` means "leave placement alone" (no ``jax.default_device`` wrap):

    - explicit ``device`` (``"cpu"``/``"gpu"``/``"cuda"``/``"rocm"``/``"tpu"``
      or a ``jax.Device``) is always honored — missing hardware raises;
    - ``device=None`` applies :func:`recommended_device` **unless** the user
      pinned ``JAX_PLATFORMS``/``JAX_PLATFORM_NAME`` (never override an
      explicit choice), the recommended platform is not available, or the
      recommendation already matches the default backend.
    """
    if device is None:
        if _user_pinned_platform():
            return None
        kind = recommended_device(resolution)
        default = jax.default_backend()
        if kind == "gpu":
            if default != "cpu":
                return None  # already going to run on the accelerator
            try:
                return jax.devices("gpu")[0]
            except RuntimeError:
                return None  # CPU-only machine: nothing to do
        if default == "cpu":
            return None  # already on CPU
        return jax.devices("cpu")[0]
    if hasattr(device, "platform"):  # already a jax.Device
        return device
    kind = str(device).strip().lower()
    if kind in ("gpu", "cuda", "rocm"):
        return jax.devices("gpu")[0]
    if kind in ("cpu", "tpu"):
        return jax.devices(kind)[0]
    raise ValueError(
        f"unknown device {device!r}; expected 'cpu', 'gpu', 'cuda', 'rocm', "
        "'tpu' or a jax.Device"
    )


def resolve_implicit_device(device: Any, resolution: Any):
    """Device for the implicit-gradient Jacobian / adjoint GMRES (or ``None``).

    Unlike the forward solve, the ``jac="implicit"`` path builds a per-dof
    *vmapped* forward-implicit-differentiation graph — dozens of preconditioned
    GMRES solves (each with control flow), one per boundary Fourier dof — whose
    XLA compile grows with the dof count and whose evaluation is kernel-launch
    bound.  Measured on 2x RTX A4000 (R1, ``benchmarks`` notes) it is
    *slower* on the GPU than on the CPU at every optimization size tested: a
    ``max_mode=2`` QH stage (24 dofs) did not finish a single Jacobian eval in
    37 min on the GPU, versus minutes on the CPU; the forward solve itself is a
    host callback that never touches the accelerator.  So the default here is
    always the CPU:

    - an explicit ``device`` (``"cpu"``/``"gpu"``/... or a ``jax.Device``) is
      still honored (delegated to :func:`resolve_device`);
    - a user ``JAX_PLATFORMS``/``JAX_PLATFORM_NAME`` pin stands down
      (returns ``None`` — leave placement alone);
    - otherwise pin to the CPU when the default backend is an accelerator, or
      leave placement untouched (``None``) when already on the CPU.

    ``resolution`` is accepted for signature parity with :func:`resolve_device`
    (and in case a size-dependent rule is wanted later); it is unused today.
    """
    if device is not None:
        return resolve_device(device, resolution)
    if _user_pinned_platform():
        return None
    if jax.default_backend() == "cpu":
        return None
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:  # pragma: no cover - CPU device always present
        return None


def device_context(device: Any, resolution: Any):
    """Context manager placing a solve stage on the resolved device.

    Returns ``jax.default_device(dev)`` for the :func:`resolve_device` result,
    or a null context when placement should be left untouched.
    """
    dev = resolve_device(device, resolution)
    if dev is None:
        return contextlib.nullcontext()
    return jax.default_device(dev)
