"""Backend (CPU/GPU) selection policy for the core solve lanes.

Measured basis: ``benchmarks/gpu_baseline.json`` (see its ``meta.notes`` for
the per-deck timings).  Per-iteration throughput favours the GPU (up to 3x
wall on NuhrenbergZille-class decks), but the GPU pays fixed per-solve
overheads (~0.2-0.4 s dispatch/transfer floor plus compile/cache-load on
cold processes), so small decks that converge in well under a second of CPU
work finish faster on the CPU.

The rule uses the per-iteration work proxy ``ns * mnmax * nznt`` (radial
surfaces x spectral modes x angular grid — the cost driver of the
``totzsps/tomnsps`` batched matmuls that dominate one ``funct3d`` pass).
The measured decks split into two clusters: work proxies up to ~24e3 where
the CPU wins and any misclassification costs < 0.5 s either way, and
proxies >= ~490e3 where the GPU wins by 2-3x.
:data:`GPU_MIN_ITERATION_WORK` = ``100_000`` sits between them (their
geometric mean is ~109e3).

Mode count is an independent guard: the measured GPU winners have at most
162 active Fourier modes, while a high-resolution HSX deck (``mnmax=858``)
ran ~3.4x *slower* on the GPU even warm, despite a large work proxy.
:data:`GPU_MAX_SPECTRAL_MODES` = 512 sits between the largest measured GPU
winner (288 modes) and that high-mode CPU winner; the intermediate range is
not calibrated, and the cutoff is not claimed as a hardware-independent
crossover — explicit placement remains available to measure newer hardware.

The policy is a *default* only: an explicit ``device=`` argument to
``solve``/``solve_multigrid`` always wins, while ``device=None`` follows
JAX placement.  The automatic policy stands down when the user selected a
JAX default device or platform themselves.
"""

from __future__ import annotations

import contextlib
from typing import Any

import jax
import numpy as np

__all__ = [
    "AUTO",
    "GPU_MIN_ITERATION_WORK",
    "GPU_MAX_SPECTRAL_MODES",
    "iteration_work",
    "recommended_device",
    "resolve_device",
    "resolve_implicit_device",
    "resolve_mirror_device",
    "device_context",
    "device_scope",
    "mirror_device_context",
]

#: Apply VMEX's measured placement policy.  ``None`` deliberately has the
#: usual JAX meaning: do not add a placement context.
AUTO = "auto"

#: Minimum ``ns * mnmax * nznt`` per-iteration work for the GPU to be the
#: recommended default (see the measured table in the module docstring).
GPU_MIN_ITERATION_WORK = 100_000

#: Above this many active Fourier modes, the measured high-mode HSX solve is
#: faster on the host CPU even though its aggregate work proxy is large.
GPU_MAX_SPECTRAL_MODES = 512


def iteration_work(resolution: Any) -> int:
    """Per-iteration work proxy ``ns * mnmax * nznt`` of a ``Resolution``."""
    return int(resolution.ns) * int(resolution.mnmax) * int(resolution.nznt)


def recommended_device(resolution: Any) -> str:
    """``"cpu"`` or ``"gpu"``: the measured-rule recommendation for one stage.

    Purely resolution-based (the benchmark thresholds in the module
    docstring); does **not** check what hardware is present — use
    :func:`resolve_device` for the availability- and pin-aware decision.
    """
    if (
        iteration_work(resolution) < GPU_MIN_ITERATION_WORK
        or int(resolution.mnmax) > GPU_MAX_SPECTRAL_MODES
    ):
        return "cpu"
    return "gpu"


def _user_selected_placement() -> bool:
    """True when the user selected a JAX default device or platform."""
    return (
        jax.config.jax_default_device is not None
        or bool(jax.config.jax_platforms)
        or bool(jax.config.values.get("jax_platform_name"))
    )


def resolve_device(device: Any = AUTO, resolution: Any = None):
    """Map a ``device=`` argument to a concrete ``jax.Device`` (or ``None``).

    ``None`` means "leave placement alone" (no ``jax.default_device`` wrap):

    - explicit ``device`` (``"cpu"``/``"gpu"``/``"cuda"``/``"rocm"``/``"tpu"``
      or a ``jax.Device``) is always honored — missing hardware raises;
    - ``device=None`` does not intervene in JAX placement;
    - ``device="auto"`` applies :func:`recommended_device` **unless** the user
      selected an active :func:`jax.default_device` context or pinned
      ``JAX_PLATFORMS``/``JAX_PLATFORM_NAME``, the recommended platform is not
      available, or it already matches the default backend.
    """
    if device is None:
        return None
    if hasattr(device, "platform"):  # already a jax.Device
        return device
    kind = str(device).strip().lower()
    if kind == AUTO:
        if _user_selected_placement():
            return None
        if resolution is None:
            raise ValueError("resolution is required when device='auto'")
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
    if kind in ("gpu", "cuda", "rocm"):
        return jax.devices("gpu")[0]
    if kind in ("cpu", "tpu"):
        return jax.devices(kind)[0]
    raise ValueError(
        f"unknown device {device!r}; expected 'auto', None, 'cpu', 'gpu', "
        "'cuda', 'rocm', 'tpu' or a jax.Device"
    )


def resolve_implicit_device(device: Any = AUTO, resolution: Any = None):
    """Device for the implicit-gradient Jacobian / adjoint GMRES (or ``None``).

    Unlike the forward solve, the ``jac="implicit"`` path builds a per-dof
    *vmapped* forward-implicit-differentiation graph — dozens of preconditioned
    GMRES solves (each with control flow), one per boundary Fourier dof — whose
    XLA compile grows with the dof count and whose evaluation is kernel-launch
    bound.  Measured (``benchmarks`` notes), it is *slower* on the GPU than on
    the CPU at every optimization size tested: a ``max_mode=2`` QH stage
    (24 dofs) did not finish a single Jacobian eval in 37 min on the GPU,
    versus minutes on the CPU.  The forward equilibrium callback uses the
    solver's independent automatic per-stage policy; this resolver controls
    only the residual/Jacobian work.  So the default here is always the CPU:

    - explicit devices are honored (delegated to :func:`resolve_device`);
    - ``None`` leaves placement to JAX;
    - ``"auto"`` stands down for an active JAX device/platform selection and
      otherwise pins to CPU on an accelerator backend.

    ``resolution`` is accepted for signature parity with :func:`resolve_device`
    (and in case a size-dependent rule is wanted later); it is unused today.
    """
    if device is None:
        return None
    if not (isinstance(device, str) and device.strip().lower() == AUTO):
        return resolve_device(device, resolution)
    if _user_selected_placement():
        return None
    if jax.default_backend() == "cpu":
        return None
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:  # pragma: no cover - CPU device always present
        return None


def resolve_mirror_device(device: Any = AUTO):
    """Device for mirror solves with host SciPy control flow.

    The mirror fixed/free-boundary solvers repeatedly cross between SciPy and
    exact JAX value/JVP/VJP callbacks.  The measured ``15x15`` case is faster
    on CPU (35.2 s versus 44.2 s on GPU), so ``"auto"`` selects CPU unless
    the user has chosen a JAX placement.  Explicit devices and ``None``
    retain the same meanings as in :func:`resolve_device`.
    """
    if device is None:
        return None
    if not (isinstance(device, str) and device.strip().lower() == AUTO):
        return resolve_device(device)
    if _user_selected_placement() or jax.default_backend() == "cpu":
        return None
    return jax.devices("cpu")[0]


def device_context(device: Any = AUTO, resolution: Any = None):
    """Context manager placing a solve stage on the resolved device.

    Returns ``jax.default_device(dev)`` for the :func:`resolve_device` result,
    or a null context when placement should be left untouched.
    """
    dev = resolve_device(device, resolution)
    if dev is None:
        return contextlib.nullcontext()
    return jax.default_device(dev)


def device_scope(device: Any):
    """Hold ``jax.default_device(device)`` around building AND executing a
    JAX transformation — the belt-and-braces supported path for running a
    whole program (forward solves, ``jax.grad``/``jax.value_and_grad``/
    ``jax.jacrev`` over :func:`vmex.core.implicit.run`, optimization
    drivers) on an explicit non-default device::

        gpu1 = jax.devices("gpu")[1]
        with device_scope(gpu1):
            p0 = im.params_from_input(inp, device=gpu1)
            grad = jax.grad(lambda p: im.run(inp, p, device=gpu1).wb)(p0)

    Raw ``jax.grad`` without the scope is supported too: the implicit module
    binds its own stages to the carried config device internally — the
    ``pure_callback`` host solve, the cached runtime template, the custom-VJP
    boundary pins, and the host-eager adjoint Krylov solve (see
    ``vmex.core.implicit._adjoint_solve_gcrot``).  The scope is the *robust*
    path on top of that: it additionally steers every caller-side eager
    constant and JAX's own transformation machinery — placement layers
    outside vmex's control — so it is recommended whenever an entire
    workflow should live on one non-default device.

    Accepts ``"cpu"``/``"gpu"``/``"cuda"``/``"rocm"``/``"tpu"`` or a
    ``jax.Device``; ``None`` returns a null context (leave placement to
    JAX).  ``"auto"`` is rejected — the scope exists precisely to express an
    explicit placement choice.
    """
    if device is None:
        return contextlib.nullcontext()
    if isinstance(device, str) and device.strip().lower() == AUTO:
        raise ValueError(
            "device_scope needs an explicit device ('cpu', 'gpu', a "
            "jax.Device, ...); 'auto' expresses no placement to hold"
        )
    dev = resolve_device(device)
    if dev is None:  # pragma: no cover - only device=None maps to None here
        return contextlib.nullcontext()
    return jax.default_device(dev)


def mirror_device_context(device: Any = AUTO):
    """Context manager applying :func:`resolve_mirror_device`."""
    dev = resolve_mirror_device(device)
    if dev is None:
        return contextlib.nullcontext()
    return jax.default_device(dev)


def _placement_device(device: Any = AUTO, resolution: Any = None):
    """Concrete target for already-committed input arrays, or ``None``."""
    dev = resolve_device(device, resolution)
    if dev is not None or device is None:
        return dev
    configured = jax.config.jax_default_device
    return configured if configured is not None else jax.devices()[0]


def _mirror_placement_device(device: Any = AUTO):
    """Concrete mirror target for committed input arrays, or ``None``."""
    dev = resolve_mirror_device(device)
    if dev is not None or device is None:
        return dev
    configured = jax.config.jax_default_device
    return configured if configured is not None else jax.devices()[0]


def _put_numeric_leaves(value: Any, device: Any):
    """Move registered-pytree array leaves while preserving metadata/objects."""
    if value is None or device is None:
        return value
    return jax.tree.map(
        lambda leaf: jax.device_put(leaf, device)
        if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
        value,
    )
