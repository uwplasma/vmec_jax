"""JAX environment defaults + persistent compilation-cache policy.

The old JAX/NumPy backend shim is gone (the core is JAX-only); what remains —
and is actually used — is:

- :func:`_configure_jax_environment` (run at import, i.e. before
  ``vmex/__init__`` does ``import jax``): environment defaults that must
  be set before JAX/XLA initializes — float64 (``JAX_ENABLE_X64``, VMEC
  parity), synchronous CPU dispatch, quiet XLA/PjRt C++ logging, GPU
  demand allocation, the machine-scoped persistent compilation-cache
  directory, and the XLA:CPU fast-compile flags;
- the compilation-cache policy helpers
  :func:`_default_compilation_cache_dir` / :func:`_cache_machine_fingerprint`
  / :func:`_configure_compilation_cache`, consumed by ``vmex/__init__``
  and re-applied by ``core.solver._harden_compilation_cache`` on every solve
  path (namespace-package shadowing guard).
"""

from __future__ import annotations

from typing import Any
import hashlib
from importlib import metadata as importlib_metadata
import sys

import os
import platform


_CACHE_FORMAT_VERSION = "2"
_DEFAULT_CACHE_MAX_SIZE = 1 << 30


def _env(name: str, default: str = "") -> str:
    """Read ``VMEX_<name>``, falling back to the legacy ``VMEC_JAX_<name>``.

    The package was renamed vmec_jax -> vmex; environment variables a user may
    have set in their shell profile (the ``*_COMPILATION_CACHE*`` knobs in
    particular) keep working under their old names for one release.
    """
    val = os.environ.get(f"VMEX_{name}")
    if val is not None:
        return val
    return os.environ.get(f"VMEC_JAX_{name}", default)


def _cache_machine_fingerprint() -> str:
    """Return a short cache key for host-specific XLA CPU executables.

    XLA CPU persistent-cache entries are native executables.  On shared home
    directories, reusing an entry compiled on another CPU can trigger XLA AOT
    loader errors or even illegal-instruction failures.  The fingerprint keeps
    vmex's default cache portable by separating entries by OS, machine, and
    CPU-feature/model signature.  Users who deliberately want a shared cache can
    still set ``VMEX_COMPILATION_CACHE_DIR`` or ``JAX_COMPILATION_CACHE_DIR``.
    """

    parts = [
        f"vmex-cache={_CACHE_FORMAT_VERSION}",
        platform.system(),
        platform.machine(),
        platform.processor(),
        f"python={sys.version_info.major}.{sys.version_info.minor}",
    ]
    for package in ("jax", "jaxlib"):
        try:
            parts.append(f"{package}={importlib_metadata.version(package)}")
        except Exception:
            pass
    try:
        if os.path.exists("/proc/cpuinfo"):
            wanted = ("model name", "cpu family", "model", "stepping", "flags", "Features")
            seen: set[str] = set()
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if ":" not in line:
                        continue
                    key, value = (part.strip() for part in line.split(":", 1))
                    if key in wanted and key not in seen:
                        parts.append(f"{key}={value}")
                        seen.add(key)
    except Exception:
        pass
    # macOS has no /proc/cpuinfo — capture the CPU brand + microarchitecture via
    # sysctl so Intel/Apple-Silicon (and different chip generations) never share
    # an XLA:CPU AOT cache entry.
    if platform.system() == "Darwin":
        try:
            import subprocess
            for key in ("machdep.cpu.brand_string", "hw.optional.arm.FEAT_SME",
                        "hw.cpufamily"):
                out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                     text=True, timeout=2)
                if out.returncode == 0 and out.stdout.strip():
                    parts.append(f"{key}={out.stdout.strip()}")
        except Exception:
            pass
    if not any(str(part).strip() for part in parts[:3]):
        try:
            parts.append(platform.node())
        except Exception:
            pass
    digest = hashlib.sha256("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    return f"{system}-{machine}-{digest}"


def _default_compilation_cache_dir() -> str | None:
    """Return the configured JAX compilation-cache directory.

    The persistent cache is enabled **by default on every backend** (CPU too)
    so repeated cold-process CLI/API runs reuse compiled kernels instead of
    recompiling (a solovev CLI rerun drops 4.3 s -> 1.2 s).  The XLA:CPU
    host-feature-mismatch hazard (AOT executables tied to a specific
    instruction set, dangerous on shared home filesystems) is handled by
    :func:`_cache_machine_fingerprint`, so heterogeneous machines never share
    a cache entry.  Opt out with ``VMEX_COMPILATION_CACHE=disabled`` (or
    ``VMEX_COMPILATION_CACHE_DIR=disabled``); point it elsewhere with
    ``JAX_COMPILATION_CACHE_DIR=/path``.
    """
    # Already set by the user — respect it.
    if "JAX_COMPILATION_CACHE_DIR" in os.environ:
        val = os.environ["JAX_COMPILATION_CACHE_DIR"].strip()
        if val.lower() in ("", "disabled", "0", "false", "no"):
            return None
        return val

    # User can opt out via VMEX_COMPILATION_CACHE_DIR=disabled
    vmec_val = _env("COMPILATION_CACHE_DIR").strip()
    if vmec_val.lower() in ("disabled", "0", "false", "no"):
        return None
    if vmec_val:
        return vmec_val

    cache_flag = _env("COMPILATION_CACHE").strip().lower()
    if cache_flag in ("disabled", "0", "false", "no", "off"):
        return None

    # Default: ~/.cache/vmex/jax_cache/<machine-fingerprint> (see
    # _cache_machine_fingerprint for the XLA:CPU AOT-reuse hazard).
    try:
        import pathlib
        return str(
            pathlib.Path.home()
            / ".cache"
            / "vmex"
            / "jax_cache"
            / _cache_machine_fingerprint()
        )
    except Exception:
        return None


def _configure_compilation_cache(jax_module: Any, cache_dir: str | None) -> None:
    """Apply vmex's persistent-cache defaults to an imported JAX module."""
    if cache_dir is None:
        return
    try:
        jax_module.config.update("jax_enable_compilation_cache", True)
    except Exception:
        pass
    try:
        jax_module.config.update("jax_compilation_cache_dir", cache_dir)
    except Exception:
        pass
    try:
        # Keep JAX's useful default: tiny elementwise kernels are cheaper to
        # rebuild than to store and were creating tens of thousands of files.
        min_compile = _env("CACHE_MIN_COMPILE_TIME_SECS", "1")
        jax_module.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile))
    except Exception:
        pass
    try:
        min_entry = _env("CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
        jax_module.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry))
    except Exception:
        pass
    try:
        xla_caches = _env("PERSISTENT_CACHE_XLA_CACHES").strip()
        if not xla_caches:
            platform_name = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
            platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
            visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower()
            gpu_requested = (
                platform_name in ("gpu", "cuda")
                or any(part.strip() in ("gpu", "cuda") for part in platforms.split(","))
                or visible_cuda not in ("", "-1", "none", "no")
            )
            xla_caches = "xla_gpu_per_fusion_autotune_cache_dir" if gpu_requested else "none"
        if xla_caches.lower() not in ("", "none", "0", "false", "no", "off"):
            jax_module.config.update("jax_persistent_cache_enable_xla_caches", xla_caches)
    except Exception:
        pass
    try:
        # JAX's file cache takes its cross-process lock only when eviction is
        # enabled.  A finite default therefore prevents concurrent VMEX runs
        # from writing the same executable at once, as well as bounding disk.
        max_size = _env("COMPILATION_CACHE_MAX_SIZE", str(_DEFAULT_CACHE_MAX_SIZE))
        if max_size:
            jax_module.config.update("jax_compilation_cache_max_size", int(max_size))
    except Exception:
        pass
    try:
        explain = _env("EXPLAIN_CACHE_MISSES")
        if explain.strip().lower() not in ("", "0", "false", "no"):
            jax_module.config.update("jax_explain_cache_misses", True)
    except Exception:
        pass


def _configure_jax_environment() -> None:
    """Set JAX/XLA environment defaults, then import + configure JAX.

    Runs once at ``vmex._compat`` import time — before
    ``vmex/__init__`` (or anything else in the package) imports JAX — so
    the env-var defaults reliably reach XLA backend initialization.  Every
    default uses ``setdefault``: an explicit user environment always wins.
    """
    try:
        # Enable x64 by default for VMEC parity unless the user opted out.
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        # VMEC/JAX optimization callbacks immediately materialize most results
        # on the host (SciPy residuals/Jacobians, history, wout writing).  On
        # CPU, asynchronous dispatch can leave completed XLA/PjRt work and
        # executable state queued across many exact-Jacobian callbacks in one
        # long-lived process.  Default CPU dispatch to synchronous execution so
        # memory is reclaimed at callback boundaries; users can still override
        # this before import with JAX_CPU_ENABLE_ASYNC_DISPATCH=true.
        os.environ.setdefault("JAX_CPU_ENABLE_ASYNC_DISPATCH", "false")
        # Suppress harmless informational C++ logs from XLA/PjRt (e.g.
        # repeated "Assume version compatibility..." on persistent-cache
        # hits).  Level 0=INFO, 1=WARNING, 2=ERROR — default to ERROR-only so
        # genuine errors still surface.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("GLOG_minloglevel", "2")
        # JAX's default GPU allocator preallocates most device memory.  That
        # hurts vmex's exact-optimizer workload in practice: it prevents
        # concurrent profiling/worker processes from starting and can make the
        # accepted-point replay path much slower.  Default to demand allocation
        # unless the user already set JAX's allocator env var or explicitly
        # asks vmex to keep JAX's preallocation default.
        _vmec_gpu_prealloc = _env("GPU_PREALLOCATE").strip().lower()
        if (
            "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
            and _vmec_gpu_prealloc not in ("1", "true", "yes", "on")
        ):
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        # Enable the JAX disk compilation cache in a machine-scoped directory
        # (see _default_compilation_cache_dir for the AOT-reuse hazard).
        _cache_dir = _default_compilation_cache_dir()
        if _cache_dir is not None:
            os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)

        # XLA:CPU compile-time flags.  The differentiable/optimization pipeline
        # is COMPILE-dominated (the fused adjoint VJP + GMRES graph dominates a
        # cold ``value_and_grad``); backend optimization level 1 plus disabling
        # expensive LLVM passes cuts compile wall-time ~1.3-2x at the cost of
        # slightly slower *warm* kernels.  Applied on CPU only (LLVM codegen),
        # never with fast-math (that would break float64 parity/determinism),
        # skipped if the user set XLA_FLAGS, and opt-in via
        # VMEX_FAST_COMPILE=1.  Pre-import environment hints cannot reliably
        # distinguish a normally discovered GPU installation, so VMEX must
        # not inject CPU-only XLA flags by default.
        _fast_compile = _env("FAST_COMPILE", "0").strip().lower()
        _accel_req = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
        _accel_reqs = os.environ.get("JAX_PLATFORMS", "").strip().lower()
        _cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        _on_accel = (
            any(a in f"{_accel_req} {_accel_reqs}" for a in ("cuda", "gpu", "tpu", "rocm"))
            or (_cuda_vis not in ("", "-1"))
        )
        if (
            _fast_compile not in ("0", "false", "no", "off")
            and "XLA_FLAGS" not in os.environ
            and not _on_accel
        ):
            os.environ["XLA_FLAGS"] = (
                "--xla_backend_optimization_level=1 "
                "--xla_llvm_disable_expensive_passes=true"
            )

        import jax

        try:
            jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "0") == "1")
        except Exception:
            pass
        try:
            _cpu_async = os.environ.get("JAX_CPU_ENABLE_ASYNC_DISPATCH", "true")
            jax.config.update(
                "jax_cpu_enable_async_dispatch",
                _cpu_async.strip().lower() not in ("0", "false", "no", "off"),
            )
        except Exception:
            pass

        # Wire up the compilation cache via jax.config too; the env-var path
        # alone does not cover all JAX/JAXLIB versions and cache thresholds.
        _configure_compilation_cache(jax, _cache_dir)
    except Exception:
        # Never block a vmex import over environment tuning (e.g. docs
        # builds with a mocked JAX): core.solver enforces the hard
        # requirements (x64, cache hardening) on every solve path anyway.
        pass


_configure_jax_environment()
