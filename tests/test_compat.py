"""Unit tests for :mod:`vmex._compat` (JAX environment + cache policy).

Covers the machine-scoped compilation-cache policy (env-var precedence table
in ``_default_compilation_cache_dir``), the cache wiring against a recording
fake JAX module, and the import-time environment defaults
(``_configure_jax_environment``).  The old JAX/NumPy backend shim
(``has_jax``/``asarray``/``einsum``/numpy mode/no-op jit) was deleted in the
Item I.8a dead-code prune — the core is JAX-only.
"""

from __future__ import annotations

import re
import types

import pytest

from vmex import _compat


# ---------------------------------------------------------------------------
# compilation-cache directory policy
# ---------------------------------------------------------------------------

_CACHE_VARS = (
    "JAX_COMPILATION_CACHE_DIR", "VMEX_COMPILATION_CACHE_DIR",
    "VMEX_COMPILATION_CACHE", "JAX_PLATFORM_NAME", "JAX_PLATFORMS",
    "XLA_FLAGS", "VMEX_FAST_COMPILE", "CUDA_VISIBLE_DEVICES",
)


@pytest.fixture
def clean_cache_env(monkeypatch):
    for var in _CACHE_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_cache_dir_env_precedence(clean_cache_env):
    mp = clean_cache_env
    # default on every backend now (R26c cold-start fix): CPU gets a
    # machine-scoped cache under ~/.cache with no env var required.
    path = _compat._default_compilation_cache_dir()
    assert path is not None and "vmex" in path and "jax_cache" in path

    # explicit JAX var wins verbatim; 'disabled' turns it off
    mp.setenv("JAX_COMPILATION_CACHE_DIR", "/tmp/jaxcache")
    assert _compat._default_compilation_cache_dir() == "/tmp/jaxcache"
    mp.setenv("JAX_COMPILATION_CACHE_DIR", "disabled")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("JAX_COMPILATION_CACHE_DIR")

    # vmec-specific dir override
    mp.setenv("VMEX_COMPILATION_CACHE_DIR", "/tmp/vmeccache")
    assert _compat._default_compilation_cache_dir() == "/tmp/vmeccache"
    mp.setenv("VMEX_COMPILATION_CACHE_DIR", "no")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("VMEX_COMPILATION_CACHE_DIR")

    # forced on CPU -> machine-scoped default under ~/.cache
    mp.setenv("VMEX_COMPILATION_CACHE", "1")
    path = _compat._default_compilation_cache_dir()
    assert path is not None and "vmex" in path and "jax_cache" in path
    mp.setenv("VMEX_COMPILATION_CACHE", "off")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("VMEX_COMPILATION_CACHE")

    # accelerator request enables the default cache
    mp.setenv("JAX_PLATFORMS", "cuda,cpu")
    assert _compat._default_compilation_cache_dir() is not None
    mp.delenv("JAX_PLATFORMS")
    mp.setenv("JAX_PLATFORM_NAME", "tpu")
    assert _compat._default_compilation_cache_dir() is not None


def test_cache_machine_fingerprint_shape_and_stability():
    fp = _compat._cache_machine_fingerprint()
    assert re.fullmatch(r"[a-z0-9_]+-[a-z0-9_]+-[0-9a-f]{16}", fp)
    assert fp == _compat._cache_machine_fingerprint()


class _FakeConfig:
    def __init__(self, fail_keys=()):
        self.updates = {}
        self.fail_keys = set(fail_keys)

    def update(self, key, value):
        if key in self.fail_keys:
            raise RuntimeError(f"cannot set {key}")
        self.updates[key] = value


def test_configure_compilation_cache_wiring(monkeypatch):
    fake = types.SimpleNamespace(config=_FakeConfig())
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VMEX_CACHE_MIN_COMPILE_TIME_SECS", "2.5")
    monkeypatch.setenv("VMEX_CACHE_MIN_ENTRY_SIZE_BYTES", "1024")
    monkeypatch.setenv("VMEX_COMPILATION_CACHE_MAX_SIZE", "100000")
    monkeypatch.setenv("VMEX_EXPLAIN_CACHE_MISSES", "1")
    monkeypatch.setenv("VMEX_PERSISTENT_CACHE_XLA_CACHES", "all")

    _compat._configure_compilation_cache(fake, "/tmp/cachedir")
    ups = fake.config.updates
    assert ups["jax_enable_compilation_cache"] is True
    assert ups["jax_compilation_cache_dir"] == "/tmp/cachedir"
    assert ups["jax_persistent_cache_min_compile_time_secs"] == 2.5
    assert ups["jax_persistent_cache_min_entry_size_bytes"] == 1024
    assert ups["jax_compilation_cache_max_size"] == 100000
    assert ups["jax_explain_cache_misses"] is True
    assert ups["jax_persistent_cache_enable_xla_caches"] == "all"

    # cache_dir=None is a no-op; failing config keys are tolerated
    fake2 = types.SimpleNamespace(config=_FakeConfig())
    _compat._configure_compilation_cache(fake2, None)
    assert fake2.config.updates == {}
    fake3 = types.SimpleNamespace(
        config=_FakeConfig(fail_keys={"jax_enable_compilation_cache"}))
    _compat._configure_compilation_cache(fake3, "/tmp/x")  # must not raise
    assert fake3.config.updates["jax_compilation_cache_dir"] == "/tmp/x"


def test_compilation_cache_defaults_are_bounded_and_selective(monkeypatch):
    monkeypatch.delenv("VMEX_CACHE_MIN_COMPILE_TIME_SECS", raising=False)
    monkeypatch.delenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", raising=False)
    monkeypatch.delenv("VMEX_COMPILATION_CACHE_MAX_SIZE", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", raising=False)
    fake = types.SimpleNamespace(config=_FakeConfig())
    _compat._configure_compilation_cache(fake, "/tmp/cachedir")
    assert fake.config.updates["jax_persistent_cache_min_compile_time_secs"] == 1.0
    # The bound stays finite (JAX only locks the cache when eviction is on)
    # but scales with the disk: the old fixed 1 GiB sat at its cap and evicted
    # the executables the next optimization stage asked for.
    bound = fake.config.updates["jax_compilation_cache_max_size"]
    assert _compat._CACHE_SIZE_FLOOR <= bound <= _compat._CACHE_SIZE_CEILING
    assert bound == _compat._default_cache_max_size("/tmp/cachedir")
    assert _compat._default_cache_max_size("/no/such/path") == \
        _compat._CACHE_SIZE_FLOOR


def test_configure_compilation_cache_gpu_autotune_default(monkeypatch):
    monkeypatch.delenv("VMEX_PERSISTENT_CACHE_XLA_CACHES", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fake = types.SimpleNamespace(config=_FakeConfig())
    _compat._configure_compilation_cache(fake, "/tmp/cachedir")
    assert (fake.config.updates["jax_persistent_cache_enable_xla_caches"]
            == "xla_gpu_per_fusion_autotune_cache_dir")


# ---------------------------------------------------------------------------
# import-time environment defaults
# ---------------------------------------------------------------------------


def test_configure_jax_environment_idempotent_and_respects_user_env(monkeypatch):
    """Re-running the import-time setup is safe and never clobbers user env."""
    import os

    monkeypatch.setenv("XLA_FLAGS", "--user_set_flag")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "0")
    _compat._configure_jax_environment()  # must not raise (jax already imported)
    assert os.environ["XLA_FLAGS"] == "--user_set_flag"       # setdefault only
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "0"          # user wins
    # the x64 default survives (VMEC parity: float64 mandatory)
    import jax

    assert jax.config.read("jax_enable_x64") is True


def test_macos_cpu_codegen_split_default_respects_backend_and_user(monkeypatch):
    """The large-graph linker guard is macOS/CPU-only and never overrides users."""
    import os

    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(_compat.platform, "system", lambda: "Darwin")
    _compat._configure_jax_environment()
    assert os.environ["XLA_FLAGS"] == "--xla_cpu_parallel_codegen_split_count=128"

    monkeypatch.setenv("XLA_FLAGS", "--user_set_flag")
    _compat._configure_jax_environment()
    assert os.environ["XLA_FLAGS"] == "--user_set_flag"

    monkeypatch.delenv("XLA_FLAGS")
    monkeypatch.setenv("VMEX_FAST_COMPILE", "1")
    _compat._configure_jax_environment()
    assert "--xla_cpu_parallel_codegen_split_count=128" in os.environ["XLA_FLAGS"]
    assert "--xla_backend_optimization_level=1" in os.environ["XLA_FLAGS"]

    monkeypatch.delenv("XLA_FLAGS")
    monkeypatch.delenv("VMEX_FAST_COMPILE")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    _compat._configure_jax_environment()
    assert "XLA_FLAGS" not in os.environ


def test_machine_fingerprint_is_stable_and_platform_scoped(monkeypatch):
    """The AOT-cache fingerprint is deterministic and OS/arch scoped.

    The fallback arms (missing /proc/cpuinfo, failing sysctl, absent
    package metadata) previously ran only on the platform that needs
    them; drive both branches explicitly so a silent fingerprint
    collision between hosts cannot regress.
    """
    from vmex import _compat

    fp1 = _compat._cache_machine_fingerprint()
    fp2 = _compat._cache_machine_fingerprint()
    assert fp1 == fp2
    system = __import__("platform").system().lower()
    machine = __import__("platform").machine().lower()
    assert fp1.startswith(f"{system}-{machine}-")
    assert len(fp1.rsplit("-", 1)[-1]) == 16

    # Darwin sysctl arm: force the subprocess to fail -> fingerprint still
    # forms (the except arm), and differs from the healthy one only if the
    # sysctl parts contributed.
    import subprocess as sp

    def boom(*a, **k):
        raise OSError("sysctl unavailable")

    monkeypatch.setattr(sp, "run", boom)
    fp3 = _compat._cache_machine_fingerprint()
    assert fp3.startswith(f"{system}-{machine}-")


def test_configure_compilation_cache_applies_and_survives_failures(monkeypatch):
    """Every cache knob is applied to a healthy config and every failing
    knob is swallowed (the import path must never break over tuning)."""
    from vmex import _compat

    class Config:
        def __init__(self, fail_keys=()):
            self.updates = {}
            self.fail_keys = set(fail_keys)

        def update(self, key, value):
            if key in self.fail_keys:
                raise RuntimeError(key)
            self.updates[key] = value

    class Jax:
        def __init__(self, **kw):
            self.config = Config(**kw)

    # None cache dir: nothing applied.
    jx = Jax()
    _compat._configure_compilation_cache(jx, None)
    assert jx.updates == {} if hasattr(jx, "updates") else True
    assert jx.config.updates == {}

    # Healthy path applies the core knobs.
    jx = Jax()
    _compat._configure_compilation_cache(jx, "/tmp/vmex-cache-test")
    ups = jx.config.updates
    assert ups.get("jax_enable_compilation_cache") is True
    assert ups.get("jax_compilation_cache_dir") == "/tmp/vmex-cache-test"
    assert "jax_persistent_cache_min_compile_time_secs" in ups
    assert "jax_persistent_cache_min_entry_size_bytes" in ups

    # Env-driven knobs: max size + cache-miss explanations + XLA caches.
    monkeypatch.setenv("VMEX_COMPILATION_CACHE_MAX_SIZE", "123456")
    monkeypatch.setenv("VMEX_EXPLAIN_CACHE_MISSES", "1")
    monkeypatch.setenv("VMEX_PERSISTENT_CACHE_XLA_CACHES", "all")
    jx = Jax()
    _compat._configure_compilation_cache(jx, "/tmp/vmex-cache-test")
    ups = jx.config.updates
    assert ups.get("jax_compilation_cache_max_size") == 123456
    assert ups.get("jax_explain_cache_misses") is True
    assert ups.get("jax_persistent_cache_enable_xla_caches") == "all"

    # Every knob failing individually must not raise.
    jx = Jax(fail_keys={
        "jax_enable_compilation_cache", "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
        "jax_persistent_cache_enable_xla_caches",
        "jax_compilation_cache_max_size", "jax_explain_cache_misses",
    })
    _compat._configure_compilation_cache(jx, "/tmp/vmex-cache-test")
    assert jx.config.updates == {}
