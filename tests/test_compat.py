"""Unit tests for :mod:`vmec_jax._compat` (JAX environment + cache policy).

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

from vmec_jax import _compat


# ---------------------------------------------------------------------------
# compilation-cache directory policy
# ---------------------------------------------------------------------------

_CACHE_VARS = (
    "JAX_COMPILATION_CACHE_DIR", "VMEC_JAX_COMPILATION_CACHE_DIR",
    "VMEC_JAX_COMPILATION_CACHE", "JAX_PLATFORM_NAME", "JAX_PLATFORMS",
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
    assert path is not None and "vmec_jax" in path and "jax_cache" in path

    # explicit JAX var wins verbatim; 'disabled' turns it off
    mp.setenv("JAX_COMPILATION_CACHE_DIR", "/tmp/jaxcache")
    assert _compat._default_compilation_cache_dir() == "/tmp/jaxcache"
    mp.setenv("JAX_COMPILATION_CACHE_DIR", "disabled")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("JAX_COMPILATION_CACHE_DIR")

    # vmec-specific dir override
    mp.setenv("VMEC_JAX_COMPILATION_CACHE_DIR", "/tmp/vmeccache")
    assert _compat._default_compilation_cache_dir() == "/tmp/vmeccache"
    mp.setenv("VMEC_JAX_COMPILATION_CACHE_DIR", "no")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("VMEC_JAX_COMPILATION_CACHE_DIR")

    # forced on CPU -> machine-scoped default under ~/.cache
    mp.setenv("VMEC_JAX_COMPILATION_CACHE", "1")
    path = _compat._default_compilation_cache_dir()
    assert path is not None and "vmec_jax" in path and "jax_cache" in path
    mp.setenv("VMEC_JAX_COMPILATION_CACHE", "off")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("VMEC_JAX_COMPILATION_CACHE")

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
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "2.5")
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "1024")
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "100000")
    monkeypatch.setenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "1")
    monkeypatch.setenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", "all")

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


def test_configure_compilation_cache_gpu_autotune_default(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", raising=False)
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
