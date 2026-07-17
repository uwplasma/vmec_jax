"""Unit tests for :mod:`vmex.core.device` — the CPU/GPU placement policy.

Pure host-side logic (no solves), so this is fast.  On a CPU-only runner the
GPU branches resolve to ``None`` (nothing to place); the tests assert the
decision, not the presence of an accelerator.
"""

from __future__ import annotations

import contextlib

import jax
import pytest

from vmex.core import device as dev
from vmex.core.fourier import Resolution


def _res(ns: int, mpol: int, ntor: int, nfp: int = 1) -> Resolution:
    return Resolution(mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6,
                      nzeta=(2 * ntor + 4) if ntor else 1, nfp=nfp,
                      lasym=False, ns=ns)


def test_iteration_work_and_recommendation_threshold():
    small = _res(ns=11, mpol=6, ntor=0)          # tiny tokamak-like
    big = _res(ns=101, mpol=12, ntor=12, nfp=4)  # reactor-scale 3D
    assert dev.iteration_work(small) < dev.GPU_MIN_ITERATION_WORK
    assert dev.iteration_work(big) >= dev.GPU_MIN_ITERATION_WORK
    assert dev.recommended_device(small) == "cpu"
    assert dev.recommended_device(big) == "gpu"


def test_pinned_platform_is_never_overridden(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    # even a GPU-recommended resolution is left alone when the user pinned.
    assert dev.resolve_device(None, _res(ns=201, mpol=16, ntor=16, nfp=5)) is None


def test_auto_cpu_recommendation_is_a_noop_on_cpu(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    # CPU-recommended + default backend already CPU -> nothing to place.
    if jax.default_backend() == "cpu":
        assert dev.resolve_device(None, _res(ns=11, mpol=6, ntor=0)) is None


def test_auto_gpu_recommendation_without_accelerator(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    # GPU-recommended but CPU-only machine -> None (nothing to do).
    if jax.default_backend() == "cpu":
        assert dev.resolve_device(None, _res(ns=201, mpol=16, ntor=16, nfp=5)) is None


def test_explicit_cpu_returns_a_cpu_device():
    resolved = dev.resolve_device("cpu", _res(ns=11, mpol=6, ntor=0))
    assert resolved is not None and resolved.platform == "cpu"


def test_explicit_jax_device_is_returned_unchanged():
    cpu0 = jax.devices("cpu")[0]
    assert dev.resolve_device(cpu0, _res(ns=11, mpol=6, ntor=0)) is cpu0


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="unknown device"):
        dev.resolve_device("quantum", _res(ns=11, mpol=6, ntor=0))


def test_gpu_request_on_cpu_machine_raises():
    # explicit accelerator request is honored -> missing hardware raises.
    if jax.default_backend() == "cpu":
        with pytest.raises(RuntimeError):
            dev.resolve_device("gpu", _res(ns=11, mpol=6, ntor=0))


def test_resolve_implicit_device_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    res = _res(ns=35, mpol=6, ntor=6, nfp=4)  # a QH-class stage
    resolved = dev.resolve_implicit_device(None, res)
    if jax.default_backend() == "cpu":
        # already on CPU -> nothing to place (the implicit path stays put).
        assert resolved is None
    else:
        # on an accelerator the launch-bound Jacobian is pinned to the CPU.
        assert resolved is not None and resolved.platform == "cpu"


def test_resolve_implicit_device_honors_explicit_and_pin(monkeypatch):
    res = _res(ns=35, mpol=6, ntor=6, nfp=4)
    # explicit device is honored (delegated to resolve_device).
    explicit = dev.resolve_implicit_device("cpu", res)
    assert explicit is not None and explicit.platform == "cpu"
    # a user platform pin stands the auto policy down.
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    assert dev.resolve_implicit_device(None, res) is None


def test_device_context_is_a_nullcontext_when_placement_untouched(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    ctx = dev.device_context(None, _res(ns=11, mpol=6, ntor=0))
    assert isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass


def test_device_context_wraps_default_device_for_explicit_cpu():
    ctx = dev.device_context("cpu", _res(ns=11, mpol=6, ntor=0))
    # jax.default_device(...) returns a context manager that is not nullcontext.
    assert not isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass
