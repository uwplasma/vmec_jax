"""Unit tests for :mod:`vmex.core.device` — the CPU/GPU placement policy.

Pure host-side logic (no solves), so this is fast.  On a CPU-only runner the
GPU branches resolve to ``None`` (nothing to place); the tests assert the
decision, not the presence of an accelerator.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from vmex.core import device as dev
from vmex.core import freeboundary, multigrid, solver
from vmex.core.fourier import Resolution


def _res(ns: int, mpol: int, ntor: int, nfp: int = 1) -> Resolution:
    return Resolution(mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6,
                      nzeta=(2 * ntor + 4) if ntor else 1, nfp=nfp,
                      lasym=False, ns=ns)


def test_iteration_work_and_recommendation_threshold():
    small = _res(ns=11, mpol=6, ntor=0)          # tiny tokamak-like
    big = _res(ns=101, mpol=12, ntor=12, nfp=4)  # reactor-scale 3D
    high_mode = _res(ns=101, mpol=18, ntor=24, nfp=4)
    assert dev.iteration_work(small) < dev.GPU_MIN_ITERATION_WORK
    assert dev.iteration_work(big) >= dev.GPU_MIN_ITERATION_WORK
    assert high_mode.mnmax > dev.GPU_MAX_SPECTRAL_MODES
    assert dev.recommended_device(small) == "cpu"
    assert dev.recommended_device(big) == "gpu"
    assert dev.recommended_device(high_mode) == "cpu"


@pytest.mark.parametrize("key", ["jax_platforms", "jax_platform_name"])
def test_selected_jax_platform_is_never_overridden(key):
    previous = jax.config.values.get(key)
    try:
        jax.config.update(key, "cpu")
        # Even a GPU-recommended resolution is left alone when JAX is pinned.
        assert dev.resolve_device(
            dev.AUTO, _res(ns=201, mpol=16, ntor=16, nfp=5)
        ) is None
    finally:
        jax.config.update(key, previous)


def test_none_leaves_placement_to_jax(monkeypatch):
    monkeypatch.setattr(dev, "recommended_device", lambda _: pytest.fail("auto policy ran"))
    assert dev.resolve_device(None, _res(ns=11, mpol=6, ntor=0)) is None
    assert dev.resolve_implicit_device(None, None) is None
    assert dev.resolve_mirror_device(None) is None


def test_omitted_device_uses_auto_policy(monkeypatch):
    seen = []
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setattr(dev, "recommended_device", lambda res: seen.append(res) or "cpu")
    dev.resolve_device(resolution=_res(ns=11, mpol=6, ntor=0))
    assert len(seen) == 1


def test_auto_without_resolution_has_a_clear_error():
    with pytest.raises(ValueError, match="resolution is required"):
        dev.resolve_device()
    with pytest.raises(ValueError, match="resolution is required"):
        dev.device_context()


def test_default_device_context_is_never_overridden(monkeypatch):
    # Pretend the process default backend is an accelerator: absent the active
    # CPU context, AUTO would explicitly return a CPU device for this deck.
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with jax.default_device(jax.devices("cpu")[0]):
        assert dev.resolve_device(dev.AUTO, _res(ns=11, mpol=6, ntor=0)) is None
        assert dev.resolve_implicit_device(dev.AUTO, None) is None
        assert dev.resolve_mirror_device(dev.AUTO) is None


def test_auto_cpu_recommendation_is_a_noop_on_cpu(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    # CPU-recommended + default backend already CPU -> nothing to place.
    if jax.default_backend() == "cpu":
        assert dev.resolve_device(dev.AUTO, _res(ns=11, mpol=6, ntor=0)) is None


def test_synthesis_policy_uses_measured_hardware_defaults(monkeypatch):
    resolution = _res(ns=101, mpol=18, ntor=24, nfp=4)
    target = SimpleNamespace(platform="cpu")
    monkeypatch.setattr(solver, "_placement_device", lambda *_: target)
    monkeypatch.setattr(solver.platform, "machine", lambda: "x86_64")
    assert not solver._resolve_use_fft(None, "cpu", resolution)
    monkeypatch.setattr(solver.platform, "machine", lambda: "arm64")
    assert solver._resolve_use_fft(None, "cpu", resolution)
    target.platform = "gpu"
    assert solver._resolve_use_fft(None, "gpu", resolution)
    assert not solver._resolve_use_fft(
        None, "gpu", _res(ns=50, mpol=8, ntor=8, nfp=2)
    )
    assert solver._resolve_use_fft(False, "gpu", resolution) is False
    assert solver._resolve_use_fft(True, "cpu", resolution) is True


def test_synthesis_policy_respects_active_default_device(monkeypatch):
    monkeypatch.setattr(solver.platform, "machine", lambda: "x86_64")
    resolution = _res(ns=101, mpol=18, ntor=24, nfp=4)
    with jax.default_device(jax.devices("cpu")[0]):
        assert not solver._resolve_use_fft(None, None, resolution)


def test_auto_gpu_recommendation_without_accelerator(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    # GPU-recommended but CPU-only machine -> None (nothing to do).
    if jax.default_backend() == "cpu":
        assert dev.resolve_device(dev.AUTO, _res(ns=201, mpol=16, ntor=16, nfp=5)) is None


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
    resolved = dev.resolve_implicit_device(dev.AUTO, res)
    if jax.default_backend() == "cpu":
        # already on CPU -> nothing to place (the implicit path stays put).
        assert resolved is None
    else:
        # on an accelerator the launch-bound Jacobian is pinned to the CPU.
        assert resolved is not None and resolved.platform == "cpu"


def test_resolve_implicit_device_honors_explicit_and_pin():
    res = _res(ns=35, mpol=6, ntor=6, nfp=4)
    # explicit device is honored (delegated to resolve_device).
    explicit = dev.resolve_implicit_device("cpu", res)
    assert explicit is not None and explicit.platform == "cpu"
    # A JAX platform pin stands the auto policy down.
    previous = jax.config.jax_platforms
    try:
        jax.config.update("jax_platforms", "cpu")
        assert dev.resolve_implicit_device(dev.AUTO, res) is None
    finally:
        jax.config.update("jax_platforms", previous)


def test_resolve_mirror_device_defaults_to_cpu_and_honors_explicit():
    resolved = dev.resolve_mirror_device(dev.AUTO)
    if jax.default_backend() == "cpu":
        assert resolved is None
    else:
        assert resolved is not None and resolved.platform == "cpu"
    explicit = dev.resolve_mirror_device("cpu")
    assert explicit is not None and explicit.platform == "cpu"


def test_device_context_is_a_nullcontext_when_placement_untouched(monkeypatch):
    monkeypatch.setattr(dev, "_user_selected_placement", lambda: True)
    res = _res(ns=11, mpol=6, ntor=0)
    for choice in (None, dev.AUTO):
        ctx = dev.device_context(choice, res)
        assert isinstance(ctx, contextlib.nullcontext)
        with ctx:
            pass


def test_device_context_wraps_default_device_for_explicit_cpu():
    ctx = dev.device_context("cpu", _res(ns=11, mpol=6, ntor=0))
    # jax.default_device(...) returns a context manager that is not nullcontext.
    assert not isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass


def test_put_numeric_leaves_preserves_metadata_and_none_contract():
    resolution = _res(ns=11, mpol=6, ntor=0)
    assert dev._placement_device(None, resolution) is None
    cpu = jax.devices("cpu")[0]
    moved = dev._put_numeric_leaves(
        {"array": np.ones(2), "metadata": "kept"}, cpu,
    )
    assert moved["array"].device.platform == "cpu"
    assert moved["metadata"] == "kept"


def test_put_numeric_leaves_moves_registered_partial_captures():
    target = jax.devices("cpu")[0]
    field = jax.tree_util.Partial(
        lambda scale, points: scale * points,
        jax.numpy.asarray(2.0),
    )
    moved = dev._put_numeric_leaves(field, target)
    assert jax.tree.leaves(moved)[0].device.platform == "cpu"
    np.testing.assert_allclose(moved(jax.numpy.asarray([3.0])), 6.0)


def test_free_boundary_uses_shared_device_context(monkeypatch):
    resolution = _res(ns=11, mpol=6, ntor=0)
    seen = {}

    @contextlib.contextmanager
    def fake_context(device, resolved):
        seen["context"] = (device, resolved)
        yield

    def fake_solve(inp, **kwargs):
        seen["solve"] = (inp, kwargs)
        return SimpleNamespace(result="result")

    monkeypatch.setattr(freeboundary, "device_context", fake_context)
    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", fake_solve)
    inp = object()
    result = freeboundary.solve_free_boundary(
        inp, resolution=resolution, device="cpu", max_iterations=3
    )

    assert result == "result"
    assert seen["context"] == ("cpu", resolution)
    assert seen["solve"][0] is inp
    assert seen["solve"][1]["resolution"] is resolution
    assert seen["solve"][1]["max_iterations"] == 3
    assert freeboundary.solve_free_boundary.__kwdefaults__["device"] == dev.AUTO
    assert multigrid.solve_free_boundary_multigrid.__kwdefaults__["device"] == dev.AUTO
