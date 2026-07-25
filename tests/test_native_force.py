"""Value and derivative parity for the opt-in native force projection."""

from __future__ import annotations

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import native_force as nf
from vmex.core.fourier import Resolution, trig_tables
from vmex.core import implicit as im
from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_multigrid
from vmex.core.native_force import (
    _NAMES, native_force_available, project_force, require_native_cpu,
)
from vmex.core.transforms import tomnspa, tomnsps


def _case(*, lasym=False, ntor=2):
    resolution = Resolution(
        mpol=4, ntor=ntor, ntheta=16, nzeta=12 if ntor else 1,
        nfp=3, lasym=lasym, ns=6,
    )
    trig = trig_tables(resolution)
    rng = np.random.default_rng(7)
    shape = (resolution.ns, trig.ntheta3, resolution.nzeta)
    kernels = {
        f"{name}_{parity}": jnp.asarray(rng.standard_normal(shape))
        for name in _NAMES for parity in ("even", "odd")
    }
    return resolution, trig, kernels


@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("ntor", [0, 2])
def test_jax_force_projector_matches_public(asym, ntor):
    resolution, trig, kernels = _case(lasym=asym, ntor=ntor)
    expected = (tomnspa if asym else tomnsps)(
        **kernels, mpol=resolution.mpol, ntor=ntor, trig=trig,
    )
    actual = project_force(
        kernels, mpol=resolution.mpol, ntor=ntor, trig=trig,
        asym=asym, backend="jax",
    )
    for value, target in zip(
        dataclasses.astuple(actual), dataclasses.astuple(expected), strict=True
    ):
        if target is None:
            assert value is None
        else:
            np.testing.assert_allclose(value, target, rtol=2e-13, atol=2e-12)


@pytest.mark.skipif(not native_force_available(), reason="native extension not built")
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("include_edge", [False, True])
@pytest.mark.parametrize("threads", [1, 2])
def test_native_force_matches_public_jax(asym, include_edge, threads):
    resolution, trig, kernels = _case(lasym=asym)
    reference = (tomnspa if asym else tomnsps)(
        **kernels, mpol=resolution.mpol, ntor=resolution.ntor,
        trig=trig, include_edge=include_edge,
    )
    def compiled():
        result = project_force(
            kernels, mpol=resolution.mpol, ntor=resolution.ntor, trig=trig,
            include_edge=include_edge, asym=asym, backend="native", threads=threads,
        )
        return tuple(value for value in dataclasses.astuple(result) if value is not None)

    with jax.disable_jit(False), jax.default_device(jax.devices("cpu")[0]):
        actual = jax.jit(compiled)()
    expected = tuple(
        value for value in dataclasses.astuple(reference) if value is not None
    )
    for value, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, target, rtol=2e-13, atol=2e-12)


@pytest.mark.skipif(not native_force_available(), reason="native extension not built")
def test_native_force_float32_with_float64_tables():
    resolution, trig, kernels = _case()
    kernels = jax.tree.map(lambda value: value.astype(jnp.float32), kernels)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = project_force(
            kernels, mpol=resolution.mpol, ntor=resolution.ntor,
            trig=trig, backend="jax",
        )
        actual = project_force(
            kernels, mpol=resolution.mpol, ntor=resolution.ntor,
            trig=trig, backend="native",
        )
    for value, target in zip(
        dataclasses.astuple(actual), dataclasses.astuple(expected), strict=True
    ):
        if value is None:
            assert target is None
            continue
        assert value.dtype == jnp.float32
        np.testing.assert_allclose(value, target, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not native_force_available(), reason="native extension not built")
@pytest.mark.parametrize("asym", [False, True])
def test_native_force_jvp_and_transpose(asym):
    resolution, trig, kernels = _case(lasym=asym)

    def projected(scale, backend):
        scaled = jax.tree.map(lambda value: scale * value, kernels)
        scaled_trig = dataclasses.replace(trig, **{
            name: scale * getattr(trig, name)
            for name in (
                "cosmui", "sinmui", "cosmumi", "sinmumi",
                "cosnv", "sinnv", "cosnvn", "sinnvn",
            )
        })
        result = project_force(
            scaled, mpol=resolution.mpol, ntor=resolution.ntor, trig=scaled_trig,
            asym=asym, backend=backend, threads=2,
        )
        return jnp.stack([
            value for value in dataclasses.astuple(result) if value is not None
        ])

    with jax.disable_jit(False), jax.default_device(jax.devices("cpu")[0]):
        value, tangent = jax.jvp(
            lambda scale: projected(scale, "native"), (1.0,), (0.25,)
        )
        reference, reference_tangent = jax.jvp(
            lambda scale: projected(scale, "jax"), (1.0,), (0.25,)
        )
        gradient = jax.grad(
            lambda scale: jnp.sum(projected(scale, "native") ** 2)
        )(1.0)
        reference_gradient = jax.grad(
            lambda scale: jnp.sum(projected(scale, "jax") ** 2)
        )(1.0)
    np.testing.assert_allclose(value, reference, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(tangent, reference_tangent, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(
        gradient, reference_gradient, rtol=2e-13, atol=2e-11
    )


def test_force_backend_validation(monkeypatch):
    resolution, trig, kernels = _case()
    with pytest.raises(ValueError, match="backend"):
        project_force(
            kernels, mpol=resolution.mpol, ntor=resolution.ntor,
            trig=trig, backend="cuda",
        )
    with pytest.raises(ValueError, match="threads"):
        project_force(
            kernels, mpol=resolution.mpol, ntor=resolution.ntor,
            trig=trig, threads=0,
        )
    monkeypatch.setattr(nf, "_force_ffi", None)
    with pytest.raises(RuntimeError, match="not available"):
        project_force(
            kernels, mpol=resolution.mpol, ntor=resolution.ntor,
            trig=trig, backend="native",
        )


def test_native_ffi_wrapper_shape_and_jvp(monkeypatch):
    from vmex.core.forces import RealSpaceForces, spectral_mhd_forces

    resolution, trig, kernels = _case()
    seen = {}

    def ffi_call(_name, specs, **_kwargs):
        def call(*args, **attrs):
            seen.update(attrs)
            return tuple(jnp.zeros(spec.shape, args[0].dtype) for spec in specs)
        return call

    monkeypatch.setattr(nf, "_force_ffi", object())
    monkeypatch.setattr(nf.ffi, "ffi_call", ffi_call)

    def projected(scale, backend):
        result = project_force(
            jax.tree.map(lambda value: scale * value, kernels),
            mpol=resolution.mpol, ntor=resolution.ntor, trig=trig,
            backend=backend, threads=99,
        )
        return jnp.stack([
            value for value in dataclasses.astuple(result) if value is not None
        ])

    value, tangent = jax.jvp(
        lambda scale: projected(scale, "native"), (1.0,), (0.25,)
    )
    reference_tangent = jax.jvp(
        lambda scale: projected(scale, "jax"), (1.0,), (0.25,)
    )[1]
    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_allclose(tangent, reference_tangent, rtol=2e-13, atol=2e-12)
    assert seen["threads"] == resolution.ns * resolution.mpol
    spectral_mhd_forces(
        RealSpaceForces(**kernels), mpol=resolution.mpol,
        ntor=resolution.ntor, trig=trig, backend="native",
    )
    asym_resolution, asym_trig, asym_kernels = _case(lasym=True)
    spectral_mhd_forces(
        RealSpaceForces(**asym_kernels), mpol=asym_resolution.mpol,
        ntor=asym_resolution.ntor, trig=asym_trig, backend="native",
    )


def test_native_cpu_guard_and_public_entry_points(monkeypatch):
    from vmex.core import device as device_module
    from vmex.core.freeboundary import solve_free_boundary
    from vmex.core.multigrid import solve_free_boundary_multigrid
    from vmex.core import optimize as opt
    from vmex.core.setup import run_setup
    from vmex.core.solver import prepare_runtime, resolution_from_input, solve

    inp = VmecInput.from_file("examples/data/input.solovev")
    resolution, _, _ = _case()
    require_native_cpu("cpu", resolution)
    with jax.default_device(jax.devices("cpu")[0]):
        require_native_cpu(None, resolution)
    with monkeypatch.context() as patch:
        patch.setattr(
            device_module, "_placement_device",
            lambda *_args: types.SimpleNamespace(platform="gpu"),
        )
        with pytest.raises(ValueError, match="CPU-only"):
            require_native_cpu(None, resolution)

    input_resolution = resolution_from_input(inp)
    setup = run_setup(inp, input_resolution)
    with pytest.raises(ValueError, match="force_backend"):
        prepare_runtime(setup, input_resolution, force_backend="cuda")
    with pytest.raises(ValueError, match="threads"):
        prepare_runtime(setup, input_resolution, threads=0)

    class GuardReached(Exception):
        pass

    def stop(*_args):
        raise GuardReached

    monkeypatch.setattr(nf, "require_native_cpu", stop)
    calls = (
        lambda: solve(inp, force_backend="native"),
        lambda: solve_multigrid(inp, force_backend="native"),
        lambda: solve_free_boundary(inp, force_backend="native"),
        lambda: im.run(inp, force_backend="native"),
        lambda: opt.least_squares(
            [], inp, max_mode=0, jac="implicit",
            solve_kwargs={"force_backend": "native"},
        ),
    )
    for call in calls:
        with pytest.raises(GuardReached):
            call()

    freeb = VmecInput.from_file("examples/data/input.cth_like_free_bdy")
    with pytest.raises(GuardReached):
        solve_free_boundary_multigrid(
            freeb, external_field=object(), force_backend="native",
        )


@pytest.mark.gpu
def test_native_backend_rejects_gpu_placement():
    try:
        jax.devices("gpu")
    except RuntimeError:
        pytest.skip("GPU unavailable")
    resolution, _, _ = _case()
    with pytest.raises(ValueError, match="CPU-only"):
        require_native_cpu("gpu", resolution)


def test_native_implicit_forward_uses_cpu(monkeypatch):
    inp = VmecInput.from_file("examples/data/input.solovev")
    cfg = im.make_config(inp, force_backend="native")
    seen = {}

    def solve_stub(*args, **kwargs):
        seen["device"] = kwargs["device"]
        return types.SimpleNamespace(converged=True, iterations=0, state=None)

    monkeypatch.setattr(im, "solve", solve_stub)
    im._host_solve(cfg, im.params_from_input(inp, device="cpu"))
    assert seen["device"] == "cpu"


@pytest.mark.skipif(not native_force_available(), reason="native extension not built")
def test_native_fixed_boundary_matches_default():
    inp = VmecInput.from_file("examples/data/input.solovev")
    with jax.disable_jit(False), jax.default_device(jax.devices("cpu")[0]):
        reference = solve_multigrid(inp, mode="jit")
        actual = solve_multigrid(
            inp, mode="jit", force_backend="native", threads=2
        )
    assert actual.iterations == reference.iterations
    for field in dataclasses.fields(reference.state):
        np.testing.assert_allclose(
            getattr(actual.state, field.name),
            getattr(reference.state, field.name),
            rtol=2e-13,
            atol=6e-14,
        )


@pytest.mark.full
@pytest.mark.skipif(not native_force_available(), reason="native extension not built")
def test_native_implicit_gradient_matches_jax():
    inp = VmecInput.from_file("examples/data/input.solovev")
    params = im.params_from_input(inp, device="cpu")

    def gradient(backend):
        return jax.grad(
            lambda p: im.run(
                inp, p, force_backend=backend, threads=2, device="cpu"
            ).wb
        )(params)

    reference = gradient("jax")
    actual = gradient("native")
    for expected, value in zip(
        jax.tree.leaves(reference), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_allclose(value, expected, rtol=3e-10, atol=3e-11)
