"""Small, mandatory accelerator checks for the manual GPU CI lane."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from vmex.core import device as device_policy
from vmex.core import implicit as im
from vmex.core import freeboundary, multigrid, optimize, solver
from vmex.core.input import VmecInput
from vmex.core.mgrid import MgridField
from vmex.core.wout import wout_from_state
from vmex.mirror import (
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    SplineMirrorDiscretization,
    solve_beta_scan,
    solve_fixed_boundary_from_radius,
)

from tests.test_lasym_free_case import lasym_free_field, lasym_free_input


DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


try:
    GPU = jax.devices("gpu")[0]
except RuntimeError:
    GPU = None


def _gpu():
    assert GPU is not None
    return GPU


def _platform(array) -> str:
    device = array.device
    return (device() if callable(device) else device).platform


def _paraxial_mirror_field(center_field, curvature, points):
    points = jax.numpy.asarray(points)
    x, y, z = jax.numpy.moveaxis(points, -1, 0)
    return jax.numpy.stack(
        (
            -curvature * x * z,
            -curvature * y * z,
            center_field + curvature * (z**2 - 0.5 * (x**2 + y**2)),
        ),
        axis=-1,
    )


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(GPU is None, reason="GPU unavailable"),
    pytest.mark.usefixtures("_module_jit_enabled"),
]


def test_gpu_is_default_without_platform_environment_pins():
    """The dedicated lane must fail rather than silently exercise the CPU."""
    _gpu()
    assert "JAX_PLATFORMS" not in os.environ
    assert "JAX_PLATFORM_NAME" not in os.environ
    assert jax.default_backend() == "gpu"
    assert _platform(jax.numpy.ones(())) == "gpu"


def test_implicit_auto_prefers_cpu_but_none_follows_jax_gpu():
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    automatic = im.params_from_input(inp)
    following_jax = im.params_from_input(inp, device=None)

    assert {_platform(x) for x in jax.tree.leaves(automatic)} == {"cpu"}
    assert {_platform(x) for x in jax.tree.leaves(following_jax)} == {"gpu"}


def test_explicit_gpu_mirror_fixed_boundary():
    """The mirror policy honors an explicit GPU through a real solve."""
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=4, nxi=9),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    result = solve_fixed_boundary_from_radius(
        0.3,
        config,
        elements=4,
        axial_flux_derivative=0.1,
        device="gpu",
    )
    assert result.evaluated.converged
    assert _platform(result.evaluated.state.radius_scale) == "gpu"
    assert float(result.evaluated.variational.maximum) <= config.ftol


def test_explicit_gpu_mirror_free_boundary_beta_scan():
    """A finite-beta continuation and its free-boundary solves stay on GPU."""
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=200,
    )
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build_cgl(config, elements=4)
    on_axis = 0.08 + 0.02 * jax.numpy.asarray(source_grid.z) ** 2
    center = source_grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis, source_grid)
    field = jax.tree_util.Partial(
        _paraxial_mirror_field,
        jax.device_put(0.08, jax.devices("cpu")[0]),
        jax.device_put(0.02, jax.devices("cpu")[0]),
    )
    assert {_platform(x) for x in jax.tree.leaves(field)} == {"cpu"}

    results = solve_beta_scan(
        discretization.fit_boundary(boundary, source_grid),
        discretization,
        config,
        field,
        jax.numpy.asarray([0.0, 0.01]),
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
        device="gpu",
    )

    assert len(results) == 2
    assert float(results[1].pressure[0, 0, center]) > 0.0
    for result in results:
        assert result.converged
        assert _platform(result.coefficient_state.radius_coefficients) == "gpu"
        assert _platform(result.coefficient_boundary.radius_coefficients) == "gpu"
        assert float(result.variational_max) <= config.ftol


def test_explicit_forward_solve_cpu_gpu_parity():
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    results = {
        platform: solver.solve(
            inp, ftol=1e-12, max_iterations=1000, mode="jit", device=platform
        )
        for platform in ("cpu", "gpu")
    }

    assert results["cpu"].converged and results["gpu"].converged
    assert results["cpu"].iterations == results["gpu"].iterations
    assert _platform(results["cpu"].state.R_cos) == "cpu"
    assert _platform(results["gpu"].state.R_cos) == "gpu"
    np.testing.assert_allclose(
        results["gpu"].fsq_history,
        results["cpu"].fsq_history,
        rtol=5e-10,
        atol=1e-14,
    )
    for gpu_leaf, cpu_leaf in zip(
        jax.tree.leaves(results["gpu"].state),
        jax.tree.leaves(results["cpu"].state),
        strict=True,
    ):
        np.testing.assert_allclose(gpu_leaf, cpu_leaf, rtol=5e-10, atol=1e-12)

    # Committed hot starts must not defeat an explicit opposite-device request.
    for platform, seed in (("gpu", results["cpu"]), ("cpu", results["gpu"])):
        restarted = solver.solve(
            inp,
            ftol=1e-12,
            max_iterations=1000,
            mode="jit",
            initial_state=seed.state,
            device=platform,
        )
        assert restarted.converged
        assert _platform(restarted.state.R_cos) == platform


def test_multigrid_auto_moves_state_across_policy_threshold(monkeypatch):
    """A CPU coarse stage must not commit an AUTO-selected fine stage to CPU."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5, 11]),
        ftol_array=np.asarray([1e-9, 1e-12]),
        niter_array=np.asarray([1000, 1000]),
    )
    monkeypatch.setattr(device_policy, "GPU_MIN_ITERATION_WORK", 500)
    seen = []
    solve_stage = multigrid._solve_stage

    def recording_solve_stage(*args, **kwargs):
        carry = solve_stage(*args, **kwargs)
        seen.append(_platform(carry.state.R_cos))
        return carry

    monkeypatch.setattr(multigrid, "_solve_stage", recording_solve_stage)
    result = multigrid.solve_multigrid(inp, mode="jit", device="auto")
    assert result.converged
    assert seen == ["cpu", "gpu"]


def test_converged_lasym_free_boundary_cpu_gpu_parity(monkeypatch):
    """A converged LASYM ladder and every NESTOR WOUT field agree."""
    inp = lasym_free_input(DATA_DIR)
    external_field = lasym_free_field()
    results = {}
    output = {}
    active_platform = ["cpu"]
    vacuum_devices = {}
    stage_devices = {"cpu": [], "gpu": []}
    vacuum_step = freeboundary._vacuum_step
    solve_stage = freeboundary._solve_free_boundary_stage

    def recording_vacuum_step(*args, **kwargs):
        value = vacuum_step(*args, **kwargs)
        fb = kwargs["fb"]
        vacuum_devices[active_platform[0]] = {
            _platform(x)
            for name in (
                "potvac", "mode_matrix", "bvec_nonsing", "bsqvac",
                "surface_fields",
            )
            for x in jax.tree.leaves(getattr(fb, name))
        }
        return value

    monkeypatch.setattr(freeboundary, "_vacuum_step", recording_vacuum_step)

    def recording_solve_stage(*args, **kwargs):
        record = {
            "field": {_platform(x) for x in jax.tree.leaves(
                kwargs["external_field"])},
        }
        state = kwargs["initial_state"]
        if state is not None:
            record["state"] = {_platform(x) for x in jax.tree.leaves(state)}
        vacuum = kwargs["vacuum_continuation"]
        if vacuum is not None:
            record["vacuum"] = {
                _platform(x)
                for name in (
                    "bsqvac", "rbsq", "mode_matrix", "bvec_nonsing",
                    "potvac", "surface_fields",
                )
                for x in jax.tree.leaves(getattr(vacuum, name))
            }
        stage_devices[active_platform[0]].append(record)
        return solve_stage(*args, **kwargs)

    monkeypatch.setattr(
        freeboundary, "_solve_free_boundary_stage", recording_solve_stage,
    )
    for platform in ("cpu", "gpu"):
        active_platform[0] = platform
        lines = []
        results[platform] = multigrid.solve_free_boundary_multigrid(
            inp,
            external_field=external_field,
            verbose=True,
            emit=lambda *args, _lines=lines, **kwargs: _lines.append(
                args[0] if args else ""
            ),
            device=platform,
        )
        output[platform] = "".join(lines)

    for platform in ("cpu", "gpu"):
        assert results[platform].converged
        assert "VACUUM PRESSURE TURNED ON" in output[platform]
        assert _platform(results[platform].state.R_cos) == platform
        assert vacuum_devices[platform] == {platform}
        assert stage_devices[platform][0]["field"] == {platform}
        assert stage_devices[platform][1] == {
            "field": {platform}, "state": {platform}, "vacuum": {platform},
        }
    np.testing.assert_allclose(
        [results["gpu"].fsqr, results["gpu"].fsqz, results["gpu"].fsql],
        [results["cpu"].fsqr, results["cpu"].fsqz, results["cpu"].fsql],
        rtol=5e-9,
        atol=1e-12,
    )
    gpu_state = np.concatenate([
        np.asarray(x).ravel() for x in jax.tree.leaves(results["gpu"].state)
    ])
    cpu_state = np.concatenate([
        np.asarray(x).ravel() for x in jax.tree.leaves(results["cpu"].state)
    ])
    delta = gpu_state - cpu_state
    assert np.linalg.norm(delta) / np.linalg.norm(cpu_state) < 5e-6
    assert np.max(np.abs(delta)) < 1e-6
    assert results["cpu"].vacuum is not None
    assert results["gpu"].vacuum is not None
    for name in ("xmpot", "xnpot"):
        np.testing.assert_array_equal(
            getattr(results["gpu"].vacuum, name),
            getattr(results["cpu"].vacuum, name),
        )
    relative_errors = {}
    for name in ("potsin", "potcos", "bsubu", "bsubv", "bsupu", "bsupv"):
        gpu_values = np.asarray(getattr(results["gpu"].vacuum, name))
        cpu_values = np.asarray(getattr(results["cpu"].vacuum, name))
        scale = max(float(np.max(np.abs(cpu_values))), np.finfo(float).tiny)
        relative_errors[name] = (
            float(np.max(np.abs(gpu_values - cpu_values))) / scale)
    assert max(relative_errors[name] for name in (
        "potsin", "potcos", "bsubu", "bsubv",
    )) < 1e-4, relative_errors
    assert max(relative_errors[name] for name in ("bsupu", "bsupv")) < 5e-4, (
        relative_errors)

    wouts = {
        platform: wout_from_state(
            inp=inp,
            state=result.state,
            fsqr=result.fsqr,
            fsqz=result.fsqz,
            fsql=result.fsql,
            niter=result.iterations,
            converged=result.converged,
            vacuum_output=result.vacuum,
        )
        for platform, result in results.items()
    }
    for name in (
        "potsin",
        "potcos",
        "bsubumnc_sur",
        "bsubvmnc_sur",
        "bsupumnc_sur",
        "bsupvmnc_sur",
        "bsubumns_sur",
        "bsubvmns_sur",
        "bsupumns_sur",
        "bsupvmns_sur",
    ):
        cpu_values = np.asarray(getattr(wouts["cpu"], name))
        gpu_values = np.asarray(getattr(wouts["gpu"], name))
        assert np.isfinite(cpu_values).all(), name
        np.testing.assert_allclose(
            gpu_values, cpu_values, rtol=5e-4, atol=1e-9, err_msg=name
        )

    resolution = solver.resolution_from_input(inp, ns=32)
    restart_inputs = []

    def recording_restart(*args, **kwargs):
        restart_inputs.append(kwargs["initial_state"])
        return solve_stage(*args, **kwargs)

    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", recording_restart)
    for target, source in (("gpu", "cpu"), ("cpu", "gpu")):
        seed = results[source].state
        seed = dataclasses.replace(
            seed,
            R_cos=seed.R_cos.at[-1, 0].add(1.0e-8),
            R_sin=seed.R_sin.at[-1, 1].add(-2.0e-8),
            Z_sin=seed.Z_sin.at[-1, 1].add(3.0e-8),
            Z_cos=seed.Z_cos.at[-1, 0].add(-4.0e-8),
        )
        restarted = freeboundary.solve_free_boundary(
            inp,
            external_field=external_field,
            resolution=resolution,
            max_iterations=1,
            error_on_no_convergence=False, initial_state=seed, device=target,
        )
        assert _platform(restarted.state.R_cos) == target
        assert _platform(restart_inputs[-1].R_cos) == target
        for family in ("R_cos", "R_sin", "Z_sin", "Z_cos"):
            np.testing.assert_array_equal(
                getattr(restart_inputs[-1], family)[-1],
                getattr(seed, family)[-1],
            )


def test_free_boundary_multigrid_auto_relocates_every_carry(monkeypatch):
    """AUTO may cross CPU/GPU between stages without retaining committed leaves."""
    inp = VmecInput.from_file(DATA_DIR / "input.cth_like_free_bdy_lasym_small")
    resolutions = [solver.resolution_from_input(inp, ns=ns) for ns in (7, 15)]
    work = [device_policy.iteration_work(resolution) for resolution in resolutions]
    monkeypatch.setattr(device_policy, "GPU_MIN_ITERATION_WORK", sum(work) // 2)
    field = MgridField(
        *(jax.numpy.ones((1, 1, 2, 2)) for _ in range(3)),
        extcur=jax.numpy.ones(1), rmin=0.0, rmax=1.0,
        zmin=-1.0, zmax=1.0, nfp=1,
    )
    seen = []

    def fake_stage(_inp, **kwargs):
        state = kwargs["initial_state"]
        vacuum = kwargs["vacuum_continuation"]
        seen.append({
            "field": {_platform(x) for x in jax.tree.leaves(kwargs["external_field"])},
            "state": None if state is None else {
                _platform(x) for x in jax.tree.leaves(state)},
            "vacuum": None if vacuum is None else {
                _platform(getattr(vacuum, name))
                for name in ("bsqvac", "rbsq", "mode_matrix", "bvec_nonsing", "potvac")
            },
            "constraint": None if kwargs["constraint_continuation"] is None else {
                _platform(x) for x in kwargs["constraint_continuation"]},
        })
        ns, mnmax = kwargs["resolution"].ns, kwargs["resolution"].mnmax
        arrays = [jax.numpy.zeros((ns, mnmax)) for _ in range(6)]
        state = solver.SpectralState(*arrays)
        cache = jax.numpy.zeros(1)
        vacuum = freeboundary.FreeBoundaryState(
            turned_on=True, bsqvac=cache, rbsq=cache, mode_matrix=cache,
            bvec_nonsing=cache, potvac=cache,
            surface_fields=(cache, cache, cache, cache),
        )
        result = type("Result", (), {"state": state})()
        return freeboundary._FreeBoundaryStageResult(
            result, vacuum, state, cache, cache,
        )

    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", fake_stage)
    result = multigrid.solve_free_boundary_multigrid(
        inp, ns_array=[7, 15, 15], ftol_array=[1e-8], niter_array=[1],
        external_field=field, raise_on_max_iterations=False, device="auto",
    )

    assert seen == [
        {"field": {"cpu"}, "state": None, "vacuum": None, "constraint": None},
        {"field": {"gpu"}, "state": {"gpu"}, "vacuum": {"gpu"},
         "constraint": None},
        {"field": {"gpu"}, "state": {"gpu"}, "vacuum": {"gpu"},
         "constraint": {"gpu"}},
    ]
    assert _platform(result.state.R_cos) == "gpu"


def test_explicit_implicit_gradient_cpu_gpu_parity():
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")

    def value_and_gradient(platform):
        params = im.params_from_input(inp, device=platform)
        value, gradient = jax.value_and_grad(
            lambda p: im.run(
                inp,
                p,
                ftol=1e-12,
                max_iterations=1000,
                device=platform,
            ).wb
        )(params)
        return value, gradient.rbc[inp.ntor, 1], gradient

    cpu_value, cpu_gradient, cpu_tree = value_and_gradient("cpu")
    gpu_value, gpu_gradient, gpu_tree = value_and_gradient(_gpu())

    assert {_platform(x) for x in jax.tree.leaves(cpu_tree)} == {"cpu"}
    assert {_platform(x) for x in jax.tree.leaves(gpu_tree)} == {"gpu"}
    np.testing.assert_allclose(gpu_value, cpu_value, rtol=1e-11)
    np.testing.assert_allclose(gpu_gradient, cpu_gradient, rtol=2e-7, atol=1e-12)


def test_scalarized_profile_gradient_cpu_gpu_parity():
    """The bounded-storage DMerc/D_R/<J.B> optimizer uses either device."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    terms = [
        (optimize.mercier_stability_residual, 0.0, 1.0),
        (optimize.glasser_stability_residual, 0.0, 1.0),
        (optimize.jdotb_residual, 0.0, 1.0e-6),
    ]
    x0 = optimize.pack_boundary(inp, 1)
    results = {
        platform: optimize.minimize(
            terms, inp, max_mode=1, device=platform,
            bounds=list(zip(x0, x0)))
        for platform in ("cpu", "gpu")
    }
    np.testing.assert_allclose(
        results["gpu"].cost, results["cpu"].cost, rtol=1e-11)
    np.testing.assert_allclose(
        results["gpu"].jac, results["cpu"].jac, rtol=2e-7, atol=1e-12)
