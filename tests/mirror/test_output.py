"""Mirror-native output and CLI plotting tests."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.image as mpimg
import numpy as np
import pytest

pytest.importorskip("netCDF4")

from vmex.core import cli
from vmex.mirror import (
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    mout_from_result,
    read_mout,
    write_mout,
)
from vmex.mirror.forces import mirror_energy
from vmex.mirror.output import MoutData, _theta_samples


REPO = Path(__file__).resolve().parents[2]


def test_canonical_benchmarks_declare_provenance_and_promotion_status() -> None:
    paths = sorted((REPO / "benchmarks").glob("mirror_*.json"))
    assert [path.name for path in paths] == [
        "mirror_fixed_boundary.json",
        "mirror_free_boundary_axisymmetric.json",
        "mirror_free_boundary_nonaxisymmetric.json",
        "mirror_hybrid_fixed_boundary.json",
    ]
    for path in paths:
        record = json.loads(path.read_text())
        assert record["schema"] == "vmex.benchmark.mirror/1"
        assert record["status"] in {
            "active-validation",
            "deferred",
            "supported",
            "supported-through-beta-0.10",
        }
        commit = record["provenance"]["measurement_commit"]
        assert 8 <= len(commit) <= 40 and int(commit, 16) >= 0, path.name
        assert record["gates"], path.name


def test_axisymmetric_free_boundary_benchmark_declares_supported_beta_ceiling() -> None:
    path = REPO / "benchmarks" / "mirror_free_boundary_axisymmetric.json"
    record = json.loads(path.read_text())
    assert record["case"]["supported_beta_max"] == pytest.approx(0.10)
    assert [row["beta"] for row in record["refinement"] if row["passed"]] == [0.0, 0.01, 0.03, 0.10]
    assert [row["beta"] for row in record["refinement"] if not row["passed"]] == [0.25, 0.50]


def test_fixed_benchmark_separates_corrected_cut_support_status() -> None:
    path = REPO / "benchmarks" / "mirror_fixed_boundary.json"
    record = json.loads(path.read_text())
    audit = record["nonaxisymmetric_corrected_cut_audit"]
    assert audit["rotating_ellipse"]["status"] == "release-candidate"
    assert audit["straight_field_line"]["status"] == "validation-only"
    assert record["gates"]["rotating_ellipse_strong_force"]
    assert not record["gates"]["straight_field_line_independent_strong_force"]


def _sample_mout() -> MoutData:
    ns, ntheta, nxi = 3, 5, 7
    s = np.linspace(0.0, 1.0, ns)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    z = np.linspace(-1.0, 1.0, nxi)
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    boundary = 0.25 * (1.0 + 0.15 * zz**2 + 0.04 * np.cos(2.0 * tt))
    radius_scale = np.broadcast_to(boundary, (ns, ntheta, nxi)).copy()
    mod_b = np.broadcast_to(0.8 + 0.35 * zz**2, (ns, ntheta, nxi)).copy()
    b_theta = 0.04
    b_xyz = np.empty((ns, ntheta, nxi, 3))
    b_xyz[..., 0] = -b_theta * np.sin(theta)[None, :, None]
    b_xyz[..., 1] = b_theta * np.cos(theta)[None, :, None]
    b_xyz[..., 2] = 1.0
    pressure = 2.0e4 * (1.0 - s[:, None, None]) * np.ones((1, ntheta, nxi))
    angle = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    coils = np.stack([
        np.column_stack([0.55 * np.cos(angle), 0.55 * np.sin(angle), np.full_like(angle, end)])
        for end in (-1.15, 1.15)
    ])
    return MoutData(
        s=s,
        theta=theta,
        xi=z,
        z=z,
        boundary_radius=boundary,
        radius_scale=radius_scale,
        lambda_stream=np.zeros_like(radius_scale),
        mod_b=mod_b,
        b_xyz=b_xyz,
        pressure=pressure,
        history=np.column_stack([np.arange(6), np.geomspace(1.0e-3, 1.0e-12, 6)]),
        coil_xyz=coils,
        ftol=1.0e-12,
        iterations=5,
        converged=True,
        mass_scale=1.0,
        variational_max=8.0e-13,
        normal_stress_rms=2.0e-10,
        b_normal_rms=3.0e-11,
        staggered_weak_max=6.0e-13,
        pointwise_force_rms=4.0e-4,
        normalized_divergence_rms=5.0e-13,
    )


def test_mout_roundtrip(tmp_path) -> None:
    path = write_mout(tmp_path / "mout_sample.nc", _sample_mout())
    loaded = read_mout(path)
    assert loaded.schema == "vmex.mirror.mout/1"
    assert loaded.converged
    assert loaded.staggered_weak_max == pytest.approx(6.0e-13)
    assert loaded.pointwise_force_rms == pytest.approx(4.0e-4)
    assert loaded.normalized_divergence_rms == pytest.approx(5.0e-13)
    np.testing.assert_allclose(loaded.boundary_radius, _sample_mout().boundary_radius)
    np.testing.assert_allclose(loaded.b_xyz, _sample_mout().b_xyz)


def test_plot_resampling_preserves_resolved_fourier_modes() -> None:
    data = _sample_mout()
    theta_dense = np.linspace(0.0, 2.0 * np.pi, 101)
    values = np.cos(2.0 * data.theta)[:, None]

    sampled = _theta_samples(data, values, theta_dense)

    np.testing.assert_allclose(sampled[:, 0], np.cos(2.0 * theta_dense), atol=2.0e-15)


def test_mout_reads_files_before_independent_diagnostics(tmp_path) -> None:
    import netCDF4

    path = write_mout(tmp_path / "mout_legacy.nc", _sample_mout())
    with netCDF4.Dataset(path, "r+") as dataset:
        dataset.delncattr("staggered_weak_max")
        dataset.delncattr("pointwise_force_rms")
        dataset.delncattr("normalized_divergence_rms")

    loaded = read_mout(path)
    assert np.isnan(loaded.staggered_weak_max)
    assert np.isnan(loaded.pointwise_force_rms)
    assert np.isnan(loaded.normalized_divergence_rms)


def test_mout_from_solved_result_contract() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.25, grid)
    state = MirrorState.from_boundary(boundary, grid)
    shape = grid.shape
    result = SimpleNamespace(
        boundary=boundary,
        plasma_state=state,
        plasma_b_squared=np.ones(shape),
        pressure=np.zeros(shape),
        history=np.asarray([[0.0, 1.0e-13]]),
        iterations=1,
        converged=True,
        mass_scale=1.0,
        variational_max=1.0e-13,
        interface=SimpleNamespace(normal_stress_rms=2.0e-13, vacuum_b_normal_rms=3.0e-13),
        message="converged",
    )
    data = mout_from_result(result, grid, config, axial_flux_derivative=0.02)
    assert data.b_xyz.shape == (*shape, 3)
    assert np.all(np.isfinite(data.b_xyz))
    assert np.all(data.pressure == 0.0)
    assert np.isnan(data.staggered_weak_max)
    assert np.isnan(data.pointwise_force_rms)
    assert np.isnan(data.normalized_divergence_rms)


def test_mout_accepts_fixed_boundary_result() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.25, grid)
    state = MirrorState.from_boundary(boundary, grid)
    energy = mirror_energy(
        state,
        grid,
        axial_flux_derivative=0.02,
        mass_profile=np.asarray([2.0e-4, 1.0e-4, 0.0]),
    )
    result = SimpleNamespace(
        state=state,
        energy=energy,
        variational=SimpleNamespace(maximum=1.0e-13),
        history=np.asarray([[0.0, energy.total, 0.0, 0.0, 1.0e-13, 2.0e-3]]),
        iterations=0,
        converged=True,
        message="converged",
    )
    data = mout_from_result(
        result,
        grid,
        config,
        boundary=boundary,
        axial_flux_derivative=0.02,
    )

    assert data.boundary_radius.shape == (1, grid.nxi)
    assert np.all(np.isfinite(data.pressure))
    assert np.all(np.isfinite(data.mod_b))
    np.testing.assert_allclose(data.history, [[0.0, 1.0e-13]])
    assert np.isnan(data.staggered_weak_max)
    assert np.isnan(data.pointwise_force_rms)
    assert np.isnan(data.normalized_divergence_rms)


def test_command_line_plots_mout_without_toroidal_dispatch(tmp_path) -> None:
    path = write_mout(tmp_path / "mout_sample.nc", _sample_mout())
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = cli.main(["--plot", str(path), "--outdir", str(tmp_path), "--quiet"])
    assert code == 0
    expected = {
        "sample_summary.png",
        "sample_cross_sections.png",
        "sample_modB.png",
        "sample_3d.png",
    }
    assert expected == {item.name for item in tmp_path.glob("sample_*.png")}
    for name in expected:
        pixels = mpimg.imread(tmp_path / name)
        assert pixels.shape[0] > 200 and pixels.shape[1] > 300
        assert float(np.std(pixels)) > 0.03
    pixels = mpimg.imread(tmp_path / "sample_3d.png")
    cyan = (pixels[..., 0] < 0.2) & (pixels[..., 1] > 0.6) & (pixels[..., 2] > 0.7)
    assert int(np.count_nonzero(cyan)) > 200
