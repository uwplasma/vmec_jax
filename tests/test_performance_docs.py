"""Guard: docs/reference/performance.rst is generated from the benchmark artifact.

The baseline table between the generated-block markers must match what
``tools/render_performance_docs.py`` renders from
``benchmarks/baseline.json`` — the review finding this prevents: a
hand-maintained narrative table silently disagreeing with the committed
measurement artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import render_performance_docs as rpd  # noqa: E402
from benchmarks.profile_resources import (  # noqa: E402
    _mirror_ladder,
    _parser,
    _peak_rss_bytes,
    _repeat_error,
)


def test_performance_table_matches_baseline_artifact() -> None:
    baseline = json.loads(rpd.BASELINE.read_text())
    text = rpd.DOC.read_text()
    head, rest = text.split(rpd.BEGIN, 1)
    _inner, _tail = rest.split(rpd.END, 1)
    rendered = rpd.render(baseline)
    assert rpd.BEGIN + rest.split(rpd.END, 1)[0] + rpd.END == rendered, (
        "docs/reference/performance.rst baseline table is stale; run python tools/render_performance_docs.py"
    )


def test_render_marks_wins_and_footnotes() -> None:
    baseline = json.loads(rpd.BASELINE.read_text())
    rendered = rpd.render(baseline)
    assert "**" in rendered, "no winning rows marked — renderer broken?"
    assert "VMEC++" in rendered
    row_count = sum(not key.startswith("_") for key in baseline)
    assert str(row_count) in rendered  # the computed row count


def test_benchmark_scripts_import_this_checkout_from_any_cwd(
    tmp_path: Path,
) -> None:
    """A benchmark must not silently import an installed VMEX distribution."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for script in (
        "run_baseline.py",
        "run_freeboundary_multigrid.py",
        "run_high_mode_fft.py",
        "make_strong_force_comparison.py",
        "polish_preconditioner.py",
        "strong_certificate.py",
        "strong_polish.py",
        "strong_root.py",
        "profile_resources.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "benchmarks" / script), "--help"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr


def test_resource_profiler_parses_platform_memory_and_mirror_ladders() -> None:
    assert _peak_rss_bytes("12345 maximum resident set size") == 12345
    assert (
        _peak_rss_bytes("Maximum resident set size (kbytes): 12345")
        == 12345 * 1024
    )
    assert _mirror_ladder("5:7:4,9:17:9") == [(5, 7, 4), (9, 17, 9)]
    assert _parser().parse_args(["--device", "gpu", "--device-index", "1"]).device_index == 1
    absolute, relative = _repeat_error([1.0, 2.0], [1.0, 2.0 + 1e-12])
    assert absolute == pytest.approx(1e-12)
    assert relative == pytest.approx(5e-13)


def test_benchmark_artifacts_disclose_redacted_provenance() -> None:
    artifacts = (
        ROOT / "benchmarks" / "baseline.json",
        ROOT / "benchmarks" / "freeboundary_multigrid.json",
        ROOT / "benchmarks" / "high_mode_fft.json",
    )
    for artifact in artifacts:
        report = json.loads(artifact.read_text())
        provenance = report["_provenance"] if artifact.name == "baseline.json" else report["provenance"]
        assert re.fullmatch(r"[0-9a-f]{8,40}", provenance["measurement_commit"])
        assert provenance["input_data_embedded"] is False
        encoded = json.dumps(provenance)
        assert "/Users/" not in encoded
        assert "/home/" not in encoded


def test_polish_preconditioner_artifact_is_clean_and_certified() -> None:
    artifact = json.loads(
        (ROOT / "benchmarks" / "polish_preconditioner_m4.json").read_text()
    )
    assert artifact["schema"] == "vmex.polish-preconditioner-benchmark/1"
    assert re.fullmatch(r"[0-9a-f]{40}", artifact["measurement_commit"])
    assert artifact["measurement_dirty"] is False
    assert artifact["persistent_compilation_cache"] is False
    assert len(artifact["cases"]) == 3
    for case in artifact["cases"]:
        assert case["warm_forward_median_seconds"] < 1.0e-3
        assert case["warm_transpose_median_seconds"] < 1.0e-3
        assert case["transfer_roundtrip_relative_residual"] < 2.0e-12
        assert case["preconditioner_duality_relative_error"] < 2.0e-12
        assert case["low_block_relative_residual"] < 1.0e-10


def test_strong_projection_artifacts_pin_resolution_and_blocking_diagnosis() -> None:
    artifacts = [
        json.loads(
            (
                ROOT
                / "benchmarks"
                / f"strong_projection_solovev_m{mpol}_m4.json"
            ).read_text()
        )
        for mpol in (5, 8, 13)
    ]
    unresolved = []
    for expected_mpol, artifact in zip((5, 8, 13), artifacts, strict=True):
        assert artifact["schema"] == "vmex.strong-polish-benchmark/1"
        assert artifact["measurement_dirty"] is False
        assert artifact["diagnostics_only"] is True
        assert artifact["mpol"] == expected_mpol
        assert artifact["total_seconds"] < 60.0
        assert artifact["total_peak_rss_increase_mib"] < 4096.0
        projection = artifact["projection_consistency"]["initial"]
        assert projection["radial_fit_unresolved_fraction"] < 0.25
        assert projection["equation_discarded_fraction"] < 1.0e-12
        unresolved.append(projection["unresolved_fraction"])
    assert unresolved[0] > unresolved[1] > unresolved[2]

    endpoint = json.loads(
        (
            ROOT / "benchmarks" / "strong_polish_solovev_m5_direct_m4.json"
        ).read_text()
    )
    assert endpoint["measurement_dirty"] is False
    initial = endpoint["projection_consistency"]["initial"]
    final = endpoint["projection_consistency"]["final"]
    assert final["projected_residual_rms"] < initial["projected_residual_rms"]
    assert final["angular_unresolved_fraction"] > 2.0 * initial[
        "angular_unresolved_fraction"
    ]
    assert endpoint["final_certificate"]["normalized_l2"] > endpoint[
        "initial_certificate"
    ]["normalized_l2"]


def test_collocation_polish_artifact_is_independently_certified() -> None:
    artifact = json.loads(
        (
            ROOT
            / "benchmarks"
            / "strong_polish_solovev_collocation_d3_m5_m4.json"
        ).read_text()
    )
    report = artifact["polish_report"]
    final = artifact["final_certificate"]
    assert artifact["measurement_dirty"] is False
    assert artifact["collocation_least_squares"] is True
    assert artifact["degree"] == 3
    assert report["converged"] is True
    assert report["termination_reason"] == "independently-certified"
    assert final["normalized_l2"] <= artifact["validation_tolerance"]
    assert final["radial_refinement"] <= artifact[
        "radial_refinement_tolerance"
    ]
    assert report["minimum_signed_jacobian"] > 0.0
    assert report["nonlinear_iterations"] == report["residual_evaluations"]
    assert len(final["radial_profile"]["rho"]) == len(
        final["radial_profile"]["flux_surface_average_force_density"]
    )
    assert artifact["total_seconds"] < 60.0
    assert artifact["total_peak_rss_increase_mib"] < 3072.0

    native = json.loads(
        (
            ROOT
            / "benchmarks"
            / "strong_polish_solovev_solvax_d3_m5_m4.json"
        ).read_text()
    )
    native_report = native["polish_report"]
    native_final = native["final_certificate"]
    assert native["measurement_dirty"] is False
    assert native["solvax_source"]["dirty"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", native["solvax_source"]["commit"])
    assert native["solvax_least_squares"] is True
    assert native_report["converged"] is True
    assert native_report["least_squares_success"] is True
    assert native_report["least_squares_relative_optimality"] <= 1.0e-3
    assert native_final["normalized_l2"] <= native["validation_tolerance"]
    assert native_final["radial_refinement"] <= native[
        "radial_refinement_tolerance"
    ]
    assert native_report["minimum_signed_jacobian"] > 0.0
    assert native["total_seconds"] < 60.0
    assert native["total_peak_rss_increase_mib"] < 3072.0


def test_solovev_cross_code_certificates_are_clean_and_comparable() -> None:
    names = ("vmex", "vmec2000", "vmecpp", "desc")
    artifacts = {
        name: json.loads(
            (
                ROOT
                / "benchmarks"
                / f"strong_certificate_solovev_{name}_m4.json"
            ).read_text()
        )
        for name in names
    }
    reference_rho = artifacts["vmex"]["radial_profile"]["rho"]
    for name, artifact in artifacts.items():
        assert artifact["schema"] == "vmex.strong-certificate-benchmark/1"
        assert artifact["measurement_dirty"] is False
        assert len(artifact["radial_profile"]["rho"]) == 128
        assert artifact["radial_profile"]["rho"] == reference_rho
        assert len(
            artifact["radial_profile"]["flux_surface_average_force_density"]
        ) == len(reference_rho)
        assert artifact["metrics"]["radial_refinement_difference"] < 1.0e-3
        if name in ("vmec2000", "desc"):
            assert artifact["external_source"]["success"] is True

    normalized = {
        name: artifact["metrics"]["normalized_l2"]
        for name, artifact in artifacts.items()
    }
    assert normalized["vmex"] == pytest.approx(
        normalized["vmecpp"], rel=2.0e-4
    )
    assert normalized["vmec2000"] == pytest.approx(
        normalized["vmecpp"], rel=2.0e-7
    )
    assert normalized["desc"] < 0.2 * normalized["vmecpp"]


def test_readme_strong_force_figure_matches_committed_sources() -> None:
    metadata = json.loads(
        (ROOT / "benchmarks" / "strong_force_comparison_m4.json").read_text()
    )
    assert metadata["schema"] == "vmex.strong-force-readme-figure/1"
    figure = ROOT / metadata["figure"]
    assert figure.is_file()
    assert hashlib.sha256(figure.read_bytes()).hexdigest() == metadata[
        "figure_sha256"
    ]
    for source in metadata["sources"].values():
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    normalized = {
        name: source["normalized_l2"]
        for name, source in metadata["sources"].items()
    }
    assert normalized["VMEX polished"] < normalized["DESC"]
    readme = (ROOT / "README.md").read_text()
    assert metadata["figure"] in readme
    assert "--solvax-least-squares" in readme


def test_committed_reports_do_not_expose_personal_paths() -> None:
    """Release-facing text must remain portable between contributors."""
    text_suffixes = {".json", ".md", ".py", ".rst", ".toml"}
    for directory in ("benchmarks", "docs", "examples"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in text_suffixes and "_build" not in path.parts:
                text = path.read_text(errors="replace")
                assert "/Users/" not in text, path
                assert "MacBook-Pro.local" not in text, path
