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

import numpy as np
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
        "run_external_equilibrium.py",
        "run_freeboundary_multigrid.py",
        "run_high_mode_fft.py",
        "make_strong_force_comparison.py",
        "polish_implicit.py",
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


def test_collocation_polish_derivative_artifact_is_clean_and_certified() -> None:
    artifact = json.loads(
        (ROOT / "benchmarks" / "polish_implicit_m4.json").read_text()
    )
    assert artifact["schema"] == "vmex.polish-implicit-benchmark/3"
    assert re.fullmatch(r"[0-9a-f]{40}", artifact["measurement_commit"])
    assert artifact["measurement_dirty"] is False
    assert artifact["persistent_compilation_cache"] is False
    assert artifact["primal_relative_optimality"] <= 1.0e-6
    assert artifact["tangent_iterations"] <= artifact["free_dofs"]
    assert artifact["adjoint_iterations"] <= artifact["free_dofs"]
    assert artifact["tangent_residual_norm"] <= artifact["tangent_tolerance"]
    assert artifact["adjoint_residual_norm"] <= artifact["adjoint_tolerance"]
    assert artifact["objective"] == "relative field-strength variance at rho=0.7"
    assert artifact["objective_value"] > 0.0
    assert artifact["finite_difference_relative_error"] < 1.0e-3
    assert artifact["finite_difference_seconds"] > (
        100.0 * artifact["warm_custom_vjp_median_seconds"]
    )
    assert artifact["tangent_adjoint_duality_relative_error"] < 1.0e-8
    assert artifact["custom_vjp_relative_squared_error"] < 1.0e-20
    assert artifact["warm_tangent_median_seconds"] < 0.05
    assert artifact["warm_adjoint_median_seconds"] < 0.05
    assert artifact["warm_custom_vjp_median_seconds"] < 0.05
    assert artifact["cold_tangent_seconds"] < 30.0
    assert artifact["cold_adjoint_seconds"] < 30.0
    assert artifact["cold_custom_vjp_seconds"] < 30.0
    assert artifact["custom_vjp_peak_rss_increase_mib"] < 512.0


def test_solvax_polish_artifact_is_independently_certified() -> None:
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
        assert len(
            artifact["radial_profile"]["flux_surface_normalized_l2"]
        ) == len(reference_rho)
        assert np.all(
            np.isfinite(artifact["radial_profile"]["flux_surface_normalized_l2"])
        )
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
    assert metadata["schema"] == "vmex.strong-force-readme-figure/3"
    figure = ROOT / metadata["figure"]
    assert figure.is_file()
    assert hashlib.sha256(figure.read_bytes()).hexdigest() == metadata[
        "figure_sha256"
    ]
    cases = metadata["cases"]
    assert set(cases) == {"solovev_analytical", "nfp2_QA_finite_beta"}
    for case in cases.values():
        for source in case["sources"].values():
            path = ROOT / source["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    solovev = cases["solovev_analytical"]["sources"]
    assert solovev["VMEX polished"]["normalized_l2"] < solovev["DESC"]["normalized_l2"]

    stellarator = cases["nfp2_QA_finite_beta"]["sources"]
    desc = json.loads((ROOT / stellarator["DESC"]["path"]).read_text())
    assert desc["external_source"]["success"] is True
    representation = desc["external_source"]["representation"]
    assert representation["L"] >= 16
    assert representation["M"] >= 10 and representation["N"] >= 10
    assert desc["metrics"]["radial_refinement_difference"] < 1.0e-3
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
