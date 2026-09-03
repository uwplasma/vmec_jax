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
        "strong_polish_3d.py",
        "strong_root.py",
        "polish_memory.py",
        "polish_cost.py",
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


def test_polish_sweep_memory_artifact_shows_the_fix_it_claims() -> None:
    """The memory record has to carry the claim, not just the run.

    The polish OOM was reported, reproduced, and fixed without anything in
    the tree recording either number.  This artifact is that record, so the
    gate is on the comparison it exists to make: the pre-0.8.2 flat sweep
    must still show the allocation, and the shipped policy must complete on
    a small fraction of it.
    """

    artifact = json.loads(
        (ROOT / "benchmarks" / "polish_memory_w7x.json").read_text()
    )
    assert artifact["schema"] == "vmex.polish-sweep-memory/1"
    provenance = artifact["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["measurement_commit"])
    assert provenance["measurement_dirty"] is False
    assert provenance["input_data_embedded"] is False
    assert provenance["x64"] is True
    encoded = json.dumps(artifact)
    assert "/Users/" not in encoded and "/home/" not in encoded

    arms = {arm["mode"]: arm for arm in artifact["arms"]}
    assert set(arms) == {"flat", "batched", "auto"}
    gib = 1024.0**3
    shipped = arms["auto"]["detail"]
    assert arms["auto"]["completed"] is True
    assert shipped["resolution"]["mpol"] == shipped["resolution"]["ntor"] == 10
    assert shipped["sweep_policy"] == {
        "batch": True,
        "checkpoint": True,
        "max_batch": 4096,
        "min_batch": 128,
        "working_set_bytes": 512 * 1024**2,
    }
    # The automatic batch must still land on the point count this deck was
    # measured with, or the calibration in strong_force.py has drifted.
    assert shipped["sweep_batch_points"] == 4096

    # The forward sweep is the allocation users reported: the flat arm still
    # reaches tens of GiB, and the shipped certificate is a small fraction
    # of it.
    assert arms["flat"]["peak_rss_bytes"] / gib > 16.0
    assert shipped["certificate_peak_rss_bytes"] / gib < 8.0
    assert (
        arms["flat"]["peak_rss_bytes"]
        > 4.0 * shipped["certificate_peak_rss_bytes"]
    )
    # The remat boundary is the reverse-mode half: batching alone leaves the
    # chart build storing whole-grid linearization residuals.  On the
    # measurement host that arm does not survive the chart stage at all --
    # the record stops at "chart" with no chart peak -- which is the stronger
    # statement; where it does survive, it must cost well over the shipped
    # policy's chart peak.
    batched = arms["batched"]
    if batched["completed"]:
        assert (
            batched["detail"]["chart_peak_rss_bytes"]
            > 1.5 * shipped["chart_peak_rss_bytes"]
        )
    else:
        assert batched["detail"]["stage"] == "chart"
        assert "chart_peak_rss_bytes" not in batched["detail"]
    assert "polish_memory_w7x.json" in (
        ROOT / "docs" / "reference" / "performance.rst"
    ).read_text()


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
    bundle = json.loads(
        (ROOT / "benchmarks" / "strong_force_cases_m4.json").read_text()
    )
    native = bundle["cases"]["shaped_tokamak_pressure"]["sources"]["VMEX"]
    native_report = native["polish_report"]
    native_final = native["final_certificate"]
    assert native["measurement_dirty"] is False
    # solvax provenance: either a git checkout (commit + clean flag) or a
    # released wheel (recorded version, no source tree).
    if native["solvax_source"] is not None:
        assert native["solvax_source"]["dirty"] is False
        assert re.fullmatch(r"[0-9a-f]{40}", native["solvax_source"]["commit"])
    else:
        assert native["versions"]["solvax"]
    assert native["solvax_least_squares"] is True
    # Acceptance is the independent certificate (asserted below); the
    # Gauss-Newton solver's own success flag is a recorded diagnostic - a
    # certified state whose solver ran out its step budget is accepted.
    assert native_report["converged"] is True
    assert isinstance(native_report["least_squares_success"], bool)
    assert native_report["least_squares_relative_optimality"] <= 1.0e-3
    assert native_final["normalized_l2"] <= native["validation_tolerance"]
    assert native_final["radial_refinement"] <= native[
        "radial_refinement_tolerance"
    ]
    assert native_report["minimum_signed_jacobian"] > 0.0
    assert native["external_source"]["success"] is True
    # Recorded, not gated: wall time is provenance under the accuracy-only
    # figure policy, and the empty-cache first run legitimately pays full
    # compilation.
    assert native["external_source"]["timing_seconds"]["total"] > 0.0
    assert native["total_peak_rss_increase_mib"] < 5120.0


def test_cross_code_certificates_are_clean_and_comparable() -> None:
    bundle = json.loads(
        (ROOT / "benchmarks" / "strong_force_cases_m4.json").read_text()
    )
    assert bundle["schema"] == "vmex.strong-force-comparison-cases/1"
    artifacts = bundle["cases"]["shaped_tokamak_pressure"]["sources"]
    reference_rho = artifacts["VMEC2000"]["radial_profile"]["rho"]
    for name, artifact in artifacts.items():
        assert artifact["measurement_dirty"] is False
        profile = (
            artifact["final_certificate"]["radial_profile"]
            if name == "VMEX"
            else artifact["radial_profile"]
        )
        rho = np.asarray(profile["rho"])
        assert len(rho) >= 64
        assert np.all(np.diff(rho) > 0.0)
        assert 0.0 < rho[0] < rho[-1] <= 1.0
        if name != "VMEX":
            assert profile["rho"] == reference_rho
        assert len(
            profile["flux_surface_average_force_density"]
        ) == len(reference_rho)
        assert len(
            profile["flux_surface_normalized_l2"]
        ) == len(reference_rho)
        assert np.all(
            np.isfinite(profile["flux_surface_normalized_l2"])
        )
        refinement = (
            artifact["final_certificate"]["radial_refinement"]
            if name == "VMEX"
            else artifact["metrics"]["radial_refinement_difference"]
        )
        assert refinement < 1.0e-3
        assert artifact["external_source"]["success"] is True

    normalized = {
        name: (
            artifact["final_certificate"]["normalized_l2"]
            if name == "VMEX"
            else artifact["metrics"]["normalized_l2"]
        )
        for name, artifact in artifacts.items()
    }
    assert normalized["VMEC2000"] == pytest.approx(
        normalized["VMEC++"], rel=5.0e-9
    )
    assert normalized["VMEX"] < 0.2 * normalized["VMEC2000"]


def test_readme_strong_force_figure_matches_committed_sources() -> None:
    metadata = json.loads(
        (ROOT / "benchmarks" / "strong_force_comparison_m4.json").read_text()
    )
    assert metadata["schema"] == "vmex.strong-force-readme-figure/4"
    figure = ROOT / metadata["figure"]
    assert figure.is_file()
    assert hashlib.sha256(figure.read_bytes()).hexdigest() == metadata[
        "figure_sha256"
    ]
    summary_figure = ROOT / metadata["summary_figure"]
    assert summary_figure.is_file()
    assert hashlib.sha256(summary_figure.read_bytes()).hexdigest() == metadata[
        "summary_figure_sha256"
    ]
    cases = metadata["cases"]
    assert set(cases) == {"shaped_tokamak_pressure", "nfp2_QA_finite_beta"}
    for case in cases.values():
        for source in case["sources"].values():
            path = ROOT / source["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    tokamak = cases["shaped_tokamak_pressure"]["sources"]
    assert set(tokamak) == {"VMEX", "VMEC2000", "VMEC++", "DESC"}
    assert tokamak["VMEX"]["normalized_l2"] < tokamak["DESC"]["normalized_l2"]

    stellarator = cases["nfp2_QA_finite_beta"]["sources"]
    bundle = json.loads((ROOT / stellarator["DESC"]["path"]).read_text())
    desc = bundle["cases"]["nfp2_QA_finite_beta"]["sources"]["DESC"]
    assert desc["external_source"]["success"] is True
    representation = desc["external_source"]["representation"]
    assert representation["L"] >= 16
    assert representation["M"] >= 10 and representation["N"] >= 10
    assert desc["metrics"]["radial_refinement_difference"] < 1.0e-3
    # No timing-ratio claims: the README figure policy is accuracy only;
    # runtime evidence lives in benchmarks/baselines, not in guards.
    readme = (ROOT / "README.md").read_text()
    assert metadata["figure"] in readme
    assert metadata["summary_figure"] in readme
    assert metadata["summary_independent_l2"]["after"] < (
        metadata["summary_independent_l2"]["before"]
    )


def test_fresh_deck_parity_artifact_is_provenanced_and_cited() -> None:
    """The fresh-deck xvmec2000 table is a hashed record, not prose.

    Every row the docs quote must trace to a deck hash, a reference-binary
    hash, and a vmex commit, and the page must cite the record by path so
    a reader can check the numbers rather than trust the phrasing.
    """
    path = ROOT / "benchmarks" / "fresh_decks_vs_vmec2000_2026-09-02.json"
    record = json.loads(path.read_text())
    assert record["schema"] == "vmex.fresh-deck-parity/1"
    provenance = record["provenance"]
    for key in ("vmex_commit", "vmex_version", "jax", "host", "protocol"):
        assert provenance[key]
    assert re.fullmatch(r"[0-9a-f]{16}", provenance["reference"]["sha256_prefix"])
    decks = record["decks"]
    assert len(decks) == 6
    for deck in decks:
        assert re.fullmatch(r"[0-9a-f]{16}", deck["sha256_prefix"]), deck["deck"]
        walls = deck["wall_s"]
        assert 0.0 < walls["vmex_warm"] <= walls["vmex_cold"], deck["deck"]
        assert walls["xvmec2000"] > 0.0
        assert deck["max_rel_diff"] and all(
            0.0 <= value < 1.0e-9 for value in deck["max_rel_diff"].values()
        ), deck["deck"]
    page = (ROOT / "docs" / "reference" / "performance.rst").read_text()
    assert path.name in page
    assert "machine precision" not in page.split("Fresh decks against")[1].split(
        "Numerical reproducibility")[0]


#: The two committed polish force-error records: the shaped tokamak whose
#: before/after pair the README quotes, and the bundled solovev deck that
#: shows what ``eps_F`` looks like when its denominator has collapsed.
POLISH_FORCE_ERROR_RECORDS = (
    "polish_force_error_2026-09-03.json",
    "polish_force_error_solovev_2026-09-03.json",
)


def _readme_number(value: float, digits: int = 3) -> str:
    """Format a measurement the way the README prints it.

    ``f"{v:.3e}"`` pads the exponent (``1.284e-02``); prose writes
    ``1.284e-2``.  Comparing through one formatter keeps the README pinned
    to the artifact digit for digit without dictating its typography.
    """
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


@pytest.mark.parametrize("name", POLISH_FORCE_ERROR_RECORDS)
def test_polish_force_error_records_are_provenanced(name: str) -> None:
    """Each polish force-error record must stand on its own provenance.

    31.2-R3: the README's previous "26-fold" polish figure had no artifact
    behind it and its two ends came from different export meshes.  The
    replacement is measured from a clean checkout with the deck hashed, the
    commit recorded, and the invocation kept, so a reader can re-run it.
    """
    record = json.loads((ROOT / "benchmarks" / name).read_text())
    assert record["schema"] == "vmex.strong-polish-benchmark/2"
    assert record["measurement_dirty"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", record["measurement_commit"])
    assert re.fullmatch(r"[0-9a-f]{16}", record["input_sha256_prefix"])
    assert record["input_data_embedded"] is False
    assert record["vmex_module"].startswith("vmex/")
    assert record["command"].startswith("python benchmarks/strong_polish.py")
    # The recorded invocation is provenance, not a filesystem tour.
    assert "/" not in record["command"].split("--output")[1]
    for key in ("python", "vmex", "jax", "numpy", "solvax"):
        assert record["versions"][key], key

    initial = record["initial_certificate"]["normalizations"]
    final = record["final_certificate"]["normalizations"]
    for block in (initial, final):
        # A record that quotes eps_F must carry the bound with it, and the
        # measures that stay informative when eps_F pins at that bound.
        assert "bounded above by 2" in block["saturation"]
        assert block["pointwise_eps_f"]["normalized_linf"] <= 2.0
        assert block["window_normalizations"]["s_min"] == 0.1
        assert block["window_normalizations"]["s_max"] == 0.99
        assert block["window_normalizations"]["node_count"] <= block[
            "global_normalizations"
        ]["node_count"]
        for scale in ("volume_average_force", "magnetic_relative_force_error"):
            assert block["global_normalizations"][scale] > 0.0, scale
    if name == POLISH_FORCE_ERROR_RECORDS[0]:
        # The tokamak record is a before/after pair: the correction has to
        # have moved something the pointwise metric cannot express.
        assert record["polish_report"]["converged"] is True
        assert final["absolute"]["near_axis_l2"] < initial[
            "absolute"
        ]["near_axis_l2"]
    else:
        # The solovev record is a single-state certificate (the driver
        # returns the unpolished state when it does not certify, so a
        # before/after pair there would be two copies of one state).
        assert record["diagnostics_only"] is True
        assert record["polish_report"]["termination_reason"] == "diagnostics-only"
        assert final == initial


def test_solovev_record_shows_the_certificate_at_its_ceiling() -> None:
    """31.2-R1's evidence: a shipped deck where eps_F carries no information.

    ``input.solovev`` peaks at 0.125 Pa, so the pressure gradient is five
    orders of magnitude below the magnetic pressure gradient and the
    pointwise denominator has collapsed.  ``eps_F`` sits at its ceiling
    while the vacuum-safe normalization still reports a small, ordinary
    force error -- which is the entire reason both are now reported.
    """
    record = json.loads(
        (ROOT / "benchmarks" / POLISH_FORCE_ERROR_RECORDS[1]).read_text()
    )
    block = record["initial_certificate"]["normalizations"]
    assert block["pointwise_eps_f"]["normalized_linf"] == pytest.approx(
        2.0, rel=1e-7
    )
    assert block["pointwise_eps_f"]["normalized_l2"] > 1.9
    scales = block["global_normalizations"]
    assert scales["volume_average_grad_pressure"] < 1.0
    assert scales["volume_average_magnetic_pressure_gradient"] > 1.0e3
    # The Panici ratio is legitimate but useless here; the vacuum-safe one
    # is small, which is the number a reader should be given.
    assert scales["relative_force_error"] > 100.0
    assert scales["magnetic_normalized_l2"] < 0.1

    readme = (ROOT / "README.md").read_text()
    assert POLISH_FORCE_ERROR_RECORDS[1] in readme
    assert f"`{block['pointwise_eps_f']['normalized_l2']:.3f}`" in readme
    assert f"`{_readme_number(scales['magnetic_normalized_l2'], 2)}`" in readme
    assert f"`{_readme_number(scales['volume_average_force'], 2)}`" in readme
    assert f"`{_readme_number(scales['volume_average_grad_pressure'], 2)}`" in readme
    assert (
        f"`{_readme_number(scales['volume_average_magnetic_pressure_gradient'], 2)}`"
        in readme
    )


def test_readme_polish_gain_matches_the_committed_record() -> None:
    """The README's polish table must be the record, digit for digit.

    Every row is read straight out of the artifact at the precision the
    README prints, so the table cannot drift from the measurement, and the
    withdrawn "26-fold" claim cannot come back.
    """
    record = json.loads(
        (ROOT / "benchmarks" / POLISH_FORCE_ERROR_RECORDS[0]).read_text()
    )
    readme = (ROOT / "README.md").read_text()
    assert POLISH_FORCE_ERROR_RECORDS[0] in readme
    assert "26-fold" not in readme.split("Earlier versions of this section")[0]

    initial = record["initial_certificate"]["normalizations"]
    final = record["final_certificate"]["normalizations"]
    quoted = [
        (record["initial_certificate"]["normalized_l2"],
         record["final_certificate"]["normalized_l2"]),
        (initial["absolute"]["l2"], final["absolute"]["l2"]),
        (initial["global_normalizations"]["relative_force_error"],
         final["global_normalizations"]["relative_force_error"]),
        (initial["global_normalizations"]["magnetic_relative_force_error"],
         final["global_normalizations"]["magnetic_relative_force_error"]),
        (initial["window_normalizations"]["relative_force_error"],
         final["window_normalizations"]["relative_force_error"]),
        (initial["absolute"]["near_axis_l2"], final["absolute"]["near_axis_l2"]),
        (initial["absolute"]["bulk_l2"], final["absolute"]["bulk_l2"]),
        (initial["absolute"]["edge_l2"], final["absolute"]["edge_l2"]),
    ]
    for before, after in quoted:
        for value in (before, after):
            assert f"`{_readme_number(value)}`" in readme, value
        ratio = before / after
        assert (
            f"{ratio:.2f}x" in readme or f"{ratio:.1f}x" in readme
        ), ratio


def test_readme_states_the_certificate_ceiling_and_its_selection() -> None:
    """31.2-R1 and 31.2-R6: the two claims the README must not make silently.

    eps_F may not appear without its bound, and a figure built from
    hand-picked cases may not be presented as general evidence.
    """
    readme = (ROOT / "README.md").read_text()
    assert "bounded above by 2 by construction" in readme
    assert "demonstrably wins" not in readme
    assert "not evidence of\na general advantage" in readme
    assert "were selected because certified polishing\nimproves them" in readme
    page = (
        ROOT / "docs" / "explanation" / "high-order-force-balance.rst"
    ).read_text()
    assert "bounded above by 2 by construction" in page
    assert "never be quoted on its own" in page


def test_strong_polish_record_redacts_its_output_destination() -> None:
    """A committed record must not disclose where it was produced.

    ``strong_polish.py`` writes its own invocation into the artifact so the
    measurement is re-runnable from the record.  The destination is a scratch
    path on the machine that ran it, and scratch paths carry user names, so
    only the file name survives into the JSON.
    """
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from strong_polish import _redacted_argv

    assert _redacted_argv(
        ["--input", "examples/data/input.solovev", "--ns", "15",
         "--output", "/scratch/someone/run one.json"]
    ) == ["--input", "examples/data/input.solovev", "--ns", "15",
          "--output", "run one.json"]
    assert _redacted_argv(["--output=/scratch/someone/run.json"]) == [
        "--output=run.json"
    ]
    assert _redacted_argv(["--ns", "15"]) == ["--ns", "15"]
    # A trailing bare --output has no value to redact and must not crash.
    assert _redacted_argv(["--output"]) == ["--output"]


def test_committed_reports_do_not_expose_personal_paths() -> None:
    """Release-facing text must remain portable between contributors."""
    text_suffixes = {".json", ".md", ".py", ".rst", ".toml"}
    for directory in ("benchmarks", "docs", "examples"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in text_suffixes and "_build" not in path.parts:
                text = path.read_text(errors="replace")
                assert "/Users/" not in text, path
                assert "MacBook-Pro.local" not in text, path
