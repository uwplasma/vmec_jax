"""Deterministic gates for the workflow profiling harness.

CI never asserts wall times (plan section 23.3).  What it can hold
deterministic: the registry's structure, the record schema, correct
compile counting on a trivially cheap injected workflow, and the
warm-regime contract that a same-shape repeat does not recompile.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "profile_workflows", ROOT / "benchmarks" / "profile_workflows.py")
profile_workflows = importlib.util.module_from_spec(_SPEC)
# Registered before exec: the module uses postponed annotations, and the
# dataclass decorator resolves them through sys.modules[cls.__module__].
sys.modules[_SPEC.name] = profile_workflows
_SPEC.loader.exec_module(profile_workflows)


def test_registry_rows_are_well_formed():
    assert profile_workflows.WORKFLOWS
    for ident, workflow in profile_workflows.WORKFLOWS.items():
        assert workflow.ident == ident
        assert workflow.title
        assert callable(workflow.build)
        for case in workflow.cases:
            assert (profile_workflows.DATA / case).exists(), case


def test_list_and_unknown_ident_handling(capsys):
    assert profile_workflows.main(["--list"]) == 0
    listed = capsys.readouterr().out
    for ident in profile_workflows.WORKFLOWS:
        assert ident in listed
    with pytest.raises(SystemExit):
        profile_workflows.main(["F999"])
    with pytest.raises(SystemExit):
        profile_workflows.main([])


def _tiny_workflow():
    """A one-jit workflow cheap enough for the PR lane (no equilibrium)."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def kernel(x):
        return (x * x + 1.0).sum()

    import numpy as np

    state = {"x": jnp.linspace(0.0, 1.0, 64)}
    # Parameter perturbation stays in numpy: an eager jnp op here would
    # itself compile one tiny executable on first use and be counted --
    # correctly -- against the variant.  Real workflows follow the same rule.
    perturbed = jnp.asarray(np.linspace(0.0, 1.5, 64))

    def run():
        return jax.block_until_ready(kernel(state["x"]))

    def run_newparams():
        state["x"] = perturbed                 # same shape, new values
        return jax.block_until_ready(kernel(state["x"]))

    return ({"run": run}, {"warm_newparams": run_newparams})


def test_compile_counting_and_warm_contract(monkeypatch):
    """First call compiles, same-shape repeats do not; the record proves it.

    The counter must survive vmex's import-time jax_logging_level = "ERROR";
    the harness imports vmex before installing the handler for exactly that
    reason, and this test would read compiles == 0 if that ordering broke.
    """
    tiny = profile_workflows.Workflow(
        "T0", "tiny self-test kernel", _tiny_workflow, ())
    monkeypatch.setitem(profile_workflows.WORKFLOWS, "T0", tiny)

    record = profile_workflows._run_in_process("T0", "warm")
    assert record["workflow"] == "T0"
    assert record["schema"] == profile_workflows.SCHEMA
    assert record["compile"]["run"]["compiles"] >= 1
    assert record["compile"]["warm"]["compiles"] == 0
    assert record["timing_s"]["warm"] <= record["timing_s"]["run"]
    assert record["memory_bytes"]["peak_host_rss"] > 0
    assert record["jax"]["x64"] in (True, False)

    newparams = profile_workflows._run_in_process("T0", "warm_newparams")
    assert newparams["compile"]["warm_newparams"]["compiles"] == 0

    # A workflow without the requested variant must refuse the label rather
    # than report a plain warm repeat under it.
    bare = profile_workflows.Workflow(
        "T1", "tiny kernel without variants",
        lambda: (_tiny_workflow()[0], {}), ())
    monkeypatch.setitem(profile_workflows.WORKFLOWS, "T1", bare)
    with pytest.raises(ValueError, match="no reshape variant"):
        profile_workflows._run_in_process("T1", "reshape")


def test_trace_dir_captures_one_xprof_trace_per_stage(monkeypatch, tmp_path):
    tiny = profile_workflows.Workflow(
        "T0", "tiny self-test kernel", _tiny_workflow, ())
    monkeypatch.setitem(profile_workflows.WORKFLOWS, "T0", tiny)
    profile_workflows._run_in_process("T0", "warm", trace_dir=tmp_path)
    trace_files = list((tmp_path / "T0" / "run").rglob("*.xplane.pb"))
    assert trace_files, "no XProf trace was written for the stage"


def test_record_schema_is_json_serializable(monkeypatch):
    tiny = profile_workflows.Workflow(
        "T0", "tiny self-test kernel", _tiny_workflow, ())
    monkeypatch.setitem(profile_workflows.WORKFLOWS, "T0", tiny)
    record = profile_workflows._run_in_process("T0", "warm")
    text = json.dumps(record, sort_keys=True)
    for key in ("commit", "case_sha256", "timing_s", "compile",
                "memory_bytes", "platform", "jax", "regime"):
        assert key in json.loads(text)


@pytest.mark.full   # two subprocess solves: nightly, not the PR lane
def test_cold_and_cache_reload_subprocess_regimes(tmp_path):
    """The cold child really is cold, and the reload really hits the cache."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "profile_workflows.py"),
         "F6", "--regimes", "cold", "cache_reload",
         "--cache-dir", str(tmp_path / "cache")],
        capture_output=True, text=True, timeout=3000, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    records = json.loads(out.stdout)
    by_regime = {record["regime"]: record for record in records}
    cold, reload_ = by_regime["cold"], by_regime["cache_reload"]
    assert cold["cache"]["entries_before"] == 0
    assert cold["cache"]["entries_after"] > 0
    assert reload_["cache"]["entries_before"] > 0
    # The reload claim: the populated cache made the second cold process
    # materially cheaper than the first.
    # Measured 0.76x on the smallest case; 0.9 keeps the claim (a populated
    # cache makes a new process cheaper) without flaking on a loaded runner.
    assert (reload_["timing_s"]["process_wall"]
            < 0.9 * cold["timing_s"]["process_wall"])


def test_desc_native_force_error_reports_both_published_normalizations():
    """The DESC row must be able to say what DESC itself measures.

    Certifying DESC's re-exported ``wout`` measures the export mesh and the
    spline lift; the artifact records DESC's own volume-averaged force error
    beside it, in the pressure-gradient normalization of Panici et al. 2023
    and the vacuum-safe magnetic-pressure one.
    """
    import json
    from pathlib import Path

    from benchmarks.run_external_equilibrium import _desc_native_force_error

    class _Equilibrium:
        def compute(self, keys):
            values = {"<|F|>_vol": 2.0, "<|grad(p)|>_vol": 4.0,
                      "<|grad(|B|^2)|/2mu0>_vol": 8.0}
            return {key: values[key] for key in keys}

    report = _desc_native_force_error(_Equilibrium())
    assert report["mean_force_density"] == 2.0
    assert report["normalized_by_pressure_gradient"] == 0.5
    assert report["normalized_by_magnetic_pressure"] == 0.25

    # a vacuum equilibrium has no pressure gradient to normalize by, and the
    # magnetic-pressure form is the one that survives
    class _Vacuum(_Equilibrium):
        def compute(self, keys):
            values = {"<|F|>_vol": 1.0, "<|grad(p)|>_vol": 0.0,
                      "<|grad(|B|^2)|/2mu0>_vol": 4.0}
            return {key: values[key] for key in keys}

    vacuum = _desc_native_force_error(_Vacuum())
    assert "normalized_by_pressure_gradient" not in vacuum
    assert vacuum["normalized_by_magnetic_pressure"] == 0.25

    artifact = json.loads(
        Path("benchmarks/desc_native_vs_lifted_2026-09-03.json").read_text())
    native = artifact["desc_native_force_error"]["normalized_by_pressure_gradient"]
    lifted = artifact["vmex_certificate_on_the_desc_wout"]["normalized_l2"]
    assert native < 1.0e-5 < lifted, "the recorded gap is the point of the artifact"
