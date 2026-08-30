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
