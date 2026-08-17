"""Thin VMEX-to-NEO_JAX effective-ripple adapter checks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vmex.core import neoclassical

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_epsilon_effective_boozer_adapter_preserves_surface_labels(monkeypatch):
    values = np.array([1.0e-4, 2.0e-4])
    calls = {}

    def run_neo(booz, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(eps_eff=values)

    monkeypatch.setattr(neoclassical, "_neo_imports", lambda: (lambda: None, run_neo))
    s, actual = neoclassical.epsilon_effective_from_boozer(
        {"s_b": np.array([0.25, 0.75])}, config=object())
    np.testing.assert_allclose(s, [0.25, 0.75])
    np.testing.assert_allclose(actual, values)
    assert calls == {
        "config": calls["config"], "use_jax": True, "progress": False,
        "jax_surface_scan": True}


def test_epsilon_effective_rejects_lasym_before_importing_optional_backend():
    with np.testing.assert_raises_regex(NotImplementedError, "LASYM"):
        neoclassical.epsilon_effective_from_wout(SimpleNamespace(lasym=True))


@pytest.mark.full
def test_epsilon_effective_matches_neo_reference():
    """The in-memory adapter retains a NEO/STELLOPT-parity QA profile."""
    pytest.importorskip("neo_jax")
    script = f"""
import json
import vmex as vj
from vmex import optimize as opt
from vmex.core import neoclassical
equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file({str(DATA_DIR / 'input.LandremanPaul2021_QA_lowres')!r}))
s, values = neoclassical.epsilon_effective_from_wout(
    equilibrium.wout, surfaces=[0.2, 0.5, 0.8],
    config=neoclassical.diagnostic_neo_config())
print(json.dumps([list(map(float, s)), list(map(float, values))]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True, timeout=90)
    s, values = json.loads(completed.stdout.splitlines()[-1])
    np.testing.assert_allclose(s, [0.19387755, 0.5, 0.80612245], rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        values, [1.29683058e-7, 2.17541367e-7, 2.49843084e-7], rtol=5e-5)
