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


def test_epsilon_effective_reads_both_neo_result_and_surface_conventions(monkeypatch):
    """NEO_JAX names the profile ``eps_eff`` or ``epsilon_effective``.

    ``BoozerData`` carries its own surface labels in ``es``; a mapping without
    ``s_b`` has none, and the adapter then falls back to surface indices so a
    plot never mislabels the radial axis.
    """
    values = np.array([3.0e-5, 4.0e-5])
    monkeypatch.setattr(
        neoclassical, "_neo_imports",
        lambda: (lambda: None,
                 lambda booz, **_kwargs: SimpleNamespace(
                     epsilon_effective=values)))
    surfaces, actual = neoclassical.epsilon_effective_from_boozer(
        SimpleNamespace(es=np.array([0.1, 0.9])))
    np.testing.assert_allclose(surfaces, [0.1, 0.9])
    np.testing.assert_allclose(actual, values)

    surfaces, _ = neoclassical.epsilon_effective_from_boozer({"bmnc_b": None})
    np.testing.assert_allclose(surfaces, [0.0, 1.0])


def test_epsilon_effective_rejects_lasym_before_importing_optional_backend():
    with np.testing.assert_raises_regex(NotImplementedError, "LASYM"):
        neoclassical.epsilon_effective_from_wout(SimpleNamespace(lasym=True))


def test_diagnostic_config_is_the_bounded_summary_resolution(monkeypatch):
    """Summary figures ask NEO for a radial trend, not a transport number.

    The bounded settings (Nemov PoP 6, 4622 (1999) integrates along field
    lines until ``acc_req`` is met) keep one plot within seconds; publication
    numbers pass their own ``NeoConfig``.
    """
    monkeypatch.setattr(
        neoclassical, "_neo_imports", lambda: (SimpleNamespace, None))
    config = neoclassical.diagnostic_neo_config()
    assert (config.theta_n, config.phi_n, config.npart) == (16, 16, 8)
    assert config.acc_req == 0.2                      # loose on purpose
    assert config.nstep_max >= config.nstep_min > 0


def test_epsilon_effective_from_wout_snaps_surfaces_and_negates_the_nu_table(
    monkeypatch,
):
    """The in-memory adapter is a pure relabelling of a booz_xform run.

    NEO's Boozer toroidal-angle table ``pmns_b`` is booz_xform's ``numns_b``
    with the opposite sign (NEO_JAX ``BoozerData``; booz_xform stores
    ``nu = phi_B - phi``), and requested surfaces are snapped to the Boozer
    surfaces that were actually transformed rather than interpolated.
    """
    import booz_xform_jax

    numns = np.array([[0.1, -0.2], [0.3, -0.4]])
    captured = {}

    class FakeBoozXform:
        s_in = np.linspace(0.05, 0.95, 10)
        nfp = 3
        xm_b, xn_b = np.array([0, 1]), np.array([0, 3])
        iota = np.linspace(0.4, 0.5, 10)
        Boozer_I, Boozer_G = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        rmnc_b, zmns_b = np.ones((2, 2)), np.zeros((2, 2))
        numns_b, bmnc_b = numns, np.array([[1.0, 0.05], [1.0, 0.06]])
        s_b = np.array([0.25, 0.75])

        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def read_wout_data(self, wout):
            captured["wout"] = wout

        def run(self):
            captured["surfs"] = list(self.compute_surfs)

    def run_neo(booz, **_kwargs):
        captured["booz"] = booz
        return SimpleNamespace(eps_eff=np.array([1.0e-4, 2.0e-4]))

    monkeypatch.setattr(booz_xform_jax, "Booz_xform", FakeBoozXform)
    monkeypatch.setattr(neoclassical, "_neo_imports", lambda: (dict, run_neo))

    wout = SimpleNamespace(lasym=False)
    s, values = neoclassical.epsilon_effective_from_wout(
        wout, surfaces=(0.24, 0.26, 0.77), mboz=8, nboz=6)

    assert captured["init"] == {"verbose": 0, "mboz": 8, "nboz": 6}
    assert captured["wout"] is wout
    # 0.24 and 0.26 snap to the same transformed surface: three requests, two runs
    assert captured["surfs"] == [2, 7]
    booz = captured["booz"]
    assert booz["ns_b"] == 2 and booz["nfp_b"] == 3
    np.testing.assert_allclose(booz["pmns_b"], -numns)
    np.testing.assert_allclose(booz["iota_b"], FakeBoozXform.iota[[2, 7]])
    np.testing.assert_allclose(s, FakeBoozXform.s_b)
    np.testing.assert_allclose(values, [1e-4, 2e-4])


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
