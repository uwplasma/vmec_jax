"""Shared-Boozer composition of the J-invariance and maximum-J residuals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import jax.numpy as jnp

from vmex.core import maxj as maxj_mod
from vmex.core import qi as qi_mod


def _boozer(outer_mean=0.98):
    return {
        "bmnc_b": jnp.array([[1.0, 0.2], [outer_mean, 0.2]]),
        "xm_b": jnp.array([0.0, 0.0]),
        "xn_b": jnp.array([0.0, 2.0]),
        "iota_b": jnp.array([0.4, 0.45]),
        "G_b": jnp.array([2.0, 2.0]),
        "I_b": jnp.array([0.0, 0.0]),
        "nfp": 2,
        "s_b": jnp.array([0.25, 0.75]),
        "psi_b": jnp.array([0.25, 0.75]),
        "psi_edge": jnp.asarray(1.0),
    }


_OPTIONS = dict(nalpha=5, points_per_period=64, num_periods=4, max_wells=6)


def _counting_boozer(monkeypatch):
    """Patch both modules' Boozer bindings with one counting fake."""
    booz = _boozer()
    calls = {"count": 0}

    def fake(*args, **kwargs):
        calls["count"] += 1
        return booz

    monkeypatch.setattr(qi_mod, "boozer_bmnc_state", fake)
    monkeypatch.setattr(maxj_mod, "boozer_bmnc_state", fake)
    return calls


def test_composed_result_equals_independent_classes(monkeypatch):
    """One Boozer pass reproduces both class evaluations exactly."""
    calls = _counting_boozer(monkeypatch)
    surfaces, pitch = [0.25, 0.75], [1.0 / 1.1]
    state, rt = object(), object()
    eq = SimpleNamespace(state=state, runtime=rt)

    qi_term = qi_mod.JInvariantQIResidual(surfaces, pitch, **_OPTIONS)
    maxj_term = maxj_mod.MaximumJResidual(surfaces, pitch, **_OPTIONS)
    qi_rows = np.asarray(qi_term(eq))
    maxj_rows = np.asarray(maxj_term(eq))
    assert calls["count"] == 2                    # one transform per class

    composed = maxj_mod.qi_and_maximum_j_from_boozer(
        state, rt, surfaces=surfaces, pitch=pitch,
        qi_options=_OPTIONS, maxj_options=_OPTIONS)
    assert calls["count"] == 3                    # a single extra transform

    np.testing.assert_array_equal(
        np.asarray(composed["qi"]["residuals1d"]), qi_rows)
    np.testing.assert_array_equal(
        np.asarray(composed["maximum_j"]["residuals1d"]), maxj_rows)
    assert float(composed["qi"]["total"]) == pytest.approx(
        float(qi_term.total(eq)))
    assert float(composed["maximum_j"]["total"]) == pytest.approx(
        float(maxj_term.total(eq)))
    assert composed["boozer"] is not None
    assert int(composed["boozer"]["nfp"]) == 2


def test_composed_options_forward_to_each_layer(monkeypatch):
    """Per-layer options reach only their layer (weights stay shared)."""
    _counting_boozer(monkeypatch)
    composed = maxj_mod.qi_and_maximum_j_from_boozer(
        object(), object(), surfaces=[0.25, 0.75], pitch=[1.0 / 1.1],
        weights=[1.0, 4.0],
        qi_options=dict(_OPTIONS, quadrature_order=32),
        maxj_options=dict(_OPTIONS, target=-0.5))
    # the maximum-J target shifts the violation threshold; residuals differ
    # from the default-target evaluation with the same weights
    default = maxj_mod.qi_and_maximum_j_from_boozer(
        object(), object(), surfaces=[0.25, 0.75], pitch=[1.0 / 1.1],
        weights=[1.0, 4.0],
        qi_options=dict(_OPTIONS, quadrature_order=32),
        maxj_options=_OPTIONS)
    assert (float(composed["maximum_j"]["total"])
            != float(default["maximum_j"]["total"]))
    np.testing.assert_array_equal(
        np.asarray(composed["qi"]["residuals1d"]),
        np.asarray(default["qi"]["residuals1d"]))


def test_joint_residual_uses_one_transform_and_cost_weights(monkeypatch):
    calls = _counting_boozer(monkeypatch)
    term = maxj_mod.JInvariantQIAndMaximumJResidual(
        [0.25, 0.75], [1.0 / 1.1], qi_weight=4.0, maxj_weight=9.0,
        qi_options=_OPTIONS, maxj_options=_OPTIONS,
    )
    eq = SimpleNamespace(state=object(), runtime=object())
    rows = np.asarray(term(eq))
    assert calls["count"] == 1
    shared = maxj_mod.qi_and_maximum_j_from_boozer(
        eq.state, eq.runtime, surfaces=[0.25, 0.75], pitch=[1.0 / 1.1],
        qi_options=_OPTIONS, maxj_options=_OPTIONS,
    )
    expected = np.concatenate([
        2.0 * np.asarray(shared["qi"]["residuals1d"]),
        3.0 * np.asarray(shared["maximum_j"]["residuals1d"]),
    ])
    np.testing.assert_array_equal(rows, expected)
    assert calls["count"] == 2
    assert float(term.total(eq)) == pytest.approx(float(rows @ rows))
    with pytest.raises(ValueError, match="non-negative"):
        maxj_mod.JInvariantQIAndMaximumJResidual(
            [0.25, 0.75], [1.0], qi_weight=-1.0)
    with pytest.raises(ValueError, match="finite surfaces"):
        maxj_mod.JInvariantQIAndMaximumJResidual([0.25, np.nan], [1.0])
    with pytest.raises(ValueError, match="weights must be finite"):
        maxj_mod.JInvariantQIAndMaximumJResidual(
            [0.25, 0.75], [1.0], weights=[1.0, np.nan])
    with pytest.raises(ValueError, match="same length"):
        maxj_mod.JInvariantQIAndMaximumJResidual(
            [0.25, 0.75], [1.0], weights=[1.0])
    with pytest.raises(ValueError, match="must be finite"):
        maxj_mod.JInvariantQIAndMaximumJResidual(
            [0.25, 0.75], [1.0], maxj_weight=np.inf)
    with pytest.raises(ValueError, match="must be positive"):
        maxj_mod.JInvariantQIAndMaximumJResidual(
            [0.25, 0.75], [1.0], mboz=0)
