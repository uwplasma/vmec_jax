"""Analytic and differentiation checks for quasi-isodynamic residuals."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from vmex.core import qi
from vmex.core import implicit as im
from vmex.core.input import VmecInput

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


def _boozer(perturbation=0.0):
    return {
        "bmnc_b": jnp.array([[1.0, 0.2, perturbation]]),
        "xm_b": jnp.array([0.0, 0.0, 1.0]),
        "xn_b": jnp.array([0.0, 2.0, 0.0]),
        "iota_b": jnp.array([0.4]),
        "G_b": jnp.array([2.0]),
        "I_b": jnp.array([0.0]),
        "nfp": 2,
        "s_b": jnp.array([0.5]),
    }


def _action_residual(perturbation):
    booz = _boozer(perturbation)
    return qi.j_invariant_qi_residual_from_boozer(
        bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
        nfp=booz["nfp"], pitch=[1.0], nalpha=9, num_periods=4,
        points_per_period=64, max_wells=6)


def test_action_residual_zero_and_non_qi_ordering():
    exact = _action_residual(0.0)
    perturbed = _action_residual(0.06)
    assert exact["valid_pitch"][0, 0]
    assert float(exact["total"]) < 1.0e-24
    assert float(perturbed["total"]) > 1.0e-5
    asymmetric = qi.j_invariant_qi_residual_from_boozer(
        bmnc_b=_boozer()["bmnc_b"], bmns_b=[[0.0, 0.0, 0.04]],
        xm_b=_boozer()["xm_b"], xn_b=_boozer()["xn_b"],
        iota_b=_boozer()["iota_b"], G_b=_boozer()["G_b"],
        I_b=_boozer()["I_b"], nfp=2, pitch=[1.0], nalpha=9,
        points_per_period=64, num_periods=4, max_wells=6)
    assert np.isfinite(float(asymmetric["total"]))

    constant = qi.j_invariant_qi_residual_from_boozer(
        bmnc_b=[[1.0]], xm_b=[0.0], xn_b=[0.0], iota_b=[0.4],
        G_b=[2.0], I_b=[0.0], nfp=2, pitch=[1.0])
    assert not constant["valid_pitch"][0, 0]
    assert np.isnan(float(constant["total"]))

    booz = _boozer(0.06)
    totals = [
        float(qi.j_invariant_qi_residual_from_boozer(
            bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
            nfp=booz["nfp"], pitch=[1.0], nalpha=9, num_periods=4,
            points_per_period=points, max_wells=6)["total"])
        for points in (32, 64, 128)
    ]
    assert abs(totals[1] - totals[2]) < abs(totals[0] - totals[2])


def test_action_residual_ad_matches_finite_difference():
    def objective(amplitude):
        return _action_residual(amplitude)["total"]

    amplitude = jnp.asarray(0.06, dtype=jnp.float64)
    compiled = jax.jit(objective)(amplitude)
    derivative = jax.grad(objective)(amplitude)
    step = 1.0e-5
    finite_difference = (
        objective(amplitude + step) - objective(amplitude - step)) / (2.0 * step)
    assert compiled == pytest.approx(objective(amplitude))
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-4)


def test_objective_terms_share_the_composable_interface(monkeypatch):
    booz = _boozer(0.04)
    monkeypatch.setattr(qi, "boozer_bmnc_state", lambda *args, **kwargs: booz)
    eq = SimpleNamespace(state=object(), runtime=object())

    action = qi.JInvariantQIResidual(
        [0.5], [1.0], nalpha=7, points_per_period=64,
        num_periods=4, max_wells=6)
    action_rows = action(eq)
    assert np.all(np.isfinite(np.asarray(action_rows)))
    assert float(action.total(eq)) == pytest.approx(float(jnp.sum(action_rows**2)))

    constructed = qi.ConstructedQIResidual(
        [0.5], nphi=41, nalpha=7, n_bounce=7)
    constructed_rows = constructed.residuals(eq)
    assert np.all(np.isfinite(np.asarray(constructed_rows)))
    assert float(constructed.total(eq)) == pytest.approx(
        float(jnp.sum(constructed_rows**2)))


def test_objective_input_validation():
    with pytest.raises(ValueError, match="positive finite"):
        qi.JInvariantQIResidual([0.5], [0.0])
    with pytest.raises(ValueError, match="weights"):
        qi.JInvariantQIResidual([0.3, 0.7], [1.0], weights=[1.0])
    with pytest.raises(ValueError, match="weights"):
        qi.ConstructedQIResidual([0.3, 0.7], weights=[1.0])
    with pytest.raises(ValueError, match="nalpha"):
        qi.j_invariant_qi_residual_from_boozer(
            bmnc_b=[[1.0]], xm_b=[0.0], xn_b=[0.0], iota_b=[0.4],
            G_b=[2.0], I_b=[0.0], nfp=2, pitch=[1.0], nalpha=1)
    with pytest.raises(ValueError, match="bmnc_b"):
        qi.j_invariant_qi_residual_from_boozer(
            bmnc_b=[1.0], xm_b=[0.0], xn_b=[0.0], iota_b=[0.4],
            G_b=[2.0], I_b=[0.0], nfp=2, pitch=[1.0])
    with pytest.raises(ValueError, match="weights"):
        qi.j_invariant_qi_residual_from_boozer(
            bmnc_b=[[1.0], [1.0]], xm_b=[0.0], xn_b=[0.0],
            iota_b=[0.4, 0.4], G_b=[2.0, 2.0], I_b=[0.0, 0.0],
            nfp=2, pitch=[1.0], weights=[1.0])


@pytest.mark.full
def test_implicit_boundary_gradient_matches_reconverged_fd():
    inp = VmecInput.from_file(
        Path(__file__).resolve().parents[1] / "examples/data/input.nfp1_QI")
    params = im.params_from_input(inp, device=None)
    term = qi.JInvariantQIResidual(
        [0.5], [1.0 / 0.95], mboz=6, nboz=6, nalpha=5,
        points_per_period=32, num_periods=3, max_wells=6,
        quadrature_order=32)

    def objective(p):
        solution = im.run(
            inp, p, ns=15, ftol=1.0e-12, max_iterations=12000,
            adjoint_tol=1.0e-12, device=None)
        return term.total_state(solution.state, solution.runtime)

    index = (int(inp.ntor), 1)
    implicit = float(np.asarray(jax.grad(objective)(params).rbc)[index])
    step = 3.0e-4
    values = [
        objective(dataclasses.replace(
            params, rbc=params.rbc.at[index].add(sign * step)))
        for sign in (-1.0, 1.0)
    ]
    finite_difference = float((values[1] - values[0]) / (2.0 * step))
    relative = abs(implicit - finite_difference) / max(
        abs(implicit), abs(finite_difference))
    assert relative < 3.0e-3
