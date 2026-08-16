"""Scientific and differentiation checks for the maximum-J residual."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

import jax
import jax.numpy as jnp

from vmex.core import maxj
from vmex.core import implicit as im
from vmex.core.input import VmecInput

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


def _boozer(outer_mean=1.02, *, psi=(0.25, 0.75)):
    return {
        "bmnc_b": jnp.array([[1.0, 0.2], [outer_mean, 0.2]]),
        "xm_b": jnp.array([0.0, 0.0]),
        "xn_b": jnp.array([0.0, 2.0]),
        "iota_b": jnp.array([0.4, 0.45]),
        "G_b": jnp.array([2.0, 2.0]),
        "I_b": jnp.array([0.0, 0.0]),
        "nfp": 2,
        "s_b": jnp.array([0.25, 0.75]),
        "psi_b": jnp.asarray(psi),
        "psi_edge": jnp.asarray(1.0 if psi[1] > 0.0 else -1.0),
    }


def _residual(
    outer_mean=1.02, *, psi=(0.25, 0.75), pitch=(1.0 / 1.1,), **options,
):
    booz = _boozer(outer_mean, psi=psi)
    settings = dict(
        nalpha=5, points_per_period=64, num_periods=4, max_wells=6)
    settings.update(options)
    return maxj.maximum_j_residual_from_boozer(
        bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
        nfp=booz["nfp"], psi_b=booz["psi_b"],
        psi_edge=booz["psi_edge"], pitch=pitch, **settings)


def test_maximum_j_sign_and_signed_flux_convention():
    favorable = _residual(1.02)
    adverse = _residual(0.98)
    reversed_flux = _residual(1.02, psi=(-0.25, -0.75))
    assert favorable["valid_pitch_pair"][0, 0]
    assert float(favorable["total"]) == 0.0
    assert np.all(np.asarray(
        favorable["relative_slope"][favorable["matched_well_mask"]]) < 0.0)
    assert float(adverse["total"]) > 1.0e-3
    assert np.all(np.asarray(
        adverse["relative_slope"][adverse["matched_well_mask"]]) > 0.0)
    assert float(reversed_flux["total"]) == 0.0
    assert float(favorable["maximum_j_fraction"]) == pytest.approx(1.0)
    assert float(adverse["maximum_j_fraction"]) == pytest.approx(0.0)
    assert float(favorable["excluded_pitch_fraction"]) == pytest.approx(0.0)
    np.testing.assert_allclose(
        reversed_flux["relative_slope"][reversed_flux["matched_well_mask"]],
        favorable["relative_slope"][favorable["matched_well_mask"]])
    np.testing.assert_allclose(
        reversed_flux["dJ_ds"][reversed_flux["matched_well_mask"]],
        favorable["dJ_ds"][favorable["matched_well_mask"]])

    depths = _residual(1.02, pitch=(1.0 / 1.1, 1.0 / 0.85))
    assert float(jnp.min(depths["trapping_depth"])) < 0.5
    assert float(jnp.max(depths["trapping_depth"])) > 0.5
    assert float(depths["shallow_maximum_j_fraction"]) == pytest.approx(1.0)
    assert float(depths["deep_maximum_j_fraction"]) == pytest.approx(1.0)


def test_maximum_j_slope_matches_adaptive_quadrature():
    pitch = 1.0 / 1.1

    def action(mean):
        root = 0.5 * np.arccos((1.1 - mean) / 0.2)
        return 2.0 * quad(
            lambda phi: (
                np.sqrt(max(1.0 - pitch * (mean + 0.2 * np.cos(2.0 * phi)), 0.0))
                * 2.0 / (mean + 0.2 * np.cos(2.0 * phi))
            ),
            root, np.pi - root, epsabs=1.0e-13,
        )[0]

    lo, hi = action(1.0), action(1.02)
    expected = ((hi - lo) / 0.5) / (0.5 * (lo + hi))
    result = _residual(
        1.02, points_per_period=256, quadrature_order=96)
    np.testing.assert_allclose(
        result["relative_slope"][result["matched_well_mask"]],
        expected, rtol=5.0e-6)


def test_well_identity_matching_and_invalid_topology():
    sine = 0.02
    booz = _boozer(1.0)
    shifted_options = dict(
        bmnc_b=jnp.array([[1.0, 0.2], [1.0, np.sqrt(0.2**2 - sine**2)]]),
        bmns_b=jnp.array([[0.0, 0.0], [0.0, sine]]),
        xm_b=booz["xm_b"], xn_b=booz["xn_b"], iota_b=booz["iota_b"],
        G_b=booz["G_b"], I_b=booz["I_b"], nfp=booz["nfp"],
        psi_b=booz["psi_b"], psi_edge=booz["psi_edge"],
        pitch=[1.0 / 1.1], nalpha=5, points_per_period=64,
        num_periods=4, max_wells=6)
    shifted = maxj.maximum_j_residual_from_boozer(**shifted_options)
    assert shifted["valid_pitch_pair"][0, 0]
    assert int(jnp.sum(shifted["matched_well_mask"])) == 20
    assert float(jnp.nanmax(shifted["match_distance"])) < 0.06

    unmatched = maxj.maximum_j_residual_from_boozer(
        **shifted_options, match_tolerance=1.0e-4)
    assert not unmatched["valid_pitch_pair"][0, 0]
    assert np.isnan(float(unmatched["total"]))
    assert float(unmatched["excluded_pitch_fraction"]) == pytest.approx(1.0)

    nonmonotone = _residual(1.02, psi=(0.75, 0.25))
    assert not nonmonotone["valid_pitch_pair"][0, 0]
    assert np.isnan(float(nonmonotone["total"]))

    absent = maxj.maximum_j_residual_from_boozer(
        bmnc_b=[[1.0], [1.1]], xm_b=[0.0], xn_b=[0.0],
        iota_b=[0.4, 0.4], G_b=[2.0, 2.0], I_b=[0.0, 0.0], nfp=2,
        psi_b=[0.25, 0.75], psi_edge=1.0, pitch=[1.0])
    assert not absent["valid_pitch_pair"][0, 0]
    assert np.isnan(float(absent["total"]))


def test_maximum_j_jit_and_ad_match_finite_difference():
    def objective(outer_mean):
        return _residual(outer_mean)["total"]

    outer_mean = jnp.asarray(0.98)
    compiled = jax.jit(objective)(outer_mean)
    derivative = jax.grad(objective)(outer_mean)
    step = 1.0e-5
    finite_difference = (
        objective(outer_mean + step) - objective(outer_mean - step)) / (2.0 * step)
    assert compiled == pytest.approx(objective(outer_mean))
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-4)


def test_maximum_j_composable_interface(monkeypatch):
    booz = _boozer(0.98)
    monkeypatch.setattr(maxj, "boozer_bmnc_state", lambda *args, **kwargs: booz)
    term = maxj.MaximumJResidual(
        [0.25, 0.75], [1.0 / 1.1], mboz=2, nboz=2, nalpha=5,
        points_per_period=64, num_periods=4, max_wells=6)
    eq = SimpleNamespace(state=object(), runtime=object())
    rows = term(eq)
    assert np.all(np.isfinite(np.asarray(rows)))
    assert float(term.total(eq)) == pytest.approx(float(jnp.sum(rows**2)))


def test_maximum_j_input_guards():
    with pytest.raises(ValueError, match="increasing surfaces"):
        maxj.MaximumJResidual([0.5], [1.0])
    with pytest.raises(ValueError, match="positive finite"):
        maxj.MaximumJResidual([0.25, 0.75], [0.0])
    with pytest.raises(ValueError, match="weights"):
        maxj.MaximumJResidual([0.25, 0.75], [1.0], weights=[1.0])
    with pytest.raises(ValueError, match="at least two"):
        maxj.maximum_j_residual_from_boozer(
            bmnc_b=[[1.0]], xm_b=[0.0], xn_b=[0.0], iota_b=[0.4],
            G_b=[2.0], I_b=[0.0], nfp=2, psi_b=[0.5],
            psi_edge=1.0, pitch=[1.0])
    with pytest.raises(ValueError, match="match_tolerance"):
        _residual(match_tolerance=0.0)
    with pytest.raises(ValueError, match="pitch_weights"):
        _residual(pitch_weights=[1.0, 1.0])


@pytest.mark.full
def test_maximum_j_implicit_boundary_gradient_matches_reconverged_fd():
    inp = VmecInput.from_file(
        Path(__file__).resolve().parents[1] / "examples/data/input.nfp1_QI")
    params = im.params_from_input(inp, device=None)
    term = maxj.MaximumJResidual(
        [0.35, 0.65], [1.0], target=-0.5, mboz=6, nboz=6, nalpha=5,
        points_per_period=32, num_periods=3, max_wells=6,
        quadrature_order=32)

    def objective(p):
        solution = im.run(
            inp, p, ns=15, ftol=1.0e-13, max_iterations=20000,
            adjoint_tol=1.0e-13, device=None)
        return term.total_state(solution.state, solution.runtime)

    index = (int(inp.ntor), 1)
    implicit = float(np.asarray(jax.grad(objective)(params).rbc)[index])
    step = 5.0e-5
    values = [
        objective(dataclasses.replace(
            params, rbc=params.rbc.at[index].add(sign * step)))
        for sign in (-1.0, 1.0)
    ]
    finite_difference = float((values[1] - values[0]) / (2.0 * step))
    relative = abs(implicit - finite_difference) / max(
        abs(implicit), abs(finite_difference))
    assert relative < 3.0e-3
