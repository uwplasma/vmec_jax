"""Validation gates for the differentiable ideal-ballooning objective (R26h.h1).

Physics limits on bundled axisymmetric decks (small resolutions, fast):

- a zero-pressure tokamak (``input.circular_tokamak``, ``AM = 0``) must be
  ballooning-STABLE on every surface/field line (the drive term vanishes, so
  the spectrum of ``d/dη(g X')' = λ f X`` is strictly negative);
- the Solovev deck with ``pres_scale`` raised to reactor-grade beta must be
  ballooning-UNSTABLE (positive eigenvalue);
- on the finite-beta ``input.shaped_tokamak_pressure`` deck the growth-rate
  sign agrees with the Mercier expectation from the parity-proven wout
  engine (interior ``DMerc > 0`` and ballooning-stable at this beta);
- the smooth-max objective is AD-transparent: ``jax.grad`` w.r.t. a pressure
  rescale matches finite differences, and the state gradient (the piece the
  implicit-gradient lane composes with) is finite.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from vmex.core import optimize as opt
from vmex.core import stability as stab
from vmex.core.input import VmecInput

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solves: run jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
FAST = dict(npoints=81, nturns=3.0)


@pytest.fixture(scope="module")
def vacuum_eq():
    """Zero-pressure circular tokamak (AM = 0): the ballooning-stable limit."""
    eq = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.circular_tokamak"))
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def highbeta_eq():
    """Solovev deck at pres_scale = 3e4 (few-percent beta): ballooning-unstable."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    eq = opt.solve_equilibrium(dataclasses.replace(inp, pres_scale=3.0e4))
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def shaped_eq():
    """Finite-beta shaped tokamak, single 13-surface stage (fast)."""
    inp = VmecInput.from_file(DATA_DIR / "input.shaped_tokamak_pressure")
    inp = dataclasses.replace(inp, ns_array=np.array([13]),
                              ftol_array=np.array([1e-12]),
                              niter_array=np.array([2000]))
    eq = opt.solve_equilibrium(inp)
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def finite_beta_3d_eq():
    """Small finite-beta, current-constrained stellarator equilibrium."""
    eq = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.li383_low_res"))
    assert eq.result.converged
    return eq


def test_zero_pressure_case_is_ballooning_stable(vacuum_eq):
    lam = np.asarray(stab.ballooning_lambda(vacuum_eq.state, vacuum_eq.runtime, **FAST))
    assert lam.shape == (3, 4, 1)  # default surfaces x alphas x zeta0s
    assert np.all(np.isfinite(lam))
    assert np.all(lam < 0.0)


def test_high_pressure_case_is_ballooning_unstable(highbeta_eq):
    lam = np.asarray(stab.ballooning_lambda(highbeta_eq.state, highbeta_eq.runtime, **FAST))
    assert np.all(np.isfinite(lam))
    assert np.max(lam) > 0.0
    # Axisymmetry: every field line is equivalent up to the angular
    # discretization (different alphas sample the trig sums at different
    # points), so the per-surface spread over lines is truncation-level.
    spread = np.max(lam, axis=(1, 2)) - np.min(lam, axis=(1, 2))
    assert np.all(spread < 1e-2 * np.max(np.abs(lam)))


def test_growth_rate_sign_agrees_with_mercier(shaped_eq):
    """Mercier-stable finite-beta deck is also ballooning-stable (sign gate)."""
    dmerc = np.asarray(opt.d_merc(shaped_eq))
    assert np.all(dmerc[2:-1] > 0.0)  # interior Mercier-stable at this beta
    lam_max = float(stab.ballooning_growth_rate(
        shaped_eq.state, shaped_eq.runtime, reduction="max", **FAST))
    assert lam_max < 0.0


def test_traceable_dmerc_matches_wout_and_has_state_jvp(shaped_eq):
    """Pure-JAX Mercier profile retains wout parity and a finite tangent."""
    state, rt = shaped_eq.state, shaped_eq.runtime
    expected = np.asarray(opt.d_merc(shaped_eq))
    actual = np.asarray(jax.jit(stab.d_merc_state)(state, rt))
    np.testing.assert_allclose(actual[2:-1], expected[2:-1], rtol=1e-8,
                               atol=1e-13)

    tangent = jax.tree.map(jnp.zeros_like, state)
    tangent = dataclasses.replace(tangent, R_cos=jnp.ones_like(state.R_cos))
    _, dmerc_tangent = jax.jvp(lambda st: stab.d_merc_state(st, rt),
                               (state,), (tangent,))
    interior = np.asarray(dmerc_tangent)[2:-1]
    assert np.all(np.isfinite(interior))
    assert np.any(interior != 0.0)

    def total_dmerc(pressure_scale):
        setup = dataclasses.replace(rt.setup, mass=rt.setup.mass * pressure_scale)
        return jnp.sum(stab.d_merc_state(
            state, dataclasses.replace(rt, setup=setup))[2:-1])

    pressure_grad = jax.grad(total_dmerc)(1.0)
    assert np.isfinite(float(pressure_grad))
    assert float(pressure_grad) != 0.0
    h = 1e-3
    pressure_fd = (total_dmerc(1.0 + h) - total_dmerc(1.0 - h)) / (2.0 * h)
    np.testing.assert_allclose(pressure_grad, pressure_fd, rtol=1e-7)


def test_traceable_jdotb_and_glasser_profiles(shaped_eq):
    """J.B matches WOUT and D_R follows the published GGJ relation."""
    state, rt = shaped_eq.state, shaped_eq.runtime
    dmerc, jdotb, bdotb, shear, h_glasser = jax.jit(
        stab._mercier_profiles_state
    )(state, rt)
    np.testing.assert_allclose(
        np.asarray(jdotb),
        np.asarray(shaped_eq.wout.jdotb),
        rtol=1e-8,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        stab.jdotb_residual(state, rt),
        np.asarray(jdotb)[2:-1],
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(bdotb),
        np.asarray(shaped_eq.wout.bdotb),
        rtol=1e-8,
        atol=1e-12,
    )

    actual = stab.glasser_d_r_state(state, rt)
    denominator = jnp.where(shear != 0.0, shear**2, 1.0)
    expected = -dmerc + (h_glasser - 0.5 * shear**2) ** 2 / denominator
    expected = jnp.where(shear != 0.0, expected, 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    assert np.all(np.isfinite(np.asarray(actual)))

    tangent = jax.tree.map(jnp.zeros_like, state)
    tangent = dataclasses.replace(
        tangent, R_cos=jnp.ones_like(state.R_cos)
    )
    _, (jdotb_tangent, d_r_tangent) = jax.jvp(
        lambda st: (
            stab.jdotb_state(st, rt),
            stab.glasser_d_r_state(st, rt, shear_epsilon=1.0e-8),
        ),
        (state,),
        (tangent,),
    )
    for profile in (jdotb_tangent, d_r_tangent):
        interior = np.asarray(profile)[2:-1]
        assert np.all(np.isfinite(interior))
        assert np.any(interior != 0.0)


def test_mercier_stability_residual_is_smooth_interior_hinge(shaped_eq):
    """The optimizer residual excludes noisy surfaces and follows its formula."""
    state, rt = shaped_eq.state, shaped_eq.runtime
    profile = stab.d_merc_state(state, rt)
    margin, smoothing = 2.0e-6, 3.0e-6
    actual = jax.jit(
        lambda st: stab.mercier_stability_residual(
            st, rt, margin=margin, smoothing=smoothing
        )
    )(state)
    expected = smoothing * jax.nn.softplus((margin - profile[2:-1]) / smoothing)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-16)
    assert actual.shape == (profile.size - 3,)
    default = stab.mercier_stability_residual(state, rt)
    assert jnp.min(profile[2:-1]) > 6.0e-6
    assert jnp.max(default) < 2.0e-9
    with pytest.raises(ValueError, match="smoothing must be positive"):
        stab.mercier_stability_residual(state, rt, smoothing=0.0)


def test_glasser_stability_residual_is_smooth_upper_bound(shaped_eq):
    """The D_R objective penalizes positive values on validated surfaces."""
    state, rt = shaped_eq.state, shaped_eq.runtime
    margin, smoothing, shear_epsilon = 2.0e-6, 3.0e-6, 1.0e-8
    profile = stab.glasser_d_r_state(
        state, rt, shear_epsilon=shear_epsilon
    )
    actual = jax.jit(
        lambda st: stab.glasser_stability_residual(
            st,
            rt,
            margin=margin,
            smoothing=smoothing,
            shear_epsilon=shear_epsilon,
        )
    )(state)
    expected = smoothing * jax.nn.softplus(
        (profile[2:-1] + margin) / smoothing
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-16)
    np.testing.assert_array_equal(
        stab.glasser_stability_residual(state, rt),
        stab.glasser_stability_residual(
            state, rt, shear_epsilon=1.0e-8
        ),
    )
    with pytest.raises(ValueError, match="smoothing must be positive"):
        stab.glasser_stability_residual(state, rt, smoothing=0.0)
    with pytest.raises(ValueError, match="shear_epsilon must be non-negative"):
        stab.glasser_d_r_state(state, rt, shear_epsilon=-1.0)


def test_least_squares_accepts_implicit_stability_terms():
    """Profile scalarization matches the Gauss--Newton cost and gradient."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    terms = [(opt.mercier_stability_residual, 0.0, 1.0),
             (opt.glasser_stability_residual, 0.0, 1.0),
             (opt.jdotb_residual, 0.0, 1.0e-6)]
    result = opt.least_squares(
        terms,
        inp,
        max_mode=1,
        jac="implicit",
        max_nfev=1,
    )
    assert result.nfev == 1
    assert np.isfinite(result.cost)
    assert np.all(np.isfinite(result.jac))
    x0 = opt.pack_boundary(inp, 1)
    scalar = opt.minimize(
        terms, inp, max_mode=1, bounds=list(zip(x0, x0)))
    np.testing.assert_allclose(scalar.cost, result.cost, rtol=1e-12)
    np.testing.assert_allclose(
        scalar.jac, result.jac.T @ result.fun, rtol=2e-5, atol=1e-8)


@pytest.mark.full
def test_implicit_glasser_optimization_improves_margin_with_constraints():
    """A sheared campaign reduces D_R while enforcing ideal prerequisites."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    inp = dataclasses.replace(inp, ai=np.asarray([1.0, 0.1]))
    seed = opt.solve_equilibrium(inp)
    aspect0 = float(opt.aspect_ratio(seed.state, seed.runtime))
    qa = opt.QuasisymmetryRatioResidual([0.25, 0.5, 0.75], 1, 0)

    def glasser(state, runtime):
        return stab.glasser_stability_residual(
            state,
            runtime,
            smoothing=1.0e-6,
            shear_epsilon=1.0e-8,
        )

    def metrics(eq):
        dmerc = np.asarray(
            stab.d_merc_state(eq.state, eq.runtime)
        )[2:-1]
        shear = np.asarray(
            stab.mercier_shear_state(eq.state, eq.runtime)
        )[2:-1]
        profile = np.asarray(stab.glasser_d_r_state(
            eq.state, eq.runtime, shear_epsilon=1.0e-8
        ))[2:-1]
        glasser_violation = np.asarray(glasser(eq.state, eq.runtime))
        qa_norm = np.linalg.norm(np.asarray(
            qa.residuals_state(eq.state, eq.runtime)))
        return (
            float(np.min(dmerc)),
            float(np.min(np.abs(shear))),
            float(np.max(profile)),
            float(np.linalg.norm(glasser_violation)),
            qa_norm,
        )

    dmerc0, shear0, d_r0, glasser0, qa0 = metrics(seed)
    result = opt.least_squares(
        [(opt.aspect_ratio, aspect0, 100.0),
         (qa, 0.0, 1.0e5),
         (opt.mercier_stability_residual, 0.0, 1.0e5),
         (glasser, 0.0, 1.0e5)],
        inp,
        max_mode=2,
        jac="implicit",
        max_nfev=6,
    )
    best = opt.solve_equilibrium(result.input)
    dmerc1, shear1, d_r1, glasser1, qa1 = metrics(best)
    aspect1 = float(opt.aspect_ratio(best.state, best.runtime))

    assert dmerc0 > 0.0 and dmerc1 > 0.0
    assert min(shear0, shear1) > 100.0e-8
    assert d_r1 < d_r0
    assert glasser1 < 0.5 * glasser0
    assert abs(aspect1 - aspect0) < 5.0e-3
    assert qa1 <= 1.05 * qa0


def test_traceable_dmerc_matches_wout_in_3d(finite_beta_3d_eq):
    """The traceable current reconstruction retains toroidal-mode parity."""
    eq = finite_beta_3d_eq
    actual = np.asarray(stab.d_merc_state(eq.state, eq.runtime))
    expected = np.asarray(opt.d_merc(eq))
    scale = np.max(np.abs(expected[2:-1]))
    np.testing.assert_allclose(actual[2:-1], expected[2:-1], rtol=1e-10,
                               atol=1e-13 * scale)
    np.testing.assert_allclose(
        stab.jdotb_state(eq.state, eq.runtime),
        eq.wout.jdotb,
        rtol=1e-10,
        atol=1e-6,
    )
    _, _, bdotb, _, _ = stab._mercier_profiles_state(
        eq.state, eq.runtime
    )
    np.testing.assert_allclose(
        bdotb,
        eq.wout.bdotb,
        rtol=1e-10,
        atol=1e-12,
    )


def test_reductions_hard_and_smooth_max(highbeta_eq):
    lam = np.asarray(stab.ballooning_lambda(highbeta_eq.state, highbeta_eq.runtime,
                                            **FAST)).ravel()
    hard = float(stab.ballooning_growth_rate(highbeta_eq.state, highbeta_eq.runtime,
                                             reduction="max", **FAST))
    assert hard == pytest.approx(np.max(lam), rel=1e-12)
    temperature = 0.01
    soft = float(stab.ballooning_growth_rate(highbeta_eq.state, highbeta_eq.runtime,
                                             temperature=temperature, **FAST))
    assert np.max(lam) <= soft <= np.max(lam) + temperature * np.log(lam.size) + 1e-12
    with pytest.raises(ValueError):
        stab.ballooning_growth_rate(highbeta_eq.state, highbeta_eq.runtime,
                                    reduction="bogus", **FAST)


def test_grad_wrt_pressure_scale_matches_finite_differences(highbeta_eq):
    """AD through the traceable lane (frozen state, rescaled pressure profile)."""
    state, rt = highbeta_eq.state, highbeta_eq.runtime

    def growth(scale):
        setup = dataclasses.replace(rt.setup, mass=rt.setup.mass * scale)
        return stab.ballooning_growth_rate(state, dataclasses.replace(rt, setup=setup),
                                           temperature=0.01, **FAST)

    value, grad = jax.value_and_grad(growth)(1.0)
    assert np.isfinite(float(value)) and np.isfinite(float(grad))
    assert float(grad) > 0.0  # more pressure -> more ballooning-unstable
    eps = 1e-4
    fd = (growth(1.0 + eps) - growth(1.0 - eps)) / (2.0 * eps)
    assert float(grad) == pytest.approx(float(fd), rel=1e-6)


def test_grad_wrt_state_is_finite(highbeta_eq):
    """The state gradient the implicit-gradient lane composes with is finite."""
    rt = highbeta_eq.runtime
    grad = jax.grad(lambda st: stab.ballooning_growth_rate(st, rt, **FAST))(highbeta_eq.state)
    leaves = jax.tree.leaves(grad)
    assert leaves
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    assert any(np.any(np.asarray(leaf) != 0.0) for leaf in leaves)


def test_surface_index_validation(highbeta_eq):
    with pytest.raises(ValueError, match="out of range"):
        stab.ballooning_lambda(highbeta_eq.state, highbeta_eq.runtime, s_indices=(1,))
    ns = int(np.shape(highbeta_eq.state.R_cos)[0])
    with pytest.raises(ValueError, match="out of range"):
        stab.ballooning_lambda(highbeta_eq.state, highbeta_eq.runtime, s_indices=(ns - 1,))
