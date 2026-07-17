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
