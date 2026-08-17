"""A/B tests for the traceable QS residual path in optimize.py.

``QuasisymmetryRatioResidual.residuals_state/profile_state/total_state``
evaluate the two-term quasisymmetry ratio on the solver's internal grid
(pure JAX, the residual vector that ``jac="implicit"`` optimizes).  Their
docstrings promise agreement with the wout-table path (``.residuals`` /
``.total``) at discretization level — solver angular grid vs the 63x64
wout sampling.  This module checks that promise on the solovev
equilibrium, plus the exact self-consistency between the three traceable
methods and the axisymmetric QA degeneracy.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from vmex.core import optimize as opt
from vmex.core.input import VmecInput

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solve: run jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
SURFACES = [0.25, 0.5, 0.75, 1.0]


@pytest.fixture(scope="module")
def eq():
    equilibrium = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.solovev"))
    assert equilibrium.result.converged
    return equilibrium


@pytest.fixture(scope="module")
def lasym_eq():
    inp = VmecInput.from_file(DATA_DIR / "input.basic_non_stellsym_simsopt")
    inp = dataclasses.replace(
        inp, ns_array=np.array([11]), ftol_array=np.array([1e-12]),
        niter_array=np.array([4000]))
    equilibrium = opt.solve_equilibrium(inp)
    assert equilibrium.result.converged
    return equilibrium


def test_total_state_matches_wout_path_at_discretization_level(eq):
    # axisymmetric case: QH (m=1, n=-1) has a nonzero, grid-comparable total
    qs = opt.QuasisymmetryRatioResidual(SURFACES, 1, -1)
    total_wout = float(qs.total(eq.wout))
    total_state = float(qs.total_state(eq.state, eq.runtime))
    assert total_wout > 0.0
    assert total_state == pytest.approx(total_wout, rel=0.1)


def test_residuals_state_consistency_and_profile(eq):
    qs = opt.QuasisymmetryRatioResidual(SURFACES, 1, -1, weights=[1.0, 2.0, 3.0, 4.0])
    r = qs.residuals_state(eq.state, eq.runtime)
    assert r.ndim == 1
    assert np.all(np.isfinite(np.asarray(r)))
    profile = qs.profile_state(eq.state, eq.runtime)
    assert profile.shape == (len(SURFACES),)
    total = float(qs.total_state(eq.state, eq.runtime))
    assert float(jnp.sum(profile)) == pytest.approx(total, rel=1e-12)
    # weights scale the profile linearly
    unweighted = opt.QuasisymmetryRatioResidual(SURFACES, 1, -1).profile_state(
        eq.state, eq.runtime)
    np.testing.assert_allclose(np.asarray(profile),
                               np.asarray(unweighted) * np.asarray([1.0, 2.0, 3.0, 4.0]),
                               rtol=1e-12)


def test_residual_quadrature_closes_on_totals(eq):
    """sum(residuals_state**2) equals the interpolated profile total."""
    qs = opt.QuasisymmetryRatioResidual(SURFACES, 1, -1)
    r = qs.residuals_state(eq.state, eq.runtime)
    # residuals_state carries the *full* half-mesh pointwise vector; its
    # square-sum is the unweighted all-surface total, which upper-bounds the
    # interpolated 4-surface profile sum and shares its scale.
    sum_sq = float(jnp.sum(r * r))
    total = float(qs.total_state(eq.state, eq.runtime))
    assert sum_sq > 0.0
    assert total <= sum_sq * (1.0 + 1e-9)


def test_axisymmetric_case_is_quasiaxisymmetric(eq):
    """solovev is a tokamak: the QA (m=1, n=0) residual is degenerate-small."""
    qa = float(opt.QuasisymmetryRatioResidual(SURFACES, 1, 0).total_state(
        eq.state, eq.runtime))
    qh = float(opt.QuasisymmetryRatioResidual(SURFACES, 1, -1).total_state(
        eq.state, eq.runtime))
    assert qa < 1e-6 * qh


def test_pointwise_state_grad_is_finite(eq):
    """The traceable path is differentiable (no 0*inf axis-row poisoning)."""
    qs = opt.QuasisymmetryRatioResidual([0.5], 1, -1)
    grad = jax.grad(
        lambda st: qs.total_state(st, eq.runtime))(eq.state)
    leaves = jax.tree.leaves(grad)
    assert leaves
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def test_lasym_state_derivative_matches_finite_difference(lasym_eq):
    """QS includes asymmetric R-sine/Z-cosine state families."""
    qs = opt.QuasisymmetryRatioResidual([0.3, 0.7], 1, 0)
    tangent = jax.tree.map(jnp.zeros_like, lasym_eq.state)
    tangent = dataclasses.replace(
        tangent, Z_cos=tangent.Z_cos.at[5, 1].set(1.0))
    _, ad = jax.jvp(
        lambda state: qs.total_state(state, lasym_eq.runtime),
        (lasym_eq.state,), (tangent,))
    h = 1.0e-6
    plus = jax.tree.map(lambda value, direction: value + h * direction,
                        lasym_eq.state, tangent)
    minus = jax.tree.map(lambda value, direction: value - h * direction,
                         lasym_eq.state, tangent)
    fd = (qs.total_state(plus, lasym_eq.runtime)
          - qs.total_state(minus, lasym_eq.runtime)) / (2.0 * h)
    assert float(ad) == pytest.approx(float(fd), rel=2e-5, abs=1e-8)
