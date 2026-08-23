"""Physics anchors and differentiation gates for the Gamma_c proxy.

Lanes: literature-anchored ordering (axisymmetric limit, QA versus
unoptimized 3D), directional consistency with the independent NEO_JAX
effective-ripple lane on a boundary-ripple ray, implicit boundary-gradient
liveness, and the composable-class contract. Every tolerance was set from
measured values recorded in the test docstrings, with stated headroom.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vmex.core import gammac  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core import optimize as opt  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
# One evaluation budget shared by the physics tests below; the measured
# anchor values in the docstrings correspond to exactly these settings.
SETTINGS = dict(
    nalpha=7, num_transit=3, points_per_transit=64, num_pitch=24,
    quadrature_order=32)


@pytest.fixture(scope="module")
def tokamak_eq():
    eq = opt.solve_equilibrium(
        VmecInput.from_file(DATA_DIR / "input.circular_tokamak"))
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def qa_eq():
    inp = VmecInput.from_file(DATA_DIR / "input.LandremanPaul2021_QA_lowres")
    inp = dataclasses.replace(
        inp, ns_array=np.array([13]), ftol_array=np.array([1e-12]),
        niter_array=np.array([4000]))
    eq = opt.solve_equilibrium(inp)
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def ncsx_eq():
    eq = opt.solve_equilibrium(
        VmecInput.from_file(DATA_DIR / "input.li383_low_res"))
    assert eq.result.converged
    return eq


def _gamma_c(eq, **overrides):
    settings = dict(SETTINGS)
    settings.update(overrides)
    out = gammac.gamma_c_state(
        eq.state, eq.runtime, surfaces=(0.5,), **settings)
    assert float(out["excluded_fraction"][0]) < 0.05
    assert float(out["overflow_fraction"][0]) == 0.0
    return float(out["gamma_c"][0])


def test_axisymmetric_limit_and_optimization_ordering(
        tokamak_eq, qa_eq, ncsx_eq):
    """Gamma_c orders tokamak << precise QA << unoptimized 3D.

    In axisymmetry the bounce-averaged radial drift vanishes exactly
    (d J / d alpha = 0), so Gamma_c -> 0: Nemov et al. 2008; Velasco et
    al. 2021 use Gamma_c as the deviation of J contours from flux
    surfaces. Measured at these settings: 9.8e-7 (circular tokamak,
    quadrature noise only), 3.8e-3 (LandremanPaul2021 QA at ns=13; drops
    to 1.1e-3 at doubled sampling — a near-omnigenous value is an upper
    bound at this budget, hence the wide band), 4.11e-2 (li383; 4.20e-2
    at doubled sampling, stable to 2%). The li383 lane is additionally
    the DESC cross-check point: DESC's Nemov Gamma_c on the same wout at
    s=0.25 agrees to 2.8% (PR record); DESC's outer-surface values
    disagree because its wout refit inflates d|B|/drho there, which was
    refereed directly against the wout tables.
    """
    tok = _gamma_c(tokamak_eq)
    qa = _gamma_c(qa_eq)
    ncsx = _gamma_c(ncsx_eq)
    assert tok < 1.0e-4
    assert qa < 0.3 * ncsx
    assert tok < 0.1 * qa
    assert 1.0e-4 < qa < 1.0e-2
    assert 2.5e-2 < ncsx < 5.0e-2


def test_gamma_c_tracks_effective_ripple_on_a_ripple_ray(qa_eq):
    """A boundary ripple worsens Gamma_c and eps_eff together.

    A directional cross-check against the independent NEO_JAX
    effective-ripple lane, not a proportionality claim: both metrics
    integrate different weightings of the same ripple wells, and the
    literature (Bader 2021, Paul 2022) documents imperfect correlation
    between such proxies and measured energetic-particle losses.
    Measured: Gamma_c 3.8e-3 -> 1.7e-2 (x4.4, asserted at x2) and
    eps_eff^(3/2) 8.9e-9 -> 2.3e-4 under the same perturbation (the QA
    baseline is essentially ripple-free, hence the enormous eps_eff
    factor; asserted at x10).
    """
    pytest.importorskip("neo_jax")
    from vmex.core.neoclassical import epsilon_effective_from_wout

    base = _gamma_c(qa_eq)
    inp = qa_eq.inp
    ntor = int(inp.ntor)
    rbc = np.array(inp.rbc)
    rbc[ntor + 1, 0] += 0.3 * abs(rbc[ntor, 1])   # (n=1, m=0) mirror ripple
    eq = opt.solve_equilibrium(dataclasses.replace(inp, rbc=rbc))
    assert eq.result.converged
    rippled = _gamma_c(eq)
    assert rippled > 2.0 * base

    def eps(e):
        _, values = epsilon_effective_from_wout(e.wout, surfaces=(0.5,))
        return float(np.asarray(values)[0])

    assert eps(eq) > 10.0 * eps(qa_eq)


def test_boundary_gradient_liveness():
    """jax.grad through the implicit solve is finite, nonzero, FD-consistent.

    li383 at ns=13, GammaC([0.5]). Measured for this commit: at the base
    sampling (nalpha=7, 3 transits, 64 points/transit, 24 pitch) the
    objective is 2.13e-3 and the reverse boundary gradient
    d/d rbc[n=0, m=1] is -0.541 against central FD -0.427 (step 1e-4)
    and -0.385 (step 2e-5): sign agreement, magnitude ratio 1.41. One
    refinement step (13, 4, 96, 48) keeps the sign and moves the
    magnitude to -0.119, while the Gamma_c value itself is
    sampling-stable to 2 percent (anchor test) — the gradient of the
    discretized objective is exact, but its magnitude at PR-lane
    sampling carries superbanana-layer discretization scatter of up to
    ~4.6x (convergence ladder in the PR record). The assertions encode
    exactly that: finite nonzero gradients, FD consistency within a
    factor 3 at fixed resolution, and sign stability under refinement;
    optimize at one fixed resolution.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.li383_low_res")
    params = im.params_from_input(inp, device=None)
    index = (int(inp.ntor), 1)

    def objective(p, nalpha=7, num_transit=3, ppt=64, npi=24):
        term = gammac.GammaC(
            [0.5], nalpha=nalpha, num_transit=num_transit,
            points_per_transit=ppt, num_pitch=npi, quadrature_order=32)
        solution = im.run(
            inp, p, ns=13, ftol=1.0e-13, max_iterations=20000,
            adjoint_tol=1.0e-13, device=None)
        return term.total_state(solution.state, solution.runtime)

    base = float(np.asarray(jax.grad(objective)(params).rbc)[index])
    step = 1.0e-4
    values = [
        float(objective(dataclasses.replace(
            params, rbc=jnp.asarray(params.rbc).at[index].add(sign * step))))
        for sign in (-1.0, 1.0)
    ]
    finite_difference = (values[1] - values[0]) / (2.0 * step)
    assert np.isfinite(base) and base != 0.0
    assert np.sign(base) == np.sign(finite_difference)
    assert 1.0 / 3.0 < base / finite_difference < 3.0

    refined = float(np.asarray(jax.grad(
        lambda p: objective(p, 13, 4, 96, 48))(params).rbc)[index])
    assert np.isfinite(refined) and refined != 0.0
    assert np.sign(refined) == np.sign(base)
    assert 0.03 < abs(refined) < 1.5 and 0.03 < abs(base) < 1.5


def test_class_contract_and_validation():
    """GammaC is a thin binding of gamma_c_state with neighbor-style guards."""
    calls = {}

    def fake_state(state, rt, **kwargs):
        calls.update(kwargs)
        return {"gamma_c": jnp.array([0.1, 0.2])}

    term = gammac.GammaC([0.3, 0.7], weights=[1.0, 4.0], nalpha=5)
    original = gammac.gamma_c_state
    gammac.gamma_c_state = fake_state
    try:
        eq = SimpleNamespace(state=object(), runtime=object())
        rows = term(eq)
        np.testing.assert_allclose(np.asarray(rows), [0.1, 0.4])
        assert float(term.total(eq)) == pytest.approx(0.01 + 0.16)
        assert calls["nalpha"] == 5 and tuple(calls["surfaces"]) == (0.3, 0.7)
    finally:
        gammac.gamma_c_state = original

    with pytest.raises(ValueError, match="increasing surfaces"):
        gammac.GammaC([0.7, 0.3])
    with pytest.raises(ValueError, match="strictly inside"):
        gammac.GammaC([0.0, 0.5])
    with pytest.raises(ValueError, match="weights"):
        gammac.GammaC([0.3, 0.7], weights=[1.0])
    with pytest.raises(ValueError, match="positive"):
        gammac.GammaC([0.5], nalpha=0)
    with pytest.raises(ValueError, match="surfaces must be finite"):
        gammac._surface_rows([], 13)
    with pytest.raises(ValueError, match="duplicate radial rows"):
        gammac._surface_rows([0.5, 0.52], 13)
    with pytest.raises(ValueError, match="nalpha"):
        gammac.gamma_c_state(None, None, nalpha=1)
    with pytest.raises(ValueError, match="num_transit"):
        gammac.gamma_c_state(None, None, num_transit=0)
    with pytest.raises(ValueError, match="num_pitch"):
        gammac.gamma_c_state(None, None, num_pitch=1)
    with pytest.raises(ValueError, match="pitch_weights"):
        gammac.gamma_c_from_fieldlines(
            bmag=jnp.ones((2, 8)), radial_drift=0.0, radial_gradient=0.0,
            drift_correction=0.0, tangency=1.0, dl_dx=1.0, length=1.0,
            pitch=jnp.ones(3), pitch_weights=jnp.ones(2))
    with pytest.raises(ValueError, match="nline, nx"):
        gammac.gamma_c_from_fieldlines(
            bmag=jnp.ones(8), radial_drift=0.0, radial_gradient=0.0,
            drift_correction=0.0, tangency=1.0, dl_dx=1.0, length=1.0,
            pitch=jnp.ones(3), pitch_weights=jnp.ones(3))
