"""Validation gates for the SPECTRAX-GK turbulence proxies (plan.md R26h.h4).

Lanes
-----
- **Geometry adapter parity** (spectraxgk-free): the flux-tube arrays of
  :func:`vmex.core.turbulence.gk_fieldline_geometry` reproduce, at
  machine precision, the field-line geometry that
  :mod:`vmex.core.stability` assembles from the same converged state
  (identical simsopt ``vmec_fieldlines`` conventions), plus internal
  consistency: the Cauchy-Schwarz metric inequality
  ``gds21^2 <= gds2 * gds22``, the mirror-term identity
  ``bgrad = gradpar d(ln bmag)/dtheta``, the vacuum-limit
  ``cvdrift = gbdrift``, and the exactly-constant equal-arc ``gradpar``.
- **Contract gate** (needs spectraxgk): the mapping passes
  ``flux_tube_geometry_from_mapping`` with host-side validation ON —
  finite arrays and the constant-``gradpar`` equal-arc contract.
- **Proxy physics** (needs spectraxgk): on a solved finite-beta shaped
  tokamak the dominant gyrokinetic growth rate is ITG-critical-gradient
  monotone — strongly driven (``R/L_Ti = 6.9``, Cyclone-level) is unstable,
  weakly driven (``R/L_Ti = 1``) is marginal/stable — and the quasilinear
  and reduced nonlinear-window heat-flux proxies are positive for the
  unstable case, with the documented saturation-rule relations between the
  objective-vector entries reproduced exactly.
- **Differentiability** (needs spectraxgk): SPECTRAX-GK is JAX-traceable —
  both ``jax.grad`` (reverse) and ``jax.jacfwd`` (forward — the mode
  vmex's implicit Jacobian lane uses) of the growth rate w.r.t. a
  pressure-profile rescale match central finite differences, the gradient
  w.r.t. the converged state (the piece ``jac="implicit"`` composes with)
  is finite and nonzero, and the wrappers satisfy the two-positional
  ``(state, runtime)`` objective-term contract of
  :func:`vmex.core.optimize.least_squares`.  The eigenvector-weighted
  quasilinear/nonlinear proxies are value-level (``jac=None``): JAX
  declines non-symmetric eigenvector derivatives, and the gate documents
  that limitation explicitly.

The spectraxgk dependency is optional (``pip install spectraxgk``); like
``test_freeboundary_diff.py``, the dependent lanes skip cleanly without it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vmex.core import optimize as opt  # noqa: E402
from vmex.core import stability as stab  # noqa: E402
from vmex.core import turbulence as turb  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solves: run jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
LINE = dict(s_index=7, alpha=0.3)          # one interior field line, off-symmetry
GK = dict(s_index=7, ntheta=16)            # modest solver budget for the proxies


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
def vacuum_eq():
    """Zero-pressure circular tokamak: the cvdrift = gbdrift limit."""
    eq = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.circular_tokamak"))
    assert eq.result.converged
    return eq


# ---------------------------------------------------------------------------
# Geometry adapter (no spectraxgk needed)
# ---------------------------------------------------------------------------


def test_geometry_matches_stability_conventions(shaped_eq):
    """Adapter arrays == stability.py's field-line assembly (same deck/line)."""
    state, rt = shaped_eq.state, shaped_eq.runtime
    mapping = turb.gk_fieldline_geometry(state, rt, ntheta=64, equal_arc=False, **LINE)

    # Recompute the overlap set with stability.py's own point closure at the
    # identical (surface, alpha, zeta0 = 0) sample points.
    ctx = stab._ballooning_context(state, rt)
    j, hs = LINE["s_index"], ctx["hs"]
    iota = 0.5 * (ctx["iotas"][j] + ctx["iotas"][j + 1])
    diota = (ctx["iotas"][j + 1] - ctx["iotas"][j]) / hs
    dpres = (ctx["pres"][j + 1] - ctx["pres"][j]) / hs
    point = stab._make_point_fn(
        ctx["m"], ctx["xn"],
        stab._parabola(ctx["rmnc"], j, hs), stab._parabola(ctx["zmns"], j, hs),
        stab._parabola(ctx["lmns"], j, hs), iota, diota, ctx["phipf"][j])
    x = jnp.asarray(mapping["theta"])
    phi = x / iota
    lmns0 = stab._parabola(ctx["lmns"], j, hs)[0]
    theta_v = stab._theta_vmec_from_pest(LINE["alpha"] + x, phi, lmns0, ctx["m"], ctx["xn"])
    q = jnp.stack([jnp.zeros_like(theta_v), theta_v, phi], axis=-1)
    modB, b_sup_phi, gaa, bxgb_ga = jax.vmap(point)(q, phi)

    L, B = ctx["L_ref"], ctx["B_ref"]
    s_j = ctx["s"][j]
    sqrt_s = jnp.sqrt(s_j)
    expected = {
        "bmag": modB / B,
        "gds2": gaa * L * L * s_j,
        "gbdrift": -2.0 * B * L * L * sqrt_s * ctx["sign_psi"] * bxgb_ga / modB**3,
    }
    expected["cvdrift"] = expected["gbdrift"] - (
        2.0 * B * L * L * sqrt_s * dpres / (jnp.abs(ctx["psi_edge"]) * modB**2))
    gradpar_ref = jnp.abs(L * iota * b_sup_phi / modB)

    for name, ref in expected.items():
        np.testing.assert_allclose(np.asarray(mapping[name]), np.asarray(ref),
                                   rtol=1e-12, atol=1e-14, err_msg=name)
    np.testing.assert_allclose(np.asarray(mapping["gradpar"]), np.asarray(gradpar_ref),
                               rtol=1e-12, atol=1e-14)
    assert mapping["vmex"]["surface_index"] == j
    assert float(mapping["s_hat"]) == pytest.approx(float(-2.0 * s_j * diota / iota))


def test_geometry_internal_identities(shaped_eq):
    """Metric/drift/mirror identities the GS2/GX conventions must satisfy."""
    mapping = turb.gk_fieldline_geometry(shaped_eq.state, shaped_eq.runtime,
                                         ntheta=128, equal_arc=False, **LINE)
    bmag = np.asarray(mapping["bmag"])
    gds2, gds21, gds22 = (np.asarray(mapping[k]) for k in ("gds2", "gds21", "gds22"))
    assert np.all(bmag > 0.0) and np.all(np.asarray(mapping["gradpar"]) > 0.0)
    assert np.all(gds2 > 0.0) and np.all(gds22 > 0.0)
    # Cauchy-Schwarz: (grad alpha . grad psi)^2 <= |grad alpha|^2 |grad psi|^2.
    assert np.all(gds21**2 <= gds2 * gds22 * (1.0 + 1e-12))
    # Finite beta with dp/ds < 0: curvature drive exceeds grad-B drive.
    assert np.all(np.asarray(mapping["cvdrift"]) > np.asarray(mapping["gbdrift"]))
    # cvdrift0 is gbdrift0 (simsopt vmec_fieldlines).
    np.testing.assert_array_equal(np.asarray(mapping["cvdrift0"]),
                                  np.asarray(mapping["gbdrift0"]))
    # Mirror term: bgrad == gradpar d(ln bmag)/dtheta (2nd-order FD check).
    theta = np.asarray(mapping["theta"])
    d_bmag = np.gradient(bmag, theta[1] - theta[0])
    bgrad_fd = np.asarray(mapping["gradpar"]) * d_bmag / bmag
    scale = np.max(np.abs(bgrad_fd))
    assert np.max(np.abs(np.asarray(mapping["bgrad"]) - bgrad_fd)) < 0.05 * scale


def test_vacuum_limit_cvdrift_equals_gbdrift(vacuum_eq):
    """Zero pressure: the cvdrift pressure correction vanishes identically."""
    mapping = turb.gk_fieldline_geometry(vacuum_eq.state, vacuum_eq.runtime, **LINE)
    np.testing.assert_allclose(np.asarray(mapping["cvdrift"]),
                               np.asarray(mapping["gbdrift"]), rtol=0.0, atol=1e-11)


def test_equal_arc_gradpar_constant_and_values_consistent(shaped_eq):
    """Equal-arc lane: exactly uniform gradpar, same geometry values."""
    state, rt = shaped_eq.state, shaped_eq.runtime
    uniform = turb.gk_fieldline_geometry(state, rt, ntheta=128, equal_arc=False, **LINE)
    arc = turb.gk_fieldline_geometry(state, rt, ntheta=64, equal_arc=True, **LINE)
    gp = np.asarray(arc["gradpar"])
    assert np.max(gp) - np.min(gp) == 0.0                 # exactly constant
    # and equal to the flux-tube average of the uniform-angle profile
    prof = np.asarray(uniform["vmex"]["gradpar_profile"])
    # tokamak: the profile is periodic over one poloidal turn, so close the
    # trapezoid with the theta = -pi sample repeated at +pi.
    harmonic = 2.0 * np.pi / np.trapezoid(
        np.append(1.0 / prof, 1.0 / prof[0]),
        np.append(np.asarray(uniform["theta"]), np.pi))
    assert gp[0] == pytest.approx(harmonic, rel=2e-3)
    # geometry values are exact evaluations at the mapped PEST angles: bmag
    # from the two lanes must agree up to the comparison interpolation error.
    x_arc = np.asarray(arc["vmex"]["theta_pest"]) - LINE["alpha"]
    bmag_interp = np.interp(x_arc, np.asarray(uniform["theta"]),
                            np.asarray(uniform["bmag"]))
    assert np.max(np.abs(np.asarray(arc["bmag"]) - bmag_interp)) < 5e-4


def test_surface_index_validation(shaped_eq):
    with pytest.raises(ValueError, match="out of range"):
        turb.gk_fieldline_geometry(shaped_eq.state, shaped_eq.runtime, s_index=1)
    with pytest.raises(ValueError, match="ntheta"):
        turb.gk_fieldline_geometry(shaped_eq.state, shaped_eq.runtime, ntheta=4)


# ---------------------------------------------------------------------------
# SPECTRAX-GK contract + proxies (importorskip-gated, like freeboundary_diff)
# ---------------------------------------------------------------------------


def test_contract_passes_spectraxgk_validation(shaped_eq):
    """The mapping satisfies spectraxgk's validated flux-tube contract."""
    pytest.importorskip("spectraxgk")
    geom = turb.flux_tube_geometry(shaped_eq.state, shaped_eq.runtime,
                                   validate=True, ntheta=32, **LINE)
    assert type(geom).__name__ == "FluxTubeGeometryData"
    assert geom.source_model == "vmex:core.turbulence"
    assert float(geom.gradpar_value) > 0.0
    assert int(np.asarray(geom.theta).shape[0]) == 32


def test_growth_rate_is_itg_critical_gradient_monotone(shaped_eq):
    """Strong ITG drive unstable, weak drive marginal; proxies positive."""
    pytest.importorskip("spectraxgk")
    state, rt = shaped_eq.state, shaped_eq.runtime
    gamma_hi = float(turb.turbulent_growth_rate(state, rt, r_over_lt=6.9, **GK))
    gamma_lo = float(turb.turbulent_growth_rate(state, rt, r_over_lt=1.0, **GK))
    assert 0.05 < gamma_hi < 5.0            # Cyclone-level drive: robustly unstable
    assert gamma_lo < 1e-6                  # below the ITG critical gradient
    assert gamma_hi > gamma_lo + 0.05       # monotone in the ITG drive


def test_objective_vector_and_scalar_proxies_consistent(shaped_eq):
    """Vector entries reproduce the documented saturation-rule proxies."""
    pytest.importorskip("spectraxgk")
    state, rt = shaped_eq.state, shaped_eq.runtime
    vec = np.asarray(turb.turbulence_objective_vector(state, rt, **GK))
    named = dict(zip(turb.TURBULENCE_OBJECTIVE_NAMES, vec))
    assert np.all(np.isfinite(vec))
    gamma = named["gamma"]
    assert gamma == pytest.approx(float(turb.turbulent_growth_rate(state, rt, **GK)),
                                  rel=1e-8)
    # quasilinear mixing-length rule: gamma * W_Q / max(kperp_eff2, 1e-12)
    ql = float(turb.quasilinear_flux_proxy(state, rt, **GK))
    assert ql == pytest.approx(
        gamma * named["linear_heat_flux_weight"] / max(named["kperp_eff2"], 1e-12),
        rel=1e-12)
    assert ql == pytest.approx(named["mixing_length_heat_flux_proxy"], rel=1e-12)
    # reduced nonlinear-window rule (spectraxgk's smooth surrogate)
    nl = float(turb.nonlinear_heat_flux_proxy(state, rt, csat=0.85, **GK))
    gamma_plus = np.logaddexp(0.0, 18.0 * gamma) / 18.0   # smooth_positive(gamma)
    expected_nl = (0.85 * max(named["linear_heat_flux_weight"], 0.0) * 2.0 * gamma_plus
                   / (1.0 + 2.2 * max(named["kperp_eff2"], 0.0) + 0.15 * gamma_plus))
    assert nl == pytest.approx(expected_nl, rel=1e-6)
    assert ql > 0.0 and nl > 0.0            # unstable ITG: outward heat flux


def test_growth_rate_gradient_matches_finite_differences(shaped_eq):
    """spectraxgk is JAX-traceable: AD == FD through geometry + eigensolve.

    Both AD modes: reverse (``jax.grad``, the hand-written objective lane)
    and forward (``jax.jacfwd`` — what ``jac="implicit"``'s forward implicit
    Jacobian traces through the objective rows).
    """
    pytest.importorskip("spectraxgk")
    state, rt = shaped_eq.state, shaped_eq.runtime

    def gamma(scale):
        setup = dataclasses.replace(rt.setup, mass=rt.setup.mass * scale)
        return turb.turbulent_growth_rate(state, dataclasses.replace(rt, setup=setup),
                                          **GK)

    value, grad = jax.value_and_grad(gamma)(1.0)
    assert np.isfinite(float(value)) and np.isfinite(float(grad))
    eps = 1e-4
    fd = (gamma(1.0 + eps) - gamma(1.0 - eps)) / (2.0 * eps)
    assert float(grad) == pytest.approx(float(fd), rel=1e-5)
    fwd = jax.jacfwd(gamma)(1.0)                     # forward mode: implicit lane
    assert float(fwd) == pytest.approx(float(fd), rel=1e-5)


def test_eigenvector_weighted_proxies_are_value_level(shaped_eq):
    """Documented limitation: quasilinear/nonlinear proxies need jac=None.

    Their heat-flux weights depend on the dominant eigenvector of the
    non-symmetric GK operator, whose derivatives JAX declines
    (spectraxgk >= 1.6.10 evaluates them with plain ``jnp.linalg.eig``).
    Values remain finite-difference-friendly, which is what ``jac=None``
    uses; revisit if spectraxgk adopts ``enable_eigvec_derivs``.
    """
    pytest.importorskip("spectraxgk")
    state, rt = shaped_eq.state, shaped_eq.runtime

    def ql(scale):
        setup = dataclasses.replace(rt.setup, mass=rt.setup.mass * scale)
        return turb.quasilinear_flux_proxy(state, dataclasses.replace(rt, setup=setup),
                                           **GK)

    with pytest.raises(NotImplementedError, match="eigenvector"):
        jax.grad(ql)(1.0)
    eps = 1e-3                                       # FD lane stays healthy
    fd = (ql(1.0 + eps) - ql(1.0 - eps)) / (2.0 * eps)
    assert np.isfinite(float(fd))


def test_grad_wrt_state_is_finite(shaped_eq):
    """The state gradient the implicit-gradient lane composes with is finite."""
    pytest.importorskip("spectraxgk")
    rt = shaped_eq.runtime
    grad = jax.grad(lambda st: turb.turbulent_growth_rate(st, rt, **GK))(shaped_eq.state)
    leaves = jax.tree.leaves(grad)
    assert leaves
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    assert any(np.any(np.asarray(leaf) != 0.0) for leaf in leaves)


def test_wrappers_satisfy_least_squares_term_contract():
    """Two-positional (state, runtime) callables: accepted by jac='implicit'."""
    for fun in (turb.turbulent_growth_rate, turb.quasilinear_flux_proxy,
                turb.nonlinear_heat_flux_proxy, turb.turbulence_objective_vector):
        assert opt._traceable_term(fun) is fun
