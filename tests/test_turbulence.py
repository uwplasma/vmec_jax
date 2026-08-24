"""Validation gates for the optional GKX turbulence objectives.

Lanes: geometry adapter parity (gkx-free — the flux-tube arrays reproduce
:mod:`vmex.core.stability`'s field-line assembly at machine precision,
plus Cauchy-Schwarz, mirror-term, vacuum-limit and equal-arc identities);
the gkx flux-tube contract with host validation ON; proxy physics on a
finite-beta shaped tokamak (ITG-critical-gradient monotone growth rate,
positive heat-flux proxies, saturation-rule relations reproduced exactly);
and differentiability (reverse and forward AD vs central FD, finite state
gradient, the two-positional objective-term contract; the eigenvector-
weighted proxies are value-level because JAX declines non-symmetric
eigenvector derivatives).

gkx is optional (``pip install 'gkx>=1.7.1'``; the legacy ``spectraxgk``
name is not supported) — dependent lanes skip cleanly without it.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vmex.core import optimize as opt  # noqa: E402
from vmex.core import stability as stab  # noqa: E402
from vmex.core import turbulence as turb  # noqa: E402
from vmex.core.statephysics import aspect_ratio  # noqa: E402
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
# Geometry adapter (no gkx needed)
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
    point = stab._make_point_fn(ctx["m"], ctx["xn"], stab._surface_tables(ctx, j),
                                iota, diota, ctx["phipf"][j])
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
# GKX contract + proxies (importorskip-gated, like freeboundary_diff)
# ---------------------------------------------------------------------------


def _require_gkx():
    """Skip unless gkx is importable."""
    pytest.importorskip("gkx")


def _require_gkx_eigenvectors():
    """Additionally require the jax floor gkx's *eigenvector* path declares.

    gkx reaches reverse-mode eigenvector derivatives through
    ``lax_linalg.eig(enable_eigvec_derivs=...)``, which first exists in jax
    0.10.1 and which gkx declares accordingly.  gkx still imports against an
    older jax, so importorskip alone lets those reach a call-time TypeError
    that is an unsatisfied dependency contract, not a defect.

    Only the eigenvector-weighted lanes need it.  ``turbulent_growth_rate``
    reduces the operator with ``jnp.linalg.eigvals`` and works on any
    supported jax, so gating it too left the whole ITG lane dark on every host
    below the floor -- which is how the R/L-into-a/L units defect survived.
    """
    _require_gkx()
    from jax._src.lax import linalg as lax_linalg

    if "enable_eigvec_derivs" not in inspect.signature(lax_linalg.eig).parameters:
        pytest.skip("gkx needs jax >= 0.10.1 for enable_eigvec_derivs "
                    f"(installed: {jax.__version__}); install vmex[turbulence]")


def test_contract_passes_gkx_validation(shaped_eq):
    """The mapping satisfies gkx's validated flux-tube contract."""
    _require_gkx()
    geom = turb.flux_tube_geometry(shaped_eq.state, shaped_eq.runtime,
                                   validate=True, ntheta=32, **LINE)
    assert type(geom).__name__ == "FluxTubeGeometryData"
    assert geom.source_model == "vmex:core.turbulence"
    assert float(geom.gradpar_value) > 0.0
    assert int(np.asarray(geom.theta).shape[0]) == 32


def test_drive_gradients_reach_gkx_as_a_over_l(shaped_eq):
    """``r_over_lt`` is R/L; GKX's operator consumes a/L.  Pin the conversion.

    GKX's ``LinearParams`` defaults are ``tprim = 2.49``, ``fprim = 0.8`` --
    the Cyclone base case ``R/L_T = 6.9``, ``R/L_n = 2.2`` divided by that
    case's ``R/a = 2.77``.  vmex used to set the deprecated ``R_over_LTi``
    instead, which applied no normalization at all, so ``r_over_lt = 6.9``
    reached the operator as ``tprim = 6.9`` -- R/a times too strongly driven,
    on every turbulence evaluation.  This deck's aspect ratio is 2.643, so the
    Cyclone drive lands at 2.611, next to GKX's own default.

    No solver call: the defect was entirely in how the parameters were built,
    and this runs wherever gkx imports.
    """
    pytest.importorskip("gkx")
    state, rt = shaped_eq.state, shaped_eq.runtime
    aspect = float(aspect_ratio(state, rt))
    assert aspect == pytest.approx(2.6427, rel=1e-3)
    params = turb._linear_params(None, 6.9, 2.2, aspect)
    assert float(params.tprim) == pytest.approx(6.9 / aspect, rel=1e-12)
    assert float(params.fprim) == pytest.approx(2.2 / aspect, rel=1e-12)
    assert float(params.tprim) == pytest.approx(2.611, rel=1e-3)
    # The subcritical case the growth-rate test relies on is well below GKX's
    # Cyclone default, which is the whole point of the conversion.
    assert float(turb._linear_params(None, 1.0, None, aspect).tprim) < 0.4
    # params_linear is the escape hatch and must pass through untouched.
    explicit = turb._linear_params(params, None, None, aspect)
    assert explicit is params


def test_growth_rate_is_itg_critical_gradient_monotone(shaped_eq):
    """Strong ITG drive unstable, weak drive marginal; proxies positive."""
    _require_gkx()
    state, rt = shaped_eq.state, shaped_eq.runtime
    gamma_hi = float(turb.turbulent_growth_rate(state, rt, r_over_lt=6.9, **GK))
    gamma_lo = float(turb.turbulent_growth_rate(state, rt, r_over_lt=1.0, **GK))
    assert 0.05 < gamma_hi < 5.0            # Cyclone-level drive: robustly unstable
    assert gamma_lo < 1e-6                  # below the ITG critical gradient
    assert gamma_hi > gamma_lo + 0.05       # monotone in the ITG drive


def test_objective_vector_and_scalar_proxies_consistent(shaped_eq):
    """Vector entries reproduce the documented saturation-rule proxies."""
    _require_gkx_eigenvectors()
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
    # reduced nonlinear-window rule (gkx's smooth surrogate)
    nl = float(turb.nonlinear_heat_flux_proxy(state, rt, csat=0.85, **GK))
    gamma_plus = np.logaddexp(0.0, 18.0 * gamma) / 18.0   # smooth_positive(gamma)
    expected_nl = (0.85 * max(named["linear_heat_flux_weight"], 0.0) * 2.0 * gamma_plus
                   / (1.0 + 2.2 * max(named["kperp_eff2"], 0.0) + 0.15 * gamma_plus))
    assert nl == pytest.approx(expected_nl, rel=1e-6)
    assert ql > 0.0 and nl > 0.0            # unstable ITG: outward heat flux


def test_growth_rate_gradient_matches_finite_differences(shaped_eq):
    """gkx is JAX-traceable: AD == FD through geometry + eigensolve.

    Both AD modes: reverse (``jax.grad``, the hand-written objective lane)
    and forward (``jax.jacfwd`` — what ``jac="implicit"``'s forward implicit
    Jacobian traces through the objective rows).
    """
    _require_gkx()
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
    """Documented guidance: quasilinear/nonlinear proxies use ``jac=None``
    (their weights depend on the dominant eigenvector of the non-symmetric
    GK operator, whose derivatives JAX declines unless
    ``enable_eigvec_derivs``); reverse AD must either refuse with that
    error or agree with the FD lane that ``jac=None`` actually uses."""
    _require_gkx_eigenvectors()
    state, rt = shaped_eq.state, shaped_eq.runtime

    def ql(scale):
        setup = dataclasses.replace(rt.setup, mass=rt.setup.mass * scale)
        return turb.quasilinear_flux_proxy(state, dataclasses.replace(rt, setup=setup),
                                           **GK)

    eps = 1e-3
    fd = float((ql(1.0 + eps) - ql(1.0 - eps)) / (2.0 * eps))
    assert np.isfinite(fd)                      # the jac=None lane always works
    try:
        analytic = float(jax.grad(ql)(1.0))
    except NotImplementedError as err:
        # gkx 1.7.1: JAX's documented refusal of non-symmetric eigenvector
        # derivatives (jax#2748) — the reason the proxies are value-level.
        assert "enable_eigvec_derivs" in str(err)
    else:
        # A gkx that opts in must reproduce the FD gradient it replaces.
        assert np.isfinite(analytic)
        scale = max(abs(fd), 1.0e-12)
        assert abs(analytic - fd) <= 1.0e-4 * scale + 1.0e-10


def test_grad_wrt_state_is_finite(shaped_eq):
    """The state gradient the implicit-gradient lane composes with is finite."""
    _require_gkx()
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


# ---------------------------------------------------------------------------
# External oracle: simsopt vmec_fieldlines, both symmetries
# ---------------------------------------------------------------------------


_SIMSOPT_GEOMETRY = ("bmag", "gradpar_theta_pest", "gds2", "gds21", "gds22",
                     "gbdrift", "gbdrift0", "cvdrift", "cvdrift0")


def _simsopt_fieldline(wout_path, s_value, alpha, theta):
    """``vmec_fieldlines`` on a vmex-written wout at the same field line.

    vmex returns the field-line coordinate ``x = theta* - alpha``; simsopt's
    ``theta1d`` is ``theta*``, so the grids differ by ``alpha``.
    """
    from simsopt.mhd import Vmec
    from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
    return vmec_fieldlines(Vmec(str(wout_path)), s_value, alpha,
                           theta1d=theta + alpha, plot=False)


@pytest.mark.parametrize("deck,overrides,ns", [
    ("input.li383_low_res", {}, 51),
    ("input.up_down_asymmetric_tokamak",
     dict(am=np.array([1.0, -1.0]), pres_scale=5000.0), 33),
])
def test_gk_geometry_matches_simsopt_vmec_fieldlines(deck, overrides, ns, tmp_path):
    """The GK geometry against an independent implementation, both symmetries.

    ``test_geometry_matches_stability_conventions`` compares this module to
    ``stability.py``, which is self-consistency: both are vmex.  simsopt's
    ``vmec_fieldlines`` is the implementation whose conventions this module
    claims, it reads a wout rather than vmex's state, and it carries the
    sine-parity families independently -- so the asymmetric row is a real
    external check of the parity handling, not a restatement.

    **Two accuracies, and the split is physics, not slack.**  ``gradpar`` and
    the pressure term ``cvdrift - gbdrift`` are convention-sensitive and agree
    to ~3e-3.  ``|B|`` does not: vmex evaluates the exact spectral field of the
    geometry while ``vmec_fieldlines`` reads the wout ``bmnc`` Nyquist table,
    which is band-limited (on ``li383_low_res``, ``xm_nyq`` reaches 7 against a
    geometry ``xm`` of 3, and the top two ``m`` bands hold 0.34% of the table).
    The drifts take a radial derivative of ``|B|``, which amplifies that by
    ``1/hs``, so they sit a few percent apart.

    That it is the band limit and not an error in either code is measured:
    simsopt's own ``gbdrift`` moves 6.9e-3 between ns = 101 and 201 and vmex's
    moves 5.9e-3, while the gap between them *plateaus* at 3.1e-2 -- it does
    not converge away, and it does not grow.  The asymmetric deck behaves
    identically (5.8e-2, 4.4e-2, 3.9e-2 at ns = 17, 33, 65), which is the
    point: parity handling adds no error of its own.
    """
    pytest.importorskip("simsopt")
    import vmex as vj

    inp = dataclasses.replace(
        VmecInput.from_file(DATA_DIR / deck), ns_array=np.array([ns]),
        ftol_array=np.array([1e-13]), niter_array=np.array([20000]), **overrides)
    eq = opt.solve_equilibrium(inp, verbose=False)
    assert eq.result.converged
    wout_path = vj.write_wout(str(tmp_path / "wout_parity.nc"), eq.wout)

    alpha, s_index = 0.7, int(round(0.4 * (ns - 1)))
    geom = turb.gk_fieldline_geometry(eq.state, eq.runtime, s_index=s_index,
                                      alpha=alpha, ntheta=16, equal_arc=False)
    s_value = float(np.asarray(stab._ballooning_context(
        eq.state, eq.runtime)["s"])[s_index])
    theta = np.asarray(geom["theta"])
    reference = _simsopt_fieldline(wout_path, s_value, alpha, theta)

    def pair(name):
        got = np.asarray(geom["gradpar" if name == "gradpar_theta_pest" else name])
        return got.ravel(), np.asarray(getattr(reference, name)).ravel()

    for name in _SIMSOPT_GEOMETRY:
        got, ref = pair(name)
        assert np.all(np.isfinite(got))
        scale = float(np.max(np.abs(ref)))
        assert scale > 0.0
        relative = float(np.max(np.abs(got - ref))) / scale
        # Ceilings, not targets: measured 6e-4..3.2e-2 symmetric and
        # 2.6e-3..5.8e-2 asymmetric, with the split explained above.
        ceiling = 1.0e-2 if name == "gradpar_theta_pest" else 1.0e-1
        assert relative < ceiling, f"{name}: {relative:.3e}"
        # Shape, not just magnitude: a sign or ordering error would pass a
        # loose bound on the drifts but not this.
        assert np.corrcoef(got, ref)[0, 1] > 0.99, name

    # The convention-sensitive combination, held tight.  Normalized against
    # the drift it contributes to rather than against itself: the pressure
    # term is 8.7e-2 of the drift on li383 but only 6.7e-4 on the low-beta
    # asymmetric deck, where a self-relative gate would be measuring noise.
    got_p = pair("cvdrift")[0] - pair("gbdrift")[0]
    ref_p = pair("cvdrift")[1] - pair("gbdrift")[1]
    drift_scale = float(np.max(np.abs(pair("gbdrift")[1])))
    assert float(np.max(np.abs(got_p - ref_p))) < 1.0e-3 * drift_scale
