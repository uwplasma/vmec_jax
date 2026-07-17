"""Analytic unit tests for :mod:`vmex.core.profiles`.

Every supported profile *kind* (parameterized and tabulated, including the
``_i``/``_ip`` current variants) is checked against a closed-form reference:

- power series / two_power / gauss_trunc / pedestal against their defining
  formulas and exact endpoint values (f(0), f(1));
- integrated current kinds against analytic antiderivatives (the 10-point
  Gauss-Legendre rule is exact for polynomial integrands of degree <= 19,
  and hits quadrature-error level on the truncated Gaussian);
- spline/line-segment kinds against data sampled from polynomials the
  interpolant reproduces exactly (linear for all three; quadratic for the
  clamped cubic spline, whose endpoint slopes come from a quadratic fit);
- the ``pressure``/``iota``/``current`` wrappers: ``pres_scale``, ``bloat``
  clamp, ``spres_ped`` pedestal hold, ``lrfp`` reciprocal, curtor-free
  normalization, and the NotImplementedError guards;
- differentiability: ``jax.grad`` through traced coefficients/scales (the
  ``_coeffs`` tracer fallback path).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import profiles

S = np.linspace(0.0, 1.0, 17)


# ---------------------------------------------------------------------------
# parameterized pressure/iota kinds
# ---------------------------------------------------------------------------


def test_power_series_matches_polyval():
    c = [1.5, -0.5, 2.0, 0.25]
    got = profiles.evaluate_profile("power_series", c, None, None, S)
    np.testing.assert_allclose(np.asarray(got), np.polyval(c[::-1], S), rtol=1e-14)


def test_two_power_formula_and_endpoints():
    c0, c1, c2 = 3.0, 2.0, 1.5
    got = np.asarray(profiles.evaluate_profile("two_power", [c0, c1, c2], None, None, S))
    np.testing.assert_allclose(got, c0 * (1.0 - S**c1) ** c2, rtol=1e-14)
    assert got[0] == pytest.approx(c0)
    assert got[-1] == pytest.approx(0.0, abs=1e-15)


def test_gauss_trunc_formula_and_endpoints():
    c0, c1 = 2.0, 0.6
    edge = math.exp(-((1.0 / c1) ** 2))
    ref = c0 / (1.0 - edge) * (np.exp(-((S / c1) ** 2)) - edge)
    got = np.asarray(profiles.evaluate_profile("gauss_trunc", [c0, c1], None, None, S))
    np.testing.assert_allclose(got, ref, rtol=1e-14)
    assert got[0] == pytest.approx(c0)  # normalized so f(0) = c0
    assert got[-1] == pytest.approx(0.0, abs=1e-15)


def test_pedestal_reduces_to_power_series_when_c19_nonpositive():
    c = np.zeros(21)
    c[:4] = [1.0, -1.0, 0.5, 0.25]
    c[17] = 5.0  # pedestal amplitude, dropped because c[19] = 0
    got = profiles.evaluate_profile("pedestal", c, None, None, S)
    np.testing.assert_allclose(np.asarray(got), np.polyval(c[3::-1], S), rtol=1e-14)


def test_pedestal_tanh_term_endpoints():
    # A normalizes the tanh pedestal so that f(0) = series(0) + c17 and the
    # pedestal contribution vanishes identically at x = 1.
    c = np.zeros(21)
    c[0] = 2.0
    c[17], c[18], c[19] = 0.75, 0.9, 0.1
    got = np.asarray(profiles.evaluate_profile("pedestal", c, None, None, S))
    assert got[0] == pytest.approx(c[0] + c[17], rel=1e-12)
    assert got[-1] == pytest.approx(c[0], rel=1e-12)


# ---------------------------------------------------------------------------
# parameterized current kinds (pcurr)
# ---------------------------------------------------------------------------


def test_current_power_series_is_integrated_iprime():
    ac = [2.0, -3.0, 4.0]  # I'(x) -> I(x) = 2x - 3x^2/2 + 4x^3/3
    got = np.asarray(profiles.current("power_series", ac, None, None, S))
    ref = 2.0 * S - 1.5 * S**2 + (4.0 / 3.0) * S**3
    np.testing.assert_allclose(got, ref, rtol=1e-14)
    assert got[0] == 0.0


def test_current_power_series_i_direct():
    ac = [1.0, 0.5, -0.25]  # I(x) = x + 0.5 x^2 - 0.25 x^3
    got = np.asarray(profiles.current("power_series_i", ac, None, None, S))
    np.testing.assert_allclose(got, S + 0.5 * S**2 - 0.25 * S**3, rtol=1e-14)


def test_current_two_power_gl_quadrature_exact_for_polynomials():
    # I'(x) = c0 (1 - x^2)^1 -> I(x) = c0 (x - x^3/3); degree-2 integrand is
    # exact under the 10-point Gauss-Legendre rule.
    c0 = 1.75
    got = np.asarray(profiles.current("two_power", [c0, 2.0, 1.0], None, None, S))
    np.testing.assert_allclose(got, c0 * (S - S**3 / 3.0), rtol=1e-13)


def test_current_gauss_trunc_matches_erf_antiderivative():
    c0, c1 = 1.3, 1.0
    edge = math.exp(-((1.0 / c1) ** 2))
    ref = np.asarray(
        [c0 * (c1 * math.sqrt(math.pi) / 2.0 * math.erf(x / c1) - edge * x) for x in S]
    )
    got = np.asarray(profiles.current("gauss_trunc", [c0, c1], None, None, S))
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_current_pedestal_i_series_part_and_origin():
    c = np.zeros(21)
    c[:3] = [2.0, -1.0, 3.0]  # I(x) = 2x - x^2/2 + x^3 with c[8:] = 0
    got = np.asarray(profiles.current("pedestal", c, None, None, S))
    np.testing.assert_allclose(got, 2.0 * S - 0.5 * S**2 + S**3, rtol=1e-13)
    assert got[0] == pytest.approx(0.0, abs=1e-15)


def test_current_pedestal_i_tanh_terms_vanish_at_origin():
    c = np.zeros(21)
    c[0] = 1.0
    c[8], c[10], c[11] = 0.5, 0.8, 0.2  # term1 active
    c[13], c[15], c[16] = 0.3, 0.5, 0.1  # term2
    c[17], c[19], c[20] = 0.2, 0.7, 0.1  # term3
    got = np.asarray(profiles.current("pedestal", c, None, None, S))
    assert got[0] == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.isfinite(got))
    # dropping the c[11] shape parameter kills term1 only
    c2 = c.copy()
    c2[11] = 0.0
    got2 = np.asarray(profiles.current("pedestal", c2, None, None, S))
    assert not np.allclose(got2, got)
    assert got2[0] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# tabulated kinds: cubic spline, Akima spline, line segments (+ _i/_ip)
# ---------------------------------------------------------------------------

KNOTS_X = np.array([0.0, 0.2, 0.45, 0.7, 1.0])
LIN = 2.0 + 3.0 * KNOTS_X  # linear data: all three interpolants are exact
LIN_F = lambda x: 2.0 + 3.0 * x  # noqa: E731
LIN_INT = lambda x: 2.0 * x + 1.5 * x**2  # noqa: E731


@pytest.mark.parametrize("kind", ["cubic_spline", "akima_spline", "line_segment"])
def test_tabulated_exact_on_linear_data(kind):
    got = profiles.evaluate_profile(kind, None, KNOTS_X, LIN, S)
    np.testing.assert_allclose(np.asarray(got), LIN_F(S), rtol=1e-12)


@pytest.mark.parametrize("kind", ["cubic_spline", "akima_spline", "line_segment"])
def test_tabulated_i_and_ip_current_variants(kind):
    # *_i: interpolate I directly; *_ip: integrate the interpolant from 0.
    got_i = profiles.current(f"{kind}_i", None, KNOTS_X, LIN, S)
    np.testing.assert_allclose(np.asarray(got_i), LIN_F(S), rtol=1e-12)
    got_ip = profiles.current(f"{kind}_ip", None, KNOTS_X, LIN, S)
    np.testing.assert_allclose(np.asarray(got_ip), LIN_INT(S), rtol=1e-12)


def test_cubic_spline_exact_on_quadratic_data():
    # VMEC's clamped spline fixes endpoint slopes with a quadratic fit, so
    # quadratic data is reproduced exactly (evaluation and integral).
    y = 1.0 + KNOTS_X + KNOTS_X**2
    got = profiles.evaluate_profile("cubic_spline", None, KNOTS_X, y, S)
    np.testing.assert_allclose(np.asarray(got), 1.0 + S + S**2, rtol=1e-11)
    got_ip = profiles.evaluate_profile("cubic_spline_ip", None, KNOTS_X, y, S)
    np.testing.assert_allclose(np.asarray(got_ip), S + S**2 / 2.0 + S**3 / 3.0, rtol=1e-11)


@pytest.mark.parametrize("kind", ["cubic_spline", "akima_spline", "line_segment"])
def test_tabulated_clips_outside_knot_range(kind):
    x = np.array([0.1, 0.5, 0.9])
    y = LIN_F(x)
    got = np.asarray(profiles.evaluate_profile(kind, None, x, y, np.array([0.0, 1.0])))
    np.testing.assert_allclose(got, [y[0], y[-1]], rtol=1e-12)


@pytest.mark.parametrize("kind", ["cubic_spline", "akima_spline", "line_segment"])
def test_tabulated_degenerate_knot_counts(kind):
    # zero knots -> zeros; one knot -> constant (or y0*x when integrating)
    got0 = profiles.evaluate_profile(kind, None, [], [], S)
    np.testing.assert_array_equal(np.asarray(got0), np.zeros_like(S))
    got1 = profiles.evaluate_profile(kind, None, [0.5], [4.0], S)
    np.testing.assert_allclose(np.asarray(got1), np.full_like(S, 4.0))
    got1_ip = profiles.current(f"{kind}_ip", None, [0.5], [4.0], S)
    np.testing.assert_allclose(np.asarray(got1_ip), 4.0 * S)
    got0_i = profiles.current(f"{kind}_i", None, [], [], S)
    np.testing.assert_array_equal(np.asarray(got0_i), np.zeros_like(S))


def test_cubic_spline_two_knots_is_secant_line():
    got = profiles.evaluate_profile("cubic_spline", None, [0.0, 1.0], [1.0, 3.0], S)
    np.testing.assert_allclose(np.asarray(got), 1.0 + 2.0 * S, rtol=1e-13)


def test_akima_spline_small_knot_fallback_and_smooth_data():
    # n < 4 falls back to the cubic Hermite form (linear data still exact)
    got = profiles.evaluate_profile("akima_spline", None, [0.0, 0.5, 1.0],
                                    LIN_F(np.array([0.0, 0.5, 1.0])), S)
    np.testing.assert_allclose(np.asarray(got), LIN_F(S), rtol=1e-12)
    # n >= 4 interpolates the knots exactly
    y = np.sin(2.0 * KNOTS_X)
    got = profiles.evaluate_profile("akima_spline", None, KNOTS_X, y, KNOTS_X)
    np.testing.assert_allclose(np.asarray(got), y, rtol=1e-12)


# ---------------------------------------------------------------------------
# wrapper functions: pressure / iota / current
# ---------------------------------------------------------------------------


def test_pressure_scale_bloat_and_pedestal_hold():
    am = [1.0, -1.0]  # p(x) = 1 - x
    p = np.asarray(profiles.pressure("power_series", am, None, None, S, pres_scale=2.5))
    np.testing.assert_allclose(p, 2.5 * (1.0 - S), rtol=1e-14)
    # bloat clamps x = min(|s*bloat|, 1): s >= 0.5 evaluates at x = 1
    p = np.asarray(profiles.pressure("power_series", am, None, None, S, bloat=2.0))
    np.testing.assert_allclose(p, np.clip(1.0 - 2.0 * S, 0.0, None), atol=1e-14)
    # spres_ped holds p at p(spres_ped) for s > spres_ped
    p = np.asarray(profiles.pressure("power_series", am, None, None, S, spres_ped=0.5))
    np.testing.assert_allclose(p, np.where(S > 0.5, 0.5, 1.0 - S), rtol=1e-14)


def test_pressure_tabulated_kind_dispatch():
    p = np.asarray(profiles.pressure("line_segment", None, KNOTS_X, LIN, S, pres_scale=2.0))
    np.testing.assert_allclose(p, 2.0 * LIN_F(S), rtol=1e-12)


def test_iota_power_series_and_lrfp_reciprocal():
    ai = [0.5, 0.25]
    np.testing.assert_allclose(
        np.asarray(profiles.iota("power_series", ai, None, None, S)),
        0.5 + 0.25 * S, rtol=1e-14)
    # lrfp: coefficients parameterize q = 1/iota
    got = np.asarray(profiles.iota("power_series", [2.0], None, None, S, lrfp=True))
    np.testing.assert_allclose(got, 0.5)
    got0 = np.asarray(profiles.iota("power_series", [0.0], None, None, S, lrfp=True))
    assert np.all(np.isinf(got0))


def test_not_implemented_guards():
    with pytest.raises(NotImplementedError):
        profiles.evaluate_profile("nope", [1.0], None, None, S)
    with pytest.raises(NotImplementedError):
        profiles.pressure("nope", [1.0], None, None, S)
    with pytest.raises(NotImplementedError):
        profiles.iota("two_power", [1.0], None, None, S)  # not a piota kind
    with pytest.raises(NotImplementedError):
        profiles.current("nope", [1.0], None, None, S)


def test_empty_aux_knots_give_zeros():
    got = profiles.evaluate_profile("cubic_spline", None, None, None, S)
    np.testing.assert_array_equal(np.asarray(got), np.zeros_like(S))


# ---------------------------------------------------------------------------
# differentiability: traced coefficients exercise the _coeffs tracer path
# ---------------------------------------------------------------------------


def test_grad_through_pres_scale_and_coefficients():
    am = jnp.asarray([1.0, -0.5])
    s = 0.4

    d_scale = jax.grad(
        lambda ps: profiles.pressure("power_series", am, None, None, s, pres_scale=ps)
    )(2.0)
    assert float(d_scale) == pytest.approx(1.0 - 0.5 * s, rel=1e-12)

    d_am = jax.grad(
        lambda c: profiles.pressure("power_series", c, None, None, s)
    )(am)
    np.testing.assert_allclose(np.asarray(d_am), [1.0, s], rtol=1e-12)


def test_grad_through_two_power_current_coefficients():
    ac = jnp.asarray([1.5, 2.0, 1.0])
    # dI/dc0 at x: I = c0 (x - x^3/3)
    d = jax.grad(
        lambda c: jnp.sum(profiles.current("two_power", c, None, None, 0.5))
    )(ac)
    assert float(d[0]) == pytest.approx(0.5 - 0.5**3 / 3.0, rel=1e-10)
