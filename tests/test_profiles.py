"""Analytic unit tests for :mod:`vmex.core.profiles`: every profile kind
(parameterized and tabulated, incl. ``_i``/``_ip`` current variants) vs a
closed-form reference — defining formulas + exact endpoints; integrated
current kinds vs analytic antiderivatives (10-point GL exact to degree 19);
spline/line-segment kinds vs polynomials the interpolant reproduces
exactly; the ``pressure``/``iota``/``current`` wrappers (pres_scale, bloat
clamp, spres_ped hold, lrfp reciprocal, curtor-free normalization, typed
guards); and ``jax.grad`` through traced coefficients/scales.
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


# ---------------------------------------------------------------------------
# sum_atan (profile_functions.f pcurr/piota)
# ---------------------------------------------------------------------------


def _sum_atan_fortran(ac, x):
    """``profile_functions.f`` lines 375-384, transcribed literally.

    The reference is the Fortran itself rather than a simplification of it,
    including the hardcoded ``x >= 1`` branch, so a divergence in either shows
    up as a mismatch instead of being defined away.
    """
    a = np.zeros(21)
    ac = np.asarray(ac, dtype=float)
    a[: min(21, ac.size)] = ac[:21]
    if x >= 1.0:
        return a[0] + a[1] + a[5] + a[9] + a[13] + a[17]
    return a[0] + (2.0 / np.pi) * (
        a[1] * np.arctan(a[2] * x ** a[3] / (1 - x) ** a[4])
        + a[5] * np.arctan(a[6] * x ** a[7] / (1 - x) ** a[8])
        + a[9] * np.arctan(a[10] * x ** a[11] / (1 - x) ** a[12])
        + a[13] * np.arctan(a[14] * x ** a[15] / (1 - x) ** a[16])
        + a[17] * np.arctan(a[18] * x ** a[19] / (1 - x) ** a[20])
    )


#: (0) the HSX benchmark deck's own AC, (1) two groups with a negative
#: amplitude, (2) all five groups with fractional exponents, (3) a short array
#: that must pad, (4) a zero-exponent group, since ``0**0`` is 1 in Fortran.
_SUM_ATAN_CASES = {
    "hsx_deck": [0.0, 1.0, 1.00423652381532e01, 1.50747420899044e00, 1.0]
    + [0.0] * 16,
    "two_groups": [0.3, 0.8, 4.0, 2.0, 1.0, -0.25, 7.5, 1.25, 0.5] + [0.0] * 12,
    "five_groups": [0.1, 0.5, 3.0, 1.0, 1.0, 0.2, 2.0, 1.5, 0.5, -0.1, 5.0,
                    2.0, 1.5, 0.3, 1.0, 0.5, 1.0, -0.2, 8.0, 3.0, 2.0],
    "short_array": [0.5, 2.0, 3.0],
    "zero_exponent": [0.0, 1.0, 2.0, 0.0, 1.0] + [0.0] * 16,
}


@pytest.mark.parametrize("case", sorted(_SUM_ATAN_CASES))
def test_sum_atan_matches_profile_functions_f(case):
    """Values against the transcribed Fortran, endpoints included.

    The grid deliberately includes ``x = 0`` and ``x = 1`` and the points just
    inside them: those are the two places the expression is singular, and both
    are real grid points for a VMEC profile.
    """
    ac = _SUM_ATAN_CASES[case]
    x = np.concatenate([[0.0, 1e-12, 1e-6],
                        np.linspace(0.001, 0.999, 40),
                        [1.0 - 1e-9, 1.0]])
    got = np.asarray(profiles.evaluate_profile("sum_atan", ac, None, None, x))
    expected = np.array([_sum_atan_fortran(ac, xi) for xi in x])
    assert np.all(np.isfinite(got))
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    assert float(np.max(np.abs(got - expected))) / scale < 1e-13
    # The hardcoded edge, exactly as Fortran writes it.
    a = np.zeros(21)
    a[: min(21, len(ac))] = np.asarray(ac, dtype=float)[:21]
    assert float(got[-1]) == pytest.approx(
        a[0] + a[1] + a[5] + a[9] + a[13] + a[17], rel=1e-14)


def test_sum_atan_is_differentiable_at_both_endpoints():
    """No NaN in the gradient at ``x = 0`` or ``x = 1``.

    ``jnp.power`` with a non-integer exponent goes through ``exp(e log x)``,
    so a naive ``x ** c`` puts ``log(0)`` on the tape and returns a NaN
    derivative at the axis; the ``x >= 1`` branch of a ``jnp.where`` is
    evaluated too, so an unguarded ``1/(1-x)`` poisons the other branch.  Both
    are real grid points, so both matter.
    """
    ac = jnp.asarray(_SUM_ATAN_CASES["five_groups"])
    for x in (0.0, 1e-14, 0.5, 1.0 - 1e-14, 1.0):
        d = jax.grad(
            lambda xx: profiles.evaluate_profile("sum_atan", ac, None, None, xx)
        )(x)
        assert np.isfinite(float(d)), f"d/dx at x={x}"
    grad_c = jax.grad(
        lambda c: jnp.sum(profiles.evaluate_profile(
            "sum_atan", c, None, None, jnp.linspace(0.0, 1.0, 17)))
    )(ac)
    assert bool(jnp.all(jnp.isfinite(grad_c)))
    assert float(jnp.max(jnp.abs(grad_c))) > 0.0


def test_sum_atan_reaches_the_current_and_iota_wrappers():
    """``sum_atan`` is selectable as ``pcurr_type`` and ``piota_type``.

    VMEC2000 offers it for those two and not for ``pmass``
    (``profile_functions.f`` has a ``CASE('sum_atan')`` in ``pcurr`` and
    ``piota`` only), so the pressure wrapper must keep rejecting it.
    """
    ac = _SUM_ATAN_CASES["hsx_deck"]
    x = np.linspace(0.0, 1.0, 9)
    expected = np.array([_sum_atan_fortran(ac, xi) for xi in x])
    # pcurr: sum_atan parameterizes I(s) itself, so no integration.
    np.testing.assert_allclose(
        np.asarray(profiles.current("sum_atan", ac, None, None, x)),
        expected, rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(
        np.asarray(profiles.iota("sum_atan", ac, None, None, x)),
        expected, rtol=1e-13, atol=1e-300)
    with pytest.raises(NotImplementedError, match="pmass_type"):
        profiles.pressure("sum_atan", ac, None, None, x)


# ---------------------------------------------------------------------------
# The remaining VMEC2000 parameterizations
# ---------------------------------------------------------------------------


def _pad21(c):
    a = np.zeros(21)
    c = np.asarray(c, dtype=float)
    a[: min(21, c.size)] = c[:21]
    return a


def _f_two_power(b, x):
    """``functions.f`` two_power."""
    return b[0] * max(1.0 - x ** b[1], 0.0) ** b[2]


def _f_two_power_gs(b, x):
    """``functions.f`` lines 63-69: two_power times six Gaussian peaks."""
    gaussians = 1.0
    for i in range(3, 19, 3):
        gaussians += b[i] * np.exp(-(((x - b[i + 1]) / b[i + 2]) ** 2))
    return gaussians * _f_two_power(b, x)


def _f_two_lorentz(am, x):
    """``profile_functions.f`` pmass 'two_Lorentz', transcribed literally."""
    one = 1.0
    return am[0] * (
        am[1] * (one / (one + (x / am[2] ** 2) ** am[3]) ** am[4]
                 - one / (one + (one / am[2] ** 2) ** am[3]) ** am[4])
        / (one - one / (one + (one / am[2] ** 2) ** am[3]) ** am[4])
        + (one - am[1]) * (one / (one + (x / am[5] ** 2) ** am[6]) ** am[7]
                           - one / (one + (one / am[5] ** 2) ** am[6]) ** am[7])
        / (one - one / (one + (one / am[5] ** 2) ** am[6]) ** am[7]))


def _f_rational(a, x):
    """``profile_functions.f`` 'rational': c[0:10] over c[10:21], Horner."""
    numerator = 0.0
    for i in range(9, -1, -1):
        numerator = x * numerator + a[i]
    denominator = 0.0
    for i in range(20, 9, -1):
        denominator = x * denominator + a[i]
    return (numerator / denominator if denominator != 0.0
            else np.finfo(np.float64).max)


def _f_nice_quadratic(ai, x):
    """``profile_functions.f`` piota 'nice_quadratic'."""
    return ai[0] * (1.0 - x) + ai[1] * x + 4.0 * ai[2] * x * (1.0 - x)


_REMAINING_KINDS = {
    "two_power_gs": (
        _pad21([2.0, 2.0, 1.5, 0.4, 0.3, 0.12, -0.2, 0.7, 0.09, 0.15, 0.5,
                0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        _f_two_power_gs),
    "two_lorentz": (_pad21([1.5, 0.6, 0.8, 2.0, 1.5, 0.4, 3.0, 1.0]),
                    _f_two_lorentz),
    "rational": (_pad21([1.0, -0.5, 0.25] + [0.0] * 7 + [2.0, 0.5, -0.3]),
                 _f_rational),
    "nice_quadratic": (_pad21([0.9, 0.45, 0.12]), _f_nice_quadratic),
}


@pytest.mark.parametrize("kind", sorted(_REMAINING_KINDS))
def test_remaining_kinds_match_profile_functions_f(kind):
    """Each against a literal transcription, endpoints included."""
    coefficients, reference = _REMAINING_KINDS[kind]
    x = np.concatenate([[0.0], np.linspace(0.01, 0.99, 40), [1.0]])
    got = np.asarray(profiles.evaluate_profile(kind, coefficients, None, None, x))
    expected = np.array([reference(coefficients, xi) for xi in x])
    assert np.all(np.isfinite(got))
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    assert float(np.max(np.abs(got - expected))) / scale < 1e-14


def test_rational_returns_the_fortran_huge_on_a_zero_denominator():
    """VMEC returns ``HUGE`` there, not a NaN or an infinity.

    That value reaches a wout and a caller can see it, so reproducing it is
    part of the parity rather than an implementation detail.  All-zero
    denominator coefficients are the way to hit it.
    """
    got = profiles.evaluate_profile("rational", _pad21([1.0]), None, None,
                                    np.array([0.0, 0.5, 1.0]))
    assert np.all(np.asarray(got) == np.finfo(np.float64).max)


def test_two_power_gs_reduces_to_two_power_without_peaks():
    """Zero Gaussian amplitudes must leave ``two_power`` untouched.

    The peak block multiplies rather than adds, so a sign or an offset error
    in it would survive a comparison that only ever exercised nonzero peaks.
    """
    c = _pad21([2.0, 2.0, 1.5] + [0.0, 0.5, 1.0] * 6)
    x = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(
        np.asarray(profiles.evaluate_profile("two_power_gs", c, None, None, x)),
        np.asarray(profiles.evaluate_profile("two_power", c[:3], None, None, x)),
        rtol=1e-14, atol=0.0)


def test_every_vmec2000_profile_type_is_selectable():
    """The three wrappers accept exactly what ``profile_functions.f`` does.

    Enumerated from its ``SELECT CASE`` labels: ``pmass`` has nine explicit
    cases plus ``power_series`` as ``CASE DEFAULT``, ``piota`` six plus the
    default, ``pcurr`` sixteen plus the default.  All of them are implemented,
    so the counts below are the contract: if VMEC2000 gains a case, this test
    is what notices.
    """
    x = np.linspace(0.0, 1.0, 5)
    generic = _pad21([1.0, 2.0, 1.5])
    # Shape parameters that are widths or exponents cannot be zero: 'two_power_gs'
    # divides by its Gaussian widths and 'two_Lorentz' by 1 - f(1), both of which
    # are 0/0 for an all-zero tail in Fortran as well.  A generic coefficient
    # vector would be testing the input, not the code.
    per_kind = {
        "sum_cossq_s": _pad21([5.0, 1.0, 0.8, 0.5, 0.3, 0.1]),
        "sum_cossq_sqrts": _pad21([5.0, 1.0, 0.8, 0.5, 0.3, 0.1]),
        "sum_cossq_s_free": _pad21([1.0, 0.2, 0.15, 0.6, 0.55, 0.2]),
        "two_lorentz": _pad21([1.5, 0.6, 0.8, 2.0, 1.5, 0.4, 3.0, 1.0]),
        "two_power_gs": _REMAINING_KINDS["two_power_gs"][0],
        "rational": _REMAINING_KINDS["rational"][0],
        "pedestal": _pad21([1.0, -0.5]),
    }
    tabulated = ("spline", "line_segment")

    def coefficients_for(kind):
        return per_kind.get(kind, generic)

    for kind in sorted(profiles._PMASS_KINDS):
        if any(marker in kind for marker in tabulated):
            continue                      # tabulated: need knots, covered above
        assert np.all(np.isfinite(profiles.pressure(
            kind, coefficients_for(kind), None, None, x))), kind
    for kind in sorted(profiles._PIOTA_KINDS):
        if any(marker in kind for marker in tabulated):
            continue
        assert np.all(np.isfinite(profiles.iota(
            kind, coefficients_for(kind), None, None, x))), kind
    for kind in sorted(profiles._PCURR_KINDS):
        if any(marker in kind for marker in tabulated):
            continue
        assert np.all(np.isfinite(profiles.current(
            kind, coefficients_for(kind), None, None, x))), kind
    assert len(profiles._PMASS_KINDS) == 10
    assert len(profiles._PIOTA_KINDS) == 7
    assert len(profiles._PCURR_KINDS) == 17
    # A name VMEC2000 does not have must still be refused, not guessed at.
    for unknown in ("sum_cossq", "two_power_g", "quadratic"):
        with pytest.raises(NotImplementedError):
            profiles.current(unknown, generic, None, None, x)


def _f_sum_cossq_s(ac, x):
    """``profile_functions.f`` 'sum_cossq_s', transcribed literally."""
    n = int(ac[0])
    dx = 1.0 / (n - 1.0)
    xi = [(i - 1) * dx for i in range(0, n + 1)]
    start = {i: max(0.0, xi[i] - dx) for i in range(1, n + 1)}
    end = {i: min(xi[i] + dx, 1.0) for i in range(1, n + 1)}
    reached = 1
    for i in range(1, n + 1):
        if start[i] < x <= end[i]:
            reached = i
    total = 0.0
    for i in range(1, reached + 1):
        if start[i] < x <= end[i]:
            wave = dx * np.sin(np.pi * (x - xi[i]) / dx) / np.pi
            total += (0.5 * ac[i] * (x + wave) if i == 1
                      else 0.5 * ac[i] * (x - xi[i] + dx + wave))
        else:
            total += 0.5 * ac[i] * dx if i == 1 else ac[i] * dx
    return total


def _f_sum_cossq_sqrts(ac, x):
    """``profile_functions.f`` 'sum_cossq_sqrts', transcribed literally."""
    n = int(ac[0])
    root = np.sqrt(x)
    dx = 1.0 / (n - 1.0)
    dx2, pi2 = dx * dx, np.pi * np.pi
    xi = [(i - 1) * dx for i in range(0, n + 1)]
    start = {i: max(0.0, xi[i] - dx) for i in range(1, n + 1)}
    end = {i: min(xi[i] + dx, 1.0) for i in range(1, n + 1)}
    reached = 1
    for i in range(1, n + 1):
        if start[i] < root <= end[i]:
            reached = i
    total = 0.0
    for i in range(1, reached + 1):
        if start[i] < root <= end[i]:
            phase = np.pi * (root - xi[i]) / dx
            if i == 1:
                total += ac[i] * (0.25 * x
                                  + dx / (2 * np.pi) * root * np.sin(phase)
                                  + dx2 / (2 * pi2) * (np.cos(phase) - 1.0))
            else:
                total += ac[i] * (0.25 * x
                                  + dx / (2 * np.pi) * root * np.sin(phase)
                                  + dx2 / (2 * pi2) * np.cos(phase)
                                  - 0.25 * start[i] ** 2 + dx2 / (2 * pi2))
        else:
            total += (ac[i] * (0.25 - 1 / pi2) * dx2 if i == 1
                      else ac[i] * dx * xi[i])
    return total


def _f_sum_cossq_s_free(ac, x):
    """``profile_functions.f`` 'sum_cossq_s_free', transcribed literally."""
    total = 0.0
    for i in range(1, 8):
        xi, width = ac[1 + 3 * (i - 1)], ac[2 + 3 * (i - 1)]
        amplitude = ac[3 * (i - 1)]
        start, end = max(0.0, xi - width), min(xi + width, 1.0)
        if amplitude == 0.0:
            continue
        at_start = np.sin(np.pi * (start - xi) / width)
        if start < x <= end:
            total += amplitude * 0.5 * (
                (x - start) + width / np.pi
                * (np.sin(np.pi * (x - xi) / width) - at_start))
        elif x > end:
            total += amplitude * 0.5 * (
                (end - start) + width / np.pi
                * (np.sin(np.pi * (end - xi) / width) - at_start))
    return total


@pytest.mark.parametrize("kind, coefficients", [
    ("sum_cossq_s", [5.0, 1.0, 0.8, 0.5, 0.3, 0.1]),
    ("sum_cossq_s", [3.0, 2.0, -0.5, 1.2]),
    ("sum_cossq_sqrts", [5.0, 1.0, 0.8, 0.5, 0.3, 0.1]),
    ("sum_cossq_sqrts", [4.0, 0.7, 1.1, -0.4, 0.6]),
    ("sum_cossq_s_free", [1.0, 0.2, 0.15, 0.6, 0.55, 0.2, -0.3, 0.85, 0.1]),
])
def test_sum_cossq_matches_profile_functions_f(kind, coefficients):
    """The three cos-squared current forms, against literal transcriptions.

    These are ``I(s)`` in closed form: the density is a sum of cos-squared
    windows and the integral is done analytically, so a wave the argument has
    already passed contributes its whole area and at most one is partial.
    That bookkeeping is where an error would hide, which is why the grid runs
    across several window boundaries.
    """
    reference = {"sum_cossq_s": _f_sum_cossq_s,
                 "sum_cossq_sqrts": _f_sum_cossq_sqrts,
                 "sum_cossq_s_free": _f_sum_cossq_s_free}[kind]
    c = _pad21(coefficients)
    x = np.concatenate([[0.0], np.linspace(0.005, 0.995, 60), [1.0]])
    got = np.asarray(profiles.current(kind, c, None, None, x))
    expected = np.array([reference(c, xi) for xi in x])
    assert np.all(np.isfinite(got))
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    assert float(np.max(np.abs(got - expected))) / scale < 1e-14
    # I is the integral of a sum of cos-squared windows, so it is
    # non-decreasing whenever every amplitude is non-negative.  That is a
    # property of the result rather than of the transcription, so it catches a
    # sign or an interval-pairing error that a matching transcription would
    # not.  (Note I(0) is not 0 here: the window test is a strict ``>``, so at
    # x = 0 the first window already counts as fully integrated, so I(0) sits
    # *above* I of the next point.  That is a real quirk of the Fortran, the
    # comparison above pins it, and it is why the monotonicity check starts at
    # the second sample rather than the first.)
    if float(np.min(c[1:])) >= 0.0:
        assert np.all(np.diff(got[1:]) >= -1e-12)


def test_sum_cossq_rejects_an_out_of_range_wave_count():
    """VMEC stops on a count outside 1..20; vmex raises instead of guessing."""
    for bad in (0.0, 21.0, -3.0):
        with pytest.raises(ValueError, match="1..20"):
            profiles.current("sum_cossq_s", _pad21([bad, 1.0]), None, None,
                             np.array([0.5]))
