"""Radial profile evaluation for pressure, rotational transform, and current.

VMEC2000 counterparts: ``LIBSTELL/Sources/Miscel/profile_functions.f``
(functions ``pmass``, ``piota``, ``pcurr``) as used by ``profil1d.f``.

Every profile is a pure function of the normalized toroidal flux
``s in [0, 1]``.  All evaluation code is written with ``jax.numpy`` so the
functions are usable inside ``jax.jit``/``jax.grad`` closures; profile *kinds*
and knot-array shapes are static (Python-level) while coefficient values may
be traced.

Units
-----
* :func:`pressure` returns pressure in **Pascals** (``PRES_SCALE * pmass(x)``
  with the VMEC input coefficients ``AM`` in Pa).  VMEC2000's ``pmass``
  function returns ``mu0 * pres_scale * pmass`` (internal units, same as
  ``B**2``); multiply by :data:`MU0` to obtain VMEC internal units.  This
  matches the historical ``vmex.profiles.eval_profiles`` behavior where
  ``pressure_pa`` is Pa and ``pressure = MU0 * pressure_pa``.
* :func:`iota` is dimensionless (or the safety factor ``q = 1/iota`` input
  when ``lrfp=True``).
* :func:`current` returns VMEC's dimensionless current shape function
  ``I(x)``; VMEC2000 (``profil1d.f``) later normalizes it so that
  ``I(1)`` corresponds to ``CURTOR`` when ``NCURR=1``.  That scaling is left
  to the caller.

Supported kinds (``evaluate_profile``):

===================  =========================================================
kind                 definition (coefficients ``c`` ascending, ``x`` in [0,1])
===================  =========================================================
power_series         ``sum_i c[i] x**i``
two_power            ``c[0] (1 - x**c[1])**c[2]``
gauss_trunc          ``c[0]/(1-E) * (exp(-(x/c[1])**2) - E)``,
                     ``E = exp(-(1/c[1])**2)`` (normalized so f(0)=c[0])
pedestal             VMEC2000 ``pmass`` pedestal: degree-15 power series in
                     ``c[0:16]`` plus a tanh pedestal shaped by ``c[16:21]``
cubic_spline         VMEC ``spline_cubic`` through (aux_s, aux_f) knots
akima_spline         VMEC/STELLOPT ``spline_akima`` through the knots
line_segment         linear interpolation through the knots
power_series_ip      current: ``I(x) = sum_i c[i]/(i+1) x**(i+1)`` (VMEC
                     ``pcurr_type='power_series'``: c parameterizes I')
power_series_i       current: ``I(x) = sum_i c[i] x**(i+1)``
two_power_ip         current: ``I(x) = int_0^x two_power(c, t) dt``
gauss_trunc_ip       current: ``I(x) = int_0^x c[0](exp(-(t/c[1])**2)-E) dt``
pedestal_i           VMEC2000 ``pcurr`` pedestal parameterization of I(x)
cubic_spline_i(_ip)  spline of I (``_i``) or of I' integrated (``_ip``)
akima_spline_i(_ip)  idem, Akima spline
line_segment_i(_ip)  idem, line segments
===================  =========================================================

Numerical integration of the ``*_ip`` parameterized kinds uses a fixed
16-point Gauss-Legendre rule on [0, x] (VMEC2000 uses a 10-point rule; the
difference is at quadrature-error level).  Spline ``*_ip`` kinds are
integrated analytically piecewise, as in the historical implementation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "MU0",
    "evaluate_profile",
    "pressure",
    "iota",
    "current",
]

#: Vacuum permeability [N/A^2]; VMEC2000 ``stel_constants`` uses 4e-7*pi.
MU0 = 4e-7 * np.pi

# 10-point Gauss-Legendre quadrature on [0, 1] used by VMEC2000 to integrate
# parameterized I'(x) -> I(x).  The nodes/weights are copied verbatim from
# ``LIBSTELL/Sources/Miscel/profile_functions.f`` (``gln = 10``, ``glx/glw``)
# so the integrated 'two_power'/'gauss_trunc' current profiles are bit-exact
# against VMEC2000 (a higher-order rule would be *less* parity-accurate).
_GL_X = jnp.asarray([
    0.01304673574141414, 0.06746831665550774, 0.1602952158504878,
    0.2833023029353764, 0.4255628305091844, 0.5744371694908156,
    0.7166976970646236, 0.8397047841495122, 0.9325316833444923,
    0.9869532642585859,
])
_GL_W = jnp.asarray([
    0.03333567215434407, 0.0747256745752903, 0.1095431812579910,
    0.1346333596549982, 0.1477621123573764, 0.1477621123573764,
    0.1346333596549982, 0.1095431812579910, 0.0747256745752903,
    0.03333567215434407,
])


def _coeffs(coefficients):
    """Return coefficients as a static NumPy vector when possible.

    Concrete inputs become NumPy arrays so Horner loops use plain Python
    scalars; traced JAX arrays are kept as-is (the loops remain valid since
    their length is static).
    """
    try:
        arr = np.asarray(coefficients, dtype=np.float64)
        return arr.reshape(-1) if arr.ndim != 1 else arr
    except Exception:
        return jnp.ravel(jnp.asarray(coefficients))


def _coeffs_padded(coefficients, n: int):
    """Coefficients as above, zero-padded on the right to length >= ``n``."""
    c = _coeffs(coefficients)
    k = int(c.shape[0])
    if k >= n:
        return c
    if isinstance(c, np.ndarray):
        return np.pad(c, (0, n - k))
    return jnp.concatenate([c, jnp.zeros((n - k,), dtype=c.dtype)])


# ----------------------------------------------------------------------------
# Parameterized profiles (profile_functions.f pmass/piota cases)
# ----------------------------------------------------------------------------


def _power_series(coefficients, x):
    """``sum_i c[i] x**i`` by Horner (pmass/piota 'power_series' default)."""
    c = _coeffs(coefficients)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x)
    for i in range(len(c) - 1, -1, -1):
        y = y * x + c[i]
    return y


def _two_power(coefficients, x):
    """``c0 * (1 - x**c1)**c2`` (profile_functions.f 'two_power')."""
    c = _coeffs_padded(coefficients, 3)
    x = jnp.asarray(x)
    core = jnp.maximum(1.0 - x ** c[1], 0.0)
    return c[0] * core ** c[2]


def _gauss_trunc(coefficients, x):
    """Truncated Gaussian (pmass 'gauss_trunc', normalized so f(0)=c0).

    ``f(x) = c0/(1 - E) * (exp(-(x/c1)**2) - E)`` with ``E = exp(-(1/c1)**2)``.
    """
    c = _coeffs_padded(coefficients, 2)
    x = jnp.asarray(x)
    edge = jnp.exp(-((1.0 / c[1]) ** 2))
    return c[0] / (1.0 - edge) * (jnp.exp(-((x / c[1]) ** 2)) - edge)


def _pedestal(coefficients, x):
    """VMEC2000 pmass 'pedestal' (PMV Texas group; profile_functions.f).

    Degree-15 power series in ``c[0:16]`` plus a tanh pedestal:
    ``A * c[17] * (tanh(2(c18 - sqrt(x))/c19) - tanh(2(c18 - 1)/c19))`` with
    ``A = 1/(tanh(2 c18/c19) - tanh(2(c18-1)/c19))``.  When ``c[19] <= 0``
    the pedestal term is dropped (Fortran zeroes ``am(16:20)``).  Note the
    input value of ``c[20]`` is ignored (VMEC overwrites it with ``A``).
    """
    c = _coeffs_padded(coefficients, 21)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x)
    for i in range(15, -1, -1):
        y = y * x + c[i]
    c18, c19 = jnp.asarray(c[18]), jnp.asarray(c[19])
    ok = c19 > 0.0
    safe19 = jnp.where(ok, c19, 1.0)
    amp = 1.0 / (jnp.tanh(2.0 * c18 / safe19) - jnp.tanh(2.0 * (c18 - 1.0) / safe19))
    ped = amp * c[17] * (
        jnp.tanh(2.0 * (c18 - jnp.sqrt(x)) / safe19)
        - jnp.tanh(2.0 * (c18 - 1.0) / safe19)
    )
    return y + jnp.where(ok, ped, 0.0)


# ----------------------------------------------------------------------------
# Parameterized current profiles (profile_functions.f pcurr cases)
# ----------------------------------------------------------------------------


def _pcurr_power_series_ip(coefficients, x):
    """pcurr 'power_series': ``I'(x)`` power series, analytic integral.

    ``I(x) = sum_i c[i]/(i+1) x**(i+1)`` (profile_functions.f default case).
    """
    c = _coeffs(coefficients)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x)
    for i in range(len(c) - 1, -1, -1):
        y = y * x + c[i] / (i + 1)
    return x * y


def _pcurr_power_series_i(coefficients, x):
    """pcurr 'power_series_I': ``I(x) = sum_i c[i] x**(i+1)`` directly."""
    c = _coeffs(coefficients)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x)
    for i in range(len(c) - 1, -1, -1):
        y = (y + c[i]) * x
    return y


def _integrate_0_to_x(fun, coefficients, x):
    """``int_0^x fun(c, t) dt`` by VMEC2000's 10-point Gauss-Legendre rule.

    ``profile_functions.f``: ``pcurr = x * sum_i glw(i) * fun(x*glx(i))`` with
    the hard-coded ``[0, 1]`` nodes/weights above (exact VMEC2000 parity).
    """
    x = jnp.asarray(x)
    t = x[..., None] * _GL_X[None, :]
    return x * jnp.sum(_GL_W[None, :] * fun(coefficients, t), axis=-1)


def _pcurr_two_power_ip(coefficients, x):
    """pcurr 'two_power': ``I'`` is two_power, integrated numerically."""
    return _integrate_0_to_x(_two_power, coefficients, x)


def _gauss_trunc_iprime(coefficients, x):
    """pcurr 'gauss_trunc' integrand: ``c0 (exp(-(x/c1)**2) - exp(-(1/c1)**2))``.

    Note: unlike the pmass 'gauss_trunc', the pcurr integrand is *not*
    normalized by ``1 - exp(-(1/c1)**2)`` (see profile_functions.f).
    """
    c = _coeffs_padded(coefficients, 2)
    x = jnp.asarray(x)
    return c[0] * (jnp.exp(-((x / c[1]) ** 2)) - jnp.exp(-((1.0 / c[1]) ** 2)))


def _pcurr_gauss_trunc_ip(coefficients, x):
    """pcurr 'gauss_trunc': truncated-Gaussian ``I'``, integrated numerically."""
    return _integrate_0_to_x(_gauss_trunc_iprime, coefficients, x)


def _pcurr_pedestal_i(coefficients, x):
    """VMEC2000 pcurr 'pedestal' parameterization of I(x).

    ``I(x) = sum_{i=0..7} c[i]/(i+1) x**(i+1)`` plus tanh pedestal terms
    shaped by ``c[8:21]`` (profile_functions.f, SPH 2010-05-26).  The input
    value of ``c[12]`` is ignored (overwritten with the pedestal
    normalization); when ``c[11] <= 0`` the ``c[8]`` pedestal term is dropped.
    """
    c = _coeffs_padded(coefficients, 21)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x)
    for i in range(7, -1, -1):
        y = y * x + c[i] / (i + 1)
    y = x * y

    c10, c11 = jnp.asarray(c[10]), jnp.asarray(c[11])
    ok = c11 > 0.0
    safe11 = jnp.where(ok, c11, 1.0)
    amp = 1.0 / (jnp.tanh(2.0 * c10 / safe11) - jnp.tanh(2.0 * (c10 - 1.0) / safe11))
    term1 = amp * c[8] * (
        jnp.tanh(2.0 * c10 / safe11) - jnp.tanh(2.0 * (c10 - jnp.sqrt(x)) / safe11)
    )
    a8 = jnp.maximum(jnp.asarray(c[16]), 0.01)
    a12 = jnp.maximum(jnp.asarray(c[20]), 0.01)
    term2 = c[13] * (jnp.tanh((x - c[15]) / a8) - jnp.tanh((0.0 - c[15]) / a8))
    term3 = c[17] * (jnp.tanh((x - c[19]) / a12) - jnp.tanh((0.0 - c[19]) / a12))
    return y + jnp.where(ok, term1, 0.0) + term2 + term3


# ----------------------------------------------------------------------------
# Tabulated profiles: cubic spline, Akima spline, line segments
# (spline_cubic.f, spline_akima.f, line_segment.f conventions)
# ----------------------------------------------------------------------------


def _cubic_endpoint_derivatives(x_knots, y_knots):
    """Endpoint slopes used by VMEC's ``spline_cubic``.

    VMEC fixes the first derivative at each endpoint with a quadratic fit
    through the first/last three knots (not a natural spline); with only two
    knots this reduces to the secant slope.
    """
    n = int(x_knots.shape[0])
    if n <= 1:
        zero = jnp.zeros((), dtype=y_knots.dtype)
        return zero, zero
    if n == 2:
        slope = (y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0])
        return slope, slope
    c_left = (
        (y_knots[2] - y_knots[0]) / (x_knots[2] - x_knots[0])
        - (y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0])
    ) / (x_knots[2] - x_knots[1])
    yp1 = (y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0]) - c_left * (
        x_knots[1] - x_knots[0]
    )
    c_right = (
        (y_knots[-3] - y_knots[-1]) / (x_knots[-3] - x_knots[-1])
        - (y_knots[-2] - y_knots[-1]) / (x_knots[-2] - x_knots[-1])
    ) / (x_knots[-3] - x_knots[-2])
    ypn = (y_knots[-2] - y_knots[-1]) / (x_knots[-2] - x_knots[-1]) - c_right * (
        x_knots[-2] - x_knots[-1]
    )
    return yp1, ypn


def _cubic_second_derivatives(x_knots, y_knots):
    """Second derivatives at the knots for VMEC's clamped cubic spline."""
    n = int(x_knots.shape[0])
    if n <= 2:
        return jnp.zeros_like(y_knots)
    h = x_knots[1:] - x_knots[:-1]
    yp1, ypn = _cubic_endpoint_derivatives(x_knots, y_knots)
    mat = jnp.zeros((n, n), dtype=y_knots.dtype)
    rhs = jnp.zeros((n,), dtype=y_knots.dtype)
    mat = mat.at[0, 0].set(2.0 * h[0])
    mat = mat.at[0, 1].set(h[0])
    rhs = rhs.at[0].set(6.0 * ((y_knots[1] - y_knots[0]) / h[0] - yp1))
    mat = mat.at[-1, -2].set(h[-1])
    mat = mat.at[-1, -1].set(2.0 * h[-1])
    rhs = rhs.at[-1].set(6.0 * (ypn - (y_knots[-1] - y_knots[-2]) / h[-1]))
    for idx in range(1, n - 1):
        hm, hp = h[idx - 1], h[idx]
        mat = mat.at[idx, idx - 1].set(hm)
        mat = mat.at[idx, idx].set(2.0 * (hm + hp))
        mat = mat.at[idx, idx + 1].set(hp)
        rhs = rhs.at[idx].set(
            6.0
            * (
                (y_knots[idx + 1] - y_knots[idx]) / hp
                - (y_knots[idx] - y_knots[idx - 1]) / hm
            )
        )
    return jnp.linalg.solve(mat, rhs)


def _cubic_interval_integral(y0, y1, m0, m1, h, dx):
    """Integral of one cubic-spline interval from its left knot to ``dx``."""
    a = y0 - m0 * h * h / 6.0
    b = y1 - m1 * h * h / 6.0
    return (
        m0 * (h**4 - (h - dx) ** 4) / (24.0 * h)
        + m1 * dx**4 / (24.0 * h)
        + a * (dx - dx**2 / (2.0 * h))
        + b * dx**2 / (2.0 * h)
    )


def _knot_setup(x_knots, y_knots, x):
    """Shared bracket lookup: clip to the knot range and find the interval."""
    n = int(x_knots.shape[0])
    x_clipped = jnp.clip(jnp.asarray(x), x_knots[0], x_knots[-1])
    idx = jnp.clip(jnp.searchsorted(x_knots, x_clipped, side="right"), 1, n - 1) - 1
    h = x_knots[1:] - x_knots[:-1]
    return x_clipped, idx, h, x_clipped - x_knots[idx]


def _prefix_plus_partial(full_interval, partial, idx):
    """Cumulative integral: sum of full intervals before ``idx`` + partial."""
    prefix = jnp.concatenate(
        [jnp.zeros((1,), dtype=full_interval.dtype), jnp.cumsum(full_interval)]
    )
    return prefix[idx] + partial


def _cubic_spline(x_knots, y_knots, x, *, integrate: bool):
    """Evaluate (or integrate from 0) VMEC's cubic spline through the knots."""
    x = jnp.asarray(x)
    n = int(x_knots.shape[0])
    if n == 0:
        return jnp.zeros_like(x)
    if n == 1:
        return y_knots[0] * x if integrate else jnp.broadcast_to(y_knots[0], x.shape)
    _, idx, h, dx = _knot_setup(x_knots, y_knots, x)
    m = _cubic_second_derivatives(x_knots, y_knots)
    h_i, y0, y1, m0, m1 = h[idx], y_knots[idx], y_knots[idx + 1], m[idx], m[idx + 1]
    if not integrate:
        return (
            m0 * (h_i - dx) ** 3 / (6.0 * h_i)
            + m1 * dx**3 / (6.0 * h_i)
            + (y0 - m0 * h_i * h_i / 6.0) * (h_i - dx) / h_i
            + (y1 - m1 * h_i * h_i / 6.0) * dx / h_i
        )
    full = _cubic_interval_integral(y_knots[:-1], y_knots[1:], m[:-1], m[1:], h, h)
    return _prefix_plus_partial(full, _cubic_interval_integral(y0, y1, m0, m1, h_i, dx), idx)


def _line_segment(x_knots, y_knots, x, *, integrate: bool):
    """Evaluate (or integrate from 0) VMEC's line-segment profile."""
    x = jnp.asarray(x)
    n = int(x_knots.shape[0])
    if n == 0:
        return jnp.zeros_like(x)
    if n == 1:
        return y_knots[0] * x if integrate else jnp.broadcast_to(y_knots[0], x.shape)
    _, idx, h, dx = _knot_setup(x_knots, y_knots, x)
    slopes = (y_knots[1:] - y_knots[:-1]) / h
    if not integrate:
        return y_knots[idx] + slopes[idx] * dx
    full = 0.5 * h * (y_knots[:-1] + y_knots[1:])
    partial = y_knots[idx] * dx + 0.5 * slopes[idx] * dx * dx
    return _prefix_plus_partial(full, partial, idx)


def _akima_coefficients(x_knots, y_knots):
    """VMEC/STELLOPT Akima Hermite coefficients (spline_akima.f).

    VMEC stops for fewer than four knots; here small knot counts fall back to
    the cubic-spline Hermite form so tiny diagnostic grids stay usable.
    """
    n = int(x_knots.shape[0])
    if n < 4:
        m = _cubic_second_derivatives(x_knots, y_knots)
        h = x_knots[1:] - x_knots[:-1]
        y0, y1, m0, m1 = y_knots[:-1], y_knots[1:], m[:-1], m[1:]
        a = y0
        b = (y1 - y0) / h - h * (2.0 * m0 + m1) / 6.0
        c = m0 / 2.0
        d = (m1 - m0) / (6.0 * h)
        return a, b, c, d, h

    xloc = jnp.zeros((n + 4,), dtype=x_knots.dtype).at[2 : 2 + n].set(x_knots)
    yloc = jnp.zeros((n + 4,), dtype=y_knots.dtype).at[2 : 2 + n].set(y_knots)
    xloc = xloc.at[0].set(2.0 * xloc[2] - xloc[4])
    xloc = xloc.at[1].set(xloc[2] + xloc[3] - xloc[4])
    xloc = xloc.at[n + 3].set(2.0 * xloc[n + 1] - xloc[n - 1])
    xloc = xloc.at[n + 2].set(xloc[n + 1] + xloc[n] - xloc[n - 1])

    m = jnp.zeros((n + 3,), dtype=y_knots.dtype)
    slopes = (yloc[3 : 2 + n] - yloc[2 : 1 + n]) / (xloc[3 : 2 + n] - xloc[2 : 1 + n])
    m = m.at[2 : n + 1].set(slopes)

    cl = (m[3] - m[2]) / (xloc[4] - xloc[2])
    bl = m[2] - cl * (xloc[3] - xloc[2])
    cr = (m[n - 1] - m[n]) / (xloc[n + 1] - xloc[n - 1])
    br = m[n - 1] - cr * (xloc[n] - xloc[n - 1])
    yloc = yloc.at[1].set(yloc[2] + bl * (xloc[1] - xloc[2]) + cl * (xloc[1] - xloc[2]) ** 2)
    yloc = yloc.at[0].set(yloc[2] + bl * (xloc[0] - xloc[2]) + cl * (xloc[0] - xloc[2]) ** 2)
    yloc = yloc.at[n + 2].set(
        yloc[n + 1] + br * (xloc[n + 2] - xloc[n + 1]) + cr * (xloc[n + 2] - xloc[n + 1]) ** 2
    )
    yloc = yloc.at[n + 3].set(
        yloc[n + 1] + br * (xloc[n + 3] - xloc[n + 1]) + cr * (xloc[n + 3] - xloc[n + 1]) ** 2
    )

    m = m.at[0].set((yloc[1] - yloc[0]) / (xloc[1] - xloc[0]))
    m = m.at[1].set((yloc[2] - yloc[1]) / (xloc[2] - xloc[1]))
    m = m.at[n + 1].set((yloc[n + 2] - yloc[n + 1]) / (xloc[n + 2] - xloc[n + 1]))
    m = m.at[n + 2].set((yloc[n + 3] - yloc[n + 2]) / (xloc[n + 3] - xloc[n + 2]))

    dm = jnp.abs(m[1:] - m[:-1])
    tangents = []
    for idx in range(1, n + 1):
        denom = dm[idx + 1] + dm[idx - 1]
        weighted = (dm[idx + 1] * m[idx] + dm[idx - 1] * m[idx + 1]) / jnp.where(
            denom == 0.0, 1.0, denom
        )
        tangents.append(jnp.where(denom == 0.0, 0.5 * (m[idx] + m[idx + 1]), weighted))
    t = jnp.asarray(tangents)

    h = x_knots[1:] - x_knots[:-1]
    interval_slope = (y_knots[1:] - y_knots[:-1]) / h
    a = y_knots[:-1]
    b = t[:-1]
    c = (3.0 * interval_slope - t[1:] - 2.0 * t[:-1]) / h
    d = (t[1:] + t[:-1] - 2.0 * interval_slope) / (h * h)
    return a, b, c, d, h


def _akima_spline(x_knots, y_knots, x, *, integrate: bool):
    """Evaluate (or integrate from 0) VMEC's Akima spline through the knots."""
    x = jnp.asarray(x)
    n = int(x_knots.shape[0])
    if n == 0:
        return jnp.zeros_like(x)
    if n == 1:
        return y_knots[0] * x if integrate else jnp.broadcast_to(y_knots[0], x.shape)
    _, idx, h, dx = _knot_setup(x_knots, y_knots, x)
    a, b, c, d, h_all = _akima_coefficients(x_knots, y_knots)
    if not integrate:
        return a[idx] + dx * (b[idx] + dx * (c[idx] + d[idx] * dx))
    full = a * h_all + 0.5 * b * h_all**2 + (c / 3.0) * h_all**3 + 0.25 * d * h_all**4
    partial = dx * (a[idx] + dx * (0.5 * b[idx] + dx * ((c[idx] / 3.0) + 0.25 * d[idx] * dx)))
    return _prefix_plus_partial(full, partial, idx)


_TABULATED = {
    "cubic_spline": _cubic_spline,
    "akima_spline": _akima_spline,
    "line_segment": _line_segment,
}

_PARAMETERIZED = {
    "power_series": _power_series,
    "two_power": _two_power,
    "gauss_trunc": _gauss_trunc,
    "pedestal": _pedestal,
    "power_series_ip": _pcurr_power_series_ip,
    "power_series_i": _pcurr_power_series_i,
    "two_power_ip": _pcurr_two_power_ip,
    "gauss_trunc_ip": _pcurr_gauss_trunc_ip,
    "pedestal_i": _pcurr_pedestal_i,
}


def evaluate_profile(kind: str, coefficients, aux_s, aux_f, s):
    """Evaluate one VMEC profile parameterization at ``s``.

    VMEC2000 counterpart: the ``SELECT CASE`` bodies of ``pmass``/``piota``/
    ``pcurr`` in ``profile_functions.f`` (one case per ``kind``; see the
    module docstring for the full table).

    Parameters
    ----------
    kind:
        Profile family, case-insensitive (static under ``jit``).
    coefficients:
        Ascending coefficients (VMEC ``AM``/``AI``/``AC``); ignored by the
        tabulated kinds.
    aux_s, aux_f:
        Knot abscissae/values (VMEC ``*_AUX_S``/``*_AUX_F``); used only by the
        spline/line-segment kinds.  Shapes must be static.
    s:
        Evaluation points; for tabulated kinds values are clipped to the knot
        range (VMEC evaluates only inside the knot span).

    Returns
    -------
    jax.Array with the same shape as ``s`` (no ``pres_scale``/``mu0``/
    ``curtor`` scaling applied; see the wrapper functions).
    """
    key = str(kind).strip().lower()
    if key in _PARAMETERIZED:
        return _PARAMETERIZED[key](coefficients, s)
    base, _, suffix = key.rpartition("_")
    if base in _TABULATED and suffix in ("i", "ip"):
        x_knots = jnp.ravel(jnp.asarray([] if aux_s is None else aux_s, dtype=jnp.float64))
        y_knots = jnp.ravel(jnp.asarray([] if aux_f is None else aux_f, dtype=jnp.float64))
        if int(x_knots.shape[0]) == 0 or int(y_knots.shape[0]) == 0:
            return jnp.zeros_like(jnp.asarray(s, dtype=jnp.float64))
        return _TABULATED[base](x_knots, y_knots, s, integrate=(suffix == "ip"))
    if key in _TABULATED:
        x_knots = jnp.ravel(jnp.asarray([] if aux_s is None else aux_s, dtype=jnp.float64))
        y_knots = jnp.ravel(jnp.asarray([] if aux_f is None else aux_f, dtype=jnp.float64))
        if int(x_knots.shape[0]) == 0 or int(y_knots.shape[0]) == 0:
            return jnp.zeros_like(jnp.asarray(s, dtype=jnp.float64))
        return _TABULATED[key](x_knots, y_knots, s, integrate=False)
    raise NotImplementedError(
        f"profile kind {kind!r} not implemented (supported: "
        f"{sorted(_PARAMETERIZED) + sorted(_TABULATED)} and the tabulated _i/_ip variants)"
    )


def _bloated(s, bloat):
    """VMEC argument clamp ``x = min(|s * bloat|, 1)`` (profile_functions.f)."""
    return jnp.minimum(jnp.abs(jnp.asarray(s) * bloat), 1.0)


def pressure(pmass_type: str, am, am_aux_s, am_aux_f, s, *,
             pres_scale=1.0, bloat=1.0, spres_ped=1.0):
    """Pressure profile p(s) in **Pascals** (VMEC2000 ``pmass`` x ``1/mu0``).

    VMEC2000 counterpart: ``pmass(xx)`` in ``profile_functions.f`` plus the
    ``spres_ped`` pedestal clamp applied in ``profil1d.f``:
    for ``s > spres_ped`` the pressure is held at ``p(spres_ped)``.

    Returns ``pres_scale * pmass_raw(min(|s*bloat|, 1))`` in Pa.  Multiply by
    :data:`MU0` for VMEC internal units (``mu0 * Pa``, the units of ``B**2``),
    exactly as VMEC2000's ``pmass`` returns ``mu0 * pres_scale * pmass``.
    ``spres_ped`` is a static (host) float; ``pres_scale`` and the
    coefficients may be traced.
    """
    kind = str(pmass_type).strip().lower()
    if kind not in ("power_series", "two_power", "gauss_trunc", "pedestal",
                    "cubic_spline", "akima_spline", "line_segment"):
        raise NotImplementedError(f"pmass_type={pmass_type!r} not implemented")
    x = _bloated(s, bloat)
    p = pres_scale * evaluate_profile(kind, am, am_aux_s, am_aux_f, x)
    spres_ped = abs(float(spres_ped))
    if spres_ped < 1.0:
        x_ped = jnp.minimum(jnp.abs(jnp.asarray(spres_ped) * bloat), 1.0)
        p_ped = pres_scale * evaluate_profile(kind, am, am_aux_s, am_aux_f, x_ped)
        p = jnp.where(jnp.asarray(s) > spres_ped, p_ped, p)
    return p


def iota(piota_type: str, ai, ai_aux_s, ai_aux_f, s, *, bloat=1.0, lrfp=False):
    """Rotational-transform profile iota(s) (dimensionless).

    VMEC2000 counterpart: ``piota(x)`` in ``profile_functions.f``.  With
    ``lrfp=True`` the ``ai`` coefficients parameterize the safety factor
    ``q = 1/iota`` and the reciprocal is returned (infinite where q = 0).

    Note: VMEC2000's ``piota`` does not apply the ``bloat`` clamp internally;
    the historical vmex implementation applied it uniformly and this port
    keeps that behavior (identical for the default ``bloat = 1``).
    """
    kind = str(piota_type).strip().lower()
    if kind not in ("power_series", "cubic_spline", "akima_spline", "line_segment"):
        raise NotImplementedError(f"piota_type={piota_type!r} not implemented")
    x = _bloated(s, bloat)
    value = evaluate_profile(kind, ai, ai_aux_s, ai_aux_f, x)
    if lrfp:
        value = jnp.where(value != 0.0, 1.0 / value, jnp.asarray(jnp.inf, dtype=value.dtype))
    return value


#: pcurr_type -> evaluate_profile kind (profile_functions.f pcurr cases).
_PCURR_KINDS = {
    "power_series": "power_series_ip",
    "power_series_i": "power_series_i",
    "two_power": "two_power_ip",
    "gauss_trunc": "gauss_trunc_ip",
    "pedestal": "pedestal_i",
    "cubic_spline_i": "cubic_spline_i",
    "cubic_spline_ip": "cubic_spline_ip",
    "akima_spline_i": "akima_spline_i",
    "akima_spline_ip": "akima_spline_ip",
    "line_segment_i": "line_segment_i",
    "line_segment_ip": "line_segment_ip",
}


def current(pcurr_type: str, ac, ac_aux_s, ac_aux_f, s, *, bloat=1.0):
    """Enclosed-toroidal-current shape function I(s) (unnormalized).

    VMEC2000 counterpart: ``pcurr(xx)`` in ``profile_functions.f``.  Kinds
    whose VMEC name parameterizes ``I'`` (``power_series``, ``two_power``,
    ``gauss_trunc``, ``*_ip``) are integrated from 0 to
    ``x = min(|s*bloat|, 1)``; ``*_i`` kinds (and ``pedestal``) parameterize
    ``I`` directly.  VMEC2000 (``profil1d.f``) rescales the result so the
    edge value matches ``CURTOR`` when ``NCURR = 1``; that scaling is the
    caller's responsibility.
    """
    kind = _PCURR_KINDS.get(str(pcurr_type).strip().lower())
    if kind is None:
        raise NotImplementedError(f"pcurr_type={pcurr_type!r} not implemented")
    return evaluate_profile(kind, ac, ac_aux_s, ac_aux_f, _bloated(s, bloat))
