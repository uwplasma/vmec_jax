"""Exact-solution validation of the strong-force certificate (plan 31.2-R5).

``tests/test_strong_force.py`` anchors the oracle on a vacuum ``1/R`` field,
where ``grad p = 0`` makes the normalized certificate degenerate (it saturates
at exactly 2), and ``input.solovev`` is a VMEC *solve*, not an analytic
equilibrium.  This module supplies the missing anchor: a closed-form Solov'ev
equilibrium that satisfies ``J x B = grad p`` by construction, handed to the
same oracle, with the measured error under refinement.

The equilibrium
---------------
``psi(R, Z) = b (R^2 - R0^2)^2 + g Z^2`` solves the Grad-Shafranov equation

    Delta* psi = -mu0 R^2 p'(psi) - F F'(psi),    Delta* psi = 8 b R^2 + 2 g,

so ``p`` is linear in ``psi`` (``p' = -8 b / mu0``) and ``F^2 = F0^2 - 4 g
psi``.  Its surfaces are exactly ellipses in ``(R^2 - R0^2, Z)``:

    R(s, w)^2 = R0^2 (1 + a rho cos w),   Z(s, w) = c rho sin w,   rho = sqrt(s)

with the radial label ``s = psi/psi_a``, ``a = sqrt(psi_a/b)/R0^2`` and
``c = sqrt(psi_a/g)``.  The ``(s, theta, phi)`` Jacobian
``R (R_s Z_w - R_w Z_s) = R0^2 a c / 4`` is *constant*, which makes every
remaining piece elementary:

* ``chipf = dpsi/ds = psi_a``, constant;
* ``lambda(w) = 2 arctan(t tan(w/2)) - w`` with ``t = sqrt((1-x)/(1+x))`` and
  ``x = a rho``, whose Fourier series is ``2 sum_m ((-tau)^m/m) sin(m w)``
  with ``tau = 2x/((1+x)(1+t)^2)`` (the removable form of ``(1-t)/(1+t)``);
* ``phipf`` is fixed by ``<lambda_theta> = 0``, i.e.
  ``phipf = F(s) sqrt(g) / (R0^2 sqrt(1 - a^2 s))``, which makes the state's
  Clebsch field equal the closed-form ``B = F grad(phi) + grad(phi) x grad(psi)``
  identically rather than approximately.

``R``'s poloidal harmonics are not a finite series, so they are supplied as
the exact Taylor-in-``x`` expansion of ``sqrt(1 + x cos w)``: the ``x^k``
coefficient is ``binom(1/2, k) cos^k w``, whose ``cos(m w)`` content is
computed by an exact DFT of ``cos^k w``.  Only ``k = m + 2j`` reaches
``cos(m w)``, so every harmonic carries the ``rho^|m|`` prefactor that the
native state's regularity contract requires, with no cancellation near the
axis.  ``test_series_helpers_reproduce_the_closed_forms`` checks both series
against direct 4096-point DFTs at 1e-15.

Reference data: none on disk.  Every number is generated in-process from the
closed forms above, so the provenance is the derivation in this docstring plus
the two DFT cross-checks.

What is measured
----------------
The state is the exact solution *projected* onto the native basis, so the
certificate cannot be zero; it has to converge as the basis is refined.  With
``R0 = 3``, ``a = 0.25``, ``c = 0.8``, ``F0 = 6``, ``psi_a = 0.2`` — aspect
ratio 8.0, elongation 2.1, ``|iota| = 0.67``, ``p(0) = 5.03e4 Pa``, and
``|J x B| = |grad p| = 5e4 .. 1.7e5 N/m^3`` so the pointwise denominator is
nowhere near its saturation limit:

* radial refinement, spline degree 3, ``mpol = 13``, 2/4/8/16 spans:
  ``normalized_l2`` 1.082e-5 -> 2.302e-6 -> 5.382e-7 -> 1.303e-7, observed
  orders 2.234, 2.096, 2.047.  That is ``O(h^(p-1))``: the spline error in
  ``q(s)`` is ``O(h^(p+1))`` and the force takes two derivatives of the
  coordinate map.  ``absolute_l2`` follows it, 2.25 -> 0.025 N/m^3.
* degree 5 at the same 4 spans reaches 7.44e-10, 3095x below degree 3.
* Fourier refinement, degree 5, 4 spans, ``mpol - 1 = 4/8/12``: 8.234e-3 ->
  2.455e-6 -> 7.437e-10.  The two ratios are 3355 and 3300 -- equal to within
  2%, i.e. geometric, at ``tau^4 = 0.127^4 ~ 1/3800``, the analytic decay rate
  of the lambda series.

The floor, and what it means for the reported numbers
-----------------------------------------------------
Refinement stops paying at ``normalized_l2 ~ 1e-10`` (``absolute_l2 ~
1.6e-5 N/m^3`` against ``|J x B| ~ 1e5``, so ~1e-10 relative).  Measured
outside CI at degree 5: 16 spans / ``mpol = 17`` gives 1.48e-10 and 8 spans
gives 7.48e-11, and raising ``mpol`` from 17 to 21 changes both only in the
fifth digit -- so the floor is not Fourier truncation, and raising the spline
degree (7) or the span count makes it slightly *worse*.  It is the round-off
of the nested ``jacfwd`` that builds ``curl B``.  Six to seven orders of
magnitude below the ``~1e-3`` polish numbers the certificate is used to
report, so the oracle's own noise is not what limits those.
"""


from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.profiles import MU0
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import (
    HighOrderEquilibriumState,
    certify_strong_force,
    evaluate_high_order_fields,
    evaluate_strong_force,
)

# Every evaluation here is a compiled reduction over tens of thousands of
# shifted quadrature points: ~7 s jitted against ~20-30 s interpreted.
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

# ---------------------------------------------------------------------------
# closed-form Solov'ev equilibrium
# ---------------------------------------------------------------------------


def _binom_half(kmax: int) -> np.ndarray:
    """``binom(1/2, k)`` for ``k = 0..kmax`` by the exact recurrence."""
    out = np.ones((kmax + 1,), dtype=float)
    for k in range(1, kmax + 1):
        out[k] = out[k - 1] * (0.5 - k + 1) / k
    return out


def _cos_power_cosine_coefficients(kmax: int, mmax: int, ngrid: int = 512) -> np.ndarray:
    """``A[k, m]`` = the ``cos(m w)`` coefficient of ``cos(w)**k``.

    Exact for ``kmax < ngrid``: ``cos^k w`` is a trigonometric polynomial of
    degree ``k``, so the DFT reproduces its coefficients to round-off.
    """
    w = 2.0 * np.pi * np.arange(ngrid) / ngrid
    cos_w = np.cos(w)
    out = np.zeros((kmax + 1, mmax + 1), dtype=float)
    powers = np.ones_like(cos_w)
    for k in range(kmax + 1):
        spectrum = np.fft.rfft(powers) / ngrid
        out[k, 0] = spectrum[0].real
        out[k, 1 : mmax + 1] = 2.0 * spectrum[1 : mmax + 1].real
        powers = powers * cos_w
    return out


def _sqrt_series_cosine_coefficients(mmax: int, kmax: int = 80) -> np.ndarray:
    """``T[m, j]`` with ``sqrt(1 + x cos w) = sum_m x^m (sum_j T[m,j] x^(2j)) cos(m w)``.

    The ``x^k`` Taylor coefficient of ``sqrt(1 + x cos w)`` is
    ``binom(1/2, k) cos^k w``; only ``k = m + 2j`` contributes to ``cos(m w)``,
    which is why every harmonic carries the ``x^m`` (hence ``rho^m``) prefactor
    the native state requires.
    """
    binom = _binom_half(kmax)
    cospow = _cos_power_cosine_coefficients(kmax, mmax)
    jmax = (kmax - mmax) // 2
    out = np.zeros((mmax + 1, jmax + 1), dtype=float)
    for m in range(mmax + 1):
        for j in range(jmax + 1):
            k = m + 2 * j
            if k <= kmax:
                out[m, j] = binom[k] * cospow[k, m]
    return out


class _Solovev:
    """Closed-form Solov'ev equilibrium and its native-basis coefficients."""

    def __init__(
        self,
        *,
        R0: float = 3.0,
        a: float = 0.25,
        c: float = 0.8,
        F0: float = 6.0,
        psi_a: float = 0.2,
        mmax: int = 12,
        kmax: int = 80,
    ) -> None:
        self.R0, self.a, self.c, self.F0, self.psi_a = R0, a, c, F0, psi_a
        self.beta = psi_a / (a * R0**2) ** 2  # psi = beta (R^2-R0^2)^2 + gamma Z^2
        self.gamma = psi_a / c**2
        # Signed (s, theta, phi) Jacobian R (R_theta Z_s - R_s Z_theta); the
        # native convention makes it negative for Z ~ +sin(theta), as in the
        # vacuum fixture of tests/test_strong_force.py.
        self.jacobian = -(R0**2) * a * c / 4.0
        self.mmax = mmax
        self._series = _sqrt_series_cosine_coefficients(mmax, kmax)

    # -- closed forms --------------------------------------------------
    def psi(self, R, Z):
        return self.beta * (R**2 - self.R0**2) ** 2 + self.gamma * Z**2

    def toroidal_function(self, s):
        """``F(s) = R B_phi``; ``F^2 = F0^2 - 4 gamma psi``."""
        return np.sqrt(self.F0**2 - 4.0 * self.gamma * self.psi_a * np.asarray(s))

    def pressure(self, s):
        """``p(s)`` in Pa; ``p' = -8 beta / mu0`` in psi, zero at the edge."""
        return (8.0 * self.beta * self.psi_a / MU0) * (1.0 - np.asarray(s))

    def geometry(self, s, w):
        rho = np.sqrt(np.asarray(s))
        return (
            self.R0 * np.sqrt(1.0 + self.a * rho * np.cos(w)),
            self.c * rho * np.sin(w),
        )

    def cylindrical_field(self, R, Z):
        """``(B_R, B_phi, B_Z)`` of ``B = F grad(phi) + grad(phi) x grad(psi)``."""
        F = np.sqrt(self.F0**2 - 4.0 * self.gamma * self.psi(R, Z))
        return (
            2.0 * self.gamma * Z / R,
            F / R,
            -4.0 * self.beta * (R**2 - self.R0**2),
        )

    def rotational_transform(self, s):
        return self.psi_a / self.phipf(s)

    # -- native-basis profiles -----------------------------------------
    def phipf(self, s):
        """Set by ``<lambda_theta> = 0``, so the Clebsch field is the exact one."""
        s = np.asarray(s, dtype=float)
        return (
            self.toroidal_function(s)
            * self.jacobian
            / (self.R0**2 * np.sqrt(1.0 - self.a**2 * s))
        )

    def q_radius(self, m: int, s):
        """``R_m(s) / rho^m``, a truncated polynomial in ``s``."""
        j = np.arange(self._series.shape[1])
        s = np.asarray(s, dtype=float)[..., None]
        return self.R0 * np.sum(
            self._series[m] * self.a ** (m + 2 * j) * s**j, axis=-1
        )

    def q_lambda(self, m: int, s):
        """``lambda_m(s) / rho^m``; ``lambda_m = 2 (-tau)^m / m``.

        ``tau/x = 2/((1+x)(1+t)^2)`` is the removable form of
        ``tau = (1-t)/(1+t)``, so this is finite at the axis.
        """
        x = self.a * np.sqrt(np.asarray(s, dtype=float))
        t = np.sqrt((1.0 - x) / (1.0 + x))
        tau_over_x = 2.0 / ((1.0 + x) * (1.0 + t) ** 2)
        return 2.0 * ((-1.0) ** m) * (self.a * tau_over_x) ** m / m


def _analytic_state(
    eq: _Solovev, *, degree: int = 5, spans: int = 8, mmax: int | None = None
) -> HighOrderEquilibriumState:
    """Project the closed-form equilibrium onto the native spline/Fourier basis."""
    mmax = eq.mmax if mmax is None else mmax
    basis = BSplineBasis.clamped(
        np.linspace(0.0, 1.0, spans + 1), degree=degree, quadrature_order=degree + 3
    )
    nodes = np.asarray(basis.collocation_nodes)
    zeros = np.zeros((mmax + 1, basis.size))
    R_cos = zeros.copy()
    Z_sin = zeros.copy()
    L_sin = zeros.copy()
    for m in range(mmax + 1):
        R_cos[m] = basis.fit(jnp.asarray(eq.q_radius(m, nodes)))
        if m >= 1:
            L_sin[m] = basis.fit(jnp.asarray(eq.q_lambda(m, nodes)))
    Z_sin[1] = basis.fit(jnp.asarray(np.full(nodes.shape, eq.c)))
    boundary_R = np.array([float(eq.q_radius(m, np.asarray(1.0))) for m in range(mmax + 1)])
    boundary_Z = np.zeros((mmax + 1,))
    boundary_Z[1] = eq.c
    return HighOrderEquilibriumState(
        radial_basis=basis,
        m=np.arange(mmax + 1),
        n=np.zeros((mmax + 1,), dtype=int),
        nfp=1,
        R_cos=jnp.asarray(R_cos),
        R_sin=jnp.asarray(zeros),
        Z_cos=jnp.asarray(zeros),
        Z_sin=jnp.asarray(Z_sin),
        L_cos=jnp.asarray(zeros),
        L_sin=jnp.asarray(L_sin),
        phipf=basis.fit(jnp.asarray(eq.phipf(nodes))),
        chipf=basis.fit(jnp.asarray(np.full(nodes.shape, eq.psi_a))),
        pressure=basis.fit(jnp.asarray(eq.pressure(nodes))),
        jacobian_sign=-1,
        source="analytic Solov'ev",
        boundary_R_cos=jnp.asarray(boundary_R),
        boundary_R_sin=jnp.asarray(np.zeros((mmax + 1,))),
        boundary_Z_cos=jnp.asarray(np.zeros((mmax + 1,))),
        boundary_Z_sin=jnp.asarray(boundary_Z),
    )


def _cylindrical_basis(zeta):
    zeta = np.asarray(zeta)
    zero = np.zeros_like(zeta)
    one = np.ones_like(zeta)
    return (
        np.stack([np.cos(zeta), np.sin(zeta), zero], axis=-1),
        np.stack([-np.sin(zeta), np.cos(zeta), zero], axis=-1),
        np.stack([zero, zero, one], axis=-1),
    )


# ---------------------------------------------------------------------------
# one compiled sweep, shared by the assertions below
# ---------------------------------------------------------------------------

#: ``(spline degree, spans, mpol - 1)`` projections that get certified.
#: Degree 3 at four span counts is the radial ladder; degree 5 at four spans
#: with a growing poloidal cut is the Fourier ladder.  The shared corner
#: ``(5, 4, 12)`` is also the degree comparison against ``(3, 4, 12)`` and the
#: floor anchor, so every compiled configuration carries an assertion.
_RADIAL_LADDER = [(3, spans, 12) for spans in (2, 4, 8, 16)]
_FOURIER_LADDER = [(5, 4, mmax) for mmax in (4, 8, 12)]
_FINEST = (5, 4, 12)


@pytest.fixture(scope="module")
def certificates() -> dict[tuple[int, int, int], object]:
    """Certify every projection once, jitted.

    Interpreted, one certificate costs ~30 s; jitted it costs ~7 s, almost
    all of it compiling the nested ``jacfwd`` graph, which is why the
    configuration list is short and every entry is used by an assertion.
    ``angular_multiplier=1`` is quadrature-converged here: measured at
    ``(3, 8, 12)``, multipliers 1, 2 and 3 give ``normalized_l2``
    5.381837e-7, 5.381835e-7 and 5.381835e-7.
    """
    out: dict[tuple[int, int, int], object] = {}
    for key in [*_RADIAL_LADDER, *_FOURIER_LADDER]:
        if key in out:
            continue
        degree, spans, mmax = key
        state = _analytic_state(
            _Solovev(mmax=mmax), degree=degree, spans=spans, mmax=mmax
        )
        out[key] = certify_strong_force(state, angular_multiplier=1)
    return out


# ---------------------------------------------------------------------------
# the state really is the analytic equilibrium
# ---------------------------------------------------------------------------


def test_analytic_state_reproduces_the_closed_form_geometry_and_field():
    """Before certifying anything, check what was handed to the oracle."""
    eq = _Solovev(mmax=12)
    state = _analytic_state(eq, degree=5, spans=8, mmax=12)
    rho = jnp.asarray([0.2, 0.45, 0.7, 0.95])
    theta = jnp.asarray([0.3, 1.7, 3.4, 5.1])
    zeta = jnp.asarray([0.0, 0.9, 2.2, 4.4])
    fields = evaluate_high_order_fields(state, rho, theta, zeta)

    position = np.asarray(fields.position)
    R_num = np.hypot(position[:, 0], position[:, 1])
    Z_num = position[:, 2]
    s = np.asarray(rho) ** 2
    R_exact, Z_exact = eq.geometry(s, np.asarray(theta))
    np.testing.assert_allclose(R_num, R_exact, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(Z_num, Z_exact, rtol=0.0, atol=1e-11)

    # The Solov'ev (s, theta, phi) Jacobian is constant; the native one is
    # 2 rho / nfp times it.
    np.testing.assert_allclose(
        np.asarray(fields.sqrt_g), 2.0 * np.asarray(rho) * eq.jacobian,
        rtol=1e-12, atol=0.0,
    )

    B_R, B_phi, B_Z = eq.cylindrical_field(R_exact, Z_exact)
    e_R, e_phi, e_Z = _cylindrical_basis(zeta)
    B_exact = B_R[:, None] * e_R + B_phi[:, None] * e_phi + B_Z[:, None] * e_Z
    np.testing.assert_allclose(np.asarray(fields.B), B_exact, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(fields.pressure), eq.pressure(s), rtol=0.0, atol=1e-9
    )

    # A real finite-beta case: the certificate denominator is not degenerate.
    samples = evaluate_strong_force(state, rho, theta, zeta)
    lorentz = np.asarray(samples.lorentz_norm)
    grad_p = np.asarray(samples.grad_pressure_norm)
    assert np.all(lorentz > 1.0e4)
    np.testing.assert_allclose(lorentz, grad_p, rtol=1e-9, atol=0.0)
    residual = np.linalg.norm(np.asarray(samples.force), axis=-1)
    assert float(np.max(residual / lorentz)) < 1.0e-8


def test_analytic_state_is_a_realistic_finite_beta_tokamak():
    """Guard the fixture against silently drifting into a trivial regime."""
    eq = _Solovev()
    s = np.linspace(0.05, 1.0, 8)
    iota = eq.rotational_transform(s)
    assert 0.5 < float(np.min(np.abs(iota))) < 1.0
    assert float(eq.pressure(0.0)) == pytest.approx(5.0301e4, rel=1e-3)
    R_out, _ = eq.geometry(1.0, 0.0)
    R_in, _ = eq.geometry(1.0, np.pi)
    aspect = 2.0 * eq.R0 / (R_out - R_in)
    assert 7.0 < aspect < 9.0
    assert 1.5 < 2.0 * eq.c / (R_out - R_in) < 3.0  # elongation


# ---------------------------------------------------------------------------
# convergence of the certificate under refinement
# ---------------------------------------------------------------------------


def test_certificate_converges_at_the_expected_radial_order(certificates):
    """Degree 3 gives ``O(h^(p-1)) = O(h^2)``: two derivatives of the map."""
    values = [float(certificates[key].normalized_l2) for key in _RADIAL_LADDER]
    orders = [np.log2(a / b) for a, b in zip(values[:-1], values[1:], strict=True)]
    assert values == sorted(values, reverse=True), values
    assert values[0] / values[-1] > 60.0, values
    assert all(order > 1.8 for order in orders), orders
    assert all(order < 2.6 for order in orders), orders
    # The absolute channel converges with it, and is dimensional (N/m^3).
    absolute = [float(certificates[key].absolute_l2) for key in _RADIAL_LADDER]
    assert absolute[0] / absolute[-1] > 60.0, absolute
    # Unlike the vacuum fixture, whose normalized_linf is exactly 2, this
    # denominator is healthy: the pointwise metric is nowhere near saturation.
    peaks = [float(certificates[key].normalized_linf) for key in _RADIAL_LADDER]
    assert max(peaks) < 1.0e-3, peaks


def test_certificate_rewards_a_higher_order_radial_basis(certificates):
    """Same four spans, degree 5 instead of 3: three decades of certificate."""
    low = float(certificates[_RADIAL_LADDER[1]].normalized_l2)
    high = float(certificates[_FINEST].normalized_l2)
    assert low / high > 1.0e3, (low, high)


def test_certificate_converges_geometrically_under_fourier_refinement(certificates):
    """Poloidal truncation decays like ``tau^m``, the lambda-series rate."""
    values = [float(certificates[key].normalized_l2) for key in _FOURIER_LADDER]
    ratios = [a / b for a, b in zip(values[:-1], values[1:], strict=True)]
    assert values == sorted(values, reverse=True), values
    # Four extra poloidal modes cost tau^4 = 0.127^4 ~ 1/3800 each step, and
    # equal ratios across the ladder are what makes the decay geometric.
    assert all(1.0e3 < ratio < 1.0e4 for ratio in ratios), ratios
    assert 0.5 < ratios[0] / ratios[1] < 2.0, ratios


def test_certificate_floor_on_an_exact_equilibrium(certificates):
    """Anchor the absolute scale: what the oracle reports on a true solution."""
    report = certificates[_FINEST]
    normalized = float(report.normalized_l2)
    absolute = float(report.absolute_l2)
    assert normalized < 1.0e-8, normalized
    # Round-off of the nested jacfwd, not a modelling error: ~1e-10 relative
    # to |J x B| ~ 1e5 N/m^3.  Six orders below the ~1e-3 polish numbers.
    assert absolute < 1.0e-3, absolute
    assert float(report.boundary_residual) < 1.0e-12
    assert float(report.gauge_residual) == 0.0
    assert float(report.minimum_signed_jacobian) > 0.0
    assert float(report.normalized_linf) < 1.0e-5


def test_series_helpers_reproduce_the_closed_forms():
    """The two hand-derived series are checked against direct DFTs."""
    eq = _Solovev(mmax=16)
    ngrid = 4096
    w = 2.0 * np.pi * np.arange(ngrid) / ngrid
    for s in (0.0, 0.3, 1.0):
        x = eq.a * np.sqrt(s)
        spectrum = np.fft.rfft(np.sqrt(1.0 + x * np.cos(w))) / ngrid
        reference = np.concatenate([[spectrum[0].real], 2.0 * spectrum[1:17].real])
        series = np.array(
            [np.sqrt(s) ** m * eq.q_radius(m, np.asarray(s)) / eq.R0
             for m in range(17)]
        )
        np.testing.assert_allclose(series, reference, rtol=0.0, atol=1e-15)

    for s in (0.04, 1.0):
        x = eq.a * np.sqrt(s)
        t = np.sqrt((1.0 - x) / (1.0 + x))
        lam = 2.0 * np.arctan2(t * np.sin(w / 2.0), np.cos(w / 2.0)) - w
        spectrum = np.fft.rfft(lam) / ngrid
        reference = -2.0 * spectrum[1:17].imag
        series = np.array(
            [np.sqrt(s) ** m * eq.q_lambda(m, np.asarray(s)) for m in range(1, 17)]
        )
        np.testing.assert_allclose(series, reference, rtol=0.0, atol=1e-15)
