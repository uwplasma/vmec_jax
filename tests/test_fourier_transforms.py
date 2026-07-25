"""Property tests for ``vmex.core.{fourier,transforms}``.

Covers the transform algebra that the (deleted) legacy A/B suite proved
during the port:

- ``real_to_fourier(fourier_to_real(x)) == x`` on band-limited data,
- ``tomnsps`` is the exact inverse of the ``totzsps``-style synthesis
  (fixaray.f dnorm normalization, including the lasym lane),
- ``symforce_split`` recovers pure reflection parities exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from vmex.core.fourier import Resolution, mode_table, trig_tables
from vmex.core.nyquist import (
    symoutput_split,
    wrout_cos_coeffs,
    wrout_sin_coeffs,
)
from vmex.core.transforms import (
    _fourier_to_real_fft,
    fourier_to_real,
    real_to_fourier,
    symforce_split,
    tomnsps,
)

# (mpol, ntor, ntheta, nzeta, nfp, lasym)
CASES = [
    (6, 0, 18, 1, 1, False),
    (6, 3, 22, 16, 3, False),
    (9, 6, 28, 24, 4, False),
    (5, 2, 20, 18, 2, True),
]
CASE_IDS = ["axisym", "nfp3", "nfp4", "nfp2-lasym"]
NS = 7

RTOL = 1e-12


def _resolution(case) -> Resolution:
    mpol, ntor, ntheta, nzeta, nfp, lasym = case
    return Resolution(
        mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nzeta, nfp=nfp, lasym=lasym, ns=NS
    )


def _seed(case) -> int:
    return abs(hash(case)) % (2**31)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_synthesis_analysis_roundtrip(case):
    """real_to_fourier(fourier_to_real(x)) == x on band-limited data.

    On the reduced symmetric theta grid the cos and sin families are each
    internally orthogonal (but not mutually), so the round trip is exact per
    parity block; on the full lasym grid it is exact for both blocks jointly.
    """
    mpol, ntor, *_ = case
    lasym = case[5]
    res = _resolution(case)
    trig = trig_tables(res)
    modes = mode_table(mpol, ntor)

    rng = np.random.default_rng(_seed(case) + 3)
    coeff_cos = rng.standard_normal((NS, modes.mnmax))
    coeff_sin = rng.standard_normal((NS, modes.mnmax))
    # sin(m*theta - n*zeta) is identically zero for (m, n) = (0, 0).
    coeff_sin[:, (modes.m == 0) & (modes.n == 0)] = 0.0

    if lasym:
        (field,) = fourier_to_real(
            coeff_cos, coeff_sin, modes=modes, trig=trig, derivatives=("value",)
        )
        back_cos, back_sin = real_to_fourier(field, modes=modes, trig=trig, parity="both")
        np.testing.assert_allclose(np.asarray(back_cos), coeff_cos, rtol=RTOL, atol=1e-12)
        np.testing.assert_allclose(np.asarray(back_sin), coeff_sin, rtol=RTOL, atol=1e-12)
    else:
        zero = np.zeros_like(coeff_cos)
        (field_cos,) = fourier_to_real(
            coeff_cos, zero, modes=modes, trig=trig, derivatives=("value",)
        )
        back_cos, back_sin = real_to_fourier(field_cos, modes=modes, trig=trig, parity="cos")
        np.testing.assert_allclose(np.asarray(back_cos), coeff_cos, rtol=RTOL, atol=1e-12)
        assert not np.any(np.asarray(back_sin))

        (field_sin,) = fourier_to_real(
            zero, coeff_sin, modes=modes, trig=trig, derivatives=("value",)
        )
        back_cos, back_sin = real_to_fourier(field_sin, modes=modes, trig=trig, parity="sin")
        np.testing.assert_allclose(np.asarray(back_sin), coeff_sin, rtol=RTOL, atol=1e-12)
        assert not np.any(np.asarray(back_cos))


def test_lasym_nestor_surface_analysis_preserves_mode_signs():
    """The shared LASYM ``*_sur`` WOUT transform preserves cos/sin signs."""
    resolution = Resolution(
        mpol=3,
        ntor=2,
        ntheta=12,
        nzeta=10,
        nfp=2,
        lasym=True,
        ns=1,
    )
    trig = trig_tables(resolution)
    modes = mode_table(resolution.mpol, resolution.ntor)
    cos_coefficients = np.zeros((1, modes.mnmax))
    sin_coefficients = np.zeros_like(cos_coefficients)
    mode_index = np.flatnonzero((modes.m == 1) & (modes.n == 1))[0]
    cos_coefficients[0, mode_index] = 1.25
    sin_coefficients[0, mode_index] = -0.75
    (surface,) = fourier_to_real(
        cos_coefficients,
        sin_coefficients,
        modes=modes,
        trig=trig,
        derivatives=("value",),
    )

    symmetric, asymmetric = symoutput_split(
        f=np.asarray(surface), trig=trig
    )
    actual_cos = wrout_cos_coeffs(
        f=symmetric, modes=modes, trig=trig
    )
    actual_sin = wrout_sin_coeffs(
        f=asymmetric, modes=modes, trig=trig
    )
    roundtrip_atol = 8.0 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        actual_cos, cos_coefficients, rtol=0.0, atol=roundtrip_atol
    )
    np.testing.assert_allclose(
        actual_sin, sin_coefficients, rtol=0.0, atol=roundtrip_atol
    )


def test_synthesis_accepts_nyquist_extended_trig_tables():
    """WOUT evaluates main-mode coefficients on Nyquist-extended grids."""
    base = Resolution(
        mpol=6, ntor=3, ntheta=22, nzeta=16, nfp=3, lasym=False, ns=NS
    )
    extended = Resolution(
        mpol=9, ntor=8, ntheta=22, nzeta=16, nfp=3, lasym=False, ns=NS
    )
    modes = mode_table(base.mpol, base.ntor)
    rng = np.random.default_rng(4)
    coefficient_cos = rng.standard_normal((NS, modes.mnmax))
    coefficient_sin = rng.standard_normal((NS, modes.mnmax))
    kwargs = dict(modes=modes, derivatives=("value", "dtheta", "dzeta"))
    actual = _fourier_to_real_fft(
        coefficient_cos, coefficient_sin, trig=trig_tables(extended), **kwargs
    )
    expected = fourier_to_real(
        coefficient_cos, coefficient_sin, trig=trig_tables(base), **kwargs
    )
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=RTOL, atol=1e-12)


def test_fft_and_dense_synthesis_match_through_ad():
    """The fast primal and lower-memory implicit path have matching AD."""
    resolution = Resolution(
        mpol=4, ntor=2, ntheta=16, nzeta=12, nfp=3, lasym=True, ns=NS
    )
    modes = mode_table(resolution.mpol, resolution.ntor)
    trig = trig_tables(resolution)
    rng = np.random.default_rng(5)
    shape = (NS, modes.mnmax)
    primals = tuple(jnp.asarray(rng.standard_normal(shape)) for _ in range(2))
    tangents = tuple(jnp.asarray(rng.standard_normal(shape)) for _ in range(2))

    def synthesize(c, s, *, use_fft):
        transform = _fourier_to_real_fft if use_fft else fourier_to_real
        return transform(c, s, modes=modes, trig=trig)

    fast = lambda c, s: synthesize(c, s, use_fft=True)  # noqa: E731
    dense = lambda c, s: synthesize(c, s, use_fft=False)  # noqa: E731
    values, tangent = jax.jvp(fast, primals, tangents)
    dense_values, dense_tangent = jax.jvp(dense, primals, tangents)
    for left, right in zip(values + tangent, dense_values + dense_tangent):
        np.testing.assert_allclose(left, right, rtol=2e-13, atol=2e-12)
    cotangent = tuple(jnp.asarray(rng.standard_normal(x.shape)) for x in values)
    _, pullback = jax.vjp(fast, *primals)
    adjoint = pullback(cotangent)
    lhs = sum(jnp.vdot(x, y) for x, y in zip(tangent, cotangent))
    rhs = sum(jnp.vdot(x, y) for x, y in zip(tangents, adjoint))
    np.testing.assert_allclose(lhs, rhs, rtol=2e-13, atol=2e-12)


# ---------------------------------------------------------------------------
# tomnsp( totzsp(x) ) round trip
# ---------------------------------------------------------------------------


def _rz_mask(ns: int, mpol: int) -> np.ndarray:
    """Expected R/Z radial mask (jmin2 start indices, edge excluded)."""
    js = np.arange(ns) + 1
    m = np.arange(mpol)
    jmin2 = np.where(m == 0, 1, 2)[None, :]
    mask = (js[:, None] >= jmin2) & (js[:, None] <= ns - 1)
    return mask.astype(float)[:, :, None]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_tomnsps_recovers_band_limited_coefficients(case):
    """tomnsps is the exact inverse of the totzsps synthesis on band-limited data.

    Feed ``tomnsps`` a pure geometry-style field through the undifferentiated
    kernels (``armn``/``azmn``) built from internal coefficients ``x`` with the
    scaled trig tables.  The mscale/nscale normalization makes the projection
    recover ``x`` exactly (factor 1) in both symmetry modes: for
    ``lasym=True`` the reduced-interval integration carries the fixaray.f
    weight ``dnorm = 1/(nzeta*(ntheta2-1)) = 2/(nzeta*ntheta1)`` with
    endpoint half-weights, which equals the full-grid average for the
    reflection-symmetric basis products fed here.  (Before the core lasym
    dnorm fix this recovered ``x/2`` — the inherited legacy defect that
    halved every lasym force projection.)
    """
    mpol, ntor, _, _, _, lasym = case
    res = _resolution(case)
    trig = trig_tables(res)

    rng = np.random.default_rng(_seed(case) + 6)
    m_grid = np.arange(mpol)[:, None]
    n_grid = np.arange(ntor + 1)[None, :]

    coeff_R_cc = rng.standard_normal((NS, mpol, ntor + 1))
    coeff_R_ss = rng.standard_normal((NS, mpol, ntor + 1))
    coeff_R_ss *= (m_grid > 0) & (n_grid > 0)  # sin*sin basis vanishes otherwise
    coeff_Z_sc = rng.standard_normal((NS, mpol, ntor + 1))
    coeff_Z_sc *= m_grid > 0
    coeff_Z_cs = rng.standard_normal((NS, mpol, ntor + 1))
    coeff_Z_cs *= n_grid > 0

    cosmu, sinmu = np.asarray(trig.cosmu), np.asarray(trig.sinmu)
    cosnv, sinnv = np.asarray(trig.cosnv), np.asarray(trig.sinnv)
    field_R = np.einsum("smn,im,kn->sik", coeff_R_cc, cosmu, cosnv) + np.einsum(
        "smn,im,kn->sik", coeff_R_ss, sinmu, sinnv
    )
    field_Z = np.einsum("smn,im,kn->sik", coeff_Z_sc, sinmu, cosnv) + np.einsum(
        "smn,im,kn->sik", coeff_Z_cs, cosmu, sinnv
    )

    zeros = np.zeros_like(field_R)
    out = tomnsps(
        force_R_even=field_R,
        force_R_odd=field_R,
        force_R_du_even=zeros,
        force_R_du_odd=zeros,
        force_R_dv_even=zeros,
        force_R_dv_odd=zeros,
        force_Z_even=field_Z,
        force_Z_odd=field_Z,
        force_Z_du_even=zeros,
        force_Z_du_odd=zeros,
        force_Z_dv_even=zeros,
        force_Z_dv_odd=zeros,
        mpol=mpol,
        ntor=ntor,
        trig=trig,
        include_edge=False,
    )

    # Known normalization: exact recovery in both symmetry modes (fixaray.f
    # dnorm; see the docstring).
    factor = 1.0
    mask = _rz_mask(NS, mpol)

    np.testing.assert_allclose(
        np.asarray(out.force_R_cc), factor * coeff_R_cc * mask, rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(out.force_Z_sc), factor * coeff_Z_sc * mask, rtol=RTOL, atol=1e-12
    )
    if ntor > 0:
        np.testing.assert_allclose(
            np.asarray(out.force_R_ss), factor * coeff_R_ss * mask, rtol=RTOL, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(out.force_Z_cs), factor * coeff_Z_cs * mask, rtol=RTOL, atol=1e-12
        )
    else:
        assert out.force_R_ss is None
        assert out.force_Z_cs is None


# ---------------------------------------------------------------------------
# symforce split (symforce.f)
# ---------------------------------------------------------------------------


def test_symforce_split_recovers_pure_parities():
    """A reflection-even field has no antisymmetric part (and vice versa)."""
    case = CASES[3]  # the lasym case: full theta grid
    mpol, ntor, *_ = case
    res = _resolution(case)
    trig = trig_tables(res)
    rng = np.random.default_rng(_seed(case) + 7)

    cosmu, sinmu = np.asarray(trig.cosmu), np.asarray(trig.sinmu)
    cosnv = np.asarray(trig.cosnv)
    coeff = rng.standard_normal((NS, mpol, ntor + 1))

    # Even under (theta, zeta) -> (-theta, -zeta): cos*cos (and sin*sin).
    field_even = np.einsum("smn,im,kn->sik", coeff, cosmu, cosnv)
    sym, asym = symforce_split(field_even, trig=trig, reflect_even=True)
    n_theta2 = trig.ntheta2
    np.testing.assert_allclose(
        np.asarray(sym)[:, :n_theta2], field_even[:, :n_theta2], rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(asym), 0.0, atol=1e-12)

    # Odd under the reflection: sin*cos.
    field_odd = np.einsum("smn,im,kn->sik", coeff, sinmu, cosnv)
    sym, asym = symforce_split(field_odd, trig=trig, reflect_even=False)
    np.testing.assert_allclose(
        np.asarray(sym)[:, :n_theta2], field_odd[:, :n_theta2], rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(asym), 0.0, atol=1e-12)
