"""Differentiable ideal-MHD stability objectives (R26h.h1).

Infinite-n ideal-ballooning growth rate as a pure, traceable function of a
converged ``(SpectralState, SolverRuntime)`` pair — the JAX analogue of the
COBRA solve (Sanchez, Hirshman, Ware, Berry & Batchelor, J. Comput. Phys.
161, 589 (2000)) in the modern differentiable formulation of Gaur et al.,
J. Plasma Phys. 89, 905890518 (2023) (arXiv:2302.07673):

    d/dη ( g dX/dη ) + c X = λ f X ,   X(±η_b) = 0 ,   g, f > 0,

a self-adjoint second-order ODE eigenproblem along a field line, with η the
straight-field-line (PEST) poloidal angle.  ``λ > 0`` is unstable; ``λ`` is
the squared growth rate normalized to the Alfvén frequency,
``λ = (γ a_N / v_A)²`` with ``a_N`` the effective minor radius and ``v_A``
the Alfvén speed at the reference field ``B_N = 2|ψ_edge|/a_N²``.

The coefficient arrays follow the (COBRA-validated) VMEC-coordinate geometry
conventions of simsopt's ``vmec_fieldlines`` (Landreman) and of the adjoint
ballooning solver of Gaur et al.:

- ``g = |b·∇η| |∇α|² a_N³ B_N / B²`` (field-line bending),
- ``c = -2 μ0 (dp/dρ) (b×κ·∇α)``-type pressure/curvature drive, assembled
  from ``B×∇|B|·∇α`` plus the ``μ0 dp/dψ`` force-balance correction,
- ``f = |∇α|² a_N B_N³ / (B² |B·∇η| B)`` (inertia),

with ``α = θ* - ι (φ - ζ0)`` the field-line label, ``∇α`` carrying the
secular magnetic-shear term ``-ι'(s) (φ - ζ0) ∇s`` (ballooning parameter
``ζ0``).  All geometry is evaluated spectrally from the converged state's
physical Fourier coefficients: values and angular derivatives are exact
trig sums; radial dependence uses a local parabola through the three
neighbouring full-mesh surfaces, and every derivative (including the second
derivatives inside ``∇|B|``) comes from JAX automatic differentiation of the
point-evaluation functions, so the whole pipeline is jit/grad-transparent.

The discretized problem is the symmetric tridiagonal generalized
eigenproblem of the central-difference stencil (the same discretization as
COBRA/DESC), reduced to standard form with the diagonal ``f^{-1/2}``
similarity transform and solved with a batched ``jnp.linalg.eigvalsh`` over
all requested (surface, α, ζ0) field lines.

Scope notes
-----------
- The current profile supports symmetric and ``lasym`` states. Mercier,
  Glasser and ballooning stability remain stellarator-symmetric.
- Surfaces need ``ι ≠ 0`` (the field-line parameterization divides by ι).
- :func:`d_merc_state` is the traceable counterpart of the parity-proven
  wout calculation.  As in VMEC2000, its first two surfaces and edge are not
  suitable stability targets; :func:`mercier_stability_residual` selects the
  validated interior and supplies a smooth stability-violation objective.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from .solver import SolverRuntime, SpectralState, _physical_coefficients
from .statephysics import _field_chain, _iotas_half_from_fields
from .transforms import physical_to_internal_scale

__all__ = [
    "d_merc_state",
    "mercier_stability_residual",
    "jdotb_state",
    "jdotb_residual",
    "mercier_shear_state",
    "glasser_d_r_state",
    "glasser_stability_residual",
    "ballooning_lambda",
    "ballooning_growth_rate",
]

Array = Any

_NEWTON_ITERATIONS = 12  # θ_vmec(θ*) root solve; λ is small, converges fast
_MU0 = 4.0e-7 * np.pi


# ---------------------------------------------------------------------------
# Mercier profile (mercier.f / jxbforce.f)
# ---------------------------------------------------------------------------


def _mercier_bsubs(geometry, jacobian, fields, s: Array) -> Array:
    """Traceable ``bss.f`` covariant radial field on the half mesh."""
    s = jnp.asarray(s)
    sh = jnp.sqrt(
        jnp.maximum(
            jnp.concatenate([0.5 * (s[1:2] + s[:1]), 0.5 * (s[1:] + s[:-1])]),
            0.0,
        )
    )[:, None, None]
    safe_sh = jnp.where(sh != 0.0, sh, 1.0)
    rv12 = 0.5 * (
        geometry.dR_dzeta_even[1:]
        + geometry.dR_dzeta_even[:-1]
        + sh[1:] * (geometry.dR_dzeta_odd[1:] + geometry.dR_dzeta_odd[:-1])
    )
    zv12 = 0.5 * (
        geometry.dZ_dzeta_even[1:]
        + geometry.dZ_dzeta_even[:-1]
        + sh[1:] * (geometry.dZ_dzeta_odd[1:] + geometry.dZ_dzeta_odd[:-1])
    )
    rs12 = jacobian.dR_ds[1:] + 0.25 * (geometry.R_odd[1:] + geometry.R_odd[:-1]) / safe_sh[1:]
    zs12 = jacobian.dZ_ds[1:] + 0.25 * (geometry.Z_odd[1:] + geometry.Z_odd[:-1]) / safe_sh[1:]
    rs12 = jnp.concatenate([rs12[:1], rs12])
    zs12 = jnp.concatenate([zs12[:1], zs12])
    rv12 = jnp.concatenate([rv12[:1], rv12])
    zv12 = jnp.concatenate([zv12[:1], zv12])
    gsu = rs12 * jacobian.ru12 + zs12 * jacobian.zu12
    gsv = rs12 * rv12 + zs12 * zv12
    return fields.bsupu * gsu + fields.bsupv * gsv


def _mercier_current_tables(bsubu: Array, bsubv: Array, bsubs: Array, rt: SolverRuntime) -> tuple[Array, Array, Array]:
    """Traceable jxbforce filter and ``B_s`` derivatives."""
    trig = rt.trig
    if bool(rt.setup.lasym):
        mmax, nmax = int(rt.resolution.mpol) - 1, int(rt.resolution.ntor)
        nt1, nt2, nt3 = int(trig.ntheta1), int(trig.ntheta2), int(trig.ntheta3)
        nzeta = int(np.asarray(trig.cosnv).shape[0])
        cosmu = jnp.asarray(trig.cosmu[:nt2, : mmax + 1])
        sinmu = jnp.asarray(trig.sinmu[:nt2, : mmax + 1])
        cosnv = jnp.asarray(trig.cosnv[:, : nmax + 1])
        sinnv = jnp.asarray(trig.sinnv[:, : nmax + 1])
        dnorm = 1.0 / (nzeta * nt3)
        cosmui = (dnorm * cosmu).at[0].multiply(0.5).at[-1].multiply(0.5)
        sinmui = dnorm * sinmu
        dmult = jnp.ones(
            (mmax + 1, nmax + 1), dtype=jnp.asarray(bsubu).dtype)
        mnyq, nnyq = nt2 - 1, nzeta // 2
        if 0 < mnyq <= mmax:
            dmult = dmult.at[mnyq].multiply(0.5)
        if 0 < nnyq <= nmax:
            dmult = dmult.at[:, nnyq].multiply(0.5)
        reflected_theta = np.where(
            np.arange(nt2) == 0, 0, nt1 - np.arange(nt2))
        reflected_zeta = (-np.arange(nzeta)) % nzeta
        extension_theta = nt1 - np.arange(nt2, nt3)

        def split(field, *, reversed_sym=False):
            half = field[:, :nt2]
            reflected = field[:, reflected_theta][:, :, reflected_zeta]
            sign = -1.0 if reversed_sym else 1.0
            return (
                0.5 * (half + sign * reflected),
                0.5 * (half - sign * reflected),
            )

        def analyze_lasym(field, theta, zeta):
            return jnp.einsum(
                "smk,kn->smn",
                jnp.einsum("sik,im->smk", field, theta),
                zeta,
            ) * dmult

        def synth(first, theta1, zeta1, second, theta2, zeta2):
            return (
                jnp.einsum("smn,im,kn->sik", first, theta1, zeta1)
                + jnp.einsum("smn,im,kn->sik", second, theta2, zeta2)
            )

        def extend(symmetric, asymmetric):
            full = jnp.zeros(
                (symmetric.shape[0], nt3, nzeta), dtype=symmetric.dtype)
            full = full.at[:, :nt2].set(symmetric + asymmetric)
            return full.at[:, nt2:].set(
                symmetric[:, extension_theta][:, :, reflected_zeta]
                - asymmetric[:, extension_theta][:, :, reflected_zeta]
            )

        def filter_lasym(field):
            symmetric, asymmetric = split(field)
            symmetric = synth(
                analyze_lasym(symmetric, cosmui, cosnv), cosmu, cosnv,
                analyze_lasym(symmetric, sinmui, sinnv), sinmu, sinnv)
            asymmetric = synth(
                analyze_lasym(asymmetric, sinmui, cosnv), sinmu, cosnv,
                analyze_lasym(asymmetric, cosmui, sinnv), cosmu, sinnv)
            return extend(symmetric, asymmetric)

        bsubu, bsubv = filter_lasym(bsubu), filter_lasym(bsubv)
        bsubu, bsubv = filter_lasym(bsubu), filter_lasym(bsubv)
        bsubs_full = bsubs.at[1:-1].set(
            0.5 * (bsubs[1:-1] + bsubs[2:])).at[0].set(0.0)
        symmetric, asymmetric = split(bsubs_full, reversed_sym=True)
        symmetric_sin = (
            analyze_lasym(symmetric, sinmui, cosnv),
            analyze_lasym(symmetric, cosmui, sinnv),
        )
        asymmetric_cos = (
            analyze_lasym(asymmetric, cosmui, cosnv),
            analyze_lasym(asymmetric, sinmui, sinnv),
        )
        bsubsu = extend(
            synth(
                symmetric_sin[0], jnp.asarray(trig.cosmum[:nt2, :mmax + 1]), cosnv,
                symmetric_sin[1], jnp.asarray(trig.sinmum[:nt2, :mmax + 1]), sinnv),
            synth(
                asymmetric_cos[0], jnp.asarray(trig.sinmum[:nt2, :mmax + 1]), cosnv,
                asymmetric_cos[1], jnp.asarray(trig.cosmum[:nt2, :mmax + 1]), sinnv),
        )
        bsubsv = extend(
            synth(
                symmetric_sin[0], sinmu, jnp.asarray(trig.sinnvn[:, :nmax + 1]),
                symmetric_sin[1], cosmu, jnp.asarray(trig.cosnvn[:, :nmax + 1])),
            synth(
                asymmetric_cos[0], cosmu, jnp.asarray(trig.sinnvn[:, :nmax + 1]),
                asymmetric_cos[1], sinmu, jnp.asarray(trig.cosnvn[:, :nmax + 1])),
        )
        return bsubu, bsubv, (bsubsu, bsubsv)

    mmax, nmax = int(rt.resolution.mpol) - 1, int(rt.resolution.ntor)
    nt2 = int(trig.ntheta2)
    cosmu = jnp.asarray(trig.cosmu[:nt2, : mmax + 1])
    sinmu = jnp.asarray(trig.sinmu[:nt2, : mmax + 1])
    cosmui = jnp.asarray(trig.cosmui[:nt2, : mmax + 1])
    sinmui = jnp.asarray(trig.sinmui[:nt2, : mmax + 1])
    cosnv = jnp.asarray(trig.cosnv[:, : nmax + 1])
    sinnv = jnp.asarray(trig.sinnv[:, : nmax + 1])
    dmult = jnp.ones((mmax + 1, nmax + 1), dtype=jnp.asarray(bsubu).dtype)
    mnyq, nnyq = nt2 - 1, int(np.asarray(trig.cosnv).shape[0]) // 2
    if 0 < mnyq <= mmax:
        dmult = dmult.at[mnyq].multiply(0.5)
    if 0 < nnyq <= nmax:
        dmult = dmult.at[:, nnyq].multiply(0.5)

    def analyze(f, theta, zeta):
        return jnp.einsum("smk,kn->smn", jnp.einsum("sik,im->smk", f[:, :nt2], theta), zeta) * dmult

    def filter_field(f):
        c1 = jnp.einsum("smk,kn->smn", jnp.einsum("sik,im->smk", f[:, :nt2], cosmui), cosnv) * dmult
        c2 = jnp.einsum("smk,kn->smn", jnp.einsum("sik,im->smk", f[:, :nt2], sinmui), sinnv) * dmult
        return jnp.einsum("smn,im,kn->sik", c1, cosmu, cosnv) + jnp.einsum("smn,im,kn->sik", c2, sinmu, sinnv)

    bsubu, bsubv = filter_field(bsubu), filter_field(bsubv)
    bsubs_full = bsubs.at[1:-1].set(0.5 * (bsubs[1:-1] + bsubs[2:]))
    bsubs_full = bsubs_full.at[0].set(0.0)
    c1 = analyze(bsubs_full[:, :nt2], sinmui, cosnv)
    c2 = analyze(bsubs_full[:, :nt2], cosmui, sinnv)
    bsubsu = jnp.einsum("smn,im,kn->sik", c1, jnp.asarray(trig.cosmum[:nt2, : mmax + 1]), cosnv) + jnp.einsum(
        "smn,im,kn->sik", c2, jnp.asarray(trig.sinmum[:nt2, : mmax + 1]), sinnv
    )
    bsubsv = jnp.einsum("smn,im,kn->sik", c1, sinmu, jnp.asarray(trig.sinnvn[:, : nmax + 1])) + jnp.einsum(
        "smn,im,kn->sik", c2, cosmu, jnp.asarray(trig.cosnvn[:, : nmax + 1])
    )
    return bsubu, bsubv, (bsubsu, bsubsv)


def _mercier_profiles_state(
    state: SpectralState,
    rt: SolverRuntime,
) -> tuple[Array, Array, Array, Array, Array]:
    """Traceable ``(DMerc, <J.B>, <B.B>, shear, H)`` radial profiles.

    The shared pure-JAX reconstruction follows VMEC2000 ``jxbforce.f`` and
    ``mercier.f``.  The returned Glasser ``H`` uses the normalization of
    Landreman & Jorge (2020), Eqs. (51) and (53).
    """
    setup = rt.setup
    s = jnp.asarray(setup.s_full)
    ns = int(s.shape[0])
    if ns < 3:
        zero = jnp.zeros_like(s)
        return zero, zero, zero, zero, zero
    geometry, jacobian, _, fields, energies = _field_chain(state, rt)
    bsubs = _mercier_bsubs(geometry, jacobian, fields, s)
    bsubu, bsubv, (bsubsu, bsubsv) = _mercier_current_tables(fields.bsubu, fields.bsubv, bsubs, rt)

    hs = 1.0 / float(ns - 1)
    sign_jac = float(np.sign(setup.signgs)) if int(setup.signgs) != 0 else 1.0
    wint = jnp.asarray(rt.trig.wint)
    phip_real = (2.0 * jnp.pi) * jnp.asarray(setup.phips) * sign_jac
    safe_phip = jnp.where(phip_real != 0.0, phip_real, 1.0)
    vp_real = sign_jac * (2.0 * jnp.pi) ** 2 * jnp.asarray(energies.vp) / safe_phip
    vp_real = vp_real.at[0].set(0.0)
    iotas = _iotas_half_from_fields(setup, fields)

    itheta = jnp.zeros_like(bsubs).at[1:-1].set(bsubsv[1:-1] - (bsubv[2:] - bsubv[1:-1]) / hs)
    izeta = jnp.zeros_like(bsubs).at[1:-1].set(-bsubsu[1:-1] + (bsubu[2:] - bsubu[1:-1]) / hs)
    izeta = izeta.at[0].set(2.0 * izeta[1] - izeta[2])
    izeta = izeta.at[-1].set(2.0 * izeta[-2] - izeta[-3])
    bdotk = (
        jnp.zeros_like(bsubs)
        .at[1:-1]
        .set(itheta[1:-1] * 0.5 * (bsubu[2:] + bsubu[1:-1]) + izeta[1:-1] * 0.5 * (bsubv[2:] + bsubv[1:-1]))
    )

    torcur = jnp.zeros_like(s).at[1:].set(sign_jac * (2.0 * jnp.pi) * jnp.einsum("sij,ij->s", bsubu[1:], wint))
    phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
    denom = 1.0 / (hs * phip_full)
    shear = (iotas[2:] - iotas[1:-1]) * denom
    vpp = (vp_real[2:] - vp_real[1:-1]) * denom
    pres = jnp.asarray(fields.pressure)
    presp = (pres[2:] - pres[1:-1]) * denom
    ip = (torcur[2:] - torcur[1:-1]) * denom

    sqs = jnp.sqrt(s[1:-1])[:, None, None]
    r1f = geometry.R_even[1:-1] + sqs * geometry.R_odd[1:-1]
    rtf = geometry.dR_dtheta_even[1:-1] + sqs * geometry.dR_dtheta_odd[1:-1]
    ztf = geometry.dZ_dtheta_even[1:-1] + sqs * geometry.dZ_dtheta_odd[1:-1]
    rzf = geometry.dR_dzeta_even[1:-1] + sqs * geometry.dR_dzeta_odd[1:-1]
    zzf = geometry.dZ_dzeta_even[1:-1] + sqs * geometry.dZ_dzeta_odd[1:-1]
    gsqrt_raw = 0.5 * (jacobian.sqrt_g[1:-1] + jacobian.sqrt_g[2:])
    gsqrt_full = gsqrt_raw / phip_full[:, None, None]
    gtt = rtf * rtf + ztf * ztf
    gpp = gsqrt_full**2 / (gtt * r1f**2 + (rtf * zzf - rzf * ztf) ** 2)
    b2 = 2.0 * (jnp.asarray(fields.total_pressure) - pres[:, None, None])
    b2i = 0.5 * (b2[1:-1] + b2[2:])
    factor = (2.0 * jnp.pi) ** 2
    tpp = jnp.einsum("sij,ij->s", gsqrt_full / b2i, wint) * factor
    tbb = jnp.einsum("sij,ij->s", b2i * gsqrt_full * gpp, wint) * factor
    # ``itheta/izeta`` above omit jxbforce.f's 1/mu0 conversion; mercier.f
    # multiplies their resulting J.B back by mu0, so the factors cancel.
    bdotj_norm = jnp.where(gsqrt_raw != 0.0, bdotk[1:-1] / gsqrt_raw, 0.0)
    jdotb = bdotj_norm * gpp * gsqrt_full
    tjb = jnp.einsum("sij,ij->s", jdotb, wint) * factor
    tjj = jnp.einsum("sij,ij->s", jdotb * bdotj_norm / b2i, wint) * factor
    dmerc = 0.25 * shear**2 - shear * (tjb - ip * tbb) + presp * (vpp - presp * tpp) * tbb + tjb**2 - tbb * tjj
    dmerc_full = jnp.zeros_like(s).at[1:-1].set(dmerc)

    # jxbforce.f flux-surface <J.B>/<B.B> averages.  ``bdotk`` omits the
    # 1/mu0 conversion above, so ``jdotb_mu0`` is mu0 times the WOUT profile.
    vp_sum = jnp.asarray(energies.vp)[2:] + jnp.asarray(energies.vp)[1:-1]
    average_norm = jnp.where(vp_sum != 0.0, 2.0 * sign_jac / vp_sum, 0.0)
    jdotb_mu0 = average_norm * jnp.einsum("sij,ij->s", bdotk[1:-1], wint)
    sqgb2 = (
        jacobian.sqrt_g[2:] * (jnp.asarray(fields.total_pressure)[2:] - pres[2:, None, None])
        + jacobian.sqrt_g[1:-1]
        * (jnp.asarray(fields.total_pressure)[1:-1] - pres[1:-1, None, None])
    )
    bdotb = average_norm * jnp.einsum("sij,ij->s", sqgb2, wint)
    jdotb_full = jnp.zeros_like(s).at[1:-1].set(jdotb_mu0 / _MU0)
    if bool(setup.lasym):
        # Match WOUT's four factor-two LASYM output normalizations; see
        # vmex.core.wout for the VMEC2000 convention and golden comparison.
        jdotb_full = 16.0 * jdotb_full
    jdotb_full = jdotb_full.at[0].set(2.0 * jdotb_full[1] - jdotb_full[2])
    jdotb_full = jdotb_full.at[-1].set(
        2.0 * jdotb_full[-2] - jdotb_full[-3]
    )
    bdotb_full = jnp.zeros_like(s).at[1:-1].set(bdotb)
    bdotb_full = bdotb_full.at[0].set(
        2.0 * bdotb_full[2] - bdotb_full[1]
    )
    bdotb_full = bdotb_full.at[-1].set(
        2.0 * bdotb_full[-2] - bdotb_full[-3]
    )

    # In the VMEC Mercier normalization H is shear times the difference
    # between the |grad psi|^-3 J.B integral and the same B^2 integral times
    # <mu0 J.B>/<B^2>.
    surface_ratio = jnp.where(bdotb != 0.0, jdotb_mu0 / bdotb, 0.0)
    h_glasser = shear * (tjb - tbb * surface_ratio)
    shear_full = jnp.zeros_like(s).at[1:-1].set(shear)
    h_full = jnp.zeros_like(s).at[1:-1].set(h_glasser)
    return dmerc_full, jdotb_full, bdotb_full, shear_full, h_full


def d_merc_state(state: SpectralState, rt: SolverRuntime) -> Array:
    """Traceable VMEC ``DMerc`` profile on the full radial mesh.

    Positive interior values indicate Mercier stability.  This is a pure-JAX
    port of the symmetric ``jxbforce.f``/``mercier.f`` path used by
    :func:`vmex.core.nyquist.mercier_and_jxb`; it accepts a live converged
    ``(state, runtime)`` pair and supports ``jit``, JVP and reverse-mode AD.
    The axis, first near-axis surface and edge retain VMEC's zero/noisy output
    convention and should be excluded from objectives (normally ``[2:-1]``).
    """
    if bool(rt.setup.lasym):
        raise NotImplementedError(
            "traceable Mercier profiles are not independently validated "
            "for lasym equilibria"
        )
    return _mercier_profiles_state(state, rt)[0]


def jdotb_state(state: SpectralState, rt: SolverRuntime) -> Array:
    """Traceable VMEC ``jdotb = <J.B>`` profile in WOUT units."""
    return _mercier_profiles_state(state, rt)[1]


def jdotb_residual(state: SpectralState, rt: SolverRuntime) -> Array:
    """Interior ``<J.B>`` profile for least-squares current objectives."""
    return jdotb_state(state, rt)[2:-1]


def mercier_shear_state(state: SpectralState, rt: SolverRuntime) -> Array:
    """Return ``S = d(iota)/d(Phi)`` in the VMEC Mercier normalization."""
    return _mercier_profiles_state(state, rt)[3]


def glasser_d_r_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    shear_epsilon: float = 0.0,
) -> Array:
    """Traceable Glasser--Greene--Johnson ``D_R`` profile.

    Non-positive values satisfy the necessary local resistive-interchange
    stability condition on nonzero-shear surfaces.  With the strict default,
    exact zero-shear entries are set to zero because the criterion is
    undefined there.  A positive ``shear_epsilon`` replaces the denominator
    ``shear**2`` by ``shear**2 + shear_epsilon**2`` for smooth optimization;
    this regularization does not make zero-shear surfaces physically valid.
    Post-check :func:`mercier_shear_state` and require every target surface to
    satisfy ``abs(S) >> shear_epsilon`` before interpreting the result.
    As for ``DMerc``, use only validated interior surfaces (normally
    ``[2:-1]``) as optimization targets.
    """
    if bool(rt.setup.lasym):
        raise NotImplementedError(
            "traceable Glasser profiles are not independently validated "
            "for lasym equilibria"
        )
    if shear_epsilon < 0.0:
        raise ValueError(
            f"shear_epsilon must be non-negative, got {shear_epsilon}"
        )
    dmerc, _, _, shear, h_glasser = _mercier_profiles_state(state, rt)
    epsilon = jnp.asarray(shear_epsilon, dtype=shear.dtype)
    denominator = shear**2 + epsilon**2
    correction = (h_glasser - 0.5 * shear**2) ** 2 / jnp.where(
        denominator != 0.0, denominator, 1.0
    )
    d_r = -dmerc + correction
    if shear_epsilon == 0.0:
        d_r = jnp.where(shear != 0.0, d_r, 0.0)
    return d_r


def mercier_stability_residual(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    margin: float = 0.0,
    smoothing: float = 1.0e-6,
) -> Array:
    """Smooth Mercier-instability residual on ``DMerc[2:-1]``.

    Positive ``DMerc`` is stable.  For each validated interior surface this
    returns ``smoothing * softplus((margin - DMerc) / smoothing)``: it tends
    to ``max(margin - DMerc, 0)`` as ``smoothing`` tends to zero, while
    retaining a smooth gradient at the stability boundary.  At finite
    smoothing it is strictly positive but exponentially close to zero on a
    sufficiently stable surface.  Use target zero in
    :func:`vmex.core.optimize.least_squares`; ``margin > 0`` requests a finite
    stability margin.  The first two surfaces and edge are excluded.
    """
    if smoothing <= 0.0:
        raise ValueError(f"smoothing must be positive, got {smoothing}")
    violation = jnp.asarray(margin) - d_merc_state(state, rt)[2:-1]
    scale = jnp.asarray(smoothing, dtype=violation.dtype)
    return scale * jax.nn.softplus(violation / scale)


def glasser_stability_residual(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    margin: float = 0.0,
    smoothing: float = 1.0e-6,
    shear_epsilon: float = 1.0e-8,
) -> Array:
    """Smooth resistive-interchange residual on ``D_R[2:-1]``.

    Subject to the prerequisite ``DMerc > 0``, stable surfaces require
    ``D_R <= 0``.  Combine this residual with
    :func:`mercier_stability_residual`; targeting it to zero penalizes
    ``D_R > -margin`` while retaining a smooth derivative.  The
    nonzero default shear regularization makes this optimization helper finite
    on zero-shear seeds; use :func:`glasser_d_r_state` with its strict default
    for reporting.
    """
    if smoothing <= 0.0:
        raise ValueError(f"smoothing must be positive, got {smoothing}")
    violation = glasser_d_r_state(
        state, rt, shear_epsilon=shear_epsilon
    )[2:-1] + jnp.asarray(margin)
    scale = jnp.asarray(smoothing, dtype=violation.dtype)
    return scale * jax.nn.softplus(violation / scale)


# ---------------------------------------------------------------------------
# Converged-state context: physical spectra, profiles, normalizations
# ---------------------------------------------------------------------------


def _ballooning_context(state: SpectralState, rt: SolverRuntime) -> dict:
    """Traceable per-state inputs of the ballooning solve.

    Physical (wout-normalized) coefficient tables ``rmnc/zmns/lmns`` on the
    full mesh (``lmns`` rescaled by ``lamscale/phipf`` exactly as the wout
    writer does), the half-mesh ``iota``/pressure profiles (the ``ncurr = 1``
    current-constrained iota comes from the solved ``chips``, as in
    ``add_fluxes.f90``), and the GX/GS2-style normalizations ``L_ref``
    (effective minor radius, ``aspectratio.f`` quadrature) and
    ``B_ref = 2|ψ_edge|/L_ref²``.
    """
    setup = rt.setup
    if bool(setup.lasym):
        raise NotImplementedError(
            "ballooning stability supports stellarator-symmetric states only "
            "(lasym = False)")
    s = jnp.asarray(setup.s_full)
    ns = int(s.shape[0])
    if ns < 5:
        raise ValueError(f"ballooning stability needs ns >= 5, got ns = {ns}")

    # Field state (pressure profile via the mass closure; the
    # current-constrained chips feed the ncurr = 1 iota) from the shared
    # geometry->fields chain (statephysics.py); the physical coefficient
    # tables come straight from the m=1-constraint inverse (cheap, spectral).
    R_cos, _R_sin, _Z_cos, Z_sin = _physical_coefficients(
        state, modes=rt.modes, lthreed=setup.lthreed, lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    geometry, _, _, fields, _ = _field_chain(state, rt)
    iotas = _iotas_half_from_fields(setup, fields)

    # Physical (wout) coefficient tables from the internal-normalized,
    # m=1-constraint-undone spectra (wrout.f conventions; lambda carries the
    # lamscale/phipf rescale of ``lambda_wout_from_full_mesh``).
    mode_scale = jnp.asarray(1.0 / physical_to_internal_scale(rt.modes, rt.trig))
    phipf = jnp.asarray(setup.phipf)
    safe_phipf = jnp.where(phipf != 0.0, phipf, 1.0)
    lam_factor = jnp.asarray(setup.lamscale) / safe_phipf
    rmnc = jnp.asarray(R_cos) * mode_scale[None, :]
    zmns = jnp.asarray(Z_sin) * mode_scale[None, :]
    lmns = jnp.asarray(state.L_sin) * mode_scale[None, :] * lam_factor[:, None]

    # Normalizations: L_ref = Aminor_p (aspectratio.f boundary quadrature,
    # same math as the wout ``Aminor_p``), B_ref = 2|psi_edge|/L_ref^2.
    sqrts_edge = jnp.asarray(setup.sqrts)[-1]
    rb = jnp.asarray(geometry.R_even)[-1] + sqrts_edge * jnp.asarray(geometry.R_odd)[-1]
    zub = (jnp.asarray(geometry.dZ_dtheta_even)[-1]
           + sqrts_edge * jnp.asarray(geometry.dZ_dtheta_odd)[-1])
    wint = jnp.asarray(rt.trig.wint)
    area = 2.0 * jnp.pi * jnp.abs(jnp.sum(rb * zub * wint))
    L_ref = jnp.sqrt(jnp.where(area != 0.0, area, 1.0) / jnp.pi)

    hs = s[1] - s[0]
    psi_edge = hs * jnp.sum(phipf[1:])          # internal psi = phi/(2 pi)
    B_ref = 2.0 * jnp.abs(psi_edge) / (L_ref * L_ref)
    sign_psi = jnp.sign(psi_edge)

    return dict(
        s=s, hs=hs, ns=ns,
        m=jnp.asarray(np.asarray(rt.modes.m, dtype=float)),
        xn=jnp.asarray(np.asarray(rt.modes.n, dtype=float) * float(rt.resolution.nfp)),
        rmnc=rmnc, zmns=zmns, lmns=lmns,
        iotas=iotas, pres=jnp.asarray(fields.pressure),
        phipf=phipf, psi_edge=psi_edge, sign_psi=sign_psi,
        L_ref=L_ref, B_ref=B_ref,
    )


def _parabola(table: Array, j: int, hs: Array) -> Array:
    """Local radial parabola of a full-mesh coefficient table at surface j.

    Returns ``(3, mnmax)`` stacked ``(value, d/ds, d²/ds²/2)`` at ``s[j]``
    from the three neighbouring surfaces — the same second-order radial
    accuracy as VMEC's own finite differences.
    """
    c0 = table[j]
    c1 = (table[j + 1] - table[j - 1]) / (2.0 * hs)
    c2 = (table[j + 1] - 2.0 * table[j] + table[j - 1]) / (2.0 * hs * hs)
    return jnp.stack([c0, c1, c2])


def _theta_vmec_from_pest(theta_star: Array, phi: Array, lmns0: Array,
                          m: Array, xn: Array) -> Array:
    """Invert ``θ* = θ + λ(θ, φ)`` for the VMEC poloidal angle (Newton).

    The straight-field-line (PEST) angle map of ``vmec_fieldlines``; a fixed
    unrolled Newton iteration (``1 + λ_θ > 0`` on nested surfaces) keeps the
    solve reverse-mode differentiable.
    """
    ml = m * lmns0
    theta = theta_star
    for _ in range(_NEWTON_ITERATIONS):
        ang = theta[..., None] * m - phi[..., None] * xn
        lam = jnp.sin(ang) @ lmns0
        dlam = jnp.cos(ang) @ ml
        theta = theta - (theta + lam - theta_star) / (1.0 + dlam)
    return theta


# ---------------------------------------------------------------------------
# Field-line point geometry (all derivatives via JAX AD)
# ---------------------------------------------------------------------------


def _make_point_fn(m: Array, xn: Array, rtab: Array, ztab: Array, ltab: Array,
                   iota: Array, diota: Array, phipf_j: Array):
    """Point-evaluation closure for one flux surface.

    Given ``q = (t, θ, φ)`` with ``t = s - s_j`` (evaluated at ``t = 0``;
    the radial parabola makes every quantity differentiable in ``t``),
    returns ``(|B|, B^φ, |∇α|², B×∇|B|·∇α)`` at the point.  The cylindrical
    position, its Jacobian (covariant basis ``e_s, e_θ, e_φ``), the dual
    basis, ``B = ψ'[(ι - λ_φ) e_θ + (1 + λ_θ) e_φ]/√g`` and ``∇|B|`` are all
    obtained by automatic differentiation of the spectral sums.
    """

    def coeffs(tab: Array, t: Array) -> Array:
        return tab[0] + t * tab[1] + (t * t) * tab[2]

    def lam_fn(q: Array) -> Array:
        t, th, ph = q[0], q[1], q[2]
        return coeffs(ltab, t) @ jnp.sin(th * m - ph * xn)

    def pos_fn(q: Array) -> Array:
        t, th, ph = q[0], q[1], q[2]
        ang = th * m - ph * xn
        R = coeffs(rtab, t) @ jnp.cos(ang)
        Z = coeffs(ztab, t) @ jnp.sin(ang)
        return jnp.array([R * jnp.cos(ph), R * jnp.sin(ph), Z])

    def b_vector(q: Array) -> Array:
        J = jax.jacfwd(pos_fn)(q)                     # columns: e_s, e_θ, e_φ
        sqrt_g = jnp.linalg.det(J)
        lam_g = jax.grad(lam_fn)(q)                   # (λ_s, λ_θ, λ_φ)
        iota_t = iota + diota * q[0]
        return phipf_j * ((iota_t - lam_g[2]) * J[:, 1]
                          + (1.0 + lam_g[1]) * J[:, 2]) / sqrt_g

    def modb_fn(q: Array) -> Array:
        return jnp.linalg.norm(b_vector(q))

    def point(q: Array, phi_rel: Array):
        J = jax.jacfwd(pos_fn)(q)
        sqrt_g = jnp.linalg.det(J)
        dual = jnp.linalg.inv(J)                      # rows: ∇s, ∇θ, ∇φ
        lam_g = jax.grad(lam_fn)(q)
        iota_t = iota + diota * q[0]
        B = phipf_j * ((iota_t - lam_g[2]) * J[:, 1]
                       + (1.0 + lam_g[1]) * J[:, 2]) / sqrt_g
        modB = jnp.linalg.norm(B)
        dB = jax.grad(modb_fn)(q)                     # (∂|B|/∂s, ∂θ, ∂φ)
        grad_modB = dB[0] * dual[0] + dB[1] * dual[1] + dB[2] * dual[2]
        # ∇α with the secular shear term, α = θ* - ι (φ - ζ0).
        alpha_cov = jnp.array([lam_g[0] - phi_rel * diota,
                               1.0 + lam_g[1],
                               lam_g[2] - iota_t])
        grad_alpha = (alpha_cov[0] * dual[0] + alpha_cov[1] * dual[1]
                      + alpha_cov[2] * dual[2])
        b_sup_phi = phipf_j * (1.0 + lam_g[1]) / sqrt_g
        return (modB, b_sup_phi, grad_alpha @ grad_alpha,
                jnp.dot(jnp.cross(B, grad_modB), grad_alpha))

    return point


# ---------------------------------------------------------------------------
# Per-surface eigenproblem
# ---------------------------------------------------------------------------


def _max_eigenvalue_tridiag(g: Array, c: Array, f: Array, h: Array) -> Array:
    """Most-unstable eigenvalue of ``d/dη(g X')' + cX = λ fX``, ``X(±η_b)=0``.

    Central-difference stencil on the uniform η grid (COBRA/DESC/Gaur
    discretization), symmetrized to standard form with the ``f^{-1/2}``
    similarity transform (``f > 0``), then a batched dense
    ``jnp.linalg.eigvalsh``.  Leading axes of ``g/c/f`` are batch axes.
    """
    g_half = 0.5 * (g[..., 1:] + g[..., :-1]) / (h * h)
    f_in = f[..., 1:-1]
    diag = (c[..., 1:-1] - g_half[..., 1:] - g_half[..., :-1]) / f_in
    off = g_half[..., 1:-1] / jnp.sqrt(f_in[..., :-1] * f_in[..., 1:])
    n = diag.shape[-1]
    pad = jnp.concatenate([off, jnp.zeros_like(off[..., :1])], axis=-1)
    matrix = (jnp.eye(n) * diag[..., :, None]
              + jnp.eye(n, k=1) * pad[..., :, None]
              + jnp.eye(n, k=-1) * pad[..., None, :])
    return jnp.linalg.eigvalsh(matrix)[..., -1]


def _surface_lambda(ctx: dict, j: int, alphas: Array, zeta0s: Array,
                    npoints: int, nturns: float) -> Array:
    """Ballooning eigenvalues on full-mesh surface ``j`` -> ``(nalpha, nzeta0)``."""
    hs = ctx["hs"]
    s_j = ctx["s"][j]
    sqrt_s = jnp.sqrt(s_j)
    # Half-mesh neighbours of full surface j are half indices j (below) and
    # j+1 (above): centered profile values/derivatives at s[j].
    iotas, pres = ctx["iotas"], ctx["pres"]
    iota = 0.5 * (iotas[j] + iotas[j + 1])
    diota = (iotas[j + 1] - iotas[j]) / hs
    dpres = (pres[j + 1] - pres[j]) / hs           # internal units: mu0 dp/ds

    point = _make_point_fn(
        ctx["m"], ctx["xn"],
        _parabola(ctx["rmnc"], j, hs),
        _parabola(ctx["zmns"], j, hs),
        _parabola(ctx["lmns"], j, hs),
        iota, diota, ctx["phipf"][j],
    )

    theta_b = jnp.pi * float(nturns)
    x = jnp.linspace(-theta_b, theta_b, int(npoints))
    h = x[1] - x[0]
    alpha_grid, zeta0_grid = [a.ravel() for a in jnp.meshgrid(alphas, zeta0s, indexing="ij")]
    theta_star = alpha_grid[:, None] + x[None, :]           # (nlines, npoints)
    phi = zeta0_grid[:, None] + x[None, :] / iota           # field line: θ* = α + ι(φ - ζ0)
    lmns0 = _parabola(ctx["lmns"], j, hs)[0]
    theta_v = _theta_vmec_from_pest(theta_star, phi, lmns0, ctx["m"], ctx["xn"])
    q = jnp.stack([jnp.zeros_like(theta_v), theta_v, phi], axis=-1)
    phi_rel = phi - zeta0_grid[:, None]

    modB, b_sup_phi, gaa, b_cross_gradb_dot_grad_alpha = jax.vmap(jax.vmap(point))(q, phi_rel)

    L_ref, B_ref, sign_psi = ctx["L_ref"], ctx["B_ref"], ctx["sign_psi"]
    bmag = modB / B_ref
    gradpar = jnp.abs(L_ref * iota * b_sup_phi / modB)      # L_ref |b·∇θ*|
    gds2 = gaa * (L_ref * L_ref) * s_j                      # |∇α|² L² ρ²
    gbdrift = (-2.0 * B_ref * L_ref * L_ref * sqrt_s * sign_psi
               * b_cross_gradb_dot_grad_alpha / (modB ** 3))
    cvdrift = gbdrift - (2.0 * B_ref * L_ref * L_ref * sqrt_s * dpres
                         / (jnp.abs(ctx["psi_edge"]) * modB * modB))
    dp_drho = 2.0 * sqrt_s * dpres / (B_ref * B_ref)        # normalized mu0 dp/dρ

    g = gradpar * gds2 / bmag
    c = -dp_drho * cvdrift / (gradpar * bmag)
    f = gds2 / (bmag ** 3 * gradpar)
    return _max_eigenvalue_tridiag(g, c, f, h).reshape(alphas.shape[0], zeta0s.shape[0])


# ---------------------------------------------------------------------------
# Public objectives
# ---------------------------------------------------------------------------


def _validate_surface_index(s_index, ns: int) -> int:
    """Validate one full-mesh interior surface index (shared with turbulence)."""
    j = int(s_index)
    if not 2 <= j <= ns - 2:
        raise ValueError(
            f"surface index {j} out of range [2, {ns - 2}] (full-mesh interior; "
            "the radial parabola needs both neighbours and the near-axis "
            "surfaces carry the usual VMEC noise)")
    return j


def _resolve_surfaces(s_indices, ns: int) -> tuple[int, ...]:
    if s_indices is None:
        fractions = (0.35, 0.6, 0.85)
        s_indices = sorted({min(max(int(round(f * (ns - 1))), 2), ns - 2) for f in fractions})
    return tuple(_validate_surface_index(j, ns) for j in s_indices)


def ballooning_lambda(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    s_indices: Sequence[int] | None = None,
    alphas: Sequence[float] | None = None,
    zeta0s: Sequence[float] = (0.0,),
    npoints: int = 121,
    nturns: float = 3.0,
) -> jnp.ndarray:
    """Most-unstable ideal-ballooning eigenvalue per field line (traceable).

    Solves the infinite-n ideal-ballooning eigenproblem (module docstring) on
    the requested full-mesh surfaces and field lines of a converged core
    state and returns the largest eigenvalue ``λ`` of each line, shaped
    ``(len(s_indices), len(alphas), len(zeta0s))``.  ``λ > 0`` means
    ballooning-unstable with normalized squared growth rate
    ``λ = (γ a_N/v_A)²``; ``λ < 0`` is the stable (oscillating) side.

    Parameters
    ----------
    s_indices:
        Full-mesh surface indices, each in ``[2, ns - 2]``.  Default: three
        surfaces at ~35/60/85 % of the radius.
    alphas:
        Field-line labels ``α = θ* - ι (φ - ζ0)``.  Default: four lines
        uniform in ``[0, π]`` (stellarator symmetry maps ``α -> -α``).
    zeta0s:
        Ballooning parameters (the toroidal angle where the secular radial
        wavenumber vanishes).  Default ``(0,)``.
    npoints, nturns:
        Field-line grid: ``npoints`` points over ``θ* ∈ α ± nturns·π``
        (COBRA-style domain; Gaur et al. use ``5π``, 3 turns is adequate for
        optimization-grade accuracy at these resolutions).
    """
    ctx = _ballooning_context(state, rt)
    js = _resolve_surfaces(s_indices, ctx["ns"])
    dtype = ctx["s"].dtype
    if alphas is None:
        alphas_arr = jnp.asarray(np.linspace(0.0, np.pi, 4), dtype=dtype)
    else:
        alphas_arr = jnp.atleast_1d(jnp.asarray(alphas, dtype=dtype))
    zeta0_arr = jnp.atleast_1d(jnp.asarray(zeta0s, dtype=dtype))
    if int(npoints) < 7:
        raise ValueError("npoints must be >= 7")
    return jnp.stack([
        _surface_lambda(ctx, j, alphas_arr, zeta0_arr, int(npoints), float(nturns))
        for j in js
    ])


def ballooning_growth_rate(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    s_indices: Sequence[int] | None = None,
    alphas: Sequence[float] | None = None,
    zeta0s: Sequence[float] = (0.0,),
    npoints: int = 121,
    nturns: float = 3.0,
    reduction: str = "softmax",
    temperature: float = 0.05,
) -> Array:
    """Scalar ballooning objective: (smooth) max of λ over all field lines.

    Reduces :func:`ballooning_lambda` over surfaces × field lines:
    ``reduction="softmax"`` (default) uses the smooth upper bound
    ``T · logsumexp(λ/T)`` (within ``T·log(N)`` of the hard max, fully
    AD-friendly for the implicit-gradient lane); ``"max"`` returns the hard
    maximum.  Drive it negative (e.g. target ``-0.01`` in
    :func:`vmex.core.optimize.least_squares`) for stable-by-construction
    campaigns; the signature is a two-positional ``(state, runtime)``
    callable, so it works with both ``jac=None`` and ``jac="implicit"``.
    """
    lam = ballooning_lambda(
        state, rt, s_indices=s_indices, alphas=alphas, zeta0s=zeta0s,
        npoints=npoints, nturns=nturns,
    ).ravel()
    if reduction == "max":
        return jnp.max(lam)
    if reduction != "softmax":
        raise ValueError(f"reduction must be 'softmax' or 'max', got {reduction!r}")
    t = float(temperature)
    if t <= 0.0:
        raise ValueError("temperature must be positive")
    return t * jax.scipy.special.logsumexp(lam / t)
