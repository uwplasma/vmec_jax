"""Constructed-QI residual vs the independent ``qi_functions_mod`` lineage.

The ``_reference_*`` functions below are the literal NumPy/SciPy
re-implementation of the legacy ``qi_functions_mod.py`` Goodman well
construction (monotone branch envelopes + endpoint caps), ``GetBranches``
crossing search, and J-integral, extracted verbatim from the PR #82
validation workflow.  They form an independent lineage: no jnp, no
sigmoid smoothing, Python loops, linear-spline quadrature.

What is and is not compared (all tolerances measured on the recorded
resolutions and quoted at 2-3x the observed error):

- **Constructed target** — on a single-well field whose ``|B|`` maximum sits
  on the period boundary, the reference construction (envelope + caps) is
  exact and the caps vanish, so the replacement's smooth squash-and-shuffle
  reconstruction must approach it as the sigmoid softness shrinks.  The
  agreement is *smoothing-limited* (max error is at the near-top levels),
  so the assertion is on the measured softness ladder, not grid refinement.
- **J values** — the reference ``J_I`` integral equals half the bounce
  action ``J = 2 int sqrt(1 - pitch B) dl`` at ``pitch = 1/B_j``; the two
  lineages meet to the reference's linear-interpolation error.
- **Level widths** — the sharp reference branch-crossing widths and the
  replacement's sigmoid occupancy widths measure the same bounce distance
  on single-well lines; compared as deviations from the field-line mean
  (the object the residual penalizes), away from the endpoint levels.
- **NOT compared pointwise: sub-well capping.**  The two lineages
  monotonize interior sub-wells with *different* rules by design: the
  reference flood-fill caps a bump at its left-shoulder value, the
  replacement running maximum caps it at the bump height.  Inside a bump
  they legitimately differ; the test pins the relation (running-max
  envelope >= flood-fill envelope, both monotone, equal outside bumps)
  instead of asserting a vacuous pointwise match.
- **Degenerate constant ``|B|``** — out of scope for *both* lineages,
  documented rather than asserted as a signal.  The literal reference
  machinery presupposes a normalized nonconstant line: its endpoint caps
  invent boundary spikes to 1 on a featureless input and ``GetBranches``
  raises on interior levels.  The replacement's per-surface normalization
  is degenerate there (``bnorm = 0/tiny``) and its shuffle reconstruction
  leaves a finite boundary artifact (measured 1.4e-2), pinned as finite
  and bounded, not zero.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import UnivariateSpline

import jax.numpy as jnp

from vmex.core import qi as qi_mod
from vmex.core.bounce import bounce_action
from vmex.core.optimize import quasi_isodynamic_residual


# ---------------------------------------------------------------------------
# Literal qi_functions_mod re-implementation (independent NumPy lineage)
# ---------------------------------------------------------------------------


def _reference_goodman_transform(b_line: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Literal well construction from ``qi_functions_mod.py`` on one line."""

    ba = np.asarray(b_line, dtype=float).copy()
    phisa = np.asarray(phi, dtype=float)
    indmin = int(np.argmin(ba))

    bl = ba[: indmin + 1].copy()
    phisl = phisa[: indmin + 1]
    phisr = phisa[indmin:]
    indmax_l = int(np.argmax(bl))
    bl[:indmax_l] = bl[indmax_l]
    for i in range(len(bl) - 1):
        if bl[i] <= bl[i + 1]:
            jf = len(bl) - 1
            for j in range(i + 1, len(bl)):
                if bl[j] < bl[i]:
                    jf = j
                    break
            bl[i:jf] = bl[i]

    br = ba[indmin:].copy()
    indmax_r = int(np.argmax(br))
    br[indmax_r:] = br[indmax_r]
    for j in range(len(br) - 1, 1, -1):
        if br[j - 1] >= br[j]:
            kf = 0
            for k in range(j - 1, 1, -1):
                if br[k] < br[j]:
                    kf = k
                    break
            br[kf + 1 : j] = br[j]

    pmax = 50
    pmin = 15
    x1l = (phisl - phisl[0]) / max(phisl[-1] - phisl[0], 1.0e-14)
    x1r = (phisr - phisr[0]) / max(phisr[-1] - phisr[0], 1.0e-14)
    fl = np.where(
        x1l < 0.5,
        (1.0 - bl[0]) * ((np.cos(2.0 * np.pi * x1l) + 1.0) / 2.0) ** pmax,
        (-bl[-1]) * ((np.cos(2.0 * np.pi * x1l) + 1.0) / 2.0) ** pmin,
    )
    fr = np.where(
        x1r < 0.5,
        (-br[0]) * ((np.cos(2.0 * np.pi * x1r) + 1.0) / 2.0) ** pmin,
        (1.0 - br[-1]) * ((np.cos(2.0 * np.pi * x1r) + 1.0) / 2.0) ** pmax,
    )
    bl = bl + fl
    br = br + fr
    bl = bl[:-1]
    return np.concatenate((bl, br))


def _reference_get_branches(phi_bs: np.ndarray, ba: np.ndarray, bj: float) -> tuple[float, float]:
    """Literal ``GetBranches`` from ``qi_functions_mod.py``."""

    bmax = 1.0
    bmin = 0.0
    diffs = ba - bj
    diffsgn = diffs[:-1] * diffs[1:]
    inds = np.where(diffsgn < 0)[0]
    inds = np.sort(inds)

    if bj <= bmin:
        imin = int(np.argmin(ba))
        phimin = phi_bs[imin]
        return float(phimin), float(phimin)
    if bj >= bmax:
        return float(phi_bs[0]), float(phi_bs[-1])

    if len(inds) < 2:
        inds = np.where(diffsgn <= 0)[0]
        for iind in range(1, len(inds)):
            if inds[iind] != inds[iind - 1] + 1:
                inds = [inds[iind - 1], inds[-1]]
                break
    if len(inds) > 2:
        inds = [inds[0], inds[-1]]
    ind1 = int(inds[0])
    ind2 = int(inds[1])

    dy1 = ba[ind1] - ba[ind1 + 1]
    dx1 = phi_bs[ind1] - phi_bs[ind1 + 1]
    m1 = dy1 / dx1
    b1 = ba[ind1] - m1 * phi_bs[ind1]
    phi1 = (bj - b1) / m1 if m1 != 0 else phi_bs[ind1]

    dy2 = ba[ind2] - ba[ind2 + 1]
    dx2 = phi_bs[ind2] - phi_bs[ind2 + 1]
    m2 = dy2 / dx2
    b2 = ba[ind2] - m2 * phi_bs[ind2]
    phi2 = (bj - b2) / m2 if m2 != 0 else phi_bs[ind2 + 1]
    return float(phi1), float(phi2)


def _reference_j_pair(
    phi: np.ndarray,
    b_input_norm: np.ndarray,
    b_target_norm: np.ndarray,
    bj_norm: np.ndarray,
    gi_value: float,
    *,
    scale: float,
    bmin: float,
    nphi_int: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Literal J-integral pair from ``qi_functions_mod.py``."""

    ji = np.zeros((len(bj_norm),), dtype=float)
    jc = np.zeros((len(bj_norm),), dtype=float)
    p1 = np.zeros((len(bj_norm),), dtype=float)
    p2 = np.zeros((len(bj_norm),), dtype=float)
    for j, bj in enumerate(bj_norm):
        phi1, phi2 = _reference_get_branches(phi, b_target_norm, float(bj))
        p1[j] = phi1
        p2[j] = phi2
        if phi1 == phi2:
            continue
        phibounce = np.linspace(phi1, phi2, nphi_int)
        bc_phys = np.interp(phibounce, phi, b_target_norm) * scale + bmin
        bi_phys = np.interp(phibounce, phi, b_input_norm) * scale + bmin
        bj_phys = bj * scale + bmin
        bc1 = 1.0 - bc_phys / bj_phys
        bi1 = 1.0 - bi_phys / bj_phys
        bc1[np.abs(bc1) < 1.0e-10] = 0.0
        bi1[np.abs(bi1) < 1.0e-10] = 0.0
        jc_integrand = UnivariateSpline(
            phibounce, np.sqrt(np.maximum(bc1, 0.0)) * gi_value / bi_phys, k=1, s=0
        )
        ji_integrand = UnivariateSpline(
            phibounce, np.sign(bi1) * np.sqrt(np.abs(bi1)) * gi_value / bi_phys, k=1, s=0
        )
        ji[j] = float(ji_integrand.integral(phi1, phi2))
        jc[j] = float(jc_integrand.integral(phi1, phi2))
    return ji, jc, p1, p2


# ---------------------------------------------------------------------------
# Synthetic Boozer spectra (single surface, physical xn convention)
# ---------------------------------------------------------------------------

NFP = 2
NPHI, NALPHA, NBOUNCE = 101, 8, 33


def _booz(coeffs: dict[tuple[int, int], float]) -> dict:
    return {
        "bmnc_b": jnp.asarray([[v for v in coeffs.values()]], dtype=jnp.float64),
        "xm_b": jnp.asarray([m for m, _ in coeffs], dtype=jnp.float64),
        "xn_b": jnp.asarray([n for _, n in coeffs], dtype=jnp.float64),
        "iota_b": jnp.asarray([0.31]),
        "nfp": NFP,
    }


# Exactly omnigenous: |B| = 1 + 0.15 cos(nfp phi) — alpha-independent, one
# well per period, maximum on the period boundary (the caps vanish).
FIELD_QI = _booz({(0, 0): 1.0, (0, NFP): 0.15})
# Alpha-dependent single-well field: adds a (m=1, n=0) harmonic.
FIELD_AD = _booz({(0, 0): 1.0, (0, NFP): 0.15, (1, 0): 0.03})
# Interior sub-well: the second harmonic creates a secondary bump.
FIELD_SUB = _booz({(0, 0): 1.0, (0, NFP): 0.12, (0, 2 * NFP): 0.05})
FIELD_CONST = _booz({(0, 0): 1.0})


def _run(field: dict, **options) -> dict:
    settings = dict(nphi=NPHI, nalpha=NALPHA, n_bounce=NBOUNCE)
    settings.update(options)
    return quasi_isodynamic_residual(**field, **settings)


def _extract_shuffled(field: dict, **options) -> tuple[dict, np.ndarray]:
    """Recover the smooth constructed field from the isolated shuffle piece."""
    out = _run(field, width_weight=0.0, branch_width_weight=0.0,
               profile_weight=0.0, shuffle_profile_weight=1.0, **options)
    bnorm = np.asarray(out["bnorm"])                       # (nsurf, nphi, nalpha)
    nsurf, nphi, nalpha = bnorm.shape
    residuals = np.asarray(out["residuals1d"])
    n_width = nsurf * nalpha * NBOUNCE
    n_profile = nsurf * nalpha * nphi
    assert residuals.size == n_width + 2 * n_profile      # layout guard
    segment = residuals[n_width + n_profile:].reshape(nsurf, nalpha, nphi)
    shuffled = np.swapaxes(bnorm, 1, 2) + segment * np.sqrt(nalpha * nphi)
    return out, shuffled


def _extract_width_deviation(field: dict) -> tuple[dict, np.ndarray]:
    """Recover ``widths - mean_alpha(widths)`` from the isolated width piece."""
    out = _run(field, width_weight=1.0, branch_width_weight=0.0,
               profile_weight=0.0, shuffle_profile_weight=0.0)
    bnorm = np.asarray(out["bnorm"])
    nsurf, nphi, nalpha = bnorm.shape
    residuals = np.asarray(out["residuals1d"])
    n_width = nsurf * nalpha * NBOUNCE
    assert residuals.size == n_width + nsurf * nalpha * nphi  # layout guard
    segment = residuals[:n_width].reshape(nsurf, nalpha, NBOUNCE)
    return out, segment * np.sqrt(nalpha * NBOUNCE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_constructed_class_matches_functional_layer(monkeypatch):
    """ConstructedQIResidual is exactly the validated functional layer."""
    booz = dict(FIELD_QI, G_b=jnp.asarray([2.0]), I_b=jnp.asarray([0.0]),
                s_b=jnp.asarray([0.5]))
    monkeypatch.setattr(qi_mod, "boozer_bmnc_state", lambda *a, **k: booz)
    term = qi_mod.ConstructedQIResidual(
        [0.5], nphi=NPHI, nalpha=NALPHA, n_bounce=NBOUNCE)
    eq = SimpleNamespace(state=object(), runtime=object())
    direct = _run(FIELD_QI)
    np.testing.assert_array_equal(
        np.asarray(term.residuals(eq)), np.asarray(direct["residuals1d"]))


def test_reference_confirms_constructed_target_on_qi_field():
    """Both lineages recognize the analytic QI field; targets agree.

    The reference construction returns the input line *exactly* (envelope
    == input for a single well with the maximum on the boundary, caps == 0),
    proving the field omnigenous in the reference lineage.  The smooth
    reconstruction is smoothing-limited: measured max|shuffled - target| =
    1.33e-2 / rms 4.5e-3 at softness 8e-3 (vs 3.86e-2 / 1.42e-2 at the 2e-2
    default), with the maximum at the near-top levels.
    """
    out, shuffled = _extract_shuffled(
        FIELD_QI, shuffle_profile_softness=8.0e-3)
    phi = np.asarray(out["phi"])
    line = np.asarray(out["bnorm"])[0, :, 0]
    target = _reference_goodman_transform(line, phi)

    assert np.abs(target - line).max() < 1.0e-12   # oracle: exactly omnigenous

    error = np.abs(shuffled[0, 0] - target)
    assert error.max() < 3.0e-2
    assert np.sqrt(np.mean(error**2)) < 1.0e-2

    # smoothing-limited: shrinking the sigmoid softness shrinks the error
    _, coarse = _extract_shuffled(FIELD_QI, shuffle_profile_softness=2.0e-2)
    coarse_error = np.abs(coarse[0, 0] - target)
    assert error.max() < coarse_error.max()
    assert np.sqrt(np.mean(error**2)) < np.sqrt(np.mean(coarse_error**2))

    # and the full residual agrees the field is (nearly) omnigenous:
    # measured total 2.0e-4 at the default weights (smoothing floor).
    assert float(_run(FIELD_QI)["total"]) < 1.0e-3


def test_reference_j_pair_matches_bounce_action_kernel():
    """Cross-lineage J values: reference spline integral vs bounce_action.

    ``J_I = int sqrt(1 - B/B_j) (gi/B) dphi`` over one branch pair equals
    half the replacement's ``J = 2 int sqrt(1 - pitch B) dl`` at
    ``pitch = 1/B_j`` with ``dl/dphi = gi/B``.  Measured max relative
    difference 5.8e-4, limited by the reference's linear interpolation.
    On the omnigenous field the reference's own pair obeys J_I == J_C to
    round-off (measured 0.0).
    """
    out = _run(FIELD_QI)
    phi = np.asarray(out["phi"])
    line = np.asarray(out["bnorm"])[0, :, 0]
    target = _reference_goodman_transform(line, phi)

    b_low, gi = 6.0, 2.0
    scale = 3.0                                   # pretend physical T range
    levels = np.linspace(0.0, 1.0, NBOUNCE)
    ji, jc, _, _ = _reference_j_pair(
        phi, line, target, levels, gi, scale=scale, bmin=b_low, nphi_int=141)
    inner = slice(2, NBOUNCE - 2)
    assert np.abs(ji[inner] - jc[inner]).max() < 1.0e-10 * np.abs(jc).max()

    selected = levels[8:-4]                       # away from well top/bottom
    pitch = 1.0 / (selected * scale + b_low)
    b_phys = line * scale + b_low
    result = bounce_action(
        jnp.asarray(b_phys), jnp.asarray(pitch[::-1]),
        dl_dphi=jnp.asarray(gi / b_phys), length=float(phi[-1] - phi[0]),
        periodic=False, quadrature_order=96)
    action = np.asarray(result["action"])
    usable = np.asarray(result["usable_mask"])
    totals = np.array([action[k][usable[k]].sum() for k in range(len(pitch))])
    reference = 2.0 * ji[8:-4][::-1]
    np.testing.assert_allclose(totals, reference, rtol=3.0e-3)


def test_width_deviation_matches_reference_crossings():
    """Occupancy-width deviations track the reference crossing widths.

    On the alpha-dependent single-well field the deviation-from-line-mean of
    the replacement's sigmoid occupancy widths matches the reference
    ``GetBranches`` interval widths (evaluated per line in the line's own
    normalization) to measured max 4.9e-3 against a deviation scale of 0.119
    — a 4% relative agreement at softness 2e-2, asserted at 1.5e-2.
    """
    out, width_dev = _extract_width_deviation(FIELD_AD)
    bnorm = np.asarray(out["bnorm"])
    levels = np.asarray(out["levels"])
    phi = np.asarray(out["phi"])
    period = phi[-1] - phi[0]
    nalpha = bnorm.shape[2]

    reference = np.zeros((nalpha, levels.size))
    for a in range(nalpha):
        line = bnorm[0, :, a]
        low, high = line.min(), line.max()
        line_norm = (line - low) / (high - low)
        target = _reference_goodman_transform(line_norm, phi)
        for il, level in enumerate(levels):
            level_line = float(np.clip((level - low) / (high - low), 0.0, 1.0))
            left, right = _reference_get_branches(phi, target, level_line)
            reference[a, il] = (right - left) / period
    reference_dev = reference - reference.mean(axis=0, keepdims=True)

    interior = (levels > 0.15) & (levels < 0.85)  # endpoint caps out of scope
    scale = np.abs(reference_dev[:, interior]).max()
    assert scale > 0.05                            # non-vacuous comparison
    error = np.abs(width_dev[0][:, interior] - reference_dev[:, interior])
    assert error.max() < 1.5e-2


def test_non_qi_ordering_agrees_between_lineages():
    """Both lineages rank the alpha-dependent field as far less omnigenous.

    Measured: replacement totals 2.0e-4 (QI) vs 7.2e-3 (alpha-dependent),
    a 36x ratio; reference rms(J_I - J_C)/max|J_C| is exactly 0 (QI) vs
    7.9e-4 (alpha-dependent).
    """
    total_qi = float(_run(FIELD_QI)["total"])
    total_ad = float(_run(FIELD_AD)["total"])
    assert total_ad > 10.0 * total_qi

    def reference_misfit(field: dict) -> float:
        out = _run(field)
        bnorm = np.asarray(out["bnorm"])
        phi = np.asarray(out["phi"])
        levels = np.linspace(0.0, 1.0, NBOUNCE)
        ji_all, jc_all = [], []
        for a in range(bnorm.shape[2]):
            line = bnorm[0, :, a]
            low, high = line.min(), line.max()
            line_norm = (line - low) / max(high - low, 1.0e-14)
            target = _reference_goodman_transform(line_norm, phi)
            ji, jc, _, _ = _reference_j_pair(
                phi, line_norm, target, levels, 1.0,
                scale=3.0, bmin=6.0, nphi_int=101)
            ji_all.append(ji)
            jc_all.append(jc)
        spread = np.asarray(ji_all) - np.asarray(jc_all)
        return float(np.sqrt(np.mean(spread**2))
                     / max(np.abs(np.asarray(jc_all)).max(), 1.0e-14))

    assert reference_misfit(FIELD_QI) < 1.0e-12
    assert reference_misfit(FIELD_AD) > 1.0e-4


def test_constant_field_is_a_documented_degenerate_regime():
    """Degenerate constant |B|: both lineages leave their domain of validity.

    Analytically a constant field is trivially omnigenous with ``J_I = J_C``
    for every trapped class, but neither implementation certifies that.
    The literal reference construction assumes a per-line-normalized
    nonconstant input: on the constant line the argmin lands on the first
    sample, so the whole left branch degenerates and is dropped, the right
    endpoint cap invents a spike to exactly 1 on a featureless input, and
    ``GetBranches`` raises ``IndexError`` on interior levels (no
    sign-change crossing exists).  The replacement's per-surface
    normalization gives ``bnorm = 0/tiny`` and its shuffle reconstruction
    leaves a finite boundary artifact — measured total 1.39e-2 at the
    default weights, pinned here as *finite and bounded*, not zero.
    Conclusion: the constant-|B| regime carries no valid QI signal in
    either lineage and must not be scored by the constructed residual.
    """
    out = _run(FIELD_CONST)
    phi = np.asarray(out["phi"])
    line = np.asarray(out["bnorm"])[0, :, 0]
    assert np.abs(line).max() == 0.0               # degenerate normalization

    target = _reference_goodman_transform(line, phi)
    assert target.size == line.size
    assert target[0] == pytest.approx(0.0)         # left branch degenerated
    assert target[-1] == pytest.approx(1.0)        # invented boundary spike
    assert np.abs(target[len(target) // 2]) < 1.0e-6
    with pytest.raises(IndexError):                # crossing search undefined
        _reference_get_branches(phi, target, 0.5)

    total = float(out["total"])
    assert np.isfinite(total)
    assert total < 0.1


def test_subwell_envelopes_differ_by_design():
    """Interior sub-wells: flood-fill vs running-maximum monotonization.

    The reference flood-fill caps a secondary bump at its shoulder value;
    the replacement's running maximum caps it at the bump height.  Both are
    monotone on each branch and identical wherever the raw line is its own
    envelope; measured max difference 6.2e-2 inside the bump, equal on 64%
    of the grid.  This is the one regime where the constructions differ by
    design, so no pointwise agreement is asserted there.
    """
    out = _run(FIELD_SUB, nalpha=4)
    phi = np.asarray(out["phi"])
    line = np.asarray(out["bnorm"])[0, :, 0]
    n_minima = int(np.sum((line[1:-1] < line[:-2]) & (line[1:-1] < line[2:])))
    assert n_minima == 2                           # the sub-well is present

    flood = _reference_goodman_transform(line, phi)   # caps vanish: boundary max
    imin = int(np.argmin(line))
    running_max = np.concatenate([
        np.maximum.accumulate(line[: imin + 1][::-1])[::-1][:-1],
        np.maximum.accumulate(line[imin:]),
    ])

    def monotone(envelope: np.ndarray) -> bool:
        return bool(np.all(np.diff(envelope[: imin + 1]) <= 1.0e-12)
                    and np.all(np.diff(envelope[imin:]) >= -1.0e-12))

    assert monotone(flood) and monotone(running_max)
    assert np.all(running_max >= flood - 1.0e-12)
    equal = np.abs(running_max - flood) < 1.0e-12
    assert equal.mean() > 0.5                      # identical outside the bump
    difference = np.abs(running_max - flood).max()
    assert 1.0e-2 < difference < 2.0e-1            # measured 6.2e-2 inside it
