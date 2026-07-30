"""Tests for ``vmex.core.preconditioner``: ``precondn`` / ``lamcal`` /
``scalfor_matrices`` / ``scalfor`` / ``tridiagonal_solve``.

The full A/B against the legacy scalfor.f/precondn.f/lamcal.f90 port
(matrix elements, assembly, solved forces at rtol 1e-12 on synthetic and
real solver states) retired with the legacy tree; kept here are the
VMEC2000 convention pins (edge pedestal, ZC(0,0) stabilization, lamcal axis
row), the tridiagonal solver vs dense references, and jit equivalence.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import preconditioner as newp

RTOL = 1e-12
ATOL = 1e-13

# (ns, mpol, ntor, ntheta, nzeta, nfp, lasym)
CASES = [
    (13, 6, 0, 18, 1, 1, False),
    (11, 7, 3, 22, 16, 3, False),
    (9, 5, 2, 20, 18, 2, True),
]
CASE_IDS = ["axisym", "nfp3-3d", "nfp2-lasym-3d"]


def _allclose(actual, desired, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Synthetic (well-conditioned) inputs with the VMEC shapes and signs
# ---------------------------------------------------------------------------


def _synthetic_case(case):
    """Build (bc, k, cfg, s, weights) shaped like the solver's bcovar payload."""
    ns, mpol, ntor, ntheta, nzeta, nfp, lasym = case
    lthreed = ntor > 0
    weights = newp.angular_integration_weights(ntheta=ntheta, nzeta=nzeta, lasym=lasym)
    ntheta_eff = int(weights.shape[0])
    shape = (ns, ntheta_eff, nzeta)
    rng = np.random.default_rng(abs(hash(case)) % (2**31))

    def uniform(lo, hi):
        return rng.uniform(lo, hi, size=shape)

    def smallnormal(scale=0.5):
        return scale * rng.standard_normal(size=shape)

    jac = SimpleNamespace(
        # Negative Jacobian, bounded away from zero (VMEC signgs = -1).
        sqrtg=-uniform(0.7, 1.7),
        r12=uniform(0.9, 1.4),
        tau=smallnormal(),
        rs=smallnormal(),
        zs=smallnormal(),
        ru12=smallnormal(),
        zu12=smallnormal(),
    )
    bc = SimpleNamespace(
        jac=jac,
        guu=uniform(0.8, 1.8),
        guv=0.3 * rng.standard_normal(size=shape),
        gvv=uniform(1.5, 2.5),
        bsq=uniform(1.2, 2.2),
        bsupv=uniform(0.6, 1.2),
        lamscale=0.7321,
    )
    k = SimpleNamespace(
        pr1_odd=smallnormal(),
        pz1_odd=smallnormal(),
        pru_even=smallnormal(),
        pru_odd=smallnormal(),
        pzu_even=smallnormal(),
        pzu_odd=smallnormal(),
    )
    cfg = SimpleNamespace(
        mpol=mpol,
        ntor=ntor,
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=nfp,
        lasym=lasym,
        lthreed=lthreed,
    )
    s = np.linspace(0.0, 1.0, ns)
    return bc, k, cfg, s, weights


def _new_precondn_from_bc(bc, k, weights, s):
    """Call the new precondn for the R- and Z-force families (VMEC call sites)."""
    ns = int(np.asarray(s).shape[0])
    delta_s = float(np.asarray(s)[1] - np.asarray(s)[0])
    common = dict(
        r12_half=jnp.asarray(bc.jac.r12)[1:],
        bsq_half=jnp.asarray(bc.bsq)[1:],
        bsupv_half=jnp.asarray(bc.bsupv)[1:],
        sqrt_g_half=jnp.asarray(bc.jac.sqrtg)[1:],
        angular_weight=weights,
        delta_s=delta_s,
        ns=ns,
    )
    # R force <- Z geometry (VMEC: arm/ard/brm/brd/crd).
    coeffs_r = newp.precondn(
        dxds_half=jnp.asarray(bc.jac.zs)[1:],
        dxdu_half=jnp.asarray(bc.jac.zu12)[1:],
        dxdu_even_full=jnp.asarray(k.pzu_even),
        dxdu_odd_full=jnp.asarray(k.pzu_odd),
        x_odd_full=jnp.asarray(k.pz1_odd),
        **common,
    )
    # Z force <- R geometry (VMEC: azm/azd/bzm/bzd).
    coeffs_z = newp.precondn(
        dxds_half=jnp.asarray(bc.jac.rs)[1:],
        dxdu_half=jnp.asarray(bc.jac.ru12)[1:],
        dxdu_even_full=jnp.asarray(k.pru_even),
        dxdu_odd_full=jnp.asarray(k.pru_odd),
        x_odd_full=jnp.asarray(k.pr1_odd),
        **common,
    )
    return coeffs_r, coeffs_z, delta_s


# ---------------------------------------------------------------------------
# lamcal conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_lamcal_axis_row_convention(case):
    """lamcal.f90 axis row: zero except the (0,0) chip/iota slot."""
    bc, _k, cfg, s, weights = _synthetic_case(case)
    new_faclam = newp.lamcal(
        guu_half=bc.guu,
        guv_half=bc.guv,
        gvv_half=bc.gvv,
        sqrt_g_half=bc.jac.sqrtg,
        lamscale=bc.lamscale,
        angular_weight=weights,
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        nfp=int(cfg.nfp),
        lthreed=bool(cfg.lthreed),
    )
    new_np = np.asarray(new_faclam)
    assert np.all(np.isfinite(new_np))
    assert np.all(new_np[0, 1:, :] == 0.0)
    assert np.all(new_np[0, 0, 1:] == 0.0)
    assert new_np[0, 0, 0] != 0.0


def test_edge_pedestal_and_zc00_values():
    """The scalfor.f edge constants: 0.05 pedestal and the 0.25 ZC00 factor."""
    case = CASES[0]
    bc, k, cfg, s, weights = _synthetic_case(case)
    ns = int(np.asarray(s).shape[0])
    coeffs_r, coeffs_z, delta_s = _new_precondn_from_bc(bc, k, weights, s)

    kwargs = dict(
        delta_s=delta_s, mpol=int(cfg.mpol), ntor=int(cfg.ntor),
        nfp=int(cfg.nfp), ns=ns, jmax=ns,
    )
    plain = newp.scalfor_matrices(coeffs_z, stabilize_edge_zc00=False, **kwargs)
    stabilized = newp.scalfor_matrices(coeffs_z, stabilize_edge_zc00=True, **kwargs)
    # Only the (m,n) = (0,0) edge diagonal differs, by (1-mult_fac)/(1+pedestal).
    diff = np.asarray(stabilized.dx) - np.asarray(plain.dx)
    assert np.count_nonzero(diff) == 1
    mult_fac = min(0.25, 0.25 * delta_s * 15.0)
    expected = np.asarray(plain.dx)[ns - 1, 0, 0] * (1.0 - mult_fac) / (1.0 + 0.05)
    _allclose(np.asarray(stabilized.dx)[ns - 1, 0, 0], expected)

    # The pedestal itself: edge diagonal vs an assembly with jmax=ns and no
    # edge treatment possible (compare against re-deriving from coefficients).
    no_edge = newp.scalfor_matrices(
        coeffs_z, stabilize_edge_zc00=False,
        delta_s=delta_s, mpol=int(cfg.mpol), ntor=int(cfg.ntor),
        nfp=int(cfg.nfp), ns=ns + 1, jmax=ns,  # edge branch inactive (jmax < ns)
    )
    ped = np.asarray(plain.dx)[ns - 1] / np.asarray(no_edge.dx)[ns - 1]
    assert np.allclose(ped[0:2, :], 1.05, rtol=1e-12)
    assert np.allclose(ped[2:, :], 1.10, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tridiagonal solver vs dense reference
# ---------------------------------------------------------------------------


def test_tridiagonal_solve_matches_dense_numpy():
    rng = np.random.default_rng(42)
    ns, mpol, nrange, nrhs = 17, 8, 5, 3
    shape = (ns, mpol, nrange)
    superd = rng.uniform(-0.5, 0.5, size=shape)
    subd = rng.uniform(-0.5, 0.5, size=shape)
    diag = 3.0 + rng.uniform(0.0, 1.0, size=shape)  # diagonally dominant
    rhs = rng.standard_normal(size=shape + (nrhs,))

    solution = np.asarray(
        newp.tridiagonal_solve(
            jnp.asarray(superd), jnp.asarray(diag), jnp.asarray(subd), jnp.asarray(rhs)
        )
    )
    assert solution.shape == rhs.shape

    for m in range(mpol):
        for n in range(nrange):
            dense = (
                np.diag(diag[:, m, n])
                + np.diag(superd[:-1, m, n], k=1)
                + np.diag(subd[1:, m, n], k=-1)
            )
            expected = np.linalg.solve(dense, rhs[:, m, n, :])
            np.testing.assert_allclose(solution[:, m, n, :], expected, rtol=1e-11, atol=1e-12)


def test_tridiagonal_solve_broadcast_rank():
    """Shared (per-column) coefficients with extra trailing RHS axes."""
    rng = np.random.default_rng(3)
    ns, ncol = 9, 4
    superd = rng.uniform(-0.4, 0.4, size=(ns, ncol))
    subd = rng.uniform(-0.4, 0.4, size=(ns, ncol))
    diag = 2.5 + rng.uniform(0.0, 1.0, size=(ns, ncol))
    rhs = rng.standard_normal(size=(ns, ncol, 6))
    got = np.asarray(newp.tridiagonal_solve(superd, diag, subd, rhs))
    for c in range(ncol):
        dense = (
            np.diag(diag[:, c]) + np.diag(superd[:-1, c], k=1) + np.diag(subd[1:, c], k=-1)
        )
        np.testing.assert_allclose(
            got[:, c, :], np.linalg.solve(dense, rhs[:, c, :]), rtol=1e-11, atol=1e-12
        )


def test_scalfor_checked_solve_reports_well_conditioned_system() -> None:
    """The production solve accepts a stable radial system without fallback."""
    ns, mpol, nrange = 6, 2, 3
    matrices = newp.TridiagonalMatrices(
        ax=jnp.full((ns, mpol, nrange), -0.25),
        bx=jnp.full((ns, mpol, nrange), -0.25),
        dx=jnp.full((ns, mpol, nrange), 2.0),
    )
    force = jnp.arange(ns * mpol * nrange, dtype=float).reshape(
        ns, mpol, nrange
    )
    solved, safe = newp.scalfor(force, matrices, jmax=ns, return_safe=True)
    reference = np.asarray(force).copy()
    reference[:, 0] = np.asarray(newp.tridiagonal_solve(
        matrices.ax[:, 0],
        matrices.dx[:, 0],
        matrices.bx[:, 0],
        force[:, 0],
    ))
    reference[1:, 1] = np.asarray(newp.tridiagonal_solve(
        matrices.ax[1:, 1],
        matrices.dx[1:, 1],
        matrices.bx[1:, 1],
        force[1:, 1],
    ))
    reference[0, 1] = 0.0
    assert bool(safe)
    np.testing.assert_allclose(np.asarray(solved), reference)


def test_scalfor_checked_solve_falls_back_without_nonfinite_update() -> None:
    """A singular radial block returns its RHS and a typed unsafe status."""
    ns, mpol, nrange = 5, 2, 2
    matrices = newp.TridiagonalMatrices(
        ax=jnp.zeros((ns, mpol, nrange)),
        bx=jnp.zeros((ns, mpol, nrange)),
        dx=jnp.zeros((ns, mpol, nrange)),
    )
    force = jnp.arange(ns * mpol * nrange, dtype=float).reshape(
        ns, mpol, nrange
    )
    solved, safe = newp.scalfor(force, matrices, jmax=ns, return_safe=True)
    assert not bool(safe)
    assert np.all(np.isfinite(np.asarray(solved)))
    # SOLVAX's identity fallback leaves each rejected RHS column unchanged;
    # scalfor still enforces VMEC's m>0 axis constraint.
    np.testing.assert_array_equal(np.asarray(solved[:, 0]), np.asarray(force[:, 0]))
    np.testing.assert_array_equal(np.asarray(solved[1:, 1]), np.asarray(force[1:, 1]))
    np.testing.assert_array_equal(np.asarray(solved[0, 1]), 0.0)


# ---------------------------------------------------------------------------
# jit compatibility
# ---------------------------------------------------------------------------


def test_jit_compatibility_end_to_end():
    case = CASES[1]
    bc, k, cfg, s, weights = _synthetic_case(case)
    ns = int(np.asarray(s).shape[0])
    delta_s = float(s[1] - s[0])

    @partial(jax.jit, static_argnames=("ns", "mpol", "ntor", "nfp", "jmax"))
    def preconditioned_force(
        dxds, dxdu, dxdu_e, dxdu_o, x_o, r12, bsq, bsupv, sqrtg, force,
        *, ns, mpol, ntor, nfp, jmax,
    ):
        coeffs = newp.precondn(
            dxds_half=dxds, dxdu_half=dxdu, dxdu_even_full=dxdu_e,
            dxdu_odd_full=dxdu_o, x_odd_full=x_o, r12_half=r12,
            bsq_half=bsq, bsupv_half=bsupv, sqrt_g_half=sqrtg,
            angular_weight=weights, delta_s=delta_s, ns=ns,
        )
        mats = newp.scalfor_matrices(
            coeffs, delta_s=delta_s, mpol=mpol, ntor=ntor, nfp=nfp, ns=ns, jmax=jmax,
        )
        return newp.scalfor(force, mats, jmax=jmax)

    rng = np.random.default_rng(11)
    force = jnp.asarray(rng.standard_normal((ns, int(cfg.mpol), int(cfg.ntor) + 1)))
    args = (
        jnp.asarray(bc.jac.zs)[1:], jnp.asarray(bc.jac.zu12)[1:],
        jnp.asarray(k.pzu_even), jnp.asarray(k.pzu_odd), jnp.asarray(k.pz1_odd),
        jnp.asarray(bc.jac.r12)[1:], jnp.asarray(bc.bsq)[1:],
        jnp.asarray(bc.bsupv)[1:], jnp.asarray(bc.jac.sqrtg)[1:], force,
    )
    kwargs = dict(ns=ns, mpol=int(cfg.mpol), ntor=int(cfg.ntor), nfp=int(cfg.nfp), jmax=ns - 1)
    jitted = preconditioned_force(*args, **kwargs)

    coeffs_r, _, _ = _new_precondn_from_bc(bc, k, weights, s)
    mats_r = newp.scalfor_matrices(
        coeffs_r, delta_s=delta_s, mpol=int(cfg.mpol), ntor=int(cfg.ntor),
        nfp=int(cfg.nfp), ns=ns, jmax=ns - 1,
    )
    eager = newp.scalfor(force, mats_r, jmax=ns - 1)
    _allclose(jitted, eager, rtol=RTOL, atol=1e-14)


def test_jit_compatibility_lamcal():
    case = CASES[1]
    bc, _k, cfg, s, weights = _synthetic_case(case)

    @jax.jit
    def jitted_lamcal(guu, guv, gvv, sqrtg, lamscale):
        return newp.lamcal(
            guu_half=guu, guv_half=guv, gvv_half=gvv, sqrt_g_half=sqrtg,
            lamscale=lamscale, angular_weight=weights,
            mpol=int(cfg.mpol), ntor=int(cfg.ntor), nfp=int(cfg.nfp),
            lthreed=bool(cfg.lthreed),
        )

    got = jitted_lamcal(
        jnp.asarray(bc.guu), jnp.asarray(bc.guv), jnp.asarray(bc.gvv),
        jnp.asarray(bc.jac.sqrtg), jnp.asarray(bc.lamscale),
    )
    eager = newp.lamcal(
        guu_half=bc.guu, guv_half=bc.guv, gvv_half=bc.gvv,
        sqrt_g_half=bc.jac.sqrtg, lamscale=bc.lamscale, angular_weight=weights,
        mpol=int(cfg.mpol), ntor=int(cfg.ntor), nfp=int(cfg.nfp),
        lthreed=bool(cfg.lthreed),
    )
    _allclose(got, eager, rtol=RTOL, atol=1e-14)
