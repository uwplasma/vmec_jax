"""Targeted unit tests for :mod:`vmex.core.setup` corner branches.

Covers the ``APHI`` toroidal-flux polynomial map (identity short-circuit
and a genuine polynomial, checked against the exact antiderivative), the
degenerate ``ns < 2`` radial grid, and the lasym delta-rotation guards
(``convert_sym``/``convert_asym`` m=1 phase alignment in readin.f).
"""

from __future__ import annotations

import numpy as np
import pytest

from vmex.core import setup as su


def test_torflux_identity_short_circuit():
    torflux, deriv = su._torflux_functions(np.asarray([1.0, 0.0, 0.0]))
    x = np.linspace(0.0, 1.0, 7)
    np.testing.assert_allclose(np.asarray(torflux(x)), x, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(deriv(x)), np.ones_like(x), rtol=0, atol=0)
    # empty APHI defaults to the identity too
    torflux, deriv = su._torflux_functions(np.asarray([]))
    np.testing.assert_allclose(np.asarray(deriv(0.3)), 1.0)


def test_torflux_polynomial_matches_antiderivative():
    # aphi = [0.5, 0.5]: torflux_deriv(x) = 0.5 + 1.0*x (i * aphi_i * x**(i-1));
    # the 101-point trapezoid rule is exact for a linear integrand.
    torflux, deriv = su._torflux_functions(np.asarray([0.5, 0.5]))
    x = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(np.asarray(deriv(x)), 0.5 + x, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(torflux(x)), 0.5 * x + 0.5 * x**2,
                               rtol=1e-12, atol=1e-14)


def test_radial_grids_degenerate_ns():
    grids = su.radial_grids(1)
    assert grids.s_full.shape == (1,)
    np.testing.assert_array_equal(np.asarray(grids.s_full), [0.0])
    np.testing.assert_array_equal(np.asarray(grids.sqrts), [1.0])
    assert float(grids.hs) == 1.0


def _m1_arrays(ntor=1, mpol=3):
    rbc = np.zeros((2 * ntor + 1, mpol))
    rbs = np.zeros_like(rbc)
    zbc = np.zeros_like(rbc)
    zbs = np.zeros_like(rbc)
    return rbc, rbs, zbc, zbs


def test_lasym_delta_rotation_guards():
    ntor = 1
    # mpol < 2: unrotated
    rbc, rbs, zbc, zbs = (np.zeros((3, 1)),) * 4
    out = su._lasym_delta_rotation(rbc, rbs, zbc, zbs, mpol=1, ntor=ntor)
    assert out[0] is rbc

    # degenerate m=1 content (denominator zero): unrotated
    rbc, rbs, zbc, zbs = _m1_arrays()
    out = su._lasym_delta_rotation(rbc, rbs, zbc, zbs, mpol=3, ntor=ntor)
    assert out[0] is rbc

    # rbs(0,1) == zbc(0,1) -> delta = 0: unrotated
    rbc, rbs, zbc, zbs = _m1_arrays()
    rbc[ntor, 1] = 1.0
    rbs[ntor, 1] = 0.25
    zbc[ntor, 1] = 0.25
    out = su._lasym_delta_rotation(rbc, rbs, zbc, zbs, mpol=3, ntor=ntor)
    assert out[0] is rbc


def test_lasym_delta_rotation_aligns_m1_modes():
    """After rotation the m=1 antisymmetric content satisfies rbs = zbc."""
    ntor = 1
    rbc, rbs, zbc, zbs = _m1_arrays()
    rbc[ntor, 1] = 1.0
    zbs[ntor, 1] = 0.8
    rbs[ntor, 1] = 0.3
    zbc[ntor, 1] = 0.1
    rbc2, rbs2, zbc2, zbs2 = su._lasym_delta_rotation(
        rbc, rbs, zbc, zbs, mpol=3, ntor=ntor)
    assert rbs2[ntor, 1] == pytest.approx(zbc2[ntor, 1], abs=1e-12)
    # the rotation is a pure phase: the m=1 quadratic invariant is preserved
    inv0 = rbc[ntor, 1] ** 2 + rbs[ntor, 1] ** 2 + zbc[ntor, 1] ** 2 + zbs[ntor, 1] ** 2
    inv1 = rbc2[ntor, 1] ** 2 + rbs2[ntor, 1] ** 2 + zbc2[ntor, 1] ** 2 + zbs2[ntor, 1] ** 2
    assert inv1 == pytest.approx(inv0, rel=1e-12)
