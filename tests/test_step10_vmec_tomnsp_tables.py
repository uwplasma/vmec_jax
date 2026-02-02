from __future__ import annotations

import numpy as np

from vmec_jax.vmec_tomnsp import vmec_angle_grid, vmec_theta_sizes, vmec_trig_tables


def test_vmec_theta_sizes_match_read_indata_logic():
    ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(22, lasym=False)
    assert ntheta1 == 22
    assert ntheta2 == 12
    assert ntheta3 == 12

    ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(22, lasym=True)
    assert ntheta1 == 22
    assert ntheta2 == 12
    assert ntheta3 == 22


def test_vmec_angle_grid_half_interval_includes_pi_when_symmetric():
    g = vmec_angle_grid(ntheta=22, nzeta=5, nfp=3, lasym=False)
    assert g.theta.size == 12
    assert np.isclose(g.theta[0], 0.0)
    assert np.isclose(g.theta[-1], np.pi)
    assert g.zeta.size == 5
    assert np.isclose(g.zeta[0], 0.0)
    assert np.isclose(g.zeta[-1], 2.0 * np.pi * (4.0 / 5.0))


def test_vmec_trig_tables_include_nfp_in_derivative_tables():
    t = vmec_trig_tables(ntheta=22, nzeta=8, nfp=3, mmax=4, nmax=4, lasym=False)
    # For n=1: cosnvn = (n*nfp)*cosnv, sinnvn = -(n*nfp)*sinnv.
    n = 1
    assert np.allclose(np.asarray(t.cosnvn)[:, n], (n * 3) * np.asarray(t.cosnv)[:, n])
    assert np.allclose(np.asarray(t.sinnvn)[:, n], -(n * 3) * np.asarray(t.sinnv)[:, n])


def test_vmec_trig_tables_cosmui3_matches_fixaray_behavior():
    # lasym=False: ntheta3==ntheta2, so cosmui3 is the same as cosmui (endpoint half-weights).
    t = vmec_trig_tables(ntheta=10, nzeta=7, nfp=3, mmax=6, nmax=4, lasym=False)
    assert t.ntheta2 == t.ntheta3
    assert np.allclose(np.asarray(t.cosmui3), np.asarray(t.cosmui))

    # lasym=True: ntheta3==ntheta1, and cosmui3 remains un-halved (full-interval integration).
    ta = vmec_trig_tables(ntheta=10, nzeta=7, nfp=3, mmax=6, nmax=4, lasym=True)
    assert ta.ntheta3 == ta.ntheta1
    assert np.allclose(np.asarray(ta.cosmui3[0, :]), np.asarray(ta.cosmu[0, :]) * ta.dnorm)
