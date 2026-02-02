from __future__ import annotations

import numpy as np
import pytest


def test_step8_wout_state_is_nearly_stationary_for_total_energy():
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    from vmec_jax._compat import enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.field import TWOPI, b2_from_bsup, bsup_from_geom, lamscale_from_phips
    from vmec_jax.geom import eval_geom
    from vmec_jax.solve import _mask_grad_for_constraints, _mode00_index
    from vmec_jax.static import build_static
    from vmec_jax.wout import read_wout, state_from_wout
    from vmec_jax._compat import jax, jnp

    enable_x64(True)

    cfg, _indata = load_config("examples/input.LandremanSenguptaPlunk_section5p3_low_res")
    static = build_static(cfg)
    wout = read_wout("examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc")
    st = state_from_wout(wout)

    # Sanity: VMEC's own reported force residuals are tiny for the reference equilibrium.
    assert float(wout.fsqr) < 1e-8
    assert float(wout.fsqz) < 1e-8
    assert float(wout.fsql) < 1e-8

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta = theta[1] - theta[0]
    dzeta = zeta[1] - zeta[0]
    weight = ds * dtheta * dzeta

    phipf = jnp.asarray(wout.phipf)
    chipf = jnp.asarray(wout.chipf)
    signgs = int(wout.signgs)
    pressure = jnp.asarray(wout.presf)
    lamscale = lamscale_from_phips(jnp.asarray(wout.phips), s)

    gamma = float(wout.gamma)
    if abs(gamma - 1.0) < 1e-14:
        gamma = 0.0

    nfp = int(cfg.nfp)

    def _objective(state):
        g = eval_geom(state, static)
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        return wb + wp / (gamma - 1.0)

    val, grad = jax.value_and_grad(_objective)(st)
    assert np.isfinite(float(np.asarray(val)))

    idx00 = _mode00_index(static.modes)
    gradm = _mask_grad_for_constraints(grad, static, idx00=idx00)

    # "Stationarity" should be reasonably small at the VMEC equilibrium, though it
    # is not expected to be 1e-12 because our objective differs from VMEC's full
    # residual formulation and uses a simple uniform-grid quadrature.
    g_arrs = [
        np.asarray(gradm.Rcos),
        np.asarray(gradm.Rsin),
        np.asarray(gradm.Zcos),
        np.asarray(gradm.Zsin),
        np.asarray(gradm.Lcos),
        np.asarray(gradm.Lsin),
    ]
    ss = float(sum(np.sum(a * a) for a in g_arrs))
    nn = int(sum(a.size for a in g_arrs))
    grad_rms = float(np.sqrt(ss / nn))
    assert grad_rms < 1e-3

