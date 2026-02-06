from __future__ import annotations

import numpy as np
import pytest


def test_step10_force_residuals_are_smaller_for_wout_equilibrium(load_case_li383_low_res):
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    from vmec_jax._compat import enable_x64
    from vmec_jax.energy import flux_profiles_from_indata
    from vmec_jax.energy import FluxProfiles
    from vmec_jax.field import chips_from_chipf, lamscale_from_phips, signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.profiles import eval_profiles
    from vmec_jax.residuals import force_residuals_from_state
    from vmec_jax.wout import read_wout, state_from_wout

    enable_x64(True)

    cfg, indata, static, _bdy, st0 = load_case_li383_low_res
    g0 = eval_geom(st0, static)
    signgs0 = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)
    flux0 = flux_profiles_from_indata(indata, static.s, signgs=signgs0)
    prof0 = eval_profiles(indata, static.s)
    pressure0 = prof0.get("pressure", np.zeros_like(np.asarray(static.s)))
    gamma0 = float(indata.get_float("GAMMA", 0.0))

    r0 = force_residuals_from_state(st0, static, flux=flux0, pressure=pressure0, gamma=gamma0)
    assert np.isfinite(r0.fsq_like)

    wout = read_wout("examples/data/wout_li383_low_res_reference.nc")
    st_w = state_from_wout(wout)
    # Use wout profiles/scalars for a consistent equilibrium objective.
    flux_w = FluxProfiles(
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(chips_from_chipf(wout.chipf)),
        phips=np.asarray(wout.phips),
        signgs=int(wout.signgs),
        lamscale=lamscale_from_phips(np.asarray(wout.phips), np.asarray(static.s)),
    )
    r_w = force_residuals_from_state(st_w, static, flux=flux_w, pressure=np.asarray(wout.presf), gamma=float(wout.gamma))

    assert np.isfinite(r_w.fsq_like)
    # The VMEC equilibrium should be closer to stationary than our crude initial guess.
    assert r_w.fsq_like < r0.fsq_like
