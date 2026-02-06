from __future__ import annotations

import numpy as np


def test_b_cartesian_matches_metric_b2(load_case_li383_low_res):
    from vmec_jax.energy import flux_profiles_from_indata
    from vmec_jax.field import b2_from_bsup, b_cartesian_from_bsup, bsup_from_geom, signgs_from_sqrtg
    from vmec_jax.geom import eval_geom

    cfg, indata, static, _bdy, st0 = load_case_li383_low_res

    g = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g.sqrtg))
    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    bsupu, bsupv = bsup_from_geom(
        g,
        phipf=flux.phipf,
        chipf=flux.chipf,
        nfp=cfg.nfp,
        signgs=signgs,
        lamscale=flux.lamscale,
    )

    B2_metric = np.asarray(b2_from_bsup(g, bsupu, bsupv))
    B = np.asarray(b_cartesian_from_bsup(g, bsupu, bsupv, zeta=static.grid.zeta, nfp=cfg.nfp))
    B2_cart = np.sum(B**2, axis=-1)

    # Skip the magnetic axis surface (coordinate singularity); both representations
    # should agree to numerical precision away from it.
    assert np.allclose(B2_cart[1:], B2_metric[1:], rtol=1e-10, atol=1e-12)

