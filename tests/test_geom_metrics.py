import numpy as np


def test_geom_metrics_runs(load_case_li383_low_res):
    """Smoke test: metric/Jacobian kernel runs and is finite."""
    cfg, indata, static, bdy, st0 = load_case_li383_low_res

    from vmec_jax.geom import eval_geom

    g = eval_geom(st0, static)

    sqrtg = np.asarray(g.sqrtg)
    assert np.all(np.isfinite(sqrtg))

    # For a sensible initial guess, Jacobian magnitude should not be tiny everywhere.
    assert np.max(np.abs(sqrtg)) > 1e-6
