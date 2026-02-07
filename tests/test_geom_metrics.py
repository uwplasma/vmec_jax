import numpy as np


def test_geom_metrics_runs():
    """Smoke test: metric/Jacobian kernel runs and is finite."""
    pytest = __import__("pytest")
    pytest.importorskip("jax")

    from vmec_jax.boundary import BoundaryCoeffs
    from vmec_jax.config import VMECConfig
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.namelist import InData
    from vmec_jax.static import build_static

    cfg = VMECConfig(ns=7, mpol=3, ntor=0, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=12, nzeta=3)
    static = build_static(cfg)
    K = int(static.modes.K)
    Rcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)
    # (m,n)=(0,0) and (1,0)
    k00 = int(np.where((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0][0])
    k10 = int(np.where((np.asarray(static.modes.m) == 1) & (np.asarray(static.modes.n) == 0))[0][0])
    Rcos[k00] = 3.0
    Rcos[k10] = 1.0
    Zsin[k10] = 0.6
    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=np.zeros_like(Rcos), Z_cos=np.zeros_like(Rcos), Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [3.0], "ZAXIS_CS": [0.0]}, indexed={})
    st0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)

    g = eval_geom(st0, static)

    sqrtg = np.asarray(g.sqrtg)
    assert np.all(np.isfinite(sqrtg))

    # For a sensible initial guess, Jacobian magnitude should not be tiny everywhere.
    assert np.max(np.abs(sqrtg)) > 1e-6
