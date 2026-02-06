import numpy as np
import pytest


def test_vmec_realspace_geom_shapes_and_finite():
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.vmec_tomnsp import vmec_trig_tables
    from vmec_jax.vmec_realspace import vmec_realspace_geom_from_state
    from vmec_jax.wout import read_wout, state_from_wout

    cfg, _ = load_config("examples/data/input.circular_tokamak")
    static = build_static(cfg)
    wout = read_wout("examples/data/wout_circular_tokamak_reference.nc")
    st = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=cfg.ntheta,
        nzeta=cfg.nzeta,
        nfp=cfg.nfp,
        mmax=cfg.mpol - 1,
        nmax=cfg.ntor,
        lasym=cfg.lasym,
    )

    geom = vmec_realspace_geom_from_state(state=st, modes=static.modes, trig=trig)
    assert geom["R"].shape == (cfg.ns, trig.ntheta3, cfg.nzeta)
    assert geom["Z"].shape == (cfg.ns, trig.ntheta3, cfg.nzeta)
    assert np.isfinite(np.asarray(geom["R"])).all()
    assert np.isfinite(np.asarray(geom["Z"])).all()


def test_vmec_half_mesh_jacobian_shapes_and_finite():
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.vmec_jacobian import vmec_half_mesh_jacobian_from_state
    from vmec_jax.vmec_tomnsp import vmec_trig_tables
    from vmec_jax.wout import read_wout, state_from_wout

    cfg, _ = load_config("examples/data/input.circular_tokamak")
    static = build_static(cfg)
    wout = read_wout("examples/data/wout_circular_tokamak_reference.nc")
    st = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=cfg.ntheta,
        nzeta=cfg.nzeta,
        nfp=cfg.nfp,
        mmax=cfg.mpol - 1,
        nmax=cfg.ntor,
        lasym=cfg.lasym,
    )

    jac = vmec_half_mesh_jacobian_from_state(
        state=st,
        modes=static.modes,
        trig=trig,
        s=np.asarray(static.s),
    )
    assert jac.sqrtg.shape == (cfg.ns, trig.ntheta3, cfg.nzeta)
    assert np.isfinite(np.asarray(jac.sqrtg)).all()
