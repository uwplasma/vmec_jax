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


def test_vmec_realspace_analysis_roundtrip():
    pytest.importorskip("jax")

    from vmec_jax.modes import vmec_mode_table
    from vmec_jax.vmec_realspace import vmec_realspace_analysis, vmec_realspace_synthesis
    from vmec_jax.vmec_tomnsp import vmec_trig_tables

    mpol = 4
    ntor = 3
    ns = 3
    ntheta = 10
    nzeta = 8

    modes = vmec_mode_table(mpol, ntor)
    trig = vmec_trig_tables(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=1,
        mmax=mpol - 1,
        nmax=ntor,
        lasym=False,
    )

    rng = np.random.default_rng(0)
    coeff_cos = rng.normal(scale=0.2, size=(ns, modes.K))
    coeff_sin = np.zeros_like(coeff_cos)

    f_cos = vmec_realspace_synthesis(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=modes, trig=trig)
    c2, s2 = vmec_realspace_analysis(f=f_cos, modes=modes, trig=trig, parity="cos")

    assert np.allclose(np.asarray(c2), coeff_cos, rtol=1e-10, atol=1e-10)
    assert np.allclose(np.asarray(s2), coeff_sin, rtol=1e-10, atol=1e-10)

    # sin-only round-trip (with the m=n=0 sin term forced to zero).
    coeff_cos = np.zeros_like(coeff_cos)
    coeff_sin = rng.normal(scale=0.2, size=(ns, modes.K))
    coeff_sin[:, 0] = 0.0

    f_sin = vmec_realspace_synthesis(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=modes, trig=trig)
    c2, s2 = vmec_realspace_analysis(f=f_sin, modes=modes, trig=trig, parity="sin")

    assert np.allclose(np.asarray(c2), coeff_cos, rtol=1e-10, atol=1e-10)
    assert np.allclose(np.asarray(s2), coeff_sin, rtol=1e-10, atol=1e-10)
