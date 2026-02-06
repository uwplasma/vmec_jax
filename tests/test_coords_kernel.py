import numpy as np

from vmec_jax.config import load_config
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.static import build_static
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.coords import eval_coords
from vmec_jax.fourier import eval_fourier


def test_boundary_matches_state_surface():
    cfg, indata = load_config("examples/data/input.li383_low_res")
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    state0 = initial_guess_from_boundary(static, bdy, indata)

    coords = eval_coords(state0, static.basis)
    R = np.asarray(coords.R)
    Z = np.asarray(coords.Z)

    Rb = np.asarray(eval_fourier(bdy.R_cos, bdy.R_sin, static.basis))
    Zb = np.asarray(eval_fourier(bdy.Z_cos, bdy.Z_sin, static.basis))

    # Should be identical (up to floating point) since s=1 uses boundary coefficients.
    assert np.max(np.abs(R[-1] - Rb)) < 1e-12
    assert np.max(np.abs(Z[-1] - Zb)) < 1e-12
