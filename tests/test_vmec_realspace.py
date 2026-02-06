from __future__ import annotations

import numpy as np

from vmec_jax._compat import enable_x64
from vmec_jax.fourier import build_helical_basis, eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys
from vmec_jax.modes import vmec_mode_table
from vmec_jax.vmec_realspace import (
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
    vmec_realspace_synthesis_dzeta_phys,
)
from vmec_jax.vmec_tomnsp import vmec_angle_grid, vmec_trig_tables


def test_vmec_realspace_matches_eval_fourier():
    enable_x64()

    rng = np.random.default_rng(1)
    ns = 3
    mpol = 5
    ntor = 3
    nfp = 2
    ntheta = 16
    nzeta = 12

    modes = vmec_mode_table(mpol, ntor)
    grid = vmec_angle_grid(ntheta=ntheta, nzeta=nzeta, nfp=nfp, lasym=False)
    basis = build_helical_basis(modes, grid)

    coeff_cos = rng.standard_normal((ns, modes.m.size))
    coeff_sin = rng.standard_normal((ns, modes.m.size))

    trig = vmec_trig_tables(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=nfp,
        mmax=int(np.max(modes.m)),
        nmax=int(np.max(np.abs(modes.n))),
        lasym=False,
        dtype=np.float64,
    )

    ref = eval_fourier(coeff_cos, coeff_sin, basis)
    real = vmec_realspace_synthesis(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=modes, trig=trig)
    np.testing.assert_allclose(real, ref, rtol=1e-10, atol=1e-10)

    ref_dtheta = eval_fourier_dtheta(coeff_cos, coeff_sin, basis)
    real_dtheta = vmec_realspace_synthesis_dtheta(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=modes, trig=trig)
    np.testing.assert_allclose(real_dtheta, ref_dtheta, rtol=1e-10, atol=1e-10)

    ref_dzeta = eval_fourier_dzeta_phys(coeff_cos, coeff_sin, basis)
    real_dzeta = vmec_realspace_synthesis_dzeta_phys(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=modes, trig=trig)
    np.testing.assert_allclose(real_dzeta, ref_dzeta, rtol=1e-10, atol=1e-10)
