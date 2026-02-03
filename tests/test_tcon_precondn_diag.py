from __future__ import annotations

import numpy as np
import pytest


def test_tcon_from_bcovar_precondn_diag_matches_reference_formula():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.vmec_constraints import tcon_from_bcovar_precondn_diag
    from vmec_jax.vmec_tomnsp import vmec_trig_tables

    # Small synthetic problem with well-conditioned norms.
    ns = 6
    ntheta = 8
    nzeta = 4
    nfp = 1
    mmax = 4
    nmax = 2
    lasym = False

    s = jnp.linspace(0.0, 1.0, ns)
    trig = vmec_trig_tables(ntheta=ntheta, nzeta=nzeta, nfp=nfp, mmax=mmax, nmax=nmax, lasym=lasym, dtype=jnp.float64)

    # Shapes (ns, ntheta3, nzeta).
    bsq = jnp.ones((ns, int(trig.ntheta3), nzeta), dtype=jnp.float64) * 2.0
    r12 = jnp.ones_like(bsq) * 1.3
    sqrtg = jnp.ones_like(bsq) * 0.9
    ru12 = jnp.ones_like(bsq) * 0.7
    zu12 = jnp.ones_like(bsq) * 0.4

    # ru0/zu0 should vary on angles so norms are nontrivial.
    th = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, int(trig.ntheta3), endpoint=False))
    ze = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False))
    ru0 = (1.0 + 0.1 * jnp.sin(th)[:, None] + 0.05 * jnp.cos(ze)[None, :])[None, :, :] * jnp.ones((ns, 1, 1))
    zu0 = (0.9 + 0.08 * jnp.cos(th)[:, None] + 0.03 * jnp.sin(ze)[None, :])[None, :, :] * jnp.ones((ns, 1, 1))

    tcon0 = 0.3
    out = tcon_from_bcovar_precondn_diag(
        tcon0=tcon0,
        trig=trig,
        s=s,
        signgs=1,
        lasym=lasym,
        bsq=bsq,
        r12=r12,
        sqrtg=sqrtg,
        ru12=ru12,
        zu12=zu12,
        ru0=ru0,
        zu0=zu0,
    )

    out_np = np.asarray(out)
    assert out_np.shape == (ns,)
    assert np.isfinite(out_np).all()

    # Reference computation (numpy), mirroring bcovar.f scaling and the reduced precondn diagonal.
    hs = float(np.asarray(s[1] - s[0]))
    ohs = 1.0 / hs
    pfactor = -4.0 * float(trig.r0scale) ** 2
    wint = np.asarray(trig.cosmui3[:, 0]) / float(np.asarray(trig.mscale[0]))
    wint3 = wint[None, :, None] * np.ones((1, 1, nzeta))

    ptau = (pfactor * (np.asarray(r12) ** 2) * np.asarray(bsq) * wint3) / np.asarray(sqrtg)
    ax_r = np.sum(ptau * ((np.asarray(zu12) * ohs) ** 2), axis=(1, 2))
    ax_z = np.sum(ptau * ((np.asarray(ru12) * ohs) ** 2), axis=(1, 2))
    ax_r[0] = 0.0
    ax_z[0] = 0.0
    ard1 = ax_r + np.concatenate([ax_r[1:], np.zeros((1,))])
    azd1 = ax_z + np.concatenate([ax_z[1:], np.zeros((1,))])

    arnorm = np.sum((np.asarray(ru0) ** 2) * wint3, axis=(1, 2))
    aznorm = np.sum((np.asarray(zu0) ** 2) * wint3, axis=(1, 2))

    tcon0_clamped = min(abs(tcon0), 1.0)
    ns_f = float(ns)
    tcon_mul = tcon0_clamped * (1.0 + ns_f * (1.0 / 60.0 + ns_f / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)

    ref = np.zeros((ns,), dtype=float)
    for js in range(1, ns - 1):
        ref[js] = min(abs(ard1[js]) / arnorm[js], abs(azd1[js]) / aznorm[js]) * (tcon_mul * (32.0 * hs) ** 2)
    ref[-1] = 0.5 * ref[-2]

    assert np.allclose(out_np, ref, rtol=2e-12, atol=2e-12)

