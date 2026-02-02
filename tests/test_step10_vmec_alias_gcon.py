from __future__ import annotations

import numpy as np

from vmec_jax.vmec_constraints import alias_gcon
from vmec_jax.vmec_tomnsp import vmec_trig_tables


def _alias_ref_symmetric(*, ztemp, trig, ntor: int, mpol: int, signgs: int, tcon):
    # Direct translation of `VMEC2000/Sources/General/alias.f` (lasym=False) for testing.
    ns, ntheta3, nzeta = ztemp.shape
    nt2 = trig.ntheta2

    cosmui = np.asarray(trig.cosmui)[:nt2, :mpol]
    sinmui = np.asarray(trig.sinmui)[:nt2, :mpol]
    cosmu = np.asarray(trig.cosmu)[:nt2, :mpol]
    sinmu = np.asarray(trig.sinmu)[:nt2, :mpol]
    cosnv = np.asarray(trig.cosnv)[:, : (ntor + 1)]
    sinnv = np.asarray(trig.sinnv)[:, : (ntor + 1)]

    faccon = np.zeros((mpol,), dtype=float)
    for m in range(1, mpol - 1):
        denom = ((m + 1) * m) ** 2  # xmpq(m+1,1)^2
        faccon[m] = (-0.25 * signgs) / denom

    gcon = np.zeros((ns, ntheta3, nzeta), dtype=float)

    for js in range(ns):
        for m in range(1, mpol - 1):
            work1 = np.zeros((nzeta,), dtype=float)
            work2 = np.zeros((nzeta,), dtype=float)
            for i in range(nt2):
                work1 += ztemp[js, i, :] * cosmui[i, m]
                work2 += ztemp[js, i, :] * sinmui[i, m]

            gcs = np.zeros((ntor + 1,), dtype=float)
            gsc = np.zeros((ntor + 1,), dtype=float)
            if js > 0:
                for n in range(ntor + 1):
                    gcs[n] += tcon[js] * np.sum(work1 * sinnv[:, n])
                    gsc[n] += tcon[js] * np.sum(work2 * cosnv[:, n])

            work3 = np.zeros((nzeta,), dtype=float)
            work4 = np.zeros((nzeta,), dtype=float)
            if js > 0:
                for n in range(ntor + 1):
                    work3 += gcs[n] * sinnv[:, n]
                    work4 += gsc[n] * cosnv[:, n]

            for i in range(nt2):
                gcon[js, i, :] += (work3 * cosmu[i, m] + work4 * sinmu[i, m]) * faccon[m]

    return gcon


def test_step10_alias_gcon_symmetric_matches_reference_loops():
    rng = np.random.default_rng(0)
    ntheta = 10
    nzeta = 8
    nfp = 3
    mpol = 6
    ntor = 3
    ns = 4

    trig = vmec_trig_tables(ntheta=ntheta, nzeta=nzeta, nfp=nfp, mmax=mpol - 1, nmax=ntor, lasym=False)
    ztemp = rng.normal(size=(ns, trig.ntheta3, nzeta))
    tcon = np.linspace(0.0, 0.5, ns)
    signgs = -1

    g_ref = _alias_ref_symmetric(ztemp=ztemp, trig=trig, ntor=ntor, mpol=mpol, signgs=signgs, tcon=tcon)
    g = np.asarray(
        alias_gcon(
            ztemp=ztemp,
            trig=trig,
            ntor=ntor,
            mpol=mpol,
            signgs=signgs,
            tcon=tcon,
            lasym=False,
        )
    )

    assert g.shape == g_ref.shape
    # Exact match should hold to floating point noise since both are discrete sums.
    assert np.max(np.abs(g - g_ref)) < 1e-12


def _alias_ref_lasym(*, ztemp, trig, ntor: int, mpol: int, signgs: int, tcon):
    # Direct translation of `alias_par` (lasym=True) for testing.
    ns, ntheta3, nzeta = ztemp.shape
    ntheta1 = trig.ntheta1
    nt2 = trig.ntheta2

    assert ntheta3 == ntheta1

    cosmui = np.asarray(trig.cosmui)[:nt2, :mpol]
    sinmui = np.asarray(trig.sinmui)[:nt2, :mpol]
    cosmu = np.asarray(trig.cosmu)[:nt2, :mpol]
    sinmu = np.asarray(trig.sinmu)[:nt2, :mpol]
    cosnv = np.asarray(trig.cosnv)[:, : (ntor + 1)]
    sinnv = np.asarray(trig.sinnv)[:, : (ntor + 1)]

    faccon = np.zeros((mpol,), dtype=float)
    for m in range(1, mpol - 1):
        denom = ((m + 1) * m) ** 2
        faccon[m] = (-0.25 * signgs) / denom

    kk_map = (nzeta - np.arange(nzeta)) % nzeta

    gcons = np.zeros((ns, ntheta3, nzeta), dtype=float)
    gcona = np.zeros((ns, ntheta3, nzeta), dtype=float)

    for js in range(ns):
        for m in range(1, mpol - 1):
            work1 = np.zeros((nzeta,), dtype=float)
            work2 = np.zeros((nzeta,), dtype=float)
            work3 = np.zeros((nzeta,), dtype=float)
            work4 = np.zeros((nzeta,), dtype=float)
            for i in range(nt2):
                work1 += ztemp[js, i, :] * cosmui[i, m]
                work2 += ztemp[js, i, :] * sinmui[i, m]
                ir = 0 if i == 0 else (ntheta1 - i)
                work3 += ztemp[js, ir, kk_map] * cosmui[i, m]
                work4 += ztemp[js, ir, kk_map] * sinmui[i, m]

            gcs = np.zeros((ntor + 1,), dtype=float)
            gsc = np.zeros((ntor + 1,), dtype=float)
            gss = np.zeros((ntor + 1,), dtype=float)
            gcc = np.zeros((ntor + 1,), dtype=float)
            if js > 0:
                for n in range(ntor + 1):
                    gcs[n] += 0.5 * tcon[js] * np.sum((work1 - work3) * sinnv[:, n])
                    gsc[n] += 0.5 * tcon[js] * np.sum((work2 - work4) * cosnv[:, n])
                    gss[n] += 0.5 * tcon[js] * np.sum((work2 + work4) * sinnv[:, n])
                    gcc[n] += 0.5 * tcon[js] * np.sum((work1 + work3) * cosnv[:, n])

            w3 = np.zeros((nzeta,), dtype=float)
            w4 = np.zeros((nzeta,), dtype=float)
            w1 = np.zeros((nzeta,), dtype=float)
            w2 = np.zeros((nzeta,), dtype=float)
            if js > 0:
                for n in range(ntor + 1):
                    w3 += gcs[n] * sinnv[:, n]
                    w4 += gsc[n] * cosnv[:, n]
                    w1 += gcc[n] * cosnv[:, n]
                    w2 += gss[n] * sinnv[:, n]

            for i in range(nt2):
                gcons[js, i, :] += (w3 * cosmu[i, m] + w4 * sinmu[i, m]) * faccon[m]
                gcona[js, i, :] += (w1 * cosmu[i, m] + w2 * sinmu[i, m]) * faccon[m]

    # Extend into theta=pi..2pi.
    for js in range(ns):
        for i in range(nt2, ntheta1):
            ir = ntheta1 - i
            gcons[js, i, :] = -gcons[js, ir, kk_map] + gcona[js, ir, kk_map]

    gcons[:, :nt2, :] = gcons[:, :nt2, :] + gcona[:, :nt2, :]
    return gcons


def test_step10_alias_gcon_lasym_matches_reference_loops():
    rng = np.random.default_rng(0)
    ntheta = 10
    nzeta = 7
    nfp = 2
    mpol = 6
    ntor = 2
    ns = 3

    trig = vmec_trig_tables(ntheta=ntheta, nzeta=nzeta, nfp=nfp, mmax=mpol - 1, nmax=ntor, lasym=True)
    ztemp = rng.normal(size=(ns, trig.ntheta3, nzeta))
    tcon = np.linspace(0.0, 0.3, ns)
    signgs = 1

    g_ref = _alias_ref_lasym(ztemp=ztemp, trig=trig, ntor=ntor, mpol=mpol, signgs=signgs, tcon=tcon)
    g = np.asarray(
        alias_gcon(
            ztemp=ztemp,
            trig=trig,
            ntor=ntor,
            mpol=mpol,
            signgs=signgs,
            tcon=tcon,
            lasym=True,
        )
    )
    assert g.shape == g_ref.shape
    assert np.max(np.abs(g - g_ref)) < 1e-12
