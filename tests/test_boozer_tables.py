"""A/B tests for :mod:`vmex.core.boozer_tables` vs the host wout engine.

``boozer_input_tables`` promises (see its docstring) wout-convention
single-surface tables computed entirely in JAX: ``bmnc`` matching the host
wout engine at ~1e-10 relative (identical quadrature), ``lmns``/``bsub*``
at the wout engine's own half-mesh finite-difference level, and traced
``iota``/``G``/``I`` equal to the wout ``iotas``/``bvco``/``buco`` rows.
This module checks exactly those claims on the solovev deck, plus
jit-compatibility of the whole table construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from vmex.core import solver
from vmex.core.boozer_tables import boozer_input_tables
from vmex.core.input import VmecInput
from vmex.core.wout import wout_from_state

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solve: run jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def solved():
    inp = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    resolution = solver.resolution_from_input(inp)
    result = solver.solve(inp, resolution, ftol=1e-14, max_iterations=3000,
                          mode="cli")
    assert result.converged
    rt = solver.prepare_runtime(inp, resolution, ftol=1e-14, max_iterations=3000)
    wout = wout_from_state(
        inp=inp, state=result.state, fsqr=float(result.fsqr),
        fsqz=float(result.fsqz), fsql=float(result.fsql),
        niter=int(result.iterations), converged=bool(result.converged),
    )
    ns = int(np.asarray(wout.iotas).shape[0])
    j = ns // 2
    tables = boozer_input_tables(result.state, rt, j)
    return wout, tables, j


@pytest.fixture(scope="module")
def symmetric_eq():
    """Solovev with the objective-lane ``state``/``runtime``/``wout`` handles."""
    from vmex.core import optimize as opt

    eq = opt.solve_equilibrium(VmecInput.from_file(str(DATA_DIR / "input.solovev")))
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def lasym_solved():
    """Small genuinely non-stellarator-symmetric equilibrium (nfp=1, ns=11)."""
    import dataclasses

    from vmex.core import optimize as opt

    inp = VmecInput.from_file(str(DATA_DIR / "input.basic_non_stellsym_simsopt"))
    inp = dataclasses.replace(
        inp, ns_array=np.array([11]), ftol_array=np.array([1e-12]),
        niter_array=np.array([4000]))
    eq = opt.solve_equilibrium(inp)
    assert eq.result.converged
    return eq


def _match_wout_row(wout_xm, wout_xn, wout_2d, j, xm, xn):
    """wout table row ``j`` on the (xm, xn) modes; mask = modes wout carries.

    The traceable tables include every grid-representable mode, which can
    exceed the wout (non-Nyquist) mode table — compare on the overlap.
    """
    index = {(int(m), int(n)): k
             for k, (m, n) in enumerate(zip(np.asarray(wout_xm), np.asarray(wout_xn)))}
    rows = np.asarray(wout_2d)
    out = np.zeros(len(xm))
    mask = np.zeros(len(xm), dtype=bool)
    for k, (m, n) in enumerate(zip(xm, xn)):
        pos = index.get((int(m), int(n)))
        if pos is not None:
            out[k] = rows[j, pos]
            mask[k] = True
    assert mask.any()
    return out, mask


def test_bmnc_matches_wout_engine(solved):
    wout, tables, j = solved
    ref, mask = _match_wout_row(wout.xm_nyq, wout.xn_nyq, wout.bmnc, j,
                                tables["xm"], tables["xn"])
    got = np.asarray(tables["bmnc"])
    np.testing.assert_allclose(got[mask], ref[mask], rtol=1e-8,
                               atol=1e-10 * np.max(np.abs(ref)))


def test_bsub_and_lmns_match_at_half_mesh_fd_level(solved):
    wout, tables, j = solved
    scale = float(np.max(np.abs(np.asarray(wout.bmnc)[j])))
    for key, wout_xm, wout_xn, wout_arr in (
        ("bsubumnc", wout.xm_nyq, wout.xn_nyq, wout.bsubumnc),
        ("bsubvmnc", wout.xm_nyq, wout.xn_nyq, wout.bsubvmnc),
        ("lmns", wout.xm, wout.xn, wout.lmns),
    ):
        ref, mask = _match_wout_row(wout_xm, wout_xn, wout_arr, j,
                                    tables["xm"], tables["xn"])
        got = np.asarray(tables[key])
        # solovev ns=11: the wout engine's own half-mesh FD level (loose)
        np.testing.assert_allclose(got[mask], ref[mask], rtol=5e-2,
                                   atol=5e-3 * max(scale, 1e-30), err_msg=key)


def test_iota_g_i_match_wout_rows(solved):
    wout, tables, j = solved
    assert float(tables["iota"]) == pytest.approx(float(np.asarray(wout.iotas)[j]),
                                                  rel=1e-9)
    assert float(tables["G"]) == pytest.approx(float(np.asarray(wout.bvco)[j]),
                                               rel=1e-8)
    assert float(tables["I"]) == pytest.approx(float(np.asarray(wout.buco)[j]),
                                               abs=1e-10 + 1e-6 * abs(float(np.asarray(wout.buco)[j])))


def test_rz_tables_interpolate_full_mesh_parity(solved):
    """rmnc/zmns are the sqrt(s)-parity half-mesh average of full-mesh rows."""
    wout, tables, j = solved
    s_full = np.linspace(0.0, 1.0, np.asarray(wout.rmnc).shape[0])
    sqrt_s = np.sqrt(s_full)
    xm = np.asarray(tables["xm"])
    for key, wout_arr in (("rmnc", wout.rmnc), ("zmns", wout.zmns)):
        rows = np.asarray(wout_arr)
        ref_jm1, mask = _match_wout_row(wout.xm, wout.xn, rows, j - 1,
                                        tables["xm"], tables["xn"])
        ref_j, _ = _match_wout_row(wout.xm, wout.xn, rows, j,
                                   tables["xm"], tables["xn"])
        even = 0.5 * (ref_jm1 + ref_j)
        # odd-m modes carry the sqrt(s) parity factor through the average
        sq = 0.5 * (ref_jm1 / max(sqrt_s[j - 1], 1e-30) + ref_j / sqrt_s[j])
        s_half = np.sqrt(0.5 * (s_full[j] + s_full[j - 1]))
        ref = np.where(xm % 2 == 1, sq * s_half, even)
        got = np.asarray(tables[key])
        np.testing.assert_allclose(got[mask], ref[mask], rtol=1e-8,
                                   atol=1e-10 * np.max(np.abs(rows)), err_msg=key)


def test_lasym_tables_match_the_wout_asymmetric_families(lasym_solved):
    """``LASYM`` states keep the full circle and both spectral families.

    A non-stellarator-symmetric run stores the whole poloidal grid, so the
    tables must be projected without the symmetric ``[0, pi]`` mirror and must
    carry the wout sine/cosine partners (``rmns``/``zmnc``/``bmns``) that the
    asymmetric Boozer transform consumes.  Each family is compared against the
    independent host wout engine at that engine's own accuracy level.
    """
    eq = lasym_solved
    wout = eq.wout
    ns = int(np.asarray(wout.iotas).shape[0])
    j = ns // 2
    tables = boozer_input_tables(eq.state, eq.runtime, j)
    scale = float(np.max(np.abs(np.asarray(wout.bmnc)[j])))
    assert wout.bmns is not None                       # the deck is asymmetric

    # |B| sine partner: same quadrature as the engine, so it is tight.
    ref, mask = _match_wout_row(wout.xm_nyq, wout.xn_nyq, wout.bmns, j,
                                tables["xm"], tables["xn"])
    got = np.asarray(tables["bmns"])
    assert np.max(np.abs(ref[mask])) > 1.0e-3 * scale  # genuinely asymmetric
    np.testing.assert_allclose(got[mask], ref[mask], rtol=1e-8,
                               atol=1e-10 * scale)

    # Geometry partners come from the sqrt(s)-parity half-mesh average.
    s_full = np.linspace(0.0, 1.0, np.asarray(wout.rmnc).shape[0])
    sqrt_s = np.sqrt(s_full)
    xm = np.asarray(tables["xm"])
    for key, wout_arr in (("rmns", wout.rmns), ("zmnc", wout.zmnc)):
        rows = np.asarray(wout_arr)
        ref_jm1, mask = _match_wout_row(wout.xm, wout.xn, rows, j - 1,
                                        tables["xm"], tables["xn"])
        ref_j, _ = _match_wout_row(wout.xm, wout.xn, rows, j,
                                   tables["xm"], tables["xn"])
        even = 0.5 * (ref_jm1 + ref_j)
        sq = 0.5 * (ref_jm1 / max(sqrt_s[j - 1], 1e-30) + ref_j / sqrt_s[j])
        s_half = np.sqrt(0.5 * (s_full[j] + s_full[j - 1]))
        ref = np.where(xm % 2 == 1, sq * s_half, even)
        np.testing.assert_allclose(np.asarray(tables[key])[mask], ref[mask],
                                   rtol=1e-8, atol=1e-10 * np.max(np.abs(rows)),
                                   err_msg=key)


def _synthesize(cos_table, sin_table, xm, xn, nfp, n_angles=64):
    """Field on a periodic (theta, zeta) grid from a cos/sin mode table."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    theta, zeta = np.meshgrid(angles, angles / int(nfp), indexing="ij")
    phase = (np.asarray(xm, dtype=float)[:, None, None] * theta[None]
             - np.asarray(xn, dtype=float)[:, None, None] * zeta[None])
    return (np.tensordot(np.asarray(cos_table), np.cos(phase), axes=1)
            + np.tensordot(np.asarray(sin_table), np.sin(phase), axes=1))


def _boozer_range_matches_wout(eq, booz, tolerance=0.1):
    """Compare surface |B| extrema in Boozer angles against the wout tables.

    The default tolerance is a tenth of the surface's own ``|B|`` range: at
    the small ``mboz``/``nboz`` used here the Boozer spectrum is truncated,
    which shaves the extrema slightly, while a broken transform misses them
    by much more.
    """
    s_full = np.asarray(eq.runtime.setup.s_full)
    s_half = 0.5 * (s_full[1:] + s_full[:-1])
    row = int(np.argmin(np.abs(s_half - float(booz["s_b"][0])))) + 1
    wout = eq.wout
    nfp = int(eq.runtime.resolution.nfp)
    sine = (np.zeros_like(np.asarray(wout.bmnc)[row]) if wout.bmns is None
            else np.asarray(wout.bmns)[row])
    reference = _synthesize(np.asarray(wout.bmnc)[row], sine,
                            wout.xm_nyq, wout.xn_nyq, nfp)
    boozer = _synthesize(np.asarray(booz["bmnc_b"])[0],
                         np.asarray(booz["bmns_b"])[0],
                         booz["xm_b"], booz["xn_b"], nfp)
    span = float(reference.max() - reference.min())
    assert span > 0.0
    assert abs(boozer.min() - reference.min()) < tolerance * span
    assert abs(boozer.max() - reference.max()) < tolerance * span
    assert float(booz["iota_b"][0]) == pytest.approx(
        float(np.asarray(wout.iotas)[row]), rel=1e-8)
    assert float(booz["G_b"][0]) == pytest.approx(
        float(np.asarray(wout.bvco)[row]), rel=1e-8)
    return row, span


def test_boozer_transform_preserves_the_field_strength_range(symmetric_eq):
    """The Boozer map is a relabelling: |B| extrema on a surface are invariant.

    Re-synthesizing ``|B|`` from ``boozer_bmnc_state`` on a fine Boozer grid
    must recover the same surface minimum and maximum as the wout tables in
    VMEC angles (booz_xform, Sanchez et al. 2000: the transform changes the
    angle labels, not the field).  ``solovev`` is axisymmetric, so every
    ``n != 0`` Boozer harmonic must also vanish.
    """
    from vmex.core import omnigenity as omn

    eq = symmetric_eq
    booz = omn.boozer_bmnc_state(eq.state, eq.runtime, surfaces=[0.5],
                                 mboz=6, nboz=2)
    _boozer_range_matches_wout(eq, booz)
    # Axisymmetric deck: no toroidal harmonics, and no sine family at all.
    assert np.all(np.asarray(booz["xn_b"], dtype=float) == 0.0)
    bmnc = np.asarray(booz["bmnc_b"])[0]
    assert np.max(np.abs(bmnc)) > 0.0
    np.testing.assert_allclose(np.asarray(booz["bmns_b"])[0], 0.0, atol=1e-12)


def test_lasym_boozer_transform_preserves_the_field_strength_range(lasym_solved):
    """``LASYM`` states keep the same invariant through the asymmetric route.

    ``boozer_bmnc_state`` routes them through booz_xform_jax's asymmetric
    transform, so the surface extrema must still match the wout tables and the
    sine family must carry the deck's actual asymmetry rather than vanish.
    """
    from vmex.core import omnigenity as omn
    from vmex.core import optimize as opt

    eq = lasym_solved
    booz = omn.boozer_bmnc_state(eq.state, eq.runtime, surfaces=[0.5],
                                 mboz=6, nboz=6)
    row, span = _boozer_range_matches_wout(eq, booz)
    # The sine family is populated at the amplitude the wout engine reports;
    # a symmetric-only transform would return zeros here.
    boozer_sine = float(np.max(np.abs(np.asarray(booz["bmns_b"])[0])))
    wout_sine = float(np.max(np.abs(np.asarray(eq.wout.bmns)[row])))
    assert wout_sine > 0.0
    assert 0.2 < boozer_sine / wout_sine < 5.0

    # The host booz_xform route on the same surface agrees on the extrema.
    nfp = int(eq.runtime.resolution.nfp)
    host = opt.boozer_modes_from_wout(eq, surfaces=[0.5], mboz=6, nboz=6)
    host_field = _synthesize(np.asarray(host["bmnc_b"])[0],
                             np.asarray(host["bmns_b"])[0],
                             host["xm_b"], host["xn_b"], nfp)
    traced_field = _synthesize(np.asarray(booz["bmnc_b"])[0],
                               np.asarray(booz["bmns_b"])[0],
                               booz["xm_b"], booz["xn_b"], nfp)
    assert abs(host_field.max() - traced_field.max()) < 0.05 * span
    assert abs(host_field.min() - traced_field.min()) < 0.05 * span


def test_tables_are_jittable(solved):
    wout, tables, j = solved
    # re-derive the runtime pieces to jit the full construction
    inp = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    resolution = solver.resolution_from_input(inp)
    result = solver.solve(inp, resolution, ftol=1e-14, max_iterations=3000,
                          mode="cli")
    rt = solver.prepare_runtime(inp, resolution, ftol=1e-14, max_iterations=3000)

    jitted = jax.jit(lambda s: boozer_input_tables(s, rt, j)["bmnc"])
    got = np.asarray(jitted(result.state))
    # jit-vs-eager reassociation noise only
    np.testing.assert_allclose(got, np.asarray(tables["bmnc"]), rtol=1e-6,
                               atol=1e-14)
