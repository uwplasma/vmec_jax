"""A/B tests for :mod:`vmex.core.boozer_tables` vs the host wout engine.

``boozer_input_tables`` promises (see its docstring) wout-convention
single-surface tables computed entirely in JAX: ``bmnc`` and the
``gmnc``/``bsupumnc``/``bsupvmnc``/``bsubsmns`` families matching the host
wout engine at quadrature level (identical grid and mode weights),
``lmns``/``bsub*`` at the wout engine's own half-mesh finite-difference
level, and traced ``iota``/``G``/``I`` equal to the wout
``iotas``/``bvco``/``buco`` rows.  This module checks exactly those claims
on the solovev deck (plus a finite-pressure LASYM deck for the field
tables), jit-compatibility of the whole table construction, and the forward
boundary tangent of the field tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from vmex.core import solver
from vmex.core.boozer_tables import boozer_input_tables, high_order_boozer_input_tables
from vmex.core.input import VmecInput
from vmex.core.omnigenity import boozer_bmnc_high_order, boozer_bmnc_state
from vmex.core.strong_force import lift_high_order_state
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


@pytest.fixture(scope="module")
def lasym_beta_eq():
    """Finite-pressure up-down-asymmetric tokamak (test_stability deck, ns=13)."""
    import dataclasses

    from vmex.core import optimize as opt

    inp = VmecInput.from_file(str(DATA_DIR / "input.up_down_asymmetric_tokamak"))
    inp = dataclasses.replace(
        inp, ns_array=np.array([13]), ftol_array=np.array([1e-10]),
        niter_array=np.array([5000]), am=np.array([1.0, -1.0]),
        pres_scale=5000.0)
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


@pytest.mark.parametrize("deck", ["symmetric_eq", "lasym_beta_eq"])
def test_field_tables_match_wout_rows_at_quadrature_level(deck, request):
    """``gmnc``/``bsup*``/``bsubs*`` equal the wout engine rows exactly.

    ``wrout.f`` writes ``sqrt(g)`` and ``B^u``/``B^v`` unfiltered on the full
    Nyquist set and ``B_s`` (``bss.f``) as the full-mesh average of
    consecutive half-mesh rows, so — unlike the jxbforce-filtered
    ``bsubumnc``/``lmns`` — these tables share the engine's quadrature and
    must match every mode of every interior row at round-off level
    (measured <= 7.8e-15 relative on both decks).
    """
    eq = request.getfixturevalue(deck)
    wout = eq.wout
    ns = int(np.asarray(wout.iotas).shape[0])
    lasym = wout.gmns is not None
    tables = {j: boozer_input_tables(eq.state, eq.runtime, j)
              for j in range(1, ns)}
    xm, xn = tables[1]["xm"], tables[1]["xn"]

    half_mesh = [("gmnc", wout.gmnc), ("bsupumnc", wout.bsupumnc),
                 ("bsupvmnc", wout.bsupvmnc)]
    full_mesh = [("bsubsmns", wout.bsubsmns)]
    if lasym:
        half_mesh += [("gmns", wout.gmns), ("bsupumns", wout.bsupumns),
                      ("bsupvmns", wout.bsupvmns)]
        full_mesh += [("bsubsmnc", wout.bsubsmnc)]

    for key, wout_arr in half_mesh:
        scale = float(np.max(np.abs(np.asarray(wout_arr)[1:])))
        assert scale > 0.0, key                        # the family is populated
        for j in range(1, ns):
            ref, mask = _match_wout_row(wout.xm_nyq, wout.xn_nyq, wout_arr, j,
                                        xm, xn)
            np.testing.assert_allclose(
                np.asarray(tables[j][key])[mask], ref[mask], rtol=0.0,
                atol=1e-13 * scale, err_msg=f"{key} row {j}")

    # B_s is half-mesh native here; the wout file stores the wrout.f full-mesh
    # convention (row i = mean of half-mesh rows i and i+1), and the
    # projection is linear, so the averaged tables must land on the wout rows.
    for key, wout_arr in full_mesh:
        scale = float(np.max(np.abs(np.asarray(wout_arr)[1:-1])))
        assert scale > 0.0, key
        for i in range(1, ns - 1):
            ref, mask = _match_wout_row(wout.xm_nyq, wout.xn_nyq, wout_arr, i,
                                        xm, xn)
            got = 0.5 * (np.asarray(tables[i][key])
                         + np.asarray(tables[i + 1][key]))
            np.testing.assert_allclose(got[mask], ref[mask], rtol=0.0,
                                       atol=1e-13 * scale,
                                       err_msg=f"{key} row {i}")


@pytest.mark.parametrize("deck", ["symmetric_eq", "lasym_solved"])
def test_projection_closes_at_the_grid_nyquist_band(deck, request):
    """The mode set reaches the grid Nyquist, with the self-conjugate weight.

    ``wrout.f`` sizes the wout Nyquist table from the grid itself
    (``mnyq = ntheta1/2``, ``nnyq = nzeta/2``) and halves the closing
    ``cosmui``/``cosnv`` column, because on an even grid that row and column
    are self-conjugate: ``(m, n)`` and ``(m, -n)`` share a single grid basis
    function.  Stopping one mode short of it leaves every surviving mode
    exact, so the loss is silent: on ``input.basic_non_stellsym_simsopt``
    (ntheta = nzeta = 10) it is 2.5% of ``bmnc`` and 2.6% of ``bmns`` in
    relative L2.
    """
    eq = request.getfixturevalue(deck)
    wout, res = eq.wout, eq.runtime.resolution
    nfp, mnyq, nnyq = int(res.nfp), res.ntheta1 // 2, res.nzeta // 2
    j = int(np.asarray(wout.iotas).shape[0]) // 2
    tables = boozer_input_tables(eq.state, eq.runtime, j)
    xm = np.asarray(tables["xm"], dtype=int)
    xn = np.asarray(tables["xn"], dtype=int)

    # the projection carries exactly the wout Nyquist mode table
    assert (xm.max(), np.abs(xn).max()) == (mnyq, nnyq * nfp)
    assert set(zip(xm.tolist(), xn.tolist())) == set(
        zip(np.asarray(wout.xm_nyq, dtype=int).tolist(),
            np.asarray(wout.xn_nyq, dtype=int).tolist()))

    # ... and the band the old m/n limits dropped matches the wout row exactly
    band = (xm > mnyq - 1) | (np.abs(xn) > max(nnyq - 1, 0) * nfp)
    assert band.any()
    for key, wout_arr in (("bmnc", wout.bmnc), ("bmns", wout.bmns)):
        if wout_arr is None:
            continue
        ref, mask = _match_wout_row(wout.xm_nyq, wout.xn_nyq, wout_arr, j, xm, xn)
        atol = 1e-13 * float(np.max(np.abs(ref[mask])))
        assert np.max(np.abs(ref[band])) > 1e3 * atol   # the band is not noise
        np.testing.assert_allclose(np.asarray(tables[key])[band], ref[band],
                                   rtol=0.0, atol=atol, err_msg=key)

    # Both angles fold on the self-conjugate corners, which are real on the
    # grid: their sine partners must be exact zeros, not sin(pi*i) round-off.
    if wout.bmns is not None:
        corner = np.isin(xm, [0, mnyq]) & np.isin(np.abs(xn), [0, nnyq * nfp])
        assert corner.sum() >= 4
        np.testing.assert_array_equal(np.asarray(tables["bmns"])[corner], 0.0)


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


def test_field_tables_jit_matches_eager(symmetric_eq):
    """jit of the field-table construction returns the eager values."""
    eq = symmetric_eq
    j = int(np.asarray(eq.runtime.setup.s_full).shape[0]) // 2
    eager = boozer_input_tables(eq.state, eq.runtime, j)
    jitted = jax.jit(lambda s: boozer_input_tables(s, eq.runtime, j))(eq.state)
    for key in ("gmnc", "bsupumnc", "bsupvmnc", "bsubsmns"):
        scale = float(np.max(np.abs(np.asarray(eager[key]))))
        assert scale > 0.0, key
        np.testing.assert_allclose(np.asarray(jitted[key]),
                                   np.asarray(eager[key]), rtol=1e-9,
                                   atol=1e-13 * scale, err_msg=key)


def test_high_order_boozer_matches_live_state_without_file_io(symmetric_eq):
    eq = symmetric_eq
    ns = int(eq.runtime.resolution.ns)
    row = ns // 2
    surface = float(
        0.5
        * (eq.runtime.setup.s_full[row] + eq.runtime.setup.s_full[row - 1])
    )
    native_state = lift_high_order_state(
        eq.state,
        eq.runtime,
        degree=3,
        max_spans=4,
    )
    with pytest.raises(ValueError, match="projection grids"):
        high_order_boozer_input_tables(native_state, np.sqrt(surface), ntheta=1)
    with pytest.raises(ValueError, match="0 < s <= 1"):
        boozer_bmnc_high_order(native_state, surfaces=[0.0])
    live = boozer_bmnc_state(
        eq.state,
        eq.runtime,
        surfaces=[surface],
        mboz=8,
        nboz=2,
        oversample=1,
    )
    native = boozer_bmnc_high_order(
        native_state,
        surfaces=[surface],
        mboz=8,
        nboz=2,
        ntheta=20,
        nzeta=8,
    )
    live_modes = {
        (int(m), int(n)): index
        for index, (m, n) in enumerate(zip(live["xm_b"], live["xn_b"]))
    }
    native_modes = {
        (int(m), int(n)): index
        for index, (m, n) in enumerate(zip(native["xm_b"], native["xn_b"]))
    }
    common = sorted(live_modes.keys() & native_modes.keys())
    live_B = np.asarray(
        [live["bmnc_b"][0, live_modes[mode]] for mode in common]
    )
    native_B = np.asarray(
        [native["bmnc_b"][0, native_modes[mode]] for mode in common]
    )
    relative = np.linalg.norm(native_B - live_B) / np.linalg.norm(live_B)
    assert relative < 5.0e-4
    np.testing.assert_allclose(native["iota_b"], live["iota_b"], rtol=2e-12)
    np.testing.assert_allclose(native["G_b"], live["G_b"], rtol=5e-4)
    np.testing.assert_allclose(native["I_b"], live["I_b"], rtol=5e-4)


def test_bsupvmnc_jvp_from_boundary_tangent_is_live(symmetric_eq):
    """A boundary-coefficient tangent drives the traceable ``bsupvmnc``.

    Downstream loss-fraction objectives differentiate the field tables in the
    boundary dofs, so the forward tangent must be live: push the ``RBC(0,1)``
    direction through the implicit-function-theorem tangent system
    ``dz = -(dF/dz)^{-1} dF/dp t`` at the solved solovev state, then one
    ``jax.jvp`` of the table construction.  Measured max ``|d(bsupvmnc)|``
    = 4.8e-2 against a table scale of 5.1e-2 (2026-08-22, x64 CPU).
    """
    import dataclasses

    import jax.numpy as jnp

    from vmex.core import implicit as im

    eq = symmetric_eq
    inp = eq.inp
    cfg = im.make_config(inp, ftol=1e-14, max_iterations=3000)
    p0 = im.params_from_input(inp)
    j = int(np.asarray(eq.runtime.setup.s_full).shape[0]) // 2

    rt0 = im.runtime_from_params(p0, cfg)
    mask = im._dof_mask(eq.state, rt0, cfg)
    P = im._dof_projector(cfg, mask)
    edge_mask = im._edge_mask(cfg)
    frozen = jax.lax.stop_gradient(eq.state)
    F = im.residual_fn(cfg, frozen, mask)
    z0 = P(frozen)

    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent = dataclasses.replace(
        zero, rbc=zero.rbc.at[int(inp.ntor), 1].set(1.0))
    b = jax.jvp(lambda prm: F(z0, prm), (p0,), (tangent,))[1]
    dz, _ = im._adjoint_solve(
        lambda t: jax.jvp(lambda zz: F(zz, p0), (z0,), (t,))[1],
        jax.tree.map(jnp.negative, b), cfg)

    def table(zz, prm):
        rt = im.runtime_from_params(prm, cfg)
        state = im._assemble(zz, rt, frozen, P, edge_mask)
        return boozer_input_tables(state, rt, j)["bsupvmnc"]

    _, dot = jax.jvp(table, (z0, p0), (P(dz), tangent))
    dot = np.asarray(dot)
    assert np.all(np.isfinite(dot))
    assert float(np.max(np.abs(dot))) > 1e-3


def test_refine_booz_grids_is_the_identity_at_oversample_one():
    """``oversample = 1`` returns the transform's own grid untouched.

    The refinement multiplies ``booz_xform_jax``'s pinned
    ``2*(2*mboz+1)`` by ``2*(2*nboz+1)`` quadrature, so asking for no
    refinement has to hand back the very same constants and grids rather than
    rebuild an equivalent pair -- the transform reads its Fourier
    normalization back off those counts.
    """
    from vmex.core.omnigenity import _refine_booz_grids

    constants, grids = object(), object()
    same_constants, same_grids = _refine_booz_grids(constants, grids, 1, 3)
    assert same_constants is constants
    assert same_grids is grids
