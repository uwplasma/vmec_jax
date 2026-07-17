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
