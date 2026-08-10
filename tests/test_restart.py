"""Hot restart from wout files (``restart_from`` / RESTART_WOUT / --restart).

Measured baselines (2026-08-10, local CPU, x64):

- solovev (ns=11, ftol 1e-14): cold 215 iterations; restart from its own
  converged VMEX wout: **1** iteration.
- cth_like_fixed_bdy (ns=15, ftol 1e-14): cold ~435 iterations; restart
  from the *VMEC2000-written* golden wout: **1** iteration, ``wb`` matching
  the golden file to 1.4e-14.
- solovev ns=11 wout -> ns=25 solve (ftol 1e-12): 150 vs 223 cold
  iterations (radial ``interp.f`` transfer error dominates).
- free-boundary cth_like (NS 7/15): restart from the converged wout skips
  the coarse rung and re-converges in 99 vs 340 final-stage iterations
  (vacuum activation repeats by reset-file semantics, so it is not
  1-iteration like fixed boundary).

The reconstruction itself (``state_from_wout``) is exact: R/Z round-trip at
machine precision and lambda on every interior surface, because the wrout.f
output maps are inverted rather than approximated.  Only the odd-m lambda
axis row differs (synthesis never reads it — ``jmin1`` rules).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from vmex.core import solver  # noqa: E402
from vmex.core.errors import VmecInputError  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.multigrid import (  # noqa: E402
    solve_free_boundary_multigrid, solve_multigrid,
)
from vmex.core.restart import (  # noqa: E402
    restart_state, skip_ladder_rungs, state_from_wout,
)
from vmex.core.wout import read_wout, wout_from_state, write_wout  # noqa: E402
from tests.conftest import resolve_golden_dir  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
FB_DECK = DATA / "input.cth_like_free_bdy_lasym_small"
FB_MGRID = DATA / "mgrid_cth_like_lasym_small.nc"
FB_CONV_DECK = DATA / "input.cth_like_free_bdy"
FB_CONV_MGRID = DATA / "mgrid_cth_like.nc"


@pytest.fixture(scope="module")
def solovev_case(tmp_path_factory):
    """Converged solovev solve + its wout file (shared across tests)."""
    inp = VmecInput.from_file(DATA / "input.solovev")
    cold = solver.solve(inp, ftol=1e-14, max_iterations=5000)
    assert cold.converged
    data = wout_from_state(
        inp=inp, state=cold.state,
        fsqr=cold.fsqr, fsqz=cold.fsqz, fsql=cold.fsql,
    )
    path = write_wout(
        tmp_path_factory.mktemp("restart") / "wout_solovev.nc", data
    )
    return inp, cold, path


# ---------------------------------------------------------------------------
# Exact state reconstruction + round-trip restart (VMEX wout)
# ---------------------------------------------------------------------------


def test_lambda_half_mesh_inversion_is_exact():
    """lambda_full_mesh_from_wout inverts lambda_wout_from_full_mesh."""
    from vmex.core import postprocess as pp

    rng = np.random.default_rng(7)
    ns, m_modes = 17, np.array([0, 1, 2, 3, 4, 5])
    s = np.linspace(0.0, 1.0, ns)
    phipf = 0.5 + 0.1 * s
    lam = rng.normal(size=(ns, m_modes.size))
    lam[0, m_modes >= 2] = 0.0        # VMEC never evolves axis lambda, m >= 2
    half = pp.lambda_wout_from_full_mesh(
        lam_full=lam, m_modes=m_modes, s=s, phipf_internal=phipf,
        lamscale=1.7,
    )
    back = pp.lambda_full_mesh_from_wout(
        lmns_half=half, m_modes=m_modes, s=s, phipf_internal=phipf,
        lamscale=1.7,
    )
    # every interior surface is reconstructed exactly; the m <= 1 axis row
    # follows the forward map's substitution (copy of surface 2).
    np.testing.assert_allclose(back[1:], lam[1:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        back[0, m_modes >= 2], lam[0, m_modes >= 2], rtol=0, atol=0
    )
    # degenerate cases return zeros, matching the forward map
    assert not np.any(pp.lambda_full_mesh_from_wout(
        lmns_half=half, m_modes=m_modes, s=s, phipf_internal=phipf,
        lamscale=0.0,
    ))


def test_state_from_wout_reconstructs_converged_state(solovev_case):
    inp, cold, path = solovev_case
    state = state_from_wout(path, inp=inp)
    for name in ("R_cos", "Z_sin"):
        np.testing.assert_allclose(
            np.asarray(getattr(state, name)),
            np.asarray(getattr(cold.state, name)),
            rtol=0, atol=1e-14,
        )
    # lambda: exact on every interior surface (axis row is never synthesized
    # for odd m; see the module docstring).
    np.testing.assert_allclose(
        np.asarray(state.L_sin)[1:], np.asarray(cold.state.L_sin)[1:],
        rtol=0, atol=1e-13,
    )


def test_round_trip_restart_converges_in_a_few_iterations(solovev_case):
    """Same-deck restart from a converged wout: <= 5% of the cold count."""
    inp, cold, path = solovev_case
    hot = solver.solve(inp, ftol=1e-14, restart_from=path)
    assert hot.converged
    assert hot.iterations <= max(3, 0.05 * cold.iterations), (
        f"restart took {hot.iterations} vs cold {cold.iterations}"
    )  # measured: 1 vs 215
    assert abs(hot.wb / cold.wb - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Cross-code restart (VMEC2000-written golden wout)
# ---------------------------------------------------------------------------


GOLDEN_DIR = resolve_golden_dir()


@pytest.mark.skipif(
    GOLDEN_DIR is None, reason="golden VMEC2000 fixtures unavailable (offline?)"
)
def test_cross_code_restart_from_vmec2000_wout():
    golden = GOLDEN_DIR / "cth_like_fixed_bdy" / "wout_cth_like_fixed_bdy.nc"
    if not golden.exists():
        pytest.skip(f"missing golden file {golden}")
    reference = read_wout(golden)
    inp = VmecInput.from_file(DATA / "input.cth_like_fixed_bdy")
    hot = solver.solve(inp, ftol=1e-14, max_iterations=25000,
                       restart_from=golden)
    assert hot.converged
    # measured: 1 iteration (cold: ~435); generous headroom for BLAS drift
    assert hot.iterations <= 20
    assert abs(hot.wb / reference.wb - 1.0) < 1e-10   # measured 1.4e-14


# ---------------------------------------------------------------------------
# Radial resampling (wout ns != target ns)
# ---------------------------------------------------------------------------


def test_restart_resamples_radially_up_and_down(solovev_case):
    inp, _, path = solovev_case
    inp25 = dataclasses.replace(inp, ns_array=[25])
    cold = solver.solve(inp25, ftol=1e-12, max_iterations=5000)
    hot = solver.solve(inp25, ftol=1e-12, max_iterations=5000,
                       restart_from=path)
    assert hot.converged
    assert hot.iterations < cold.iterations          # measured 150 vs 223
    assert abs(hot.wb / cold.wb - 1.0) < 1e-9
    # downsampling also works (ns 11 -> 7)
    inp7 = dataclasses.replace(inp, ns_array=[7])
    down = solver.solve(inp7, ftol=1e-12, max_iterations=5000,
                        restart_from=path)
    assert down.converged
    assert int(np.shape(down.state.R_cos)[0]) == 7


# ---------------------------------------------------------------------------
# Multigrid rung skipping
# ---------------------------------------------------------------------------


def test_skip_ladder_rungs_policy():
    assert skip_ladder_rungs(np.array([5, 9, 15]), 9) == 1    # meets rung 2
    assert skip_ladder_rungs(np.array([5, 9, 15]), 10) == 2   # between rungs
    assert skip_ladder_rungs(np.array([5, 9, 15]), 3) == 0    # coarser: full
    assert skip_ladder_rungs(np.array([5, 9, 15]), 40) == 2   # finer: last
    assert skip_ladder_rungs(np.array([15]), 15) == 0


def test_multigrid_restart_skips_coarse_rungs(solovev_case):
    inp, cold, path = solovev_case
    lines: list[str] = []
    result = solve_multigrid(
        inp, ns_array=[7, 11], ftol_array=[1e-8, 1e-14],
        niter_array=[2000, 2000], verbose=True,
        emit=lambda value="", end="\n": lines.append(str(value)),
        restart_from=path,
    )
    banners = [ln for ln in lines if re.search(r"NS =\s+\d+ NO\. FOURIER", ln)]
    assert len(banners) == 1 and "NS =   11" in banners[0]
    assert result.converged
    assert result.iterations <= max(3, 0.05 * cold.iterations)


def test_restart_from_and_initial_state_are_exclusive(solovev_case):
    inp, cold, path = solovev_case
    with pytest.raises(ValueError, match="not both"):
        solve_multigrid(inp, restart_from=path, initial_state=cold.state)
    with pytest.raises(ValueError, match="not both"):
        solver.solve(inp, restart_from=path, initial_state=cold.state)
    fb_inp = VmecInput.from_file(FB_DECK)
    with pytest.raises(ValueError, match="not both"):
        solve_free_boundary_multigrid(
            fb_inp, restart_from=path, initial_state=cold.state,
        )
    # a prebuilt RunSetup has no deck to remap a wout against
    from vmex.core.setup import run_setup
    from vmex.core.solver import resolution_from_input

    resolution = resolution_from_input(inp)
    setup = run_setup(inp, resolution)
    with pytest.raises(ValueError, match="VmecInput"):
        solver.solve(setup, resolution, restart_from=path)


# ---------------------------------------------------------------------------
# Free boundary
# ---------------------------------------------------------------------------


def test_free_boundary_restart_seeds_forces_and_free_edge(tmp_path):
    """FB restart: first residual orders below cold, free edge carried."""
    inp = VmecInput.from_file(FB_DECK)
    cold = solve_free_boundary_multigrid(
        inp, mgrid_path=FB_MGRID, ftol_array=[1e-8], niter_array=[200],
        raise_on_max_iterations=False, verbose=False,
    )
    data = wout_from_state(
        inp=inp, state=cold.state, fsqr=cold.fsqr, fsqz=cold.fsqz,
        fsql=cold.fsql, converged=False, vacuum_output=cold.vacuum,
    )
    path = write_wout(tmp_path / "wout_fb.nc", data)
    hot = solve_free_boundary_multigrid(
        inp, mgrid_path=FB_MGRID, ftol_array=[1e-8], niter_array=[10],
        raise_on_max_iterations=False, verbose=False, restart_from=path,
    )
    # forces are evaluated on the seeded state: measured 3.3e-3 vs 0.235 cold
    assert hot.fsq_history[0, 0] < 0.05 * cold.fsq_history[0, 0]
    # the wout's free edge seeds the run (fixed-boundary clamping would snap
    # back to the deck's LCFS): the wout edge is the cold run's evolved edge.
    seed = state_from_wout(path, inp=inp)
    np.testing.assert_allclose(
        np.asarray(seed.R_cos[-1]), np.asarray(cold.state.R_cos[-1]),
        rtol=0, atol=1e-13,
    )
    assert not np.allclose(
        np.asarray(seed.R_cos[-1]), np.asarray(seed.R_cos[-1]) * 0.0
    )


@pytest.mark.full
def test_free_boundary_converged_restart_reconverges(tmp_path):
    """Converged FB restart skips the coarse rung and re-converges fast."""
    if not FB_CONV_MGRID.exists():
        pytest.skip("mgrid_cth_like.nc asset unavailable (tools/fetch_assets.py)")
    inp = VmecInput.from_file(FB_CONV_DECK)
    kwargs = dict(ns_array=[7, 15], ftol_array=[1e-8, 1e-10],
                  niter_array=[1000, 2500], mgrid_path=FB_CONV_MGRID,
                  verbose=False)
    cold = solve_free_boundary_multigrid(inp, **kwargs)
    assert cold.converged
    data = wout_from_state(
        inp=inp, state=cold.state, fsqr=cold.fsqr, fsqz=cold.fsqz,
        fsql=cold.fsql, vacuum_output=cold.vacuum,
    )
    path = write_wout(tmp_path / "wout_fb_cth.nc", data)
    hot = solve_free_boundary_multigrid(inp, restart_from=path, **kwargs)
    assert hot.converged
    # measured 99 vs 340 final-stage iterations (vacuum turn-on repeats)
    assert hot.iterations < 0.5 * cold.iterations
    assert abs(hot.wb / cold.wb - 1.0) < 1e-5        # measured 3.1e-7


# ---------------------------------------------------------------------------
# Error cases + input/CLI surface
# ---------------------------------------------------------------------------


def test_missing_restart_file_raises(solovev_case):
    inp, _, _ = solovev_case
    with pytest.raises(VmecInputError, match="not found"):
        solver.solve(inp, restart_from="wout_does_not_exist.nc")


def test_incompatible_lasym_and_nfp_raise(solovev_case):
    inp, _, path = solovev_case
    wout = read_wout(path)
    zeros = np.zeros_like(wout.rmnc)
    asym = dataclasses.replace(
        wout, lasym=True, rmns=zeros, zmnc=zeros, lmnc=zeros,
        raxis_cs=np.zeros(inp.ntor + 1), zaxis_cc=np.zeros(inp.ntor + 1),
    )
    with pytest.raises(VmecInputError, match="symmetric"):
        state_from_wout(asym, inp=inp)
    with pytest.raises(VmecInputError, match="NFP"):
        state_from_wout(dataclasses.replace(wout, nfp=3), inp=inp)
    # a symmetric wout may seed an LASYM deck (asym blocks start at zero)
    inp_asym = dataclasses.replace(inp, lasym=True)
    state = state_from_wout(wout, inp=inp_asym)
    assert not np.any(np.asarray(state.R_sin))
    assert np.any(np.asarray(state.R_cos))


def test_restart_state_rejects_bad_sources(solovev_case):
    inp, cold, _ = solovev_case
    with pytest.raises(VmecInputError, match="unsupported restart source"):
        restart_state(object(), inp)
    truncated = dataclasses.replace(
        cold.state, R_cos=cold.state.R_cos[:, :3],
    )
    with pytest.raises(VmecInputError, match="modes"):
        restart_state(truncated, inp)
    # SolveResult and SpectralState both normalize to the same seed
    a = restart_state(cold, inp, ns=11)
    b = restart_state(cold.state, inp, ns=11)
    np.testing.assert_array_equal(np.asarray(a.R_cos), np.asarray(b.R_cos))
    # bare states resample radially too
    c = restart_state(cold.state, inp, ns=7)
    assert int(np.shape(c.R_cos)[0]) == 7


def _regrid_input(inp, *, mpol: int, ntor: int):
    """Rebuild a deck at a different mpol/ntor (arrays must match shapes)."""
    old_ntor = int(inp.ntor)
    shape = (2 * ntor + 1, mpol)

    def pad(grid):
        out = np.zeros(shape)
        g = np.asarray(grid, dtype=float)
        n_lo, n_hi = max(-ntor, -old_ntor), min(ntor, old_ntor)
        cols = min(mpol, g.shape[1])
        out[n_lo + ntor:n_hi + ntor + 1, :cols] = (
            g[n_lo + old_ntor:n_hi + old_ntor + 1, :cols]
        )
        return out

    def axis(vec):
        out = np.zeros(ntor + 1)
        v = np.asarray(vec, dtype=float).ravel()
        out[: min(v.size, ntor + 1)] = v[: min(v.size, ntor + 1)]
        return out

    return dataclasses.replace(
        inp, mpol=mpol, ntor=ntor,
        rbc=pad(inp.rbc), zbs=pad(inp.zbs), rbs=pad(inp.rbs), zbc=pad(inp.zbc),
        raxis_c=axis(inp.raxis_c), zaxis_s=axis(inp.zaxis_s),
        raxis_s=axis(inp.raxis_s), zaxis_c=axis(inp.zaxis_c),
    )


def test_mode_table_superset_and_subset_remap(solovev_case):
    """mpol/ntor changes: zero-fill new modes, truncate removed ones."""
    inp, cold, path = solovev_case
    from vmex.core.fourier import mode_table

    bigger = _regrid_input(inp, mpol=8, ntor=2)
    state = state_from_wout(path, inp=bigger)
    modes = mode_table(8, 2)
    assert int(state.R_cos.shape[1]) == modes.mnmax
    # shared (m, n=0) columns carry the wout data; new columns are zero
    shared = [k for k, (m, n) in enumerate(zip(modes.m, modes.n))
              if n == 0 and m < int(inp.mpol)]
    fresh = [k for k in range(modes.mnmax) if k not in shared]
    assert np.any(np.asarray(state.R_cos)[:, shared])
    assert not np.any(np.asarray(state.R_cos)[:, fresh])
    smaller = _regrid_input(inp, mpol=3, ntor=0)
    state_small = state_from_wout(path, inp=smaller)
    assert int(state_small.R_cos.shape[1]) == mode_table(3, 0).mnmax


def test_indata_restart_wout_key_and_writers(tmp_path):
    deck = tmp_path / "input.key"
    src = (DATA / "input.solovev").read_text()
    deck.write_text(src.replace(
        "&INDATA", "&INDATA\nRESTART_WOUT = 'wout_seed.nc'", 1))
    inp = VmecInput.from_file(deck)
    assert inp.restart_wout == "wout_seed.nc"
    # both writers keep the key; a cold deck omits it
    out = tmp_path / "input.rewritten"
    inp.to_indata(out)
    assert "RESTART_WOUT" in out.read_text()
    assert VmecInput.from_file(out).restart_wout == "wout_seed.nc"
    jpath = tmp_path / "input.json"
    inp.to_json(jpath)
    assert VmecInput.from_file(jpath).restart_wout == "wout_seed.nc"
    cold = VmecInput.from_file(DATA / "input.solovev")
    cold_out = tmp_path / "input.cold"
    cold.to_indata(cold_out)
    assert "RESTART_WOUT" not in cold_out.read_text()


def test_cli_restart_flag_wins_over_deck_key(tmp_path):
    from vmex.core.cli import _restart_source, build_parser

    deck = tmp_path / "input.case"
    deck.write_text("&INDATA\n/\n")
    inp_key = dataclasses.replace(
        VmecInput.from_file(DATA / "input.solovev"),
        restart_wout="sub/wout_deck.nc",
    )
    args = build_parser().parse_args([str(deck)])
    # deck key: resolved relative to the input file's directory
    assert _restart_source(args, inp_key, deck) == tmp_path / "sub/wout_deck.nc"
    # absolute deck key: used verbatim
    inp_abs = dataclasses.replace(inp_key, restart_wout="/abs/wout.nc")
    assert _restart_source(args, inp_abs, deck) == Path("/abs/wout.nc")
    # CLI flag: wins, resolved like any command-line path
    args = build_parser().parse_args([str(deck), "--restart", "wout_cli.nc"])
    assert _restart_source(args, inp_key, deck) == Path("wout_cli.nc")
    # neither: cold start
    args = build_parser().parse_args([str(deck)])
    assert _restart_source(args, VmecInput.from_file(deck), deck) is None


def test_cli_end_to_end_restart(tmp_path, solovev_case, capsys):
    from vmex.core.cli import main

    _, cold, path = solovev_case
    deck = tmp_path / "input.solovev_hot"
    deck.write_text((DATA / "input.solovev").read_text())
    rc = main([str(deck), "--restart", str(path), "--outdir", str(tmp_path)])
    assert rc == 0
    assert "RESTART : seeding from" in capsys.readouterr().out
    wout = read_wout(tmp_path / "wout_solovev_hot.nc")
    assert wout.ier_flag == 0
    assert wout.niter <= max(3, 0.05 * cold.iterations)
    assert abs(wout.wb / cold.wb - 1.0) < 1e-12
