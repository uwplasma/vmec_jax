"""Free-boundary radial-ladder, continuation, and hot-restart regressions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmex.core import freeboundary as FB  # noqa: E402
from vmex.core import multigrid as MG  # noqa: E402
from vmex.core.errors import (  # noqa: E402
    BAD_JACOBIAN_FLAG, NORM_TERM_FLAG, SUCCESSFUL_TERM_FLAG,
    VmecJacobianError,
)
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.fourier import mode_table  # noqa: E402
from vmex.core.multigrid import (  # noqa: E402
    interpolate_state, solve_free_boundary_multigrid,
)
from vmex.core.preconditioner_2d import Prec2DConfig  # noqa: E402
from vmex.core.solver import SpectralState, resolution_from_input  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy_lasym_small"
MGRID = REPO / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"
CONV_DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy"
CONV_MGRID = REPO / "examples" / "data" / "mgrid_cth_like.nc"
CONV_WOUT = REPO / "examples" / "data" / "single_grid" / "wout_cth_like_free_bdy.nc"


def _state(ns: int, mnmax: int, value: float) -> SpectralState:
    a = jnp.full((ns, mnmax), value, dtype=jnp.float64)
    return SpectralState(a, a, a, a, a, a)


def test_stage_transfer_carries_vacuum_and_interpolates_final_xc(monkeypatch) -> None:
    """User seeds and increasing/equal grids continue from final xc."""
    inp = VmecInput.from_file(DECK)
    modes = mode_table(int(inp.mpol), int(inp.ntor))
    seed = _state(5, modes.mnmax, 3.0)
    calls = []
    vacua = []

    def fake_stage(_inp, **kwargs):
        assert kwargs["time_step"] == 0.25
        assert kwargs["tcon0"] == 1.2
        assert kwargs["gamma"] == 0.1
        assert kwargs["nstep"] == 17
        assert kwargs["lconm1"] is False
        assert kwargs["precon_type"] == "NONE"
        assert kwargs["prec2d_threshold"] == 3e-7
        if len(calls) < 2:
            assert kwargs["reuse_vacuum_cache"] is False
        else:
            assert kwargs["reuse_vacuum_cache"] is True
            np.testing.assert_array_equal(
                np.asarray(kwargs["constraint_continuation"][0]), [2.0])
        ns = kwargs["resolution"].ns
        incoming = kwargs["initial_state"]
        continuation = kwargs["vacuum_continuation"]
        if calls:
            assert kwargs["residual_continuation"] == (
                float(len(calls)),
                float(len(calls)) + 0.1,
                float(len(calls)) + 0.2,
            )
        else:
            assert kwargs["residual_continuation"] is None
        calls.append((ns, incoming, continuation))
        vacuum = FB.FreeBoundaryState(
            ivac=5 + len(calls), nvacskip=11 + len(calls), nvskip0=9,
            turned_on=True, delbsq=0.1 * len(calls),
        )
        vacua.append(vacuum)
        # Deliberately make current xc and xstore distinguishable.
        current = _state(ns, kwargs["resolution"].mnmax, 10.0 + len(calls))
        xstore = _state(ns, kwargs["resolution"].mnmax, 20.0 + len(calls))
        result = SimpleNamespace(
            state=current,
            marker=ns,
            fsqr=float(len(calls)),
            fsqz=float(len(calls)) + 0.1,
            fsql=float(len(calls)) + 0.2,
        )
        return SimpleNamespace(
            result=result, continuation_state=xstore, vacuum=vacuum,
            rcon0=jnp.asarray([len(calls)], dtype=float),
            zcon0=jnp.asarray([-len(calls)], dtype=float),
        )

    monkeypatch.setattr(FB, "_solve_free_boundary_stage", fake_stage)
    field = object()
    result = solve_free_boundary_multigrid(
        inp, ns_array=[7, 15, 15], ftol_array=[1e-4], niter_array=[2],
        external_field=field, initial_state=seed,
        raise_on_max_iterations=False,
        time_step=0.25, tcon0=1.2, gamma=0.1, nstep=17, lconm1=False,
        precon_type="NONE", prec2d_threshold=3e-7,
    )

    assert result.marker == 15
    assert [c[0] for c in calls] == [7, 15, 15]
    expected_seed = interpolate_state(seed, ns_fine=7, modes=modes)
    np.testing.assert_allclose(
        np.asarray(calls[0][1].R_cos), np.asarray(expected_seed.R_cos))
    assert calls[0][2] is None
    assert calls[1][1].R_cos.shape[0] == 15
    # allocate_ns.f overwrites newly allocated xstore from old xc before
    # initialize_radial.f calls interp.f.  The source is therefore stage 1's
    # final xc (=11), not its best-residual restart checkpoint (=21).
    expected = interpolate_state(
        _state(7, modes.mnmax, 11.0), ns_fine=15, modes=modes)
    np.testing.assert_allclose(
        np.asarray(calls[1][1].R_cos), np.asarray(expected.R_cos))
    assert calls[1][2] is vacua[0]
    # Equal NS returns early in initialize_radial.f and keeps current xc (=12).
    np.testing.assert_allclose(np.asarray(calls[2][1].R_cos), 12.0)
    assert calls[2][2] is vacua[1]
    assert calls[2][1].R_cos.shape[0] == 15


def test_initial_bad_jacobian_restarts_free_ladder_through_ns3(
    monkeypatch,
) -> None:
    """Free boundary gets the same one-shot coarse-axis recovery."""
    inp = VmecInput.from_file(DECK)
    original = solve_free_boundary_multigrid

    def bad_first_stage(*_args, **_kwargs):
        raise VmecJacobianError(
            "INITIAL JACOBIAN CHANGED SIGN!",
            ier_flag=BAD_JACOBIAN_FLAG,
        )

    monkeypatch.setattr(FB, "_solve_free_boundary_stage", bad_first_stage)
    stopped = []
    joined = []

    class Stop:
        def set(self):
            stopped.append(True)

    class Worker:
        def join(self):
            joined.append(True)

    monkeypatch.setattr(
        FB,
        "_launch_free_lane_prefetch",
        lambda *_args, **_kwargs: (Worker(), Stop()),
    )
    marker = object()
    retried = {}

    def record_retry(_inp, **kwargs):
        retried.update(kwargs)
        return marker

    monkeypatch.setattr(MG, "solve_free_boundary_multigrid", record_retry)
    result = original(
        inp,
        ns_array=[7, 15],
        ftol_array=[1.0e-6, 1.0e-10],
        niter_array=[80, 200],
        external_field=object(),
        verbose=True,
        prefetch_compile=True,
    )

    assert result is marker
    np.testing.assert_array_equal(retried["ns_array"], [3, 7, 15])
    np.testing.assert_allclose(retried["ftol_array"], [1.0e-4, 1.0e-6, 1.0e-10])
    np.testing.assert_array_equal(retried["niter_array"], [80, 80, 200])
    assert retried["coarse_grid_retry"] is False
    assert stopped == [True]
    assert joined == [True]

    with pytest.raises(VmecJacobianError):
        original(
            inp,
            ns_array=[7],
            ftol_array=[1.0e-6],
            niter_array=[80],
            external_field=object(),
            coarse_grid_retry=False,
        )


def test_vmec2000_niter_exhaustion_is_not_converged() -> None:
    from benchmarks.run_freeboundary_multigrid import _vmec_converged

    stdout = """\
 Try increasing NITER or PRE_NITER if the preconditioner is on.
 EXECUTION TERMINATED NORMALLY
"""
    assert not _vmec_converged(stdout)


@pytest.mark.full
def test_public_two_stage_free_boundary_rebuilds_and_stays_finite() -> None:
    """The bundled non-confidential LASYM case crosses vacuum turn-on."""
    inp = VmecInput.from_file(DECK)
    lines: list[str] = []

    def emit(value="", end="\n"):
        lines.append(str(value) + end)

    result = solve_free_boundary_multigrid(
        inp, ns_array=[7, 15], ftol_array=[1e-10, 1e-10],
        niter_array=[60, 5], mgrid_path=MGRID, verbose=True, emit=emit,
        raise_on_max_iterations=False,
    )
    output = "".join(lines)
    assert output.count("VACUUM PRESSURE TURNED ON") == 1
    assert "NS =    7" in output and "NS =   15" in output
    assert result.state.R_cos.shape[0] == 15
    assert np.all(np.isfinite(result.fsq_history))
    assert np.all(np.isfinite(np.asarray(result.state.R_cos)))
    # Regression for the old turn-on-ordering blow-up (~5 km major radius).
    assert 0.5 < result.r00 < 1.0


def test_niter_exhausted_free_stage_transfers_final_xc_vmec2000_parity() -> None:
    """The pre-vacuum free ladder continues from final xc after coarse NITER."""
    inp = VmecInput.from_file(DECK)
    result = solve_free_boundary_multigrid(
        inp, ns_array=[7, 15], ftol_array=[1e-30, 1e-30],
        niter_array=[2, 1], mgrid_path=MGRID,
        raise_on_max_iterations=False,
    )
    # Local xvmec2000/PARVMEC 9.0 on these public assets gives the same first
    # ns=15 row (vacuum is deliberately not yet active).
    np.testing.assert_allclose(
        result.fsq_history[0, :3],
        [0.03192713, 0.00284142, 0.00899460],
        rtol=2e-6,
    )
    assert result.r00 == pytest.approx(0.7430588672, rel=2e-10)


def test_lforbal_free_ladder_matches_vmec2000_before_vacuum_activation() -> None:
    """The shared LFORBAL force map survives free coarse-to-fine transfer."""
    inp = replace(
        VmecInput.from_file(DECK), lforbal=True, lmove_axis=False
    )
    result = solve_free_boundary_multigrid(
        inp,
        ns_array=[7, 15],
        ftol_array=[1e-30, 1e-30],
        niter_array=[2, 1],
        mgrid_path=MGRID,
        raise_on_max_iterations=False,
        device="cpu",
    )
    # Fresh local xvmec2000/PARVMEC 9.0 prints on the ns=15 pass:
    #   1  1.89E-02  3.61E-03  8.96E-03 ... WMHD 5.0791E-02
    # Vacuum is deliberately not active yet (DEL-BSQ remains 1).
    np.testing.assert_allclose(
        result.fsq_history[0, :3],
        [1.89e-2, 3.61e-3, 8.96e-3],
        rtol=7e-3,
    )
    # The full-mesh chipf reconstruction used by calc_fbal/add_fluxes also
    # restores the VMEC2000 energy row (5.079117E-02) rather than the former
    # half-mesh-substitution value.
    assert result.r00 == pytest.approx(0.7430400335, rel=2e-10)
    assert result.wmhd == pytest.approx(0.05079117051, rel=2e-9)


@pytest.mark.full
def test_lforbal_free_ladder_crosses_vacuum_activation() -> None:
    """The non-variational force remains finite after a real NESTOR update."""
    inp = replace(VmecInput.from_file(DECK), lforbal=True)
    lines: list[str] = []

    def emit(value="", end="\n"):
        lines.append(str(value) + end)

    result = solve_free_boundary_multigrid(
        inp,
        ns_array=[7, 15],
        ftol_array=[1e-10, 1e-10],
        niter_array=[60, 5],
        mgrid_path=MGRID,
        verbose=True,
        emit=emit,
        raise_on_max_iterations=False,
        device="cpu",
    )
    assert "".join(lines).count("VACUUM PRESSURE TURNED ON") == 1
    assert np.all(np.isfinite(result.fsq_history))
    assert np.all(np.isfinite(np.asarray(result.state.R_cos)))
    assert 0.5 < result.r00 < 1.0


def test_single_grid_hot_restart_preserves_free_edge() -> None:
    inp = VmecInput.from_file(DECK)
    first = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, max_iterations=1,
        error_on_no_convergence=False,
    )
    seed = replace(
        first.state,
        R_cos=first.state.R_cos.at[-1, 0].add(1.0e-5),
        Z_sin=first.state.Z_sin.at[-1, 1].add(-1.0e-5),
    )
    restarted = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, max_iterations=1, initial_state=seed,
        error_on_no_convergence=False,
    )
    # Vacuum activation repeats on a reset-style user hot restart, but the
    # evolved free boundary is not replaced by the deck's original edge.
    np.testing.assert_allclose(
        np.asarray(restarted.state.R_cos[-1]),
        np.asarray(seed.R_cos[-1]), rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(restarted.state.Z_sin[-1]),
        np.asarray(seed.Z_sin[-1]), rtol=0.0, atol=0.0,
    )


def test_converged_hot_start_still_enters_vacuum_lane() -> None:
    """A fixed-boundary tolerance hit is not free-boundary convergence."""
    @dataclass(frozen=True)
    class Carry:
        done: object
        ier: object
        iteration: object

    carry = Carry(
        done=jnp.asarray(True),
        ier=jnp.asarray(SUCCESSFUL_TERM_FLAG, dtype=jnp.int64),
        iteration=jnp.asarray(1, dtype=jnp.int64),
    )
    resumed = FB._resume_for_vacuum(carry, -1)
    assert not bool(resumed.done)
    assert int(resumed.ier) == NORM_TERM_FLAG
    assert int(resumed.iteration) == 2
    assert FB._resume_for_vacuum(carry, 1) is carry


def test_rejects_fixed_boundary_input() -> None:
    inp = replace(VmecInput.from_file(DECK), lfreeb=False)
    with pytest.raises(ValueError, match="LFREEB"):
        solve_free_boundary_multigrid(inp, external_field=object())


@pytest.mark.full
def test_free_boundary_accepts_active_2d_preconditioner() -> None:
    """The fixed-boundary preconditioner controls remain live in free mode."""
    inp = VmecInput.from_file(DECK)
    cfg = Prec2DConfig(
        threshold=1e-2, start_iteration=45, step=0.05,
        gmres_restart=2, gmres_max_restarts=1, gmres_rtol=0.1,
    )
    result = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, resolution=resolution_from_input(inp, ns=7),
        max_iterations=55, error_on_no_convergence=False, prec2d=cfg,
    )
    assert result.iterations == 55
    assert np.all(np.isfinite([result.fsqr, result.fsqz, result.fsql]))


@pytest.mark.full
def test_converged_multigrid_final_state_matches_vmec2000_wout() -> None:
    if not CONV_MGRID.exists() or not CONV_WOUT.exists():
        pytest.skip("converged CTH mgrid/wout assets unavailable")
    from vmex.core.wout import read_wout
    from benchmarks.run_freeboundary_multigrid import _stage_iterations

    inp = VmecInput.from_file(CONV_DECK)
    lines: list[str] = []
    result = solve_free_boundary_multigrid(
        inp, ns_array=[7, 15], ftol_array=[1e-8, 1e-10],
        niter_array=[1000, 2500], mgrid_path=CONV_MGRID,
        verbose=True,
        emit=lambda value="", end="\n": lines.append(str(value) + end),
    )
    stages = _stage_iterations("".join(lines))
    assert len(stages) == 2
    # initialize_radial.f retains the coarse-grid residual module variables.
    # On the first fine-grid pass they activate residue.f90's medge=1 gate,
    # so the carried vacuum edge force is included immediately.  These are
    # the local VMEC2000/PARVMEC screen values for the same public assets.
    np.testing.assert_allclose(
        [
            stages[1]["first_fsqr"],
            stages[1]["first_fsqz"],
            stages[1]["first_fsql"],
        ],
        [1.73, 0.887, 1.51e-5],
        rtol=1.5e-2,
    )
    reference = read_wout(CONV_WOUT)
    mine = {(int(m), int(n)): i for i, (m, n) in enumerate(zip(result.xm, result.xn))}
    idx = np.asarray([
        mine[(int(m), int(n))] for m, n in zip(reference.xm, reference.xn)
    ])
    rerr = np.max(np.abs(result.rmnc[-1, idx] - reference.rmnc[-1])) / np.max(np.abs(reference.rmnc[-1]))
    zerr = np.max(np.abs(result.zmns[-1, idx] - reference.zmns[-1])) / np.max(np.abs(reference.zmns[-1]))
    s_mine = np.linspace(0.0, 1.0, result.iotaf.size)
    s_ref = np.linspace(0.0, 1.0, reference.iotaf.size)
    iota_ref = np.interp(s_mine, s_ref, reference.iotaf)
    ierr = np.max(np.abs(result.iotaf - iota_ref)) / np.max(np.abs(iota_ref))
    assert result.converged
    # This packaged reference is ns=151, while the fast CI ladder ends at
    # ns=15; compare the common boundary and interpolated iota at the expected
    # radial-discretization scale.  The exact ns=15 VMEC2000 comparison is
    # recorded by benchmarks/run_freeboundary_multigrid.py (<1e-3 here).
    assert rerr < 1e-2
    assert zerr < 1e-2
    assert ierr < 1.5e-2


@pytest.mark.full
def test_symmetric_multigrid_exports_final_nestor_wout(tmp_path) -> None:
    """The final ladder stage publishes VMEC-compatible vacuum tables."""
    if not CONV_MGRID.exists():
        pytest.skip("converged CTH mgrid asset unavailable")
    netCDF4 = pytest.importorskip("netCDF4")
    from vmex.core.wout import wout_from_state, write_wout

    inp = VmecInput.from_file(CONV_DECK)
    result = solve_free_boundary_multigrid(
        inp, ns_array=[7, 15], ftol_array=[1e-10, 1e-10],
        niter_array=[60, 5], mgrid_path=CONV_MGRID,
        raise_on_max_iterations=False,
    )
    vacuum = result.vacuum
    assert vacuum is not None
    assert vacuum.bsubu.shape == (
        resolution_from_input(inp, ns=15).ntheta3, int(inp.nzeta))
    assert vacuum.potsin.shape == vacuum.xmpot.shape == vacuum.xnpot.shape
    assert vacuum.potsin.size == (int(inp.mpol) + 2) * (2 * int(inp.ntor) + 1)

    wout = wout_from_state(
        inp=inp, state=result.state,
        fsqr=result.fsqr, fsqz=result.fsqz, fsql=result.fsql,
        niter=result.iterations, converged=result.converged,
        vacuum_output=vacuum,
    )
    np.testing.assert_array_equal(wout.xmpot, vacuum.xmpot)
    np.testing.assert_array_equal(wout.xnpot, vacuum.xnpot)
    for name in (
        "potsin", "bsubumnc_sur", "bsubvmnc_sur",
        "bsupumnc_sur", "bsupvmnc_sur",
    ):
        values = np.asarray(getattr(wout, name), dtype=float)
        assert values.size > 0 and np.isfinite(values).all()
        assert np.max(np.abs(values)) > 0.0

    path = write_wout(tmp_path / "wout_cth.nc", wout)
    with netCDF4.Dataset(path) as ds:
        for name in (
            "potsin", "xmpot", "xnpot", "bsubumnc_sur",
            "bsubvmnc_sur", "bsupumnc_sur", "bsupvmnc_sur",
        ):
            assert not np.ma.getmaskarray(ds[name][:]).any(), name

    fixed = wout_from_state(
        inp=replace(inp, lfreeb=False), state=result.state,
        fsqr=result.fsqr, fsqz=result.fsqz, fsql=result.fsql,
        niter=result.iterations, converged=result.converged,
    )
    assert fixed.potsin is fixed.bsubumnc_sur is None
    fixed_path = write_wout(tmp_path / "wout_fixed.nc", fixed)
    with netCDF4.Dataset(fixed_path) as ds:
        assert "potsin" not in ds.variables
        assert "bsubumnc_sur" not in ds.variables


@pytest.mark.full
def test_prefetch_compile_parity_and_bookkeeping() -> None:
    """Free-boundary ``prefetch_compile`` is cache warming only.

    A prefetched ladder must be bit-identical to the no-prefetch ladder,
    every background compile thread must be joined before the driver
    returns, and the machinery must actually have produced standalone
    free-boundary lane executables (the consumption paths fall back
    silently, so parity alone would also pass with a broken prefetch).
    """
    import threading

    from vmex.core import solver as S

    inp = VmecInput.from_file(DECK)
    # Unique NITER values keep these lane structures private to this test
    # (no other test can have compiled or registered them).
    ladder = dict(
        ns_array=[7, 15], ftol_array=[1e-30], niter_array=[741, 743],
        mgrid_path=MGRID, raise_on_max_iterations=False,
        release_stage_cache=True,
    )
    result = solve_free_boundary_multigrid(inp, prefetch_compile=True, **ladder)
    assert not [t for t in threading.enumerate()
                if t.name == "vmex-fb-lane-prefetch"]
    fb_tags = {key[0][0] for key in S._LANE_EXECUTABLES
               if isinstance(key[0], tuple)}
    assert any(str(tag).startswith("fb_") for tag in fb_tags)

    baseline = solve_free_boundary_multigrid(
        inp, prefetch_compile=False, **ladder)
    np.testing.assert_array_equal(result.fsq_history, baseline.fsq_history)
    assert float(result.wb) == float(baseline.wb)
    assert float(result.r00) == float(baseline.r00)
    assert int(result.iterations) == int(baseline.iterations)
