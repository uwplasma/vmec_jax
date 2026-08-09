"""Regression and smoke tests for ``vmex.core.optimize``: QS ratio residual
conventions on golden wouts (QH is ``(helicity_m, helicity_n) = (1, -1)``
with ``helicity_n`` in units of nfp); scalar targets vs the wout engine at
1e-8 and vs golden VMEC2000 at solver-drift tolerances; QI residual pins;
boundary dof packing, ESS scaling, and least-squares smokes on solovev.
The converged solovev state is cached in ``/tmp``; golden wout fixtures
resolve through ``conftest.resolve_golden_dir``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("netCDF4")
jax.config.update("jax_enable_x64", True)

from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.wout import read_wout  # noqa: E402
from vmex.core import optimize as opt  # noqa: E402

from tests.conftest import resolve_golden_dir  # noqa: E402

GOLDEN_DIR = resolve_golden_dir()
pytestmark = [
    pytest.mark.skipif(
        GOLDEN_DIR is None, reason="golden VMEC2000 fixtures unavailable (offline?)"),
    pytest.mark.usefixtures("_module_jit_enabled"),  # full solves: run jitted
]
DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
CACHE_DIR = Path("/tmp/vmex_test_cache_optimize")

SURFACES = [0.25, 0.5, 0.75, 1.0]


def _golden_wout(case: str):
    path = GOLDEN_DIR / case / f"wout_{case}.nc"
    if not path.exists():
        pytest.skip(f"missing golden file {path}")
    return read_wout(path)


# ---------------------------------------------------------------------------
# Converged solovev equilibrium, cached in /tmp
# ---------------------------------------------------------------------------

_STATE_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


@pytest.fixture(scope="module")
def solovev_eq() -> opt.Equilibrium:
    """Converged solovev equilibrium (core multigrid solver), /tmp-cached."""
    from vmex.core.solver import SpectralState, prepare_runtime, resolution_from_input

    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cache = CACHE_DIR / "solovev_state.npz"
    jax.config.update("jax_disable_jit", False)  # tests/conftest disables jit globally
    if cache.exists():
        data = np.load(cache)
        state = SpectralState(**{k: jax.numpy.asarray(data[k]) for k in _STATE_FIELDS})
        result = SimpleNamespace(
            fsqr=float(data["fsqr"]), fsqz=float(data["fsqz"]), fsql=float(data["fsql"]),
            iterations=int(data["iterations"]), converged=bool(data["converged"]))
        ns = int(np.shape(state.R_cos)[0])
        runtime = prepare_runtime(inp, resolution_from_input(inp, ns=ns))
        return opt.Equilibrium(inp=inp, state=state, runtime=runtime, result=result)
    eq = opt.solve_equilibrium(inp)
    assert eq.result.converged
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache,
             **{k: np.asarray(getattr(eq.state, k)) for k in _STATE_FIELDS},
             fsqr=eq.result.fsqr, fsqz=eq.result.fsqz, fsql=eq.result.fsql,
             iterations=eq.result.iterations, converged=eq.result.converged)
    return eq


# ---------------------------------------------------------------------------
# QS ratio residual: conventions and sanity on golden wout files
# ---------------------------------------------------------------------------
# helicity_n in units of nfp, the simsopt convention (nn = helicity_n * nfp):
# QA (1, 0); QH (1, -1); QP would be (0, 1).


def test_qs_solovev_axisymmetric_sanity():
    """An axisymmetric equilibrium is exactly QA: the (1, 0) residual ~ 0."""
    w = _golden_wout("solovev")
    qa = float(opt.QuasisymmetryRatioResidual(SURFACES, 1, 0).total(w))
    qh = float(opt.QuasisymmetryRatioResidual(SURFACES, 1, -1).total(w))
    assert qa < 1e-9                      # zero up to wout round-trip noise
    assert qa < 1e-8 * qh                 # and negligible vs a wrong helicity


def test_qs_helicity_sign_convention_qh():
    """nfp4_QH minimizes the (1, -1) helicity residual — pins the sign.

    ``helicity_n`` multiplies ``nfp`` internally, so the plan's "QH:
    (m, n) = (1, -nfp)" in physical mode numbers is ``helicity_n = -1`` here.
    """
    w = _golden_wout("nfp4_QH_warm_start")
    totals = {hn: float(opt.QuasisymmetryRatioResidual(SURFACES, 1, hn).total(w))
              for hn in (-1, 0, 1)}
    assert totals[-1] < totals[1]
    assert totals[-1] < totals[0]


def test_qs_residual_from_converged_state(solovev_eq):
    """QS residual through wout_from_state on a converged core state."""
    w = solovev_eq.wout
    obj = opt.QuasisymmetryRatioResidual(SURFACES, 1, 0)
    ours = obj.compute(w)
    assert np.all(np.isfinite(np.asarray(ours["residuals1d"])))
    assert float(ours["total"]) < 1e-9    # axisymmetric => QA-symmetric
    # Equilibrium objects are accepted directly (least_squares term entry).
    np.testing.assert_allclose(np.asarray(obj.J(solovev_eq)),
                               np.asarray(ours["residuals1d"]), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Scalar targets
# ---------------------------------------------------------------------------


def test_scalar_targets_match_own_wout(solovev_eq):
    """(state, runtime) scalars == the wout engine of the same state (1e-8)."""
    eq, w = solovev_eq, solovev_eq.wout
    np.testing.assert_allclose(float(opt.aspect_ratio(eq.state, eq.runtime)),
                               float(w.aspect), rtol=1e-8)
    np.testing.assert_allclose(float(opt.volume(eq.state, eq.runtime)),
                               float(w.volume_p), rtol=1e-8)
    np.testing.assert_allclose(float(opt.mean_iota(eq.state, eq.runtime)),
                               float(np.mean(np.asarray(w.iotas)[1:])), rtol=1e-8)
    np.testing.assert_allclose(float(opt.edge_iota(eq.state, eq.runtime)),
                               float(np.asarray(w.iotaf)[-1]), rtol=1e-8)
    # magnetic well against the same endpoint-extrapolation formula on wout vp
    vp = np.abs(np.asarray(w.vp, dtype=float))[1:]
    v0 = 1.5 * vp[0] - 0.5 * vp[1]
    v1 = 1.5 * vp[-1] - 0.5 * vp[-2]
    np.testing.assert_allclose(float(opt.magnetic_well(eq.state, eq.runtime)),
                               (v0 - v1) / v0, rtol=1e-8)
    mirror = float(opt.mirror_ratio(eq.state, eq.runtime))
    assert 0.0 < mirror < 1.0


def test_boundary_elongation_is_traceable_and_grid_converged(solovev_eq):
    """Equivalent-ellipse elongation is physical, resolved, and JAX-ready."""
    eq = solovev_eq
    coarse = np.asarray(
        opt.elongation_profile(eq.state, eq.runtime, ntheta=32, nphi=16)
    )
    fine = np.asarray(
        opt.elongation_profile(eq.state, eq.runtime, ntheta=64, nphi=32)
    )
    assert np.all(np.isfinite(coarse)) and np.all(coarse >= 1.0)
    np.testing.assert_allclose(np.max(coarse), 1.5602656925818226, rtol=1e-7)
    np.testing.assert_allclose(np.max(coarse), np.max(fine), rtol=1e-7)
    np.testing.assert_allclose(float(opt.max_elongation(eq.state, eq.runtime)),
                               np.max(coarse), rtol=1e-7)

    gradient = np.asarray(jax.grad(
        lambda z_sin: opt.max_elongation(
            dataclasses.replace(eq.state, Z_sin=z_sin), eq.runtime
        )
    )(eq.state.Z_sin))
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) > 0.0


def test_solve_equilibrium_forwards_verbose(monkeypatch, solovev_eq):
    """The public final-solve helper exposes the VMEC iteration table flag."""
    captured = {}
    result = SimpleNamespace(
        state=solovev_eq.state,
        fsqr=solovev_eq.result.fsqr,
        fsqz=solovev_eq.result.fsqz,
        fsql=solovev_eq.result.fsql,
        iterations=solovev_eq.result.iterations,
        converged=solovev_eq.result.converged,
    )

    def fake_solve_multigrid(inp, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(opt, "solve_multigrid", fake_solve_multigrid)
    solved = opt.solve_equilibrium(
        solovev_eq.inp,
        initial_state=solovev_eq.state,
        raise_on_max_iterations=True,
        verbose=True,
    )
    assert solved.state is solovev_eq.state
    assert captured["initial_state"] is solovev_eq.state
    assert captured["raise_on_max_iterations"] is True
    assert captured["verbose"] is True


def test_scalar_targets_vs_golden(solovev_eq):
    """Scalars vs golden VMEC2000 wout values: the golden run is an
    independently converged state (ftol 1e-14), so tolerances carry solver
    drift; the iota of this ncurr=0 deck is prescribed (AI = 1), hence exact."""
    eq = solovev_eq
    gold = _golden_wout("solovev")
    np.testing.assert_allclose(float(opt.aspect_ratio(eq.state, eq.runtime)),
                               float(gold.aspect), rtol=1e-6)
    np.testing.assert_allclose(float(opt.volume(eq.state, eq.runtime)),
                               float(gold.volume_p), rtol=1e-6)
    np.testing.assert_allclose(float(opt.mean_iota(eq.state, eq.runtime)), 1.0,
                               rtol=1e-10)
    np.testing.assert_allclose(float(opt.edge_iota(eq.state, eq.runtime)), 1.0,
                               rtol=1e-10)
    np.testing.assert_allclose(
        float(opt.mean_iota(eq.state, eq.runtime)),
        float(np.mean(np.asarray(gold.iotas)[1:])), rtol=1e-8)


def test_scalar_regression_pins(solovev_eq):
    """Regression pins (converged input.solovev, ns=11; recorded 2026-07-09,
    jax 0.x x64 CPU, deck ftol 1e-14 — loose rtol absorbs BLAS variation)."""
    eq = solovev_eq
    np.testing.assert_allclose(float(opt.aspect_ratio(eq.state, eq.runtime)),
                               3.117998343734321, rtol=1e-6)
    np.testing.assert_allclose(float(opt.mirror_ratio(eq.state, eq.runtime)),
                               0.23876209809674176, rtol=1e-5)
    np.testing.assert_allclose(float(opt.magnetic_well(eq.state, eq.runtime)),
                               -0.05903842888376773, rtol=1e-4)


def test_d_merc(solovev_eq):
    """DMerc objective: identity to the wout engine (golden-validated in
    test_wout_golden.py) plus a golden A/B on interior surfaces (near
    axis/edge carry the usual Mercier noise).  Pin recorded 2026-07-09
    (x64 CPU)."""
    eq = solovev_eq
    dm = np.asarray(opt.d_merc(eq))
    assert np.all(np.isfinite(dm))
    np.testing.assert_array_equal(dm, np.asarray(eq.wout.DMerc))
    np.testing.assert_allclose(dm[5], -5.689907338850136e-06, rtol=1e-4)
    gold = np.asarray(_golden_wout("solovev").DMerc)
    scale = float(np.max(np.abs(gold[2:-1])))
    np.testing.assert_allclose(dm[2:-1], gold[2:-1],
                               rtol=5e-2, atol=1e-3 * scale)
    # wout-like objects work too (objective usable without a solve)
    np.testing.assert_array_equal(np.asarray(opt.d_merc(_golden_wout("solovev"))), gold)


def test_l_grad_b(solovev_eq):
    """LgradB objective: finiteness, jit parity, grid convergence, pins
    (recorded 2026-07-09, x64 CPU: golden nfp4_QH 0.3238956855163282 m,
    cached solovev 2.2782393147008424 m ~ minor-radius scale)."""
    jax.config.update("jax_disable_jit", False)
    gqh = _golden_wout("nfp4_QH_warm_start")
    val = float(opt.l_grad_b(gqh))
    assert np.isfinite(val) and 0.0 < val < 100.0
    np.testing.assert_allclose(val, 0.3238956855163282, rtol=1e-8)
    np.testing.assert_allclose(float(opt.l_grad_b(solovev_eq)),
                               2.2782393147008424, rtol=1e-5)
    # jit-clean and equal to eager
    jitted = float(jax.jit(lambda: opt.l_grad_b(gqh))())
    np.testing.assert_allclose(jitted, val, rtol=1e-12)
    # angular-grid refinement changes the hard minimum only mildly
    fine = float(opt.l_grad_b(gqh, ntheta=48, nphi=48))
    np.testing.assert_allclose(fine, val, rtol=5e-2)


def test_l_grad_b_rejects_asymmetric_wout() -> None:
    """The symmetric diagnostic must not silently omit LASYM partners."""
    with pytest.raises(NotImplementedError, match="lasym = False"):
        opt.l_grad_b(SimpleNamespace(lasym=True))


# ---------------------------------------------------------------------------
# QI residual
# ---------------------------------------------------------------------------

QI_KW = dict(nphi=61, nalpha=13, n_bounce=21, include_bounce_endpoints=True,
             softness=2.0e-2, width_weight=1.0, branch_width_weight=0.5,
             branch_width_softness=2.0e-2, profile_weight=0.1,
             shuffle_profile_weight=1.0, shuffle_profile_softness=2.0e-2)


def test_qi_residual_golden_pin():
    """QI residual on the golden nfp4_QH wout: finite pin (recorded
    2026-07-09, when the port was A/B-verified at rtol 1e-8 against the
    legacy Goodman-style residual)."""
    pytest.importorskip("booz_xform_jax")
    w = _golden_wout("nfp4_QH_warm_start")
    booz = opt.boozer_modes_from_wout(w, surfaces=[0.5, 1.0], mboz=10, nboz=10)
    ours = opt.quasi_isodynamic_residual(
        bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], nfp=booz["nfp"], **QI_KW)
    res = np.asarray(ours["residuals1d"])
    assert np.all(np.isfinite(res))
    assert float(ours["total"]) > 0.0


def test_qi_regression_pin_and_jit(solovev_eq):
    """QI residual on a cached converged state: pin, finiteness, jit parity.
    Pin recorded 2026-07-09 (x64 CPU; solovev ns=11 ftol 1e-14, surfaces
    (0.5, 1.0), mboz=nboz=8) -> total = 0.13626; rtol 1e-3 because the
    residual amplifies convergence-path drift (~1e-4 between BLAS/jit
    configurations)."""
    pytest.importorskip("booz_xform_jax")
    jax.config.update("jax_disable_jit", False)
    booz = opt.boozer_modes_from_wout(solovev_eq.wout, surfaces=[0.5, 1.0],
                                      mboz=8, nboz=8)
    out = opt.quasi_isodynamic_residual(
        bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], nfp=booz["nfp"], **QI_KW)
    res = np.asarray(out["residuals1d"])
    total = float(out["total"])
    assert np.all(np.isfinite(res))
    np.testing.assert_allclose(total, 0.1362660686195369, rtol=1e-3)

    total_jit = jax.jit(
        lambda bm: opt.quasi_isodynamic_residual(
            bmnc_b=bm, xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], nfp=booz["nfp"], **QI_KW)["total"])(booz["bmnc_b"])
    np.testing.assert_allclose(float(total_jit), total, rtol=1e-12)
    # the wout-level convenience wrapper agrees (same booz configuration)
    total_wrap = float(opt.quasi_isodynamic_residual_from_wout(
        solovev_eq.wout, surfaces=[0.5, 1.0], mboz=8, nboz=8, **QI_KW)["total"])
    np.testing.assert_allclose(total_wrap, total, rtol=1e-12)


# ---------------------------------------------------------------------------
# Boundary dofs + least-squares driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck", ["solovev", "li383_low_res"])
def test_boundary_pack_roundtrip(deck):
    """VmecInput -> dofs -> VmecInput is the identity (and dofs survive edits)."""
    inp = VmecInput.from_file(DATA_DIR / f"input.{deck}")
    for max_mode in (1, 2):
        x = opt.pack_boundary(inp, max_mode)
        assert x.size == len(opt.boundary_dof_names(inp, max_mode))
        assert opt.unpack_boundary(inp, x, max_mode) == inp
        x2 = x + 1e-3 * (1.0 + np.arange(x.size))
        inp2 = opt.unpack_boundary(inp, x2, max_mode)
        np.testing.assert_array_equal(opt.pack_boundary(inp2, max_mode), x2)
        assert inp2 != inp
    # RBC(0,0) (major radius) is not a dof
    assert "RBC(0,0)" not in opt.boundary_dof_names(inp, 2)


def test_ess_scale():
    """ESS trust-region scaling: exp(-alpha*level), normalized at level 1."""
    inp = VmecInput.from_file(DATA_DIR / "input.li383_low_res")
    names = opt.boundary_dof_names(inp, 2)
    scale = opt._ess_scale(inp, 2, 1.2)
    assert scale.shape == (len(names),)
    lut = dict(zip(names, scale))
    np.testing.assert_allclose(lut["RBC(0,1)"], 1.0)             # level 1
    np.testing.assert_allclose(lut["RBC(2,2)"], np.exp(-1.2))    # level 2
    assert np.all(scale <= 1.0 + 1e-12)


def test_least_squares_smoke(solovev_eq):
    """2-iteration FD least squares on solovev (aspect target only) improves.

    max_mode=1 on this axisymmetric deck gives 2 dofs (RBC(0,1), ZBS(0,1));
    the initial aspect is ~3.118, the target 4.0, and a handful of
    finite-difference trust-region steps must strictly reduce the cost.
    """
    jax.config.update("jax_disable_jit", False)
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    aspect0 = float(opt.aspect_ratio(solovev_eq.state, solovev_eq.runtime))
    cost0 = 0.5 * (aspect0 - 4.0) ** 2
    res = opt.least_squares([(opt.aspect_ratio, 4.0, 1.0)], inp, max_mode=1,
                            max_nfev=4, diff_step=1e-4)
    assert res.cost < cost0
    assert isinstance(res.input, VmecInput)
    assert res.equilibrium is not None  # last trial solve (not necessarily res.x)
    best = opt.solve_equilibrium(res.input)
    aspect1 = float(opt.aspect_ratio(best.state, best.runtime))
    assert abs(aspect1 - 4.0) < abs(aspect0 - 4.0)
    # the optimized input reproduces the reported dofs
    np.testing.assert_array_equal(opt.pack_boundary(res.input, 1), res.x)


def test_least_squares_implicit_smoke(solovev_eq):
    """3-iteration jac='implicit' least squares on solovev improves the cost.

    Same aspect-only objective as the finite-difference smoke above, but the
    Jacobian comes from the Phase-6 implicit-gradient path
    (``vmex.core.implicit``): one hot-restarted forward solve per trial
    boundary plus one linearized-KKT solve for all dofs — gradient cost
    ~O(1 equilibrium solve) independent of the dof count.
    """
    jax.config.update("jax_disable_jit", False)
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    aspect0 = float(opt.aspect_ratio(solovev_eq.state, solovev_eq.runtime))
    cost0 = 0.5 * (aspect0 - 4.0) ** 2
    res = opt.least_squares([(opt.aspect_ratio, 4.0, 1.0)], inp, max_mode=1,
                            jac="implicit", max_nfev=3)
    assert res.cost < cost0
    best = opt.solve_equilibrium(res.input)
    aspect1 = float(opt.aspect_ratio(best.state, best.runtime))
    assert abs(aspect1 - 4.0) < abs(aspect0 - 4.0)


def test_minimize_scalarized_implicit_smoke(solovev_eq):
    """L-BFGS-B lowers the same cost with a finite reverse gradient."""
    jax.config.update("jax_disable_jit", False)
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    x0 = opt.pack_boundary(inp, 1)
    cost0 = 0.5 * (float(opt.aspect_ratio(
        solovev_eq.state, solovev_eq.runtime)) - 4.0) ** 2
    bounds = list(zip(x0 - 0.5, x0 + 0.5))
    res = opt.minimize(
        [(opt.aspect_ratio, 4.0, 1.0)], inp, max_mode=1,
        bounds=bounds, options={"maxiter": 1})
    assert res.cost < cost0
    assert np.isfinite(res.optimality)
    assert np.all(np.isfinite(res.jac))
    assert np.all(res.x >= x0 - 0.5) and np.all(res.x <= x0 + 0.5)
    assert isinstance(res.input, VmecInput)


def test_least_squares_implicit_jac_chunking(solovev_eq):
    """The R17.1 chunked implicit Jacobian matches the unchunked one:
    ``jac_chunk_size`` only changes how the per-dof columns are batched
    (:func:`solvax.chunk_map`), so the Jacobian at the initial boundary must
    be identical.  ``max_nfev=1`` keeps it cheap; solovev has 2 dofs so
    ``jac_chunk_size=1`` is a real 2-chunk pass."""
    jax.config.update("jax_disable_jit", False)
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    obj = [(opt.aspect_ratio, 4.0, 1.0)]
    # explicit None pins the unchunked reference (default is "auto")
    ref = opt.least_squares(obj, inp, max_mode=1, jac="implicit",
                            jac_chunk_size=None, max_nfev=1)
    assert ref.jac.shape[1] == 2  # RBC(0,1), ZBS(0,1)
    for chunk in (1, 2, "auto"):
        got = opt.least_squares(obj, inp, max_mode=1, jac="implicit",
                                jac_chunk_size=chunk, max_nfev=1)
        assert got.jac.shape == ref.jac.shape
        np.testing.assert_allclose(got.jac, ref.jac, rtol=1e-8, atol=1e-10,
                                   err_msg=f"chunk={chunk!r}")


def test_auto_jac_chunk_stays_bounded_with_large_device(monkeypatch):
    """A reported accelerator budget must not turn ``auto`` into one vmap."""
    monkeypatch.setattr(opt, "auto_chunk_size", lambda dim: dim)
    assert opt._auto_jac_chunk(120) == 11


def test_least_squares_implicit_jac_solver_block(solovev_eq, monkeypatch):
    """The R25.2 block-tridiagonal Jacobian (``jac_solver="block"``: colored
    jvp probes, one :func:`solvax.block_thomas_factor`, GMRES-certified
    columns) must agree with the per-dof GMRES path to the solver tolerance
    (``adjoint_tol = 1e-6``)."""
    jax.config.update("jax_disable_jit", False)
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    obj = [(opt.aspect_ratio, 4.0, 1.0)]

    def elongation_excess(state, runtime):
        return jax.numpy.maximum(
            opt.max_elongation(state, runtime) - 8.0, 0.0
        )

    ref = opt.least_squares(obj, inp, max_mode=1, jac="implicit",
                            jac_solver="gmres", max_nfev=1)
    # The public problem uses cost weights: weight=4 scales residuals and
    # their Jacobian by sqrt(4)=2.  It exposes the same block engine the
    # compatibility driver used to keep private.
    problem = opt.VmecProblem.from_tuples(
        inp,
        [(opt.aspect_ratio, 4.0, 4.0), (elongation_excess, 0.0, 1.0)],
        max_mode=1,
        implicit_jacobian_method="block_tridiagonal",
        use_ess=False,
    )
    residual, weighted_jac = problem.residual_and_jac(problem.x0)
    compiled = problem.compile_residual_and_jacobian(progress=False)
    got_jac = weighted_jac[0] / 2.0
    reverse = opt.least_squares(obj, inp, max_mode=1, jac="implicit",
                                jac_solver="reverse", max_nfev=1)
    assert got_jac.shape == ref.jac[0].shape
    np.testing.assert_allclose(residual[0] / 2.0, ref.fun[0], rtol=1e-12)
    np.testing.assert_allclose(got_jac, ref.jac[0], rtol=1e-6, atol=1e-8)
    assert residual[1] == 0.0
    assert np.all(np.isfinite(weighted_jac[1]))
    np.testing.assert_allclose(reverse.jac, ref.jac, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(problem.grad(problem.x0),
                               weighted_jac.T @ residual, rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(problem.jax_residual_jac(problem.x0))))
    assert problem.input_from_x(problem.x0) == inp
    np.testing.assert_array_equal(problem.x_from_input(inp), problem.x0)
    accepted = problem.equilibrium_from_x(problem.x0)
    assert accepted.inp == inp
    assert accepted.result.converged
    with pytest.raises(ValueError, match="ntheta"):
        opt.elongation_profile(
            accepted.state, accepted.runtime, ntheta=3, nphi=1
        )
    asymmetric_runtime = dataclasses.replace(
        accepted.runtime,
        setup=dataclasses.replace(accepted.runtime.setup, lasym=True),
    )
    asymmetric_elongation = opt.elongation_profile(
        accepted.state, asymmetric_runtime, ntheta=4, nphi=1
    )
    assert np.all(np.isfinite(asymmetric_elongation))
    assert problem.metadata["derivative_method"] == "implicit"
    assert "converged equilibrium" in problem.metadata["derivative_description"]
    assert problem.metadata["weight_semantics"] == "cost"
    assert problem.metadata["implicit_jacobian_method"] == "block_tridiagonal"
    assert problem.metadata["jacobian_batch_size"] == 1
    assert problem.metadata["input_resolution"] == {
        "mpol": inp.mpol,
        "ntor": inp.ntor,
        "ntheta": inp.ntheta,
        "nzeta": inp.nzeta,
    }
    np.testing.assert_array_equal(compiled.residual, residual)
    np.testing.assert_array_equal(compiled.jacobian, weighted_jac)
    assert (
        problem.metadata["implicit_jacobian_description"]
        == "block-tridiagonal equilibrium response"
    )
    assert (
        problem.metadata["weight_description"]
        == "weight multiplies squared cost"
    )
    scalar = opt.VmecProblem.from_loss(
        inp,
        lambda state, runtime: 0.5 * (opt.aspect_ratio(state, runtime) - 4.0) ** 2,
        max_mode=1,
        use_ess=False,
    )
    value, gradient = scalar.value_and_grad(scalar.x0)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert np.isfinite(scalar.fun(scalar.x0))
    assert scalar.fun(np.full_like(scalar.x0, np.nan)) == 1.0e12
    assert np.isfinite(float(scalar.jax_fun(scalar.x0)))

    holder = scalar.metadata["holder"]
    real_device_get = opt.jax.device_get

    def fail_device_get(_value):
        raise RuntimeError("synthetic transfer failure")

    monkeypatch.setattr(opt.jax, "device_get", fail_device_get)
    scalar._vg_cache = None
    failed_value, failed_gradient = scalar.value_and_grad(scalar.x0)
    assert failed_value == 1.0e12
    np.testing.assert_array_equal(failed_gradient, gradient)
    assert scalar.fun(scalar.x0) == 1.0e12
    holder["last_grad"] = None
    scalar._vg_cache = None
    with pytest.raises(RuntimeError, match="synthetic transfer failure"):
        scalar.value_and_grad(scalar.x0)
    with pytest.raises(RuntimeError, match="synthetic transfer failure"):
        scalar.fun(scalar.x0)

    def nan_device_get(value):
        return np.full_like(np.asarray(value), np.nan)

    monkeypatch.setattr(opt.jax, "device_get", nan_device_get)
    scalar._vg_cache = None
    with pytest.raises(FloatingPointError, match="non-finite initial"):
        scalar.value_and_grad(scalar.x0)
    holder["last_grad"] = gradient
    scalar._vg_cache = None
    failed_value, failed_gradient = scalar.value_and_grad(scalar.x0)
    assert failed_value == 1.0e12
    np.testing.assert_array_equal(failed_gradient, gradient)
    monkeypatch.setattr(opt.jax, "device_get", real_device_get)

    with pytest.raises(AttributeError, match="residuals"):
        scalar.residual(scalar.x0)
    with pytest.raises(ValueError, match="jac_solver"):
        opt.least_squares(obj, inp, max_mode=1, jac="implicit",
                          jac_solver="svd", max_nfev=1)


def test_public_problem_factory_validation():
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    term = [(opt.aspect_ratio, 4.0, 1.0)]
    with pytest.raises(ValueError, match="exactly one"):
        opt.make_problem(inp)
    with pytest.raises(ValueError, match="exactly one"):
        opt.make_problem(inp, objective_terms=term, loss=opt.aspect_ratio)
    with pytest.raises(ValueError, match="derivative_method"):
        opt.make_problem(
            inp, objective_terms=term, derivative_method="finite_difference"
        )
    with pytest.raises(ValueError, match="non-negative"):
        opt.make_problem(
            inp, objective_terms=[(opt.aspect_ratio, 4.0, -1.0)], max_mode=1
        )
    common = dict(max_mode=1, x0=None, solve_kwargs={})
    with pytest.raises(ValueError, match="weight_semantics"):
        opt._least_squares_implicit(
            term, inp, weight_semantics="unknown", **common
        )
    with pytest.raises(ValueError, match="implicit_jacobian_method"):
        opt.make_problem(
            inp,
            objective_terms=term,
            implicit_jacobian_method="block",
        )
    with pytest.raises(ValueError, match="jacobian_batch_size"):
        opt.make_problem(inp, objective_terms=term, jacobian_batch_size=0)
    with pytest.raises(ValueError, match="not both"):
        opt._least_squares_implicit(
            term, inp, scalar_objective=opt.aspect_ratio, **common
        )


def test_least_squares_implicit_warm_start_modes(solovev_eq):
    """R25.4 perturbation warm start reaches the same optimum as plain hot
    restart: ``warm_start`` only changes each trial's initial guess, never
    the fixed point, so the optimizer walks the same trust-region path;
    ``solve_stats`` exposes the effort totals the R25.4 benchmark compares."""
    jax.config.update("jax_disable_jit", False)
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    obj = [(opt.aspect_ratio, 4.0, 1.0)]
    ref = opt.least_squares(obj, inp, max_mode=1, jac="implicit",
                            warm_start="state", max_nfev=3)
    got = opt.least_squares(obj, inp, max_mode=1, jac="implicit",
                            warm_start="perturbation", max_nfev=3)
    np.testing.assert_allclose(got.cost, ref.cost, rtol=1e-10)
    np.testing.assert_allclose(got.x, ref.x, rtol=1e-8, atol=1e-12)
    for res in (ref, got):
        assert res.solve_stats is not None
        assert res.solve_stats["solves"] >= res.nfev
        assert res.solve_stats["iterations"] > 0
    with pytest.raises(ValueError, match="warm_start"):
        opt.least_squares(obj, inp, max_mode=1, jac="implicit",
                          warm_start="broyden", max_nfev=1)


def test_least_squares_max_mode_schedule():
    """Staged max_mode continuation: two ultra-short stages chain through
    result.input; the second starts from — and does not regress — the
    first stage's boundary."""
    jax.config.update("jax_disable_jit", False)
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    res = opt.least_squares([(opt.aspect_ratio, 4.0, 1.0)], inp,
                            max_mode=(1, 1), max_nfev=2, diff_step=1e-4,
                            use_ess=True)
    assert len(res.stage_results) == 2
    assert res.stage_results[-1] is res
    assert res.cost <= res.stage_results[0].cost + 1e-12
    np.testing.assert_array_equal(opt.pack_boundary(res.input, 1), res.x)


def test_equilibrium_wout_is_cached(solovev_eq):
    """Equilibrium.wout is computed once and reused (cached_property)."""
    assert solovev_eq.wout is solovev_eq.wout
    assert dataclasses.is_dataclass(solovev_eq.wout)
