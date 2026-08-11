"""Tests for ``vmex.core.{forces,residuals}`` (forces.f / residue.f90).
Stage-by-stage legacy parity was proven by the retired A/B suite; kept
here, on realistic profil3d.f initial states (sym 2D, sym 2D ncurr=1, sym
3D, lasym): the residue.f90 m=1 constrained <-> physical round trip, the
m1-zero / edge-force release conditions as traced values, and the full
funct3d pass (finite, jit == eager, finite/nonzero grad of ``fsqr``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import residuals as newr
from vmex.core.forces import apply_m1_force_balance
from vmex.core.input import VmecInput
from vmex.core.setup import run_setup
from vmex.core.solver import (
    _initial_state,
    evaluate_forces,
    prepare_runtime,
    resolution_from_input,
)
from vmex.core.transforms import SpectralForce

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

RTOL = 1e-12
ATOL = 1e-13

CASES = [
    "solovev",  # 2D sym, ncurr=0
    "cth_like_fixed_bdy",  # 2D sym, nfp=5, ncurr=1
    "li383_low_res",  # 3D sym (lthreed: crmn/czmn, m=1 constraint)
    "up_down_asymmetric_tokamak",  # lasym (symforce + tomnspa)
]


def _allclose(new, old, name, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(
        np.asarray(new), np.asarray(old), rtol=rtol, atol=atol, err_msg=f"{name} mismatch"
    )


@pytest.fixture(scope="module", params=CASES, ids=CASES)
def case(request):
    name = request.param
    inp = VmecInput.from_file(DATA_DIR / f"input.{name}")
    resolution = resolution_from_input(inp)
    # These are pure force-kernel tests, so request a regular inferred axis
    # for all-zero-axis decks.  Production keeps the supplied zero axis and
    # performs its one VMEC2000-compatible recovery in the solve driver.
    setup = run_setup(inp, resolution, infer_axis_if_missing=True)
    rt = prepare_runtime(inp, resolution, setup=setup)
    state = _initial_state(rt.setup)
    return SimpleNamespace(name=name, inp=inp, rt=rt, state=state)


# ---------------------------------------------------------------------------
# residuals.py: m=1 coefficient mappings (residue.f90 / readin.f)
# ---------------------------------------------------------------------------


def test_m1_mappings_roundtrip(case):
    """physical(constrained(x)) == x on realistic spectral coefficients."""
    rt, state = case.rt, case.state
    setup = rt.setup
    kwargs = dict(
        modes=rt.modes,
        lthreed=bool(setup.lthreed),
        lasym=bool(setup.lasym),
        lconm1=bool(setup.lconm1),
    )
    physical = newr.m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos, **kwargs
    )
    back = newr.m1_physical_to_constrained(*physical, **kwargs)
    originals = (state.R_cos, state.Z_sin, state.R_sin, state.Z_cos)
    for name, new_c, orig in zip(("R_cos", "Z_sin", "R_sin", "Z_cos"), back, originals):
        _allclose(new_c, orig, f"m1 roundtrip {name}")
    # For 2D symmetric decks (no m=1 coupling) the mappings are the identity.
    if not (bool(setup.lthreed) or bool(setup.lasym)):
        for name, phys, orig in zip(("R_cos", "Z_sin"), physical[:2], originals[:2]):
            _allclose(phys, orig, f"m1 identity {name}")


# ---------------------------------------------------------------------------
# residuals.py: release conditions (residue.f90 / funct3d.f gating)
# ---------------------------------------------------------------------------


def test_release_conditions_are_traced_values():
    zero = newr.m1_zero_condition(
        fsqz_previous=jnp.asarray(1e-7), iterations_since_restart=jnp.asarray(100)
    )
    keep = newr.m1_zero_condition(
        fsqz_previous=jnp.asarray(1e-3), iterations_since_restart=jnp.asarray(100)
    )
    startup = newr.m1_zero_condition(
        fsqz_previous=jnp.asarray(1e-3), iterations_since_restart=jnp.asarray(0)
    )
    assert bool(zero) and not bool(keep) and bool(startup)

    edge_on = newr.edge_force_condition(
        fsq_rz_previous=jnp.asarray(1e-7),
        iterations_since_restart=jnp.asarray(10),
        free_boundary=True,
    )
    edge_off_fixedb = newr.edge_force_condition(
        fsq_rz_previous=jnp.asarray(1e-7),
        iterations_since_restart=jnp.asarray(10),
        free_boundary=False,
    )
    edge_off_late = newr.edge_force_condition(
        fsq_rz_previous=jnp.asarray(1e-7),
        iterations_since_restart=jnp.asarray(60),
        free_boundary=True,
    )
    assert bool(edge_on) and not bool(edge_off_fixedb) and not bool(edge_off_late)
    # jit-compatible (traced masks, no Python branching on values).
    assert bool(
        jax.jit(lambda f, i: newr.m1_zero_condition(fsqz_previous=f, iterations_since_restart=i))(
            jnp.asarray(1e-7), jnp.asarray(100)
        )
    )


def test_lforbal_replaces_only_symmetric_m1_n0_interior() -> None:
    """tomnsp_mod.f leaves every block except frcc/fzsc(m=1,n=0) alone."""
    shape = (5, 3, 2)
    frcc = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    fzsc = 100.0 + frcc
    untouched = -frcc
    force = SpectralForce(
        force_R_cc=frcc,
        force_Z_sc=fzsc,
        force_R_ss=untouched,
        force_Z_cs=2.0 * untouched,
    )
    equif = jnp.asarray([0.0, 2.0, 4.0, 6.0, 0.0])
    factor_R = jnp.asarray([0.0, 10.0, 20.0, 30.0, 0.0])
    factor_Z = jnp.asarray([0.0, 5.0, 10.0, 15.0, 0.0])
    got = apply_m1_force_balance(
        force, equif=equif, factor_R=factor_R, factor_Z=factor_Z
    )

    expected_R = np.asarray(frcc).copy()
    expected_Z = np.asarray(fzsc).copy()
    old_R = expected_R[:, 1, 0].copy()
    old_Z = expected_Z[:, 1, 0].copy()
    work = old_R[1:-1] / np.asarray(factor_R)[1:-1] - (
        old_Z[1:-1] / np.asarray(factor_Z)[1:-1]
    )
    expected_R[1:-1, 1, 0] = (
        0.5 * np.asarray(factor_R)[1:-1]
        * (np.asarray(equif)[1:-1] + work)
    )
    expected_Z[1:-1, 1, 0] = (
        0.5 * np.asarray(factor_Z)[1:-1]
        * (np.asarray(equif)[1:-1] - work)
    )
    np.testing.assert_allclose(np.asarray(got.force_R_cc), expected_R)
    np.testing.assert_allclose(np.asarray(got.force_Z_sc), expected_Z)
    np.testing.assert_array_equal(np.asarray(got.force_R_ss), np.asarray(untouched))
    np.testing.assert_array_equal(
        np.asarray(got.force_Z_cs), np.asarray(2.0 * untouched)
    )


# ---------------------------------------------------------------------------
# Full funct3d pass: finiteness, jit-compatibility, differentiability
# ---------------------------------------------------------------------------


def test_full_chain_residuals_finite(case):
    gc, residuals, diagnostics = evaluate_forces(case.state, case.rt)
    assert not bool(diagnostics.jacobian_sign_changed)
    for name in ("fsqr", "fsqz", "fsql"):
        value = float(getattr(residuals, name))
        assert np.isfinite(value) and value > 0.0, name
    for leaf in jax.tree.leaves(gc):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_full_chain_is_jittable(case):
    def scalars(state):
        _gc, residuals, _diag = evaluate_forces(state, case.rt)
        return residuals.fsqr, residuals.fsqz, residuals.fsql

    eager = scalars(case.state)
    jitted = jax.jit(scalars)(case.state)
    for name, a, b in zip(("fsqr", "fsqz", "fsql"), jitted, eager):
        _allclose(a, b, f"jit {name}", rtol=1e-11, atol=1e-14)


def test_grad_of_fsqr_wrt_R_cos(case):
    import dataclasses

    def fsqr_of_R_cos(R_cos):
        state = dataclasses.replace(case.state, R_cos=R_cos)
        _gc, residuals, _diag = evaluate_forces(state, case.rt)
        return residuals.fsqr

    grad = jax.grad(fsqr_of_R_cos)(case.state.R_cos)
    grad_np = np.asarray(grad)
    assert grad_np.shape == np.asarray(case.state.R_cos).shape
    assert np.all(np.isfinite(grad_np))
    assert np.any(grad_np != 0.0)


# ---------------------------------------------------------------------------
# solver._evaluate: the cond-gated ns4 pivot replay (GPU latency fix)
# ---------------------------------------------------------------------------


def test_cond_refresh_gate_is_bitwise_identical(case):
    """``cond_refresh=True`` must be bit-identical to the ungated evaluation.

    The iteration lanes stage the band-frozen pivot replay under
    ``lax.cond(refresh, ...)``; on refresh iterations the taken side must
    reproduce the unconditional assembly exactly, and on non-refresh
    iterations the frozen masks must pass through untouched.
    """
    from vmex.core import solver as _solver

    rt, state = case.rt, case.state
    cache0 = _solver._zero_cache(rt)
    one = jnp.asarray(1.0)
    it1 = jnp.asarray(1)
    it2 = jnp.asarray(2)

    def run(cond, cache, it, last):
        return _solver._evaluate(
            state, cache, it, last, one, rt, one, cond_refresh=cond
        )

    def assert_equal(x, y):
        for lx, ly in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)):
            np.testing.assert_array_equal(np.asarray(lx), np.asarray(ly))

    # Jitted on purpose: the contract is about the *compiled* programs (the
    # conftest disables jit globally, which would compare eager op-by-op
    # evaluations and miss any fusion difference introduced by the cond).
    with jax.disable_jit(False):
        # Refresh iteration (iteration == iter_last_reset): gate taken.
        gated = jax.jit(lambda: run(True, cache0, it1, it1))()
        plain = jax.jit(lambda: run(False, cache0, it1, it1))()
        # Non-refresh iteration: gate skipped, frozen masks carried through.
        gated2 = jax.jit(lambda: run(True, plain.cache, it2, it1))()
        plain2 = jax.jit(lambda: run(False, plain.cache, it2, it1))()
    assert_equal((gated.cache, gated.gc, gated.pre), (plain.cache, plain.gc, plain.pre))
    assert_equal(
        (gated2.cache, gated2.gc, gated2.pre), (plain2.cache, plain2.gc, plain2.pre)
    )
    assert_equal(
        (gated2.cache.pivot_R, gated2.cache.pivot_Z),
        (plain.cache.pivot_R, plain.cache.pivot_Z),
    )


def _function_level_while_count(module_text: str) -> tuple[int, int]:
    """(total, function-body-level) ``stablehlo.while`` ops in an MLIR module.

    A while whose nearest shallower-indented line is the ``func.func``
    signature executes unconditionally; one whose nearest shallower line is a
    region boundary (``}, {`` / ``({``) sits inside a branch of a multi-region
    op (``stablehlo.case``/``if``) and only runs when that branch is taken.
    """
    lines = module_text.splitlines()
    total = at_function_level = 0
    for i, line in enumerate(lines):
        if "stablehlo.while" not in line:
            continue
        total += 1
        indent = len(line) - len(line.lstrip())
        for j in range(i - 1, -1, -1):
            enclosing = lines[j]
            if not enclosing.strip():
                continue
            if len(enclosing) - len(enclosing.lstrip()) < indent:
                if enclosing.lstrip().startswith("func.func"):
                    at_function_level += 1
                break
    return total, at_function_level


def test_cond_refresh_gate_stages_replay_off_the_iteration_path():
    """CUDA-lowered, the pivot replay must sit inside the ``lax.cond`` branch.

    ``cond_refresh=True`` is only worth its salt if the sequential ns-length
    replay scans (``stablehlo.while``) end up inside the conditional's branch
    region — executed on 1-in-25 refresh iterations — rather than in the
    unconditional function body, where they would serialize every iteration
    into ns kernel launches on GPU.  The ungated evaluation keeps them at
    function level; the counts must otherwise agree.
    """
    from jax import export

    from vmex.core import solver as _solver

    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    resolution = _solver.resolution_from_input(inp)
    setup = run_setup(inp, resolution, infer_axis_if_missing=True)
    rt = prepare_runtime(inp, resolution, setup=setup)
    state = _initial_state(rt.setup)
    cache = _solver._zero_cache(rt)
    args = (state, cache, jnp.asarray(2), jnp.asarray(1), jnp.asarray(1.0))
    shapes = jax.tree_util.tree_map(
        lambda a: jax.ShapeDtypeStruct(jnp.shape(a), jnp.result_type(a)), args
    )

    def lowered(gate):
        def fn(state, cache, it, it0, one):
            r = _solver._evaluate(
                state, cache, it, it0, one, rt, one, cond_refresh=gate
            )
            return r.cache, r.gc, r.residuals

        with jax.disable_jit(False):  # conftest disables jit; export needs it
            return export.export(jax.jit(fn), platforms=["cuda"])(*shapes).mlir_module()

    gated_total, gated_inline = _function_level_while_count(lowered(True))
    plain_total, plain_inline = _function_level_while_count(lowered(False))
    assert plain_total == plain_inline > 0  # ungated: replay on the hot path
    assert gated_total == plain_total  # same scans, relocated ...
    assert gated_inline == 0  # ... all inside the refresh branch
