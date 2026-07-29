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
