"""Tests for ``vmex.core.setup`` (profil1d.f / readin.f / profil3d.f).

Field-by-field parity with the legacy setup chain (flux profiles, boundary
processing incl. theta flip and the lconm1 m=1 conversion, initial guess,
guess_axis) was proven by the A/B suite that retired with the legacy tree.
Kept here are the analytic VMEC2000 grid conventions, well-posedness of the
initial state through the proven geometry chain, the readin.f theta-flip
behaviour, and jit-compatibility of the state-producing path.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.fourier import mode_table, trig_tables
from vmex.core.geometry import (
    half_mesh_jacobian,
    real_space_geometry,
    sqrt_s_half_mesh,
)
from vmex.core.input import VmecInput
from vmex.core.setup import (
    geometry_state,
    interior_guess,
    run_setup,
)
from vmex.core.solver import resolution_from_input

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

RTOL = 1e-12
ATOL = 1e-13

SIGNGS = -1

# 2D sym pressure-driven / 2D sym nfp=5 current-driven (two_power, phiedge<0)
# / 3D sym (lconm1 m=1, axis inference) / lasym (axis inference) / high-mpol
# multigrid deck (legacy RAXIS name).
CASES = [
    "solovev",
    "cth_like_fixed_bdy",
    "li383_low_res",
    "up_down_asymmetric_tokamak",
    "DSHAPE",
]


def _allclose(new, old, name):
    np.testing.assert_allclose(
        np.asarray(new), np.asarray(old), rtol=RTOL, atol=ATOL, err_msg=f"{name} mismatch"
    )


@pytest.fixture(scope="module", params=CASES, ids=CASES)
def case(request):
    """Build the core setup for one input deck."""
    name = request.param
    deck = DATA_DIR / f"input.{name}"

    inp = VmecInput.from_file(deck)
    res = resolution_from_input(inp)
    ns = int(res.ns)
    s = np.linspace(0.0, 1.0, ns)
    setup = run_setup(inp, res)
    modes_new = mode_table(res.mpol, res.ntor)
    trig_new = trig_tables(res)

    geom = real_space_geometry(
        **geometry_state(setup, modes=modes_new),
        modes=modes_new, trig=trig_new, s=setup.s_full,
    )
    jac = half_mesh_jacobian(geom, s=setup.s_full)

    return SimpleNamespace(
        name=name, s=s, ns=ns, inp=inp, res=res, setup=setup,
        modes_new=modes_new, trig_new=trig_new, geom=geom, jac=jac,
    )


# ---------------------------------------------------------------------------
# Radial grids (profil1d.f conventions, analytic references)
# ---------------------------------------------------------------------------


def test_radial_grids_conventions(case):
    setup, s = case.setup, case.s
    _allclose(setup.s_full, s, "s_full")
    hs = s[1] - s[0]
    _allclose(setup.hs, hs, "hs")
    # profil1d.f: s_half(i) = hs*|i-1.5| — the axis slot repeats the first
    # interior half-mesh value; interior slots are the full-mesh midpoints.
    _allclose(setup.s_half[1:], 0.5 * (s[1:] + s[:-1]), "s_half interior")
    _allclose(setup.s_half[0], 0.5 * hs, "s_half axis slot")
    sqrts_ref = np.sqrt(s)
    sqrts_ref[-1] = 1.0
    _allclose(setup.sqrts, sqrts_ref, "sqrts")
    _allclose(setup.shalf, sqrt_s_half_mesh(jnp.asarray(s)), "shalf")
    # profil1d.f sm/sp odd-m half-mesh weights.
    sm_ref = np.zeros(case.ns)
    sm_ref[1:] = np.asarray(setup.shalf)[1:] / sqrts_ref[1:]
    sp_ref = np.zeros(case.ns)
    sp_ref[1:-1] = np.asarray(setup.shalf)[2:] / sqrts_ref[1:-1]
    sp_ref[-1] = 1.0 / sqrts_ref[-1]
    sp_ref[0] = sm_ref[1]
    _allclose(setup.sm, sm_ref, "sm")
    _allclose(setup.sp, sp_ref, "sp")


def test_profiles_and_current_lane(case):
    setup = case.setup
    assert setup.signgs == SIGNGS
    assert setup.lflip is False
    # phips carries a zeroed axis slot; lamscale is finite and positive.
    assert float(np.asarray(setup.phips)[0]) == 0.0
    assert np.isfinite(float(setup.lamscale)) and float(setup.lamscale) > 0.0
    assert setup.ncurr == int(case.inp.ncurr)
    if setup.ncurr == 1 and abs(float(case.inp.curtor)) > 0 and np.any(
        np.asarray(case.inp.ac)
    ):
        # The current-driven deck must exercise a nonzero pcurr lane.
        assert np.any(np.asarray(setup.icurv) != 0.0)


# ---------------------------------------------------------------------------
# Boundary / initial state (readin.f + profil3d.f) properties
# ---------------------------------------------------------------------------


def test_edge_row_is_processed_boundary(case):
    setup = case.setup
    _allclose(setup.R_cos[-1], setup.boundary_R_cos, "edge row R_cos")
    _allclose(setup.Z_sin[-1], setup.boundary_Z_sin, "edge row Z_sin")


def test_axis_coefficients(case):
    setup = case.setup
    input_axis_given = any(
        np.any(np.asarray(a) != 0.0)
        for a in (case.inp.raxis_c, case.inp.raxis_s, case.inp.zaxis_c, case.inp.zaxis_s)
    )
    if input_axis_given:
        # Axis given in the deck: used verbatim (profil3d.f).
        _allclose(setup.raxis_c, case.inp.raxis_c, "raxis_c")
        _allclose(setup.zaxis_s, case.inp.zaxis_s, "zaxis_s")
    else:
        # Axis inferred (guess_axis.f lane): must be usable, i.e. nonzero R.
        assert float(np.asarray(setup.raxis_c)[0]) > 0.0


# ---------------------------------------------------------------------------
# Well-posedness: the initial state through the proven new geometry chain
# ---------------------------------------------------------------------------


def test_initial_state_jacobian_has_no_sign_change(case):
    assert not bool(case.jac.jacobian_sign_changed), (
        f"{case.name}: initial guess flags a Jacobian sign change"
    )
    # And the interior Jacobian sign agrees with signgs = -1 (jacobian.f).
    tau_interior = np.asarray(case.jac.tau)[1:]
    assert np.all(np.sign(tau_interior) == case.setup.signgs)


# ---------------------------------------------------------------------------
# Theta flip (readin.f lflip + init_geometry.f90 flip_theta)
# ---------------------------------------------------------------------------


def test_theta_flip_recovers_good_jacobian(case):
    if case.name != "solovev":
        pytest.skip("flip check runs once on the solovev deck")
    # Negating ZBS reverses the poloidal orientation: readin.f must detect
    # rtest*ztest < 0, flip theta -> pi - theta, and negate iotas/chips.
    flipped_inp = dataclasses.replace(case.inp, zbs=-np.asarray(case.inp.zbs))
    flipped = run_setup(flipped_inp, case.res)
    assert flipped.lflip is True
    assert flipped.signgs == SIGNGS
    _allclose(flipped.iotas, -np.asarray(case.setup.iotas), "flipped iotas")
    _allclose(flipped.chips, -np.asarray(case.setup.chips), "flipped chips")
    geom = real_space_geometry(
        **geometry_state(flipped, modes=case.modes_new),
        modes=case.modes_new, trig=case.trig_new, s=flipped.s_full,
    )
    jac = half_mesh_jacobian(geom, s=flipped.s_full)
    assert not bool(jac.jacobian_sign_changed)
    assert np.all(np.sign(np.asarray(jac.tau)[1:]) == SIGNGS)


# ---------------------------------------------------------------------------
# jit-compatibility of the state-producing path
# ---------------------------------------------------------------------------


def test_interior_guess_is_jittable(case):
    setup, modes, trig = case.setup, case.modes_new, case.trig_new

    def produce_state(args):
        b_rc, b_rs, b_zc, b_zs, r_c, r_s, z_c, z_s = args
        return interior_guess(
            boundary_R_cos=b_rc, boundary_R_sin=b_rs,
            boundary_Z_cos=b_zc, boundary_Z_sin=b_zs,
            raxis_c=r_c, raxis_s=r_s, zaxis_c=z_c, zaxis_s=z_s,
            modes=modes, trig=trig, s=setup.s_full,
        )

    args = (setup.boundary_R_cos, setup.boundary_R_sin, setup.boundary_Z_cos,
            setup.boundary_Z_sin, setup.raxis_c, setup.raxis_s,
            setup.zaxis_c, setup.zaxis_s)
    eager = produce_state(args)
    jitted = jax.jit(produce_state)(args)
    names = ("R_cos", "R_sin", "Z_cos", "Z_sin", "lambda_cos", "lambda_sin")
    for name, a, b in zip(names, eager, jitted):
        _allclose(b, a, f"jit {name}")
    # The jitted state reproduces the RunSetup state (same axis inputs).
    for name, a in zip(names, jitted):
        _allclose(a, getattr(setup, name), f"jit vs run_setup {name}")


def test_geometry_chain_from_setup_is_jittable(case):
    setup, modes, trig = case.setup, case.modes_new, case.trig_new

    def sign_flag(state_setup):
        geom = real_space_geometry(
            **geometry_state(state_setup, modes=modes),
            modes=modes, trig=trig, s=state_setup.s_full,
        )
        jac = half_mesh_jacobian(geom, s=state_setup.s_full)
        return jac.jacobian_sign_changed, jac.tau

    flag_eager, tau_eager = sign_flag(setup)
    flag_jit, tau_jit = jax.jit(sign_flag)(setup)
    assert bool(flag_eager) == bool(flag_jit) == False  # noqa: E712
    _allclose(tau_jit, tau_eager, "tau (jit)")
