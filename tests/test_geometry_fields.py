"""Tests for ``vmex.core.{geometry,fields}`` (jacobian.f / bcovar.f).

Field-by-field parity of the geometry/field chain with the legacy
parity-proven kernels (jacobian, metrics, B fields, energies/norms, surface
currents, tcon) was proven by the A/B suite that retired with the legacy
tree.  Kept here, on realistic profil3d.f initial states for sym 2D, sym 2D
ncurr=1, sym 3D and lasym decks:

- Jacobian sign-change detection (healthy state clean; an m=1 spike flips it),
- physical invariants of the field chain (finite energies, positive volume,
  signgs-consistent sqrt(g)),
- jit equivalence of the full pipeline and grad of ``wb`` w.r.t. the
  spectral coefficients.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from vmex.core.fields import (
    constraint_scaling,
    energies_and_force_norms,
    magnetic_fields,
    metric_elements,
    surface_currents,
)
from vmex.core.geometry import (
    apply_lambda_axis_closure,
    half_mesh_jacobian,
    real_space_geometry,
)
from vmex.core.input import VmecInput
from vmex.core.solver import (
    _geometry,
    _initial_state,
    prepare_runtime,
    resolution_from_input,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

RTOL = 1e-12
ATOL = 1e-13

CASES = [
    "solovev",  # 2D sym, ncurr=0, pressure profile lane
    "cth_like_fixed_bdy",  # 2D sym, nfp=5, ncurr=1 (add_fluxes lane)
    "li383_low_res",  # 3D sym (lthreed: guv/gvv, lambda closure)
    "up_down_asymmetric_tokamak",  # lasym
]


@pytest.fixture(scope="module", params=CASES, ids=CASES)
def case(request):
    """Initial state + a pure new-core pipeline closure for one deck."""
    name = request.param
    inp = VmecInput.from_file(DATA_DIR / f"input.{name}")
    rt = prepare_runtime(inp, resolution_from_input(inp))
    setup = rt.setup
    state = _initial_state(setup)
    s = setup.s_full

    (R_cos, R_sin, Z_cos, Z_sin), _geom0 = _geometry(state, rt)
    del _geom0
    new_inputs = dict(
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        lambda_cos=state.L_cos,
        lambda_sin=apply_lambda_axis_closure(
            state.L_sin, modes=rt.modes, ntor=rt.resolution.ntor
        ),
    )

    def new_pipeline(inputs):
        geom = real_space_geometry(**inputs, modes=rt.modes, trig=rt.trig, s=s)
        jac = half_mesh_jacobian(geom, s=s)
        mets = metric_elements(geom, s=s)
        mf = magnetic_fields(
            geometry=geom, jacobian=jac, metrics=mets, trig=rt.trig, s=s,
            phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
            signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
            ncurr=setup.ncurr, enclosed_current=setup.icurv,
        )
        en = energies_and_force_norms(
            jacobian=jac, metrics=mets, fields=mf, trig=rt.trig, s=s,
            signgs=setup.signgs,
        )
        tcon = constraint_scaling(
            tcon0=rt.tcon0, geometry=geom, jacobian=jac,
            total_pressure=mf.total_pressure, trig=rt.trig, s=s,
        )
        return geom, jac, mets, mf, en, tcon

    geom, jac, mets, mf, en, tcon = new_pipeline(new_inputs)
    return SimpleNamespace(
        name=name, inp=inp, rt=rt, setup=setup, s=s, state=state,
        new_inputs=new_inputs, new_pipeline=new_pipeline,
        geom=geom, jac=jac, mets=mets, mf=mf, en=en, tcon=tcon,
    )


# ---------------------------------------------------------------------------
# geometry.py: half-mesh Jacobian (jacobian.f)
# ---------------------------------------------------------------------------


def test_jacobian_sign_change_detection(case):
    # The profil3d initial states are healthy: no sign change, and the
    # interior sqrt(g) sign agrees with signgs.
    assert not bool(case.jac.jacobian_sign_changed)
    tau_interior = np.asarray(case.jac.tau)[1:]
    assert np.all(np.sign(tau_interior) == int(case.setup.signgs))

    # A large m=1 spike on one interior surface makes the flux surfaces
    # cross: tau flips sign there (VMEC irst = 2 / bad_jacobian_flag).
    m = np.asarray(case.rt.modes.m)
    k_m1 = int(np.nonzero(m == 1)[0][0])
    js_mid = int(case.s.shape[0]) // 2
    bad_inputs = dict(case.new_inputs)
    bad_inputs["R_cos"] = case.new_inputs["R_cos"].at[js_mid, k_m1].add(5.0)
    geom_bad = real_space_geometry(
        **bad_inputs, modes=case.rt.modes, trig=case.rt.trig, s=case.s
    )
    jac_bad = half_mesh_jacobian(geom_bad, s=case.s)
    assert bool(jac_bad.jacobian_sign_changed)


# ---------------------------------------------------------------------------
# fields.py: physical invariants (bcovar.f)
# ---------------------------------------------------------------------------


def test_field_chain_invariants(case):
    en, mf = case.en, case.mf
    for name in ("wb", "wp", "volume", "fnorm", "fnormL", "r1"):
        assert np.isfinite(float(getattr(en, name))), name
    assert float(en.wb) > 0.0
    assert float(en.volume) > 0.0
    assert float(en.fnorm) > 0.0
    # Interior total pressure (bsq) is positive; lamscale finite positive.
    assert np.all(np.asarray(mf.total_pressure)[1:] > 0.0)
    assert float(mf.lamscale) > 0.0
    # tcon is finite and positive on the interior surfaces.
    tcon = np.asarray(case.tcon)
    assert np.all(np.isfinite(tcon))
    assert np.all(tcon[1:-1] >= 0.0)


def test_surface_currents_finite(case):
    cur = surface_currents(
        bsubu=case.mf.bsubu, bsubv=case.mf.bsubv, trig=case.rt.trig,
        s=case.s, signgs=int(case.setup.signgs),
    )
    for name in ("buco", "bvco"):
        arr = np.asarray(getattr(cur, name))
        assert np.all(np.isfinite(arr)), name
    for name in ("ctor", "rbtor", "rbtor0"):
        assert np.isfinite(float(getattr(cur, name))), name
    # The toroidal-field profile bvco must be nonzero away from the axis.
    assert np.max(np.abs(np.asarray(cur.bvco)[1:])) > 0.0


# ---------------------------------------------------------------------------
# jit-compatibility and differentiability
# ---------------------------------------------------------------------------


def test_pipeline_is_jittable(case):
    def outputs(inputs):
        geom, jac, mets, mf, en, tcon = case.new_pipeline(inputs)
        return (
            jac.sqrt_g,
            jac.tau,
            jac.jacobian_sign_changed,
            mets.guu,
            mf.bsupu,
            mf.bsubv,
            mf.total_pressure,
            en.wb,
            en.fnorm,
            en.fnormL,
            tcon,
        )

    eager = outputs(case.new_inputs)
    jitted = jax.jit(outputs)(case.new_inputs)
    for idx, (a, b) in enumerate(zip(eager, jitted)):
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=RTOL, atol=ATOL, err_msg=f"jit output {idx}"
        )


def test_grad_of_wb_wrt_spectral_coefficients(case):
    def wb_of_R_cos(R_cos):
        inputs = dict(case.new_inputs)
        inputs["R_cos"] = R_cos
        *_, en, _tcon = case.new_pipeline(inputs)
        return en.wb

    grad = jax.grad(wb_of_R_cos)(case.new_inputs["R_cos"])
    grad_np = np.asarray(grad)
    assert grad_np.shape == np.asarray(case.new_inputs["R_cos"]).shape
    assert np.all(np.isfinite(grad_np))
    assert np.any(grad_np != 0.0)
