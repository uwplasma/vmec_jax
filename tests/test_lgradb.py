"""Traceable ``L_grad_B`` objective (plan_pre_vmex Item E, part 1).

Validation gates for :func:`vmec_jax.core.optimize.l_grad_b_state`, the
implicit-adjoint-compatible ``(state, runtime)`` lane of the wout-engine
:func:`~vmec_jax.core.optimize.l_grad_b`:

a. **Value parity** — the traceable hard-min lane vs the wout lane on three
   decks (solovev 2D, li383_low_res 3D ncurr=1, LandremanPaul2021_QA_lowres
   3D QA), same sampling surfaces.  Both lanes share the pointwise math
   (:func:`vmec_jax.core.statephysics._lgradb_grid`); the traceable lane
   rebuilds the wout coefficient tables (``rmnc/zmns`` renormalization, the
   ``wrout.f`` Nyquist analysis of ``B^u/B^v``) in jnp, so they agree to
   float round-off — measured 0 to 4.4e-16 relative on all three decks
   (2026-07-17, x64 CPU); asserted at rtol 1e-12 (headroom for BLAS/einsum
   reassociation), far inside the 1e-6 plan gate.
b. **Gradient** — ``jax.grad`` of the smooth (soft-min) objective through
   the implicit solve vs :func:`vmec_jax.core.implicit.frozen_path_directional_fd`
   on solovev, one boundary dof: measured rel 1.7e-6, gate 1e-4.
c. A ``jac="implicit"`` least-squares smoke including the new term.

Converged states are cached in ``/tmp`` (same pattern as test_optimize.py) so
repeated runs skip the solves.  No golden fixtures needed — every comparison
is self-referential against the wout engine / frozen-path FD of the same
state.  The LandremanPaul deck exhausts its iteration budget at this
resolution (``converged=False``); parity is a property of the returned state,
not of convergence, so it is asserted regardless.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("netCDF4")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vmec_jax.core.input import VmecInput  # noqa: E402
from vmec_jax.core import implicit as im  # noqa: E402
from vmec_jax.core import optimize as opt  # noqa: E402

pytestmark = [pytest.mark.usefixtures("_module_jit_enabled")]  # full solves: jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
CACHE_DIR = Path("/tmp/vmec_jax_test_cache_lgradb")

_STATE_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")

# Value-parity decks; achieved relative difference measured 2026-07-17
# (x64 CPU): solovev 4.4e-16, li383_low_res 0.0, LandremanPaul 0.0.
PARITY_DECKS = ("solovev", "li383_low_res", "LandremanPaul2021_QA_lowres")
PARITY_RTOL = 1e-12          # asserted (plan gate: 1e-6)

SOFTMIN_K = 50.0             # [1/m]; soft-min bias <= log(24*24)/k ~ 0.127 m


def _cached_eq(deck: str) -> opt.Equilibrium:
    """Converged (or budget-exhausted) equilibrium of a bundled deck, cached."""
    from vmec_jax.core.solver import (
        SpectralState,
        prepare_runtime,
        resolution_from_input,
    )

    inp = VmecInput.from_file(DATA_DIR / f"input.{deck}")
    cache = CACHE_DIR / f"{deck}_state.npz"
    if cache.exists():
        data = np.load(cache)
        state = SpectralState(**{k: jnp.asarray(data[k]) for k in _STATE_FIELDS})
        result = SimpleNamespace(
            fsqr=float(data["fsqr"]), fsqz=float(data["fsqz"]),
            fsql=float(data["fsql"]), iterations=int(data["iterations"]),
            converged=bool(data["converged"]))
        ns = int(np.shape(state.R_cos)[0])
        runtime = prepare_runtime(inp, resolution_from_input(inp, ns=ns))
        return opt.Equilibrium(inp=inp, state=state, runtime=runtime, result=result)
    eq = opt.solve_equilibrium(inp)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache,
             **{k: np.asarray(getattr(eq.state, k)) for k in _STATE_FIELDS},
             fsqr=eq.result.fsqr, fsqz=eq.result.fsqz, fsql=eq.result.fsql,
             iterations=eq.result.iterations, converged=eq.result.converged)
    return eq


# ---------------------------------------------------------------------------
# a. value parity: traceable hard-min lane vs the wout lane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck", PARITY_DECKS)
def test_value_parity_vs_wout_lane(deck):
    """Hard-min ``l_grad_b_state`` == wout-lane ``l_grad_b`` to float round-off.

    Checked on the default (edge) surface and one interior surface — the
    interior case additionally exercises the central (two-sided) radial
    stencil of the half-mesh field tables.
    """
    eq = _cached_eq(deck)
    ns = int(np.shape(eq.state.R_cos)[0])
    for s_index in (-1, ns // 2):
        ref = float(opt.l_grad_b(eq, s_index=s_index))
        got = float(opt.l_grad_b_state(eq.state, eq.runtime, s_index=s_index))
        assert np.isfinite(ref) and ref > 0.0
        np.testing.assert_allclose(
            got, ref, rtol=PARITY_RTOL,
            err_msg=f"{deck}, s_index={s_index}: traceable vs wout lane")


def test_value_parity_nondefault_grid():
    """Parity holds on a non-default angular sampling grid too."""
    eq = _cached_eq("li383_low_res")
    ref = float(opt.l_grad_b(eq, ntheta=36, nphi=30))
    got = float(opt.l_grad_b_state(eq.state, eq.runtime, ntheta=36, nphi=30))
    np.testing.assert_allclose(got, ref, rtol=PARITY_RTOL)


def test_jit_parity_and_softmin_bounds():
    """jit == eager; the soft minimum is a lower bound within log(N)/k."""
    eq = _cached_eq("solovev")
    hard = float(opt.l_grad_b_state(eq.state, eq.runtime))
    jitted = float(jax.jit(
        lambda s: opt.l_grad_b_state(s, eq.runtime))(eq.state))
    np.testing.assert_allclose(jitted, hard, rtol=1e-12)
    soft = float(opt.l_grad_b_state(eq.state, eq.runtime, softmin_k=SOFTMIN_K))
    assert soft <= hard + 1e-12
    assert hard - soft <= np.log(24 * 24) / SOFTMIN_K


def test_implicit_lane_dispatch_accepts_state_term():
    """``_traceable_term`` accepts the (state, rt) lane; still rejects the wout lane."""
    assert opt._traceable_term(opt.l_grad_b_state) is opt.l_grad_b_state
    smooth = functools.partial(opt.l_grad_b_state, softmin_k=SOFTMIN_K)
    assert opt._traceable_term(smooth) is smooth
    with pytest.raises(ValueError, match="l_grad_b_state"):
        opt._traceable_term(opt.l_grad_b)      # wout lane: FD-only, by design


# ---------------------------------------------------------------------------
# b. gradient of the smooth objective vs frozen-path FD (solovev)
# ---------------------------------------------------------------------------


def test_gradient_vs_frozen_path_fd():
    """``jax.grad`` of the soft-min objective == frozen-path central FD.

    One boundary dof (``RBC(0,1)``, h = 3e-5 — the solovev boundary step of
    test_implicit_grad.py) through the implicit solve.  Measured rel 1.7e-6
    (2026-07-17, x64 CPU); gate 1e-4.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, ftol=1e-14, max_iterations=2000)
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)

    def metric(state, rt):
        return opt.l_grad_b_state(state, rt, softmin_k=SOFTMIN_K)

    ad = float(np.asarray(jax.grad(
        lambda p: metric(im.solve_implicit(p, cfg),
                         im.runtime_from_params(p, cfg)))(p0).rbc)[ntor, 1])

    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent = dataclasses.replace(zero, rbc=zero.rbc.at[ntor, 1].set(1.0))
    fd, info = im.frozen_path_directional_fd(p0, cfg, metric, tangent, h=3e-5)
    res = max(info["newton_res"])
    rel = abs(ad / fd - 1.0)
    print(f"\n[solovev] d(softmin L_grad_B)/d(RBC(0,1)) h=3e-5: "
          f"AD={ad:+.10e}  frozen-FD={fd:+.10e}  rel={rel:.2e} "
          f"(Newton res {res:.0e})")
    assert res < 1e-8, f"frozen solve not converged (res {res:.1e})"
    assert rel <= 1e-4


# ---------------------------------------------------------------------------
# c. jac="implicit" least-squares smoke including the new term
# ---------------------------------------------------------------------------


def test_least_squares_implicit_smoke_with_lgradb():
    """3-nfev implicit least squares with an ``l_grad_b_state`` term improves.

    Aspect target plus the smooth ``L_grad_B`` term — the combination the
    objectives README row optimizes — through ``jac="implicit"`` end to end
    (term dispatch, implicit Jacobian rows, trust-region step).
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    eq0 = _cached_eq("solovev")
    term = functools.partial(opt.l_grad_b_state, softmin_k=SOFTMIN_K)
    terms = [(opt.aspect_ratio, 4.0, 1.0), (term, 3.0, 0.5)]
    cost0 = 0.5 * sum(
        (w * (float(f(eq0.state, eq0.runtime)) - t)) ** 2 for f, t, w in terms)
    res = opt.least_squares(terms, inp, max_mode=1, jac="implicit", max_nfev=3)
    assert res.nfev <= 3
    assert np.isfinite(res.cost)
    assert res.cost < cost0
    assert isinstance(res.input, VmecInput)
