"""Tests for ``vmex.core.preconditioner_2d`` (2D block preconditioner).

Fast unit tests exercise the matrix-free block operator directly:

- :func:`test_hvp_matches_dense` pins the exact Hessian-vector product
  (``jax.jvp`` of the 1D-preconditioned force map) against a dense reference
  Jacobian (``jax.jacfwd``) on a tiny real solver state — the "block assembly
  vs dense" check the roadmap (R10.2) asks for;
- :func:`test_newton_direction_solves_dense_system` pins the SOLVAX-GMRES
  Newton solve against ``jnp.linalg.solve`` on a synthetic nonsingular
  operator (independent of the VMEC physics).

The ``full``-marked :func:`test_2d_fewer_iterations_same_equilibrium` runs a
stiff single-grid solve twice (1D vs 2D) and asserts the 2D preconditioner
reaches the *same* equilibrium in *fewer* iterations (the R20 showcase claim).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from vmex.core.fourier import Resolution
from vmex.core.input import VmecInput
from vmex.core import solver as solver_module
from vmex.core.preconditioner_2d import (
    Prec2DConfig, flat_operator, newton_direction,
)
from vmex.core.solver import (
    SpectralState, _initial_state, _preconditioned_force_signed, _resolve_prec2d,
    evaluate_forces, prepare_runtime, resolution_from_input, solve,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

_ALL = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def _tiny_reduced_map(case: str = "circular_tokamak", ns: int = 6):
    """Build a tiny 1D-preconditioned force map over the evolved channels.

    Returns ``(g_reduced, x0, gc)``: the reduced force map (dict of the
    non-trivial symmetric channels -> dict), the linearization point ``x0``,
    and the ``gc`` that :func:`evaluate_forces` produced at that point (so the
    reduced map's primal is exactly the solver's force — a consistency anchor).
    """
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{case}"))
    r0 = resolution_from_input(inp)
    res = Resolution(mpol=r0.mpol, ntor=r0.ntor, ntheta=r0.ntheta, nzeta=r0.nzeta,
                     nfp=r0.nfp, lasym=r0.lasym, ns=ns)
    rt = prepare_runtime(inp, res)
    state = _initial_state(rt.setup)
    gc, _res, diag = evaluate_forces(state, rt)
    cache = diag.cache
    channels = _ALL if rt.setup.lasym else ("R_cos", "Z_sin", "L_sin")
    inactive = tuple(c for c in _ALL if c not in channels)
    it = jnp.asarray(30)
    fsqz_prev = jnp.asarray(1.0e-8)

    def to_full(reduced: dict) -> SpectralState:
        full = dict(reduced)
        for c in inactive:
            full[c] = getattr(state, c) * 0.0
        return SpectralState(**{c: full[c] for c in _ALL})

    def g_reduced(reduced: dict) -> dict:
        gc_full = _preconditioned_force_signed(
            to_full(reduced), cache, rt, iteration=it, fsqz_previous=fsqz_prev)
        return {c: getattr(gc_full, c) for c in channels}

    x0 = {c: getattr(state, c) for c in channels}
    return g_reduced, x0, gc


def test_reduced_map_reproduces_solver_force():
    """The frozen-cache force map's primal equals the solver's ``gc`` exactly."""
    g_reduced, x0, gc = _tiny_reduced_map()
    g0 = g_reduced(x0)
    for c in g0:
        np.testing.assert_allclose(
            np.asarray(g0[c]), np.asarray(getattr(gc, c)), rtol=0, atol=0,
            err_msg=f"channel {c}",
        )


def test_hvp_matches_dense():
    """Exact HVP (``jvp``) equals the dense Jacobian (``jacfwd``) at ``x0``."""
    g_reduced, x0, _gc = _tiny_reduced_map()
    matvec, unravel, n = flat_operator(g_reduced, x0)
    x0_flat, _ = ravel_pytree(x0)

    def g_flat(v):
        return ravel_pytree(g_reduced(unravel(v)))[0]

    jacobian = jax.jacfwd(g_flat)(x0_flat)
    rng = np.random.default_rng(1)
    for _ in range(3):
        v = jnp.asarray(rng.standard_normal(n))
        np.testing.assert_allclose(
            np.asarray(matvec(v)), np.asarray(jacobian @ v), rtol=1e-9, atol=1e-10,
        )


def test_newton_direction_solves_dense_system():
    """GMRES Newton solve recovers ``A^{-1} b`` on a synthetic linear map."""
    rng = np.random.default_rng(0)
    n = 12
    a = rng.standard_normal((n, n)) + n * np.eye(n)  # diagonally dominant
    b = rng.standard_normal(n)
    a_j = jnp.asarray(a)
    force_map = lambda d: {"x": a_j @ d["x"]}  # noqa: E731
    x0 = {"x": jnp.zeros(n)}
    rhs = {"x": jnp.asarray(b)}
    cfg = Prec2DConfig(threshold=1.0, gmres_restart=n, gmres_max_restarts=4,
                       gmres_rtol=1e-12)
    delta, sol = newton_direction(force_map, x0, rhs, cfg)
    ref = np.linalg.solve(a, b)
    assert bool(sol.converged)
    np.testing.assert_allclose(np.asarray(delta["x"]), ref, rtol=1e-6, atol=1e-8)


def test_prec2d_config_is_hashable():
    """The config is pytree *meta* (static): it must hash for executable reuse."""
    cfg = Prec2DConfig(threshold=1e-6)
    assert hash(cfg) == hash(Prec2DConfig(threshold=1e-6))
    assert hash(cfg) != hash(Prec2DConfig(threshold=1e-7))


def test_precon_type_has_explicit_non_aliasing_semantics() -> None:
    """NONE/DEFAULT are 1-D; other VMEC Krylov names are not GMRES aliases."""
    assert _resolve_prec2d(
        VmecInput(precon_type="NONE"), None, None, None
    ) is None
    assert _resolve_prec2d(
        VmecInput(precon_type="DEFAULT"), None, None, None
    ) is None
    assert isinstance(
        _resolve_prec2d(VmecInput(precon_type="GMRES"), None, None, None),
        Prec2DConfig,
    )
    with pytest.raises(NotImplementedError, match="CG, GMRESR, and TFQMR"):
        _resolve_prec2d(VmecInput(precon_type="TFQMR"), None, None, None)


def test_precon_none_keeps_vmec2000_radial_preconditioner(monkeypatch) -> None:
    """PRECON_TYPE=NONE disables 2-D blocks, never scalfor/faclam."""
    inp = replace(
        VmecInput.from_file(str(DATA_DIR / "input.circular_tokamak")),
        precon_type="NONE",
        raxis_c=np.asarray([6.0]),
    )
    resolution = resolution_from_input(inp, ns=7)
    runtime = prepare_runtime(inp, resolution)
    state = _initial_state(runtime.setup)
    original = solver_module.apply_radial_preconditioner
    calls = 0

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_module, "apply_radial_preconditioner", tracked)
    _gc, _residuals, diagnostics = evaluate_forces(state, runtime)
    assert calls == 1
    assert bool(diagnostics.health.pipeline.radial_preconditioner_safe)
    assert runtime.prec2d is None


@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_2d_fewer_iterations_same_equilibrium():
    """2D preconditioner: same equilibrium as 1D, strictly fewer iterations.

    Stiff high-aspect axisymmetric tokamak (``circular_tokamak_aspect_100``) at
    a moderate ``ns``.  The 2D Newton step must converge to the same ``wb`` /
    geometry as the 1D radial preconditioner while taking materially fewer
    iterations (the R20 showcase evidence).
    """
    inp = VmecInput.from_file(str(DATA_DIR / "input.circular_tokamak_aspect_100"))
    r0 = resolution_from_input(inp)
    res = Resolution(mpol=r0.mpol, ntor=r0.ntor, ntheta=r0.ntheta, nzeta=r0.nzeta,
                     nfp=r0.nfp, lasym=r0.lasym, ns=51)

    r1 = solve(inp, res, mode="jit", ftol=1e-11, max_iterations=20000)
    cfg = Prec2DConfig(threshold=1e-6, gmres_restart=60, gmres_max_restarts=3,
                       gmres_rtol=3e-3)
    r2 = solve(inp, res, mode="jit", ftol=1e-11, max_iterations=20000, prec2d=cfg)

    assert r1.converged and r2.converged
    # Same equilibrium.  wb (a stationary energy functional) is the tight
    # parity witness — it matches to ~1e-10 here; the raw geometry coordinates
    # are only pinned to the ftol-limited state scatter (~1e-6), so the
    # mode-by-mode check uses that looser, convergence-limited tolerance.
    assert abs(r1.wb - r2.wb) / abs(r1.wb) < 1e-8
    scale = float(np.max(np.abs(r1.rmnc)))
    np.testing.assert_allclose(r2.rmnc, r1.rmnc, rtol=1e-4, atol=1e-5 * scale)
    np.testing.assert_allclose(r2.zmns, r1.zmns, rtol=1e-4, atol=1e-5 * scale)
    # fewer iterations (comfortable margin; measured ~5x on this case)
    assert r2.iterations < 0.75 * r1.iterations
