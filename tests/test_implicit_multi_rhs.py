"""Parity tests for the batched reverse-AD helpers (PR #24).

``solve_implicit_with_aux`` and ``implicit_state_pullback_multi_rhs`` add a
vectorized state-cotangent pullback for callers with several objectives sharing
one fixed point (it reuses the residual/projector/VJP setup once and batches the
adjoint solves).  These check it reproduces the scalar ``solve_implicit`` VJP
exactly, so the batched path is a pure efficiency win with identical gradients.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmex.core import implicit as im
from vmex.core.input import VmecInput

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _solovev_setup():
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    cfg = im.make_config(inp, ftol=1e-13, max_iterations=2000)
    p0 = im.params_from_input(inp)
    return inp, cfg, p0


def test_solve_implicit_with_aux_matches_solve_implicit():
    """The aux helper returns the same converged state as solve_implicit."""
    _, cfg, p0 = _solovev_setup()
    state_aux, mask = im.solve_implicit_with_aux(p0, cfg)
    state_ref = im.solve_implicit(p0, cfg)
    for a, b in zip(jax.tree.leaves(state_aux), jax.tree.leaves(state_ref)):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
    # the mask is a 0/1 SpectralState of the same structure as the state
    assert jax.tree.structure(mask) == jax.tree.structure(state_ref)


def test_multi_rhs_pullback_matches_scalar_vjp():
    """Batched pullback == stacking the scalar solve_implicit VJP per cotangent."""
    _, cfg, p0 = _solovev_setup()
    x_star, mask = im.solve_implicit_with_aux(p0, cfg)

    # three distinct state cotangents (same pytree structure as x_star)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    gbars = [jax.tree.map(lambda a, k=k: jax.random.normal(k, a.shape, a.dtype), x_star)
             for k in keys]
    gbar_batch = jax.tree.map(lambda *a: jnp.stack(a), *gbars)

    g_multi = im.implicit_state_pullback_multi_rhs(p0, cfg, x_star, mask, gbar_batch)

    # scalar reference: the actual solve_implicit custom-VJP, applied per cotangent
    _, pullback = jax.vjp(lambda p: im.solve_implicit(p, cfg), p0)
    for i, gbar in enumerate(gbars):
        g_scalar = pullback(gbar)[0]
        g_multi_i = jax.tree.map(lambda a: a[i], g_multi)
        for a, b in zip(jax.tree.leaves(g_scalar), jax.tree.leaves(g_multi_i)):
            a = np.asarray(a); b = np.asarray(b)
            scale = np.max(np.abs(a)) + 1e-30
            assert np.max(np.abs(a - b)) <= 1e-8 * scale, "multi-rhs != scalar VJP"
