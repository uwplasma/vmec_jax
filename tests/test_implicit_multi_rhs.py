"""Parity tests for the batched reverse-AD helpers (PR #24).

``solve_implicit_with_aux`` and ``implicit_state_pullback_multi_rhs`` add a
vectorized state-cotangent pullback for callers with several objectives sharing
one fixed point (it reuses the residual/projector/VJP setup once and batches the
adjoint solves).  These check it reproduces the scalar ``solve_implicit`` VJP
exactly, so the batched path is a pure efficiency win with identical gradients.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import implicit as im
from vmex.core.errors import AdjointSolveError
from vmex.core.input import VmecInput

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _solovev_setup():
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    cfg = im.make_config(inp, ftol=1e-13, max_iterations=2000)
    p0 = im.params_from_input(inp)
    return inp, cfg, p0


def _small_solovev_setup():
    """Same analytic equilibrium on the smallest meaningful AD grid.

    Block assembly scales cubically with the modes per radial plane.  The
    forward/transpose/finite-difference identities tested below do not need
    the production-resolution Solovev deck, which remains covered by the
    full gradient and equilibrium parity lanes.
    """
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    inp = inp.change_resolution(
        mpol=3, ntor=0, ntheta=12, nzeta=4,
    )
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )
    cfg = im.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    return inp, cfg, im.params_from_input(inp)


def _tree_dot(left, right):
    return sum(
        jnp.vdot(a, b).real
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


def test_solve_implicit_with_aux_matches_solve_implicit():
    """The aux helper returns the same converged state as solve_implicit."""
    _, cfg, p0 = _solovev_setup()
    state_aux, mask = im.solve_implicit_with_aux(p0, cfg)
    state_ref = im.solve_implicit(p0, cfg)
    for a, b in zip(jax.tree.leaves(state_aux), jax.tree.leaves(state_ref)):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
    # the mask is a 0/1 SpectralState of the same structure as the state
    assert jax.tree.structure(mask) == jax.tree.structure(state_ref)
    with pytest.raises(ValueError, match="solver"):
        im.implicit_state_pullback_multi_rhs(
            p0, cfg, state_aux, mask, state_aux, solver="dense"
        )


@pytest.mark.full
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


def test_block_response_forward_transpose_and_fd():
    """One factorization serves tangent and transpose responses accurately."""
    inp, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    zero = jax.tree.map(jnp.zeros_like, p0)
    tangents = (
        dataclasses.replace(
            zero, rbc=zero.rbc.at[inp.ntor, 1].set(1.0)
        ),
        dataclasses.replace(zero, pres_scale=jnp.ones_like(zero.pres_scale)),
    )
    tangent_batch = jax.tree.map(lambda *x: jnp.stack(x), *tangents)
    state_tangent, report = im.implicit_state_tangent_multi_rhs(
        p0, cfg, state, mask, tangent_batch,
        probe_chunk_size=4, response_chunk_size=2,
    )
    assert np.all(np.asarray(report.converged))
    assert np.all(
        np.asarray(report.residual_norm) <= np.asarray(report.tolerance)
    )

    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    cotangents = jax.tree.map(
        lambda value: jnp.stack([
            jax.random.normal(key, value.shape, value.dtype) for key in keys
        ]),
        state,
    )
    pullback = im.implicit_state_pullback_multi_rhs(
        p0, cfg, state, mask, cotangents,
        solver="block", probe_chunk_size=4, response_chunk_size=2,
    )
    lhs = _tree_dot(state_tangent, cotangents)
    rhs = _tree_dot(tangent_batch, pullback)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-8, atol=2e-10)

    reference = im.implicit_state_pullback_multi_rhs(
        p0, cfg, state, mask, cotangents
    )
    for got, expected in zip(
        jax.tree.leaves(pullback), jax.tree.leaves(reference)
    ):
        np.testing.assert_allclose(got, expected, rtol=2e-8, atol=2e-10)

    for i, (tangent, step) in enumerate(zip(tangents, (3e-5, 1e-4))):
        directional = jax.jvp(
            lambda x, p: im.aspect_ratio(x, im.runtime_from_params(p, cfg)),
            (state, p0),
            (jax.tree.map(lambda value: value[i], state_tangent), tangent),
        )[1]
        finite_difference, info = im.frozen_path_directional_fd(
            p0, cfg, im.aspect_ratio, tangent, h=step
        )
        assert max(info["newton_res"]) < 1e-8
        np.testing.assert_allclose(
            directional, finite_difference, rtol=2e-5, atol=2e-8
        )


@pytest.mark.full
def test_block_pullback_rejects_unconverged_response():
    """The opt-in transpose path cannot return an uncertified gradient."""
    _, cfg, p0 = _solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    impossible = dataclasses.replace(
        cfg, adjoint_tol=1e-30, adjoint_maxiter=1, adjoint_restart=2
    )
    cotangent = jax.tree.map(lambda value: value[None], state)
    with pytest.raises(AdjointSolveError, match="block-preconditioned GCROT"):
        im.implicit_state_pullback_multi_rhs(
            p0, impossible, state, mask, cotangent,
            solver="block", probe_chunk_size=4,
        )


@pytest.mark.full
def test_block_response_lasym_parity():
    """All six LASYM state families share the same forward/transpose engine."""
    inp0 = VmecInput.from_file(DATA / "input.basic_non_stellsym_simsopt")
    inp = dataclasses.replace(
        inp0,
        ns_array=np.asarray([11]),
        ftol_array=np.asarray([1e-12]),
        niter_array=np.asarray([4000]),
    )
    cfg = im.make_config(inp, ftol=1e-12, max_iterations=4000)
    params = im.params_from_input(inp)
    state, mask = im.solve_implicit_with_aux(params, cfg)
    assert im._active_state_fields(cfg) == im._STATE_FIELDS

    zero = jax.tree.map(jnp.zeros_like, params)
    tangent = dataclasses.replace(
        zero, rbs=zero.rbs.at[inp.ntor + 1, 1].set(1.0)
    )
    tangent_batch = jax.tree.map(lambda value: value[None], tangent)
    state_tangent, report = im.implicit_state_tangent_multi_rhs(
        params, cfg, state, mask, tangent_batch,
        probe_chunk_size=4, response_chunk_size=1,
    )
    assert bool(np.asarray(report.converged[0]))

    cotangent = jax.tree.map(
        lambda value: jnp.linspace(
            -0.2, 0.3, value.size, dtype=value.dtype
        ).reshape((1,) + value.shape),
        state,
    )
    block = im.implicit_state_pullback_multi_rhs(
        params, cfg, state, mask, cotangent, solver="block",
        probe_chunk_size=4, response_chunk_size=1,
    )
    reference = im.implicit_state_pullback_multi_rhs(
        params, cfg, state, mask, cotangent
    )
    np.testing.assert_allclose(
        _tree_dot(state_tangent, cotangent),
        _tree_dot(tangent_batch, block),
        rtol=2e-6, atol=2e-9,
    )
    for got, expected in zip(
        jax.tree.leaves(block), jax.tree.leaves(reference)
    ):
        np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-9)
