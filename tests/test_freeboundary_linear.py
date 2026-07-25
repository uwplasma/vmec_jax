"""Coupled plasma/NESTOR bordered-operator tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vmex.core.freeboundary_linear import (
    NestorBorderedOperator, linearize_nestor_coupling,
)


def _operator():
    a = jnp.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    b = jnp.asarray([[0.2, 0.0], [0.1, -0.3], [0.0, 0.4]])
    c = jnp.asarray([[0.5, 0.1, 0.0], [0.0, -0.2, 0.6]])
    d = jnp.asarray([[2.0, 0.2], [0.2, 1.5]])
    op = NestorBorderedOperator(
        lambda x: a @ x, lambda q: b @ q,
        lambda x: c @ x, lambda q: d @ q, 3, 2,
    )
    return op, jnp.block([[a, b], [c, d]]), a, b, c, d


def test_nestor_bordered_value_transpose_jvp_and_vjp():
    op, dense, *_ = _operator()
    x = jnp.arange(5.0)
    tangent = jnp.linspace(-0.3, 0.4, 5)
    with jax.disable_jit(False):
        value, jvp = jax.jit(
            lambda y, dy: jax.jvp(op, (y,), (dy,))
        )(x, tangent)
    np.testing.assert_allclose(value, dense @ x, atol=1e-14)
    np.testing.assert_allclose(jvp, dense @ tangent, atol=1e-14)
    np.testing.assert_allclose(op.transpose(x), dense.T @ x, atol=1e-14)
    grad = jax.grad(lambda y: jnp.sum(op(y) ** 2))(x)
    np.testing.assert_allclose(grad, 2.0 * dense.T @ dense @ x, atol=1e-13)


def test_nestor_schur_and_block_inverse():
    op, dense, a, b, c, d = _operator()
    a_solve = lambda x: jnp.linalg.solve(a, x)  # noqa: E731
    schur_dense = d - c @ jnp.linalg.solve(a, b)
    schur_solve = lambda x: jnp.linalg.solve(schur_dense, x)  # noqa: E731
    q = jnp.asarray([0.3, -0.1])
    np.testing.assert_allclose(op.schur(a_solve)(q), schur_dense @ q, atol=1e-14)
    rhs = jnp.arange(1.0, 6.0)
    with jax.disable_jit(False):
        actual = jax.jit(op.preconditioner(a_solve, schur_solve))(rhs)
    np.testing.assert_allclose(actual, jnp.linalg.solve(dense, rhs), atol=1e-13)


def test_live_nestor_factory_differentiates_all_four_blocks():
    x0 = jnp.asarray([0.2, -0.3])
    q0 = jnp.asarray([0.4, 0.1])

    def plasma_residual(x, q):
        return jnp.asarray([x[0] ** 2 + q[0] * q[1], x[1] + q[1] ** 2])

    def vacuum_system(x):
        matrix = jnp.asarray([[2.0 + x[0], x[1]], [x[1], 1.5 - x[0]]])
        return matrix, jnp.asarray([1.0 + x[1], -0.2 + x[0]])

    op = linearize_nestor_coupling(plasma_residual, vacuum_system, x0, q0)

    def coupled(value):
        x, q = value[:2], value[2:]
        matrix, rhs = vacuum_system(x)
        return jnp.concatenate((plasma_residual(x, q), matrix @ q - rhs))

    base = jnp.concatenate((x0, q0))
    dense = jax.jacfwd(coupled)(base)
    np.testing.assert_allclose(jax.jacfwd(op)(jnp.zeros_like(base)), dense, atol=1e-14)
    blocks = (
        jax.jacfwd(op.plasma)(jnp.zeros_like(x0)),
        jax.jacfwd(op.vacuum_to_plasma)(jnp.zeros_like(q0)),
        jax.jacfwd(op.plasma_to_vacuum)(jnp.zeros_like(x0)),
        jax.jacfwd(op.vacuum)(jnp.zeros_like(q0)),
    )
    assert all(float(jnp.linalg.norm(block)) > 0.0 for block in blocks)
