"""Matrix-free bordered linear algebra for a coupled plasma/NESTOR root."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

Array = Any
MatVec = Callable[[Array], Array]

__all__ = ["NestorBorderedOperator", "linearize_nestor_coupling"]


@dataclass(frozen=True)
class NestorBorderedOperator:
    """The coupled linearization ``[[A, B], [C, D]]``.

    ``A`` is the plasma-force linearization, ``D`` the NESTOR potential
    equation, and ``B``/``C`` their edge coupling. All four blocks are
    matrix-free callables; only their static input sizes are stored.
    """

    plasma: MatVec
    vacuum_to_plasma: MatVec
    plasma_to_vacuum: MatVec
    vacuum: MatVec
    plasma_size: int
    vacuum_size: int

    @property
    def shape(self) -> tuple[int, int]:
        size = self.plasma_size + self.vacuum_size
        return size, size

    def __call__(self, value: Array) -> Array:
        x = value[:self.plasma_size]
        q = value[self.plasma_size:]
        return jnp.concatenate((
            self.plasma(x) + self.vacuum_to_plasma(q),
            self.plasma_to_vacuum(x) + self.vacuum(q),
        ))

    def transpose(self, value: Array) -> Array:
        """Apply the exact transpose, generated from the four linear blocks."""
        template = jnp.zeros(
            (self.plasma_size + self.vacuum_size,), dtype=jnp.asarray(value).dtype
        )
        return jax.linear_transpose(self, template)(value)[0]

    def schur(self, plasma_solve: MatVec) -> MatVec:
        """Return ``q -> (D - C A^-1 B) q`` without materializing blocks."""
        return lambda q: self.vacuum(q) - self.plasma_to_vacuum(
            plasma_solve(self.vacuum_to_plasma(q))
        )

    def preconditioner(
        self, plasma_solve: MatVec, schur_solve: MatVec
    ) -> MatVec:
        """Exact block inverse when both supplied solves are exact."""
        def apply(rhs):
            rx = rhs[:self.plasma_size]
            rq = rhs[self.plasma_size:]
            ax = plasma_solve(rx)
            q = schur_solve(rq - self.plasma_to_vacuum(ax))
            x = plasma_solve(rx - self.vacuum_to_plasma(q))
            return jnp.concatenate((x, q))

        return apply


def linearize_nestor_coupling(
    plasma_residual: Callable[[Array, Array], Array],
    vacuum_system: Callable[[Array], tuple[Array, Array]],
    plasma: Array,
    potential: Array,
) -> NestorBorderedOperator:
    """Linearize a live plasma residual and NESTOR ``A(x)q-b(x)`` equation.

    ``vacuum_system(x)`` returns the NESTOR mode matrix and right-hand side
    assembled at plasma variables ``x``; no nested vacuum solve is performed.
    The returned four matrix-free blocks are exact JVPs of the two nonlinear
    residuals at ``(plasma, potential)``.
    """
    plasma = jnp.asarray(plasma)
    potential = jnp.asarray(potential)

    def vacuum_residual(x, q):
        matrix, rhs = vacuum_system(x)
        return matrix @ q - rhs

    def block(fun, primal):
        return jax.linearize(fun, primal)[1]

    return NestorBorderedOperator(
        block(lambda x: plasma_residual(x, potential), plasma),
        block(lambda q: plasma_residual(plasma, q), potential),
        block(lambda x: vacuum_residual(x, potential), plasma),
        block(lambda q: vacuum_residual(plasma, q), potential),
        int(plasma.size), int(potential.size),
    )
