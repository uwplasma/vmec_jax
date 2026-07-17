"""2D block preconditioner: a matrix-free Newton step via ``jax.jvp`` + SOLVAX.

VMEC2000 counterpart
--------------------
``Sources/Hessian/precon2d.f`` — VMEC2000's optional 2D (``ictrl_prec2d``)
preconditioner.  It builds the full block-tridiagonal-in-radius Hessian of the
1D-preconditioned force map ``gc = g(xc)`` by finite-difference "jogs" of every
``(mode, type)`` column (``compute_blocks``), LU-factors it with BCYCLIC
(``blk3d_factor``), and in ``residue`` replaces the force by the block-solve
``gc <- -H^{-1} gc`` (``block_precond``, where ``H = d gc / d xc``).  This is a
Newton step on the already-1D-preconditioned residual; it converges stiff cases
(high beta, high aspect, high mode number) in far fewer iterations than the 1D
radial preconditioner alone.  Activation (``evolve.f``): finest grid only, once
``fsqr + fsqz + fsql < prec2d_threshold`` and ``iter2 >= 10``.

Modern (JAX) approach
---------------------
The forces are already traceable, so we get **exact** Hessian-vector products
for free from ``jax.jvp`` — no finite-difference jogs, no assembled blocks, no
BCYCLIC factorization.  Let ``g`` be the 1D-preconditioned force map
``state -> gc`` (frozen ``ns4`` cache — the 1D operator is a fixed linear
preconditioner during the Newton solve).  The Newton direction that drives
``g -> 0`` is

    delta = -J^{-1} g,     J = d g / d state   (block-tridiagonal in radius),

obtained by solving ``J delta = -g`` with **matrix-free GMRES**
(:func:`solvax.gmres`) whose matvec is ``v -> jvp(g, state, v)``.  Because the
1D preconditioner is baked into ``g`` (and hence into both ``J`` and the
right-hand side ``-g``), the ``M_1D^{-1}`` factor cancels exactly out of the
solve: ``delta`` is the *same* full Newton step ``-(dF/dstate)^{-1} F`` on the
raw force ``F`` regardless of the 1D operator — the 1D preconditioner only
accelerates GMRES (it makes ``J`` close to the identity near equilibrium, where
``M_1D`` approximates ``dF/dstate``).  This matches ``block_precond``'s sign:
``gc_out = -H^{-1} gc`` with ``H = d gc / d xc`` (``precon2d.f`` factorization
check ``block_dsave * x = -gc_save``).

The module is deliberately VMEC-agnostic: :func:`newton_direction` takes any
pytree-valued force map and a linearization point, so it is unit-testable
against a dense reference (``tests/test_preconditioner_2d.py``) and
could equally drive the block-tridiagonal :func:`solvax.block_thomas_truncated`
route if the blocks were assembled explicitly (they are not, by design — the
matrix-free HVP keeps peak memory at one force graph).

Wiring lives in :mod:`vmex.core.solver` (``_make_body``): when
``precon_type != "NONE"`` the traced iteration replaces the 1D force direction
by :func:`newton_direction` under a ``lax.cond`` gated on the activation
predicate, so the default 1D-only path is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
from jax.flatten_util import ravel_pytree

from solvax import gmres

__all__ = ["Prec2DConfig", "flat_operator", "newton_direction"]

Array = Any
PyTree = Any


@dataclass(frozen=True)
class Prec2DConfig:
    """Static configuration of the 2D block preconditioner (hashable meta).

    Attributes
    ----------
    threshold:
        Activate the Newton step once ``fsqr + fsqz + fsql < threshold`` on the
        finest grid (VMEC2000 ``prec2d_threshold`` / ``evolve.f``).  The default
        input value ``1e-30`` means "never" — the caller supplies a finite
        threshold to switch it on.
    start_iteration:
        Earliest iteration at which the Newton step may activate (VMEC2000
        skips the first 10 iterations: ``IF (iter2 < 10) ictrl_prec2d = 0``).
    step:
        Damping applied to the (full) Newton step, ``state += step * delta``.
        ``1.0`` is the undamped Newton update; VMEC2000 instead folds the step
        into its second-order Richardson stepper with ``time_step = 0.5``.
    gmres_restart:
        Arnoldi cycle size ``m`` of the inner GMRES (clamped to the problem
        size for tiny cases).
    gmres_max_restarts:
        Maximum number of GMRES restart cycles (outer iterations).
    gmres_rtol, gmres_atol:
        Inner GMRES tolerances on ``||b - J delta||``.  A loose ``rtol`` gives
        an *inexact* Newton step (cheaper, still super-linear far from the
        asymptotic regime); tighten for a near-exact step.
    finest:
        Whether this runtime is the finest multigrid stage (VMEC2000
        ``ns == ns_maxval``).  Single-grid solves are always finest; only the
        last multigrid stage sets this ``True``.
    """

    threshold: float
    start_iteration: int = 10
    step: float = 1.0
    gmres_restart: int = 40
    gmres_max_restarts: int = 3
    gmres_rtol: float = 1.0e-2
    gmres_atol: float = 0.0
    finest: bool = True


def flat_operator(
    force_map: Callable[[PyTree], PyTree], x0: PyTree
) -> tuple[Callable[[Array], Array], Callable[[Array], PyTree], int]:
    """Flat matrix-free Jacobian operator of ``force_map`` linearized at ``x0``.

    Returns ``(matvec, unravel, n)`` where ``matvec(v_flat) = J @ v_flat`` with
    ``J = d force_map / d x`` evaluated at ``x0`` (one ``jax.jvp`` — an exact
    Hessian-vector product, no finite differences), ``unravel`` reconstructs the
    pytree from a flat vector, and ``n`` is the flattened dimension.  The output
    pytree of ``force_map`` must share the structure of ``x0`` (a square
    Jacobian), as it does for the VMEC force map (state channels -> force
    channels).
    """
    x0_flat, unravel = ravel_pytree(x0)
    n = int(x0_flat.shape[0])

    def matvec(v_flat: Array) -> Array:
        _, jv = jax.jvp(force_map, (x0,), (unravel(v_flat),))
        return ravel_pytree(jv)[0]

    return matvec, unravel, n


def newton_direction(
    force_map: Callable[[PyTree], PyTree],
    x0: PyTree,
    rhs: PyTree,
    cfg: Prec2DConfig,
    *,
    guess: PyTree | None = None,
):
    """Matrix-free Newton direction: solve ``J delta = rhs`` for ``delta``.

    ``J = d force_map / d x`` at ``x0`` (exact HVP via :func:`flat_operator`),
    solved with restarted flexible GMRES (:func:`solvax.gmres`).  For the VMEC
    Newton step pass ``force_map = g`` (the 1D-preconditioned force),
    ``x0 = state`` and ``rhs = -g(state)`` so that ``delta = -J^{-1} g`` is the
    descent-consistent Newton update used in place of the raw force.

    Parameters
    ----------
    force_map, x0:
        As in :func:`flat_operator`.
    rhs:
        Right-hand side pytree, same structure as ``x0``.
    cfg:
        :class:`Prec2DConfig` supplying the GMRES parameters.
    guess:
        Optional initial-guess pytree for ``delta`` (defaults to zeros).

    Returns
    -------
    ``(delta, solution)``: the Newton-direction pytree and the raw
    :class:`solvax.krylov.KrylovSolution` (``residual_norm``, ``iterations``,
    ``converged``) for diagnostics.
    """
    matvec, unravel, n = flat_operator(force_map, x0)
    b_flat = ravel_pytree(rhs)[0]
    x0_flat = None if guess is None else ravel_pytree(guess)[0]
    restart = min(int(cfg.gmres_restart), max(n, 1))
    solution = gmres(
        matvec,
        b_flat,
        x0=x0_flat,
        restart=restart,
        rtol=float(cfg.gmres_rtol),
        atol=float(cfg.gmres_atol),
        max_restarts=int(cfg.gmres_max_restarts),
    )
    return unravel(solution.x), solution
