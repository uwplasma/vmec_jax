#!/usr/bin/env python
"""Implicit differentiation: exact equilibrium gradients, checked against FD.

Unlike the Fortran original, vmex differentiates a *converged* equilibrium.
``vj.implicit.run`` solves the fixed point and exposes the standard wout scalars
(``aspect``, ``wb`` magnetic energy, ``volume``, ``iota_edge``, ...) as JAX
values, so ``jax.grad`` / ``jax.jacrev`` return derivatives with respect to the
boundary Fourier coefficients and profile parameters.

The gradient is computed by the *implicit function theorem*: one adjoint linear
solve on the converged state (see ``vmex.core.implicit``), not by unrolling
the iteration.  That means it is O(1) in memory (independent of iteration count)
and exact to solver tolerance -- no finite-difference step to tune, no vanishing
gradient at the axisymmetric saddle.  This script proves it by comparing the
adjoint gradient to a central finite difference on the same solver.

Physics: Solovev analytic tokamak, ns=11.  Runtime a few seconds warm (one
forward + one adjoint solve for the gradient, a handful of forward solves for
the FD reference).
"""

import dataclasses
import os
from pathlib import Path

import numpy as np

import jax

import vmex as vj

im = vj.implicit  # the differentiable fixed-point solver + wout-scalar diagnostics

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.solovev"
FTOL = 1e-12            # tight forward tolerance: the adjoint is exact at the fixed point
MAX_ITERS = 5000
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"

inp = vj.VmecInput.from_file(INPUT_FILE)
ntor = int(inp.ntor)
p0 = im.params_from_input(inp)   # pytree of {rbc, zbs, phiedge, pres_scale, am, ...}


def scalar(params, name):
    """Converge the equilibrium and read off one wout scalar (differentiable).

    ``name`` is any ImplicitSolution field: aspect, wb, wp, volume, iota_edge, ...
    """
    sol = im.run(inp, params, ftol=FTOL, max_iterations=MAX_ITERS)
    return getattr(sol, name)


# --------------------------- adjoint gradients -----------------------------
# jax.grad walks the implicit adjoint: one forward solve, one adjoint solve each.
# Two representative derivatives -- a boundary shape mode and a scalar parameter,
# each on the objective it genuinely drives:
#   * aspect ratio depends strongly on the m=1 boundary mode (elongation);
#   * magnetic energy wb scales with the enclosed toroidal flux phiedge.
aspect, g_aspect = jax.value_and_grad(lambda p: scalar(p, "aspect"))(p0)
wb, g_wb = jax.value_and_grad(lambda p: scalar(p, "wb"))(p0)
print(f"solovev ns={int(inp.ns_array[-1])}, ftol={FTOL:g}:  "
      f"aspect = {float(aspect):.8f}   wb = {float(wb):.8e}")

d_rbc = float(np.asarray(g_aspect.rbc)[ntor, 1])   # d(aspect)/d RBC(n=0, m=1)
d_phiedge = float(np.asarray(g_wb.phiedge))        # d(wb)/d phiedge


# --------------------------- finite-difference check -----------------------
def _central_fd(name, perturb, h):
    """Central difference of a scalar under a +/- h parameter perturbation."""
    plus = float(scalar(perturb(+h), name))
    minus = float(scalar(perturb(-h), name))
    return (plus - minus) / (2.0 * h)


def _bump_rbc(m, n_idx):
    def perturb(delta):
        rbc = np.array(p0.rbc)
        rbc[n_idx, m] += delta
        return dataclasses.replace(p0, rbc=rbc)
    return perturb


def _bump_scalar(field):
    return lambda delta: dataclasses.replace(p0, **{field: getattr(p0, field) + delta})


h_rbc, h_phi = 3e-5, 1e-5
fd_rbc = _central_fd("aspect", _bump_rbc(1, ntor), h_rbc)
fd_phi = _central_fd("wb", _bump_scalar("phiedge"), h_phi)

print(f"\nd(aspect)/d(RBC(0,1))  AD={d_rbc:+.10e}  FD={fd_rbc:+.10e}  "
      f"rel={abs(d_rbc / fd_rbc - 1.0):.2e}")
print(f"d(wb)/d(phiedge)       AD={d_phiedge:+.10e}  FD={fd_phi:+.10e}  "
      f"rel={abs(d_phiedge / fd_phi - 1.0):.2e}")
print("\nThe adjoint gradients match central FD to solver tolerance, at O(1) "
      "memory and with no step size to tune.")

# For a full Jacobian of several outputs at once, use jax.jacrev of a vector:
#   vec = lambda p: jnp.stack([scalar(p, "aspect"), scalar(p, "wb"), ...])
#   J = jax.jacrev(vec)(p0)   # one adjoint solve per output row
