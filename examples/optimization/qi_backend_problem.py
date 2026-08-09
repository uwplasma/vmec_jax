"""Shared QI problem for the external-optimizer examples."""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual


def make_qi_problem() -> opt.VmecProblem:
    """Return one problem used unchanged by SciPy, JAXopt, and Optax."""
    data = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"
    inp = VmecInput.from_file(data)
    max_mode = int(os.environ.get("VMEX_MAX_MODE", "1"))
    inp = opt.prepare_optimization_input(inp, max_mode, minimum_mpol=5)
    qi = QIResidual(
        np.linspace(0.2, 1.0, 4),
        mboz=8,
        nboz=8,
        nphi=41,
        nalpha=9,
        n_levels=6,
    )

    def iota_floor(state, runtime):
        return jnp.maximum(0.3 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

    return opt.VmecProblem.from_tuples(
        inp,
        [
            (qi, 0.0, 1.0),
            (opt.aspect_ratio, 6.0, 0.1),
            (iota_floor, 0.0, 10.0),
        ],
        max_mode=max_mode,
        # "implicit": exact converged-equilibrium derivatives.
        # "finite_difference": independent equilibrium re-solves.
        derivative_method="implicit",
        # "auto" is recommended; advanced exact paths are
        # "block_tridiagonal", "forward_gmres", and "reverse_adjoint".
        implicit_jacobian_method="auto",
        # "auto" favors warm throughput; 1 can shorten cold compilation for
        # small problems.  Advanced users can change this one argument.
        jacobian_batch_size="auto",
        # "cost": w multiplies squared cost; "residual": w multiplies rows.
        weight_semantics="cost",
        use_ess=True,
        # Report elapsed-time heartbeats during first-use JAX preparation.
        progress=True,
    )


def iteration_budget(default: int) -> int:
    """Keep documentation smoke runs short without changing normal defaults."""
    if os.environ.get("VMEX_EXAMPLES_CI") == "1":
        return 1
    return int(os.environ.get("VMEX_MAXITER", str(default)))
