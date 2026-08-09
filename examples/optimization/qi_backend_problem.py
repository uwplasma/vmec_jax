"""Shared QI problem for the external-optimizer examples."""

from __future__ import annotations

from dataclasses import replace
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
    mpol = max(max_mode + 2, 5)
    ntor = mpol
    ntheta = 2 * mpol + 6
    nzeta = 2 * ntor + 4
    inp = replace(inp, delt=0.5)
    inp = inp.change_resolution(
        mpol=mpol,
        ntor=ntor,
        ntheta=ntheta,
        nzeta=nzeta,
    )
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

    def elongation_excess(state, runtime):
        return jnp.maximum(opt.max_elongation(state, runtime) - 8.0, 0.0)

    return opt.VmecProblem.from_tuples(
        inp,
        [
            (qi, 0.0, 1.0),
            (opt.aspect_ratio, 6.0, 0.1),
            (iota_floor, 0.0, 10.0),
            (elongation_excess, 0.0, 1.0),
        ],
        max_mode=max_mode,
        use_ess=True,
        # Report elapsed-time heartbeats while constructing the problem.
        progress=True,
    )


def iteration_budget(default: int) -> int:
    """Keep documentation smoke runs short without changing normal defaults."""
    if os.environ.get("VMEX_EXAMPLES_CI") == "1":
        return 1
    return int(os.environ.get("VMEX_MAXITER", str(default)))
