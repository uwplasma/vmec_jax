"""Startup-latency contracts of the staged fixed-point refinement.

Pins the perf behavior behind the QA_optimization evaluation-latency fix (the
``tests/test_runtime_recompile_keys.py`` idiom — lower-only, no repeated full
solves): ``_refine_step_core`` is one reusable per-config executable.  Every
per-trial quantity (iterate, residual, parameters, frozen anchor, dof mask) is
a program ARGUMENT, never a baked closure constant, so consecutive optimizer
trial boundaries share one compiled program.  The previous host-eager step
re-linearized ``F`` per call and handed ``solvax.gcrot`` a fresh closure, so
the ``lax.while_loop`` it staged missed the compile cache on every trial —
one measured ``jit(while)`` recompile per optimizer evaluation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import numpy as np

from vmex.core import implicit as im
from vmex.core.input import VmecInput

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _small_solovev_setup():
    """Smallest meaningful analytic equilibrium (one fast forward solve)."""
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    inp = inp.change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )
    cfg = im.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    return inp, cfg, im.params_from_input(inp)


def test_refine_step_lowering_is_shared_across_trial_points() -> None:
    """Two trial iterates lower to one identical refinement program.

    The lowered text is what the compile cache keys on (one cfg, fixed
    avals), so byte-identical lowerings ARE executable reuse: the second and
    every later optimizer trial run the already-compiled refinement step
    instead of restaging a fresh ``while_loop`` closure.
    """
    _, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    P = im._dof_projector(cfg, mask)
    F = im.residual_fn(cfg, state, mask)
    z0 = P(state)
    fz0 = F(z0, p0)

    # A nearby trial: different values, same structure — exactly what a
    # line-search hands the refinement on consecutive evaluations.
    z1 = jax.tree.map(lambda a: a * (1.0 + 1.0e-6), z0)
    p1 = dataclasses.replace(p0, rbc=p0.rbc * (1.0 + 1.0e-6))
    fz1 = F(z1, p1)

    text_a = im._refine_step_core.lower(z0, fz0, p0, state, mask, cfg=cfg).as_text()
    text_b = im._refine_step_core.lower(z1, fz1, p1, state, mask, cfg=cfg).as_text()
    assert text_a == text_b
    # ... while the trial values genuinely differ.
    assert not np.array_equal(np.asarray(z0.R_cos), np.asarray(z1.R_cos))
