"""Compile-budget guards for the solver lanes and the cold python-API solve.

Two creep modes have historically surfaced as user-visible cold-start
regressions months after the commit that caused them (most recently the
0.3-class -> 0.8-class CLI start repaired in #227):

- **lane-HLO creep** — the traced iteration lanes accrete ops until their
  compile time dominates the start; and
- **eager-dispatch creep** — host-eager setup/export code dispatches more
  and more single-op XLA programs (``jit(copy)``, ``jit(multiply)``, ...)
  outside the jitted passes.

Both are pinned here against measured budgets with generous (~25%)
headroom, DESC-benchmark style, so a regression fails CI loudly at the
offending commit instead of surfacing as a complaint later.  Neither guard
needs an XLA compile in-process: the lane guard stops at StableHLO
lowering, and the dispatch guard counts compile log records inside one
short subprocess solve (measured ~3 s), keeping the module honest about
cold-start state without slowing the fast lanes.

When a budget trips because of *deliberate* new physics or lane structure,
re-measure (instructions at each constant) and move the constant in the
same commit, stating the new measured value.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from vmex.core import solver
from vmex.core.input import VmecInput

ROOT = Path(__file__).resolve().parents[1]
SOLOVEV_DECK = ROOT / "examples" / "data" / "input.solovev"

#: StableHLO text-size ceilings for the two production iteration lanes,
#: lowered on the solovev deck's own resolution (ns=11, mpol=6, ntor=0).
#: Text size is a deliberately blunt but stable proxy for traced-graph
#: size: op creep in ``_make_body`` grows it roughly linearly, and it needs
#: no XLA compile to evaluate.  Measured 2026-09-01 at 03011303 (jax
#: 0.11.1, CPU, x64): ``_block_lane`` 504,324 chars, ``_while_lane``
#: 520,957 chars; ceilings are measured + ~25%.  Re-measure with
#: ``lane.lower(carry, rt).as_text()`` (see ``_lane_operands`` below).
_LANE_STABLEHLO_CEILINGS = {
    "_block_lane": 630_000,
    "_while_lane": 650_000,
}
#: A lane this size cannot legitimately shrink an order of magnitude; a
#: sub-floor text means the metric itself degenerated (wrong carry, stub
#: lowering), not that the lane got cheap.
_LANE_STABLEHLO_FLOOR = 100_000

#: Total XLA programs compiled by ONE cold python-API ``solve`` of the
#: solovev deck — the jitted lanes plus every eager single-op dispatch of
#: the setup/export/printout passes.  Measured 2026-09-01 at 03011303
#: (jax 0.11.1, CPU, x64, persistent compilation cache disabled): 98
#: programs, of which 92 are eager single-op programs; stable across
#: repeat runs.  Ceiling is measured + ~27%.  Re-measure by running
#: ``_COMPILE_COUNT_SCRIPT`` below by hand.
_COLD_SOLVE_PROGRAM_CEILING = 125


@pytest.fixture(autouse=True)
def _enable_jit():
    """Lane lowering needs JIT (the repo conftest disables it globally)."""
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


@pytest.fixture(scope="module")
def _lane_operands():
    """The exact ``lane(carry, rt)`` operands a solovev CLI solve dispatches."""
    inp = VmecInput.from_file(str(SOLOVEV_DECK))
    rt = solver.prepare_runtime(inp, solver.resolution_from_input(inp))
    carry = jax.tree.map(
        jnp.array, solver._initial_carry(solver._initial_state(rt.setup), rt, ijacob=0)
    )
    return carry, rt


@pytest.mark.parametrize("lane_name", sorted(_LANE_STABLEHLO_CEILINGS))
def test_lane_stablehlo_size_stays_under_budget(_lane_operands, lane_name):
    """Lowering only — no XLA compile — so the fast guard stays fast."""
    carry, rt = _lane_operands
    text = getattr(solver, lane_name).lower(carry, rt).as_text()
    assert "func.func public @main" in text  # a real lowered module
    size = len(text)
    assert _LANE_STABLEHLO_FLOOR < size, (
        f"{lane_name}: StableHLO shrank to {size} chars — the lowering is "
        "degenerate, or the lane got structurally cheaper; re-measure and "
        "move both the ceiling and this floor."
    )
    assert size <= _LANE_STABLEHLO_CEILINGS[lane_name], (
        f"{lane_name}: StableHLO grew to {size} chars, over the "
        f"{_LANE_STABLEHLO_CEILINGS[lane_name]}-char budget. If the growth "
        "is deliberate, re-measure and move the constant in this commit; "
        "otherwise an op crept into the traced iteration body."
    )


# The counter follows the repo-external measurement pattern: install a
# ``logging.Handler`` on ``logging.getLogger("jax")`` AFTER importing vmex
# (vmex configures JAX logging on import), enable ``jax_log_compiles`` for
# the duration, and count the per-program ``Finished XLA compilation of``
# records that jax's dispatch logger emits (jax >= 0.10 wording).  The
# persistent compilation cache is disabled so a warm on-disk cache cannot
# hide (or skip) compile records.
_COMPILE_COUNT_SCRIPT = """\
import logging
import sys

import vmex as vj  # must precede the handler: import configures JAX logging
import jax


class CompileCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.count = 0

    def emit(self, record):
        if "Finished XLA compilation of" in record.getMessage():
            self.count += 1


counter = CompileCounter()
jax_logger = logging.getLogger("jax")
jax_logger.addHandler(counter)
jax_logger.setLevel(logging.INFO)  # compile records log at WARNING
jax.config.update("jax_log_compiles", True)
jax.config.update("jax_enable_compilation_cache", False)

result = vj.solve(vj.VmecInput.from_file(sys.argv[1]))
assert result.converged, "solovev budget solve did not converge"
print("COMPILED_PROGRAMS:", counter.count)
"""


def test_cold_solve_compiled_program_count_stays_under_budget():
    """One cold subprocess solve, so suite order cannot pre-warm any cache.

    Unmarked by convention: the whole probe measures ~10 s (interpreter +
    imports + a ~3 s solovev solve), inside the repo's unmarked-medium
    band (compare ``tests/test_cli.py``); the ``pr-fast`` manifest lane
    excludes this module.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    proc = subprocess.run(
        [sys.executable, "-c", _COMPILE_COUNT_SCRIPT, str(SOLOVEV_DECK)],
        capture_output=True, text=True, timeout=600, cwd=ROOT, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    match = re.search(r"COMPILED_PROGRAMS: (\d+)", proc.stdout)
    assert match, proc.stdout + proc.stderr
    count = int(match.group(1))
    assert 0 < count, "no compile records — the log-capture pattern broke"
    assert count <= _COLD_SOLVE_PROGRAM_CEILING, (
        f"a cold solovev solve compiled {count} XLA programs, over the "
        f"{_COLD_SOLVE_PROGRAM_CEILING}-program budget. New eager "
        "single-op dispatch crept into the setup/export/printout passes "
        "(see #227); jit the new pass, or re-measure and move the "
        "constant if the growth is deliberate."
    )
