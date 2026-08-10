"""Verbose CLI trajectory emission: incremental-transfer contract.

The CLI lane mirrors trajectory rows to the host incrementally (only the
rows written since the previous block round-trip are transferred).  The
emitted screen lines must be exactly the lines a full-trajectory replay
produces — pinned here against the jit lane's complete trajectory buffer,
which is bit-identical to the CLI lane's.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pytest

from vmex.core import solver
from vmex.core.input import VmecInput

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

#: screen_line rows: iteration then adjacent fixed-width scientific fields.
_ROW = re.compile(r"^\s*\d+\s*\d\.\d{2}E[-+]\d{2}")


def _capture():
    buf = io.StringIO()

    def emit(*args, **kwargs):
        print(*args, **kwargs, file=buf)

    return buf, emit


@pytest.mark.parametrize("nstep", [1, 7])
def test_cli_verbose_lines_match_full_trajectory_replay(nstep):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    rt = solver.prepare_runtime(inp, nstep=nstep)

    cli_buf, cli_emit = _capture()
    carry_cli = solver._solve_stage(
        rt, None, mode="cli", verbose=True, emit=cli_emit
    )
    carry_jit = solver._solve_stage(
        rt, None, mode="jit", verbose=False, emit=print
    )
    assert int(carry_cli.iteration) == int(carry_jit.iteration)

    upto = int(carry_jit.iteration)
    ref_buf, ref_emit = _capture()
    solver._emit_lines(
        rt, np.asarray(carry_jit.trajectory)[:upto], upto, set(), True, ref_emit
    )

    cli_rows = [l for l in cli_buf.getvalue().splitlines() if _ROW.match(l)]
    ref_rows = ref_buf.getvalue().splitlines()
    assert ref_rows, "replay produced no screen lines"
    assert cli_rows == ref_rows
