"""Converged CPU gate for the portable LASYM free-boundary fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmex.core import multigrid
from vmex.core.wout import wout_from_state

from tests.test_lasym_free_case import lasym_free_field, lasym_free_input

pytestmark = [
    pytest.mark.full,
    pytest.mark.usefixtures("_module_jit_enabled"),
]


def test_converged_lasym_free_boundary_wout_fields():
    """The full ladder converges and exports finite nonzero sine fields."""
    data = Path(__file__).resolve().parents[1] / "examples" / "data"
    inp = lasym_free_input(data)
    result = multigrid.solve_free_boundary_multigrid(
        inp,
        external_field=lasym_free_field(),
        device="cpu",
    )
    assert result.converged
    assert max(result.fsqr, result.fsqz, result.fsql) <= 1.0e-10
    wout = wout_from_state(
        inp=inp,
        state=result.state,
        fsqr=result.fsqr,
        fsqz=result.fsqz,
        fsql=result.fsql,
        niter=result.iterations,
        converged=True,
        vacuum_output=result.vacuum,
    )
    for name in (
        "potsin",
        "potcos",
        "bsubumnc_sur",
        "bsubvmnc_sur",
        "bsupumnc_sur",
        "bsupvmnc_sur",
        "bsubumns_sur",
        "bsubvmns_sur",
        "bsupumns_sur",
        "bsupvmns_sur",
    ):
        assert np.isfinite(np.asarray(getattr(wout, name))).all(), name
    assert np.max(np.abs(np.asarray(wout.potcos))) > 0.0
    assert np.max(np.abs(np.asarray(wout.bsubumns_sur))) > 0.0
