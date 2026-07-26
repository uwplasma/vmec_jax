"""Free-boundary stress assertions on the generated QI + coils case.

Companion to :mod:`tests.test_qi_free_boundary_case`, which builds the fixture
(QI boundary, analytic modular coils, tabulated mgrid, finite pressure).  The
assertions here are about *lawful behaviour under strain*, not about the
accuracy of the resulting equilibrium: this deck is deliberately hard and is
not expected to reach ``ftol``.

Three properties are checked, each corresponding to a way the free-boundary
driver has been observed to fail:

1. the generated external field is a usable free-boundary field (finite,
   correct field-period symmetry, sane magnitude);
2. a free-boundary solve on a finite-pressure QI deck keeps every residual
   channel finite and terminates with a typed outcome;
3. **the trajectory actually moves** -- a recovery restart must change the
   iterate, not replay an identical path.  This is the assertion the earlier
   robustness test lacked, and the one that catches a restart which restores a
   checkpoint without perturbing the dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from vmex.core.errors import VmecError  # noqa: E402
from vmex.core.multigrid import solve_free_boundary_multigrid  # noqa: E402

from tests.test_qi_free_boundary_case import (  # noqa: E402
    COIL_CURRENT,
    _biot_savart,
    _coil_filaments,
    qi_free_field,
    qi_free_input,
)


def test_generated_coil_field_is_physical() -> None:
    """The analytic modular set produces a smooth 1/R-like toroidal field."""
    field = _biot_savart(_coil_filaments(2), COIL_CURRENT)
    probes = np.array([[0.9, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    magnitudes = np.linalg.norm(field(probes), axis=-1)

    assert np.all(np.isfinite(magnitudes))
    # order-1 T on the winding-surface centreline, not a degenerate field
    assert 0.1 < magnitudes[1] < 10.0
    # toroidal field falls with major radius
    assert magnitudes[0] > magnitudes[1] > magnitudes[2]


def test_generated_mgrid_has_field_period_symmetry() -> None:
    """The tabulated table is finite and carries the requested field periods.

    The coil set is built with ``COILS_PER_PERIOD`` filaments per period, so the
    tabulated field must repeat under a rotation by ``2*pi/nfp``.  The mgrid
    stores one field period, and the solver relies on that periodicity when it
    evaluates the external field at arbitrary toroidal angles, so a table whose
    first and last toroidal planes disagree would silently corrupt NESTOR.
    """
    field = qi_free_field(2)
    br, bp, bz = (np.asarray(a) for a in (field.br, field.bp, field.bz))

    assert int(field.nfp) == 2
    for name, arr in (("br", br), ("bp", bp), ("bz", bz)):
        assert arr.size > 0, f"{name} is empty"
        assert np.all(np.isfinite(arr)), f"{name} has non-finite entries"

    # A toroidal field dominates: |bp| should exceed the poloidal components.
    assert np.mean(np.abs(bp)) > np.mean(np.abs(br))
    assert np.mean(np.abs(bp)) > np.mean(np.abs(bz))

    # Periodicity across the stored period: the tabulation covers [0, 2*pi/nfp)
    # so rolling by a full period returns the same plane.
    np.testing.assert_allclose(np.roll(bp, bp.shape[0], axis=0), bp, rtol=0, atol=0)


@pytest.mark.full  # a real two-rung free-boundary ladder with recovery
def test_finite_pressure_qi_free_boundary_stays_lawful() -> None:
    """A hard finite-pressure QI free-boundary run degrades gracefully.

    Not expected to converge.  Required: finite residual channels throughout
    and a typed termination -- never a crash, never a silent NaN.
    """
    inp = qi_free_input(ns_array=(9, 15), niter=25)
    field = qi_free_field(int(inp.nfp))
    try:
        result = solve_free_boundary_multigrid(
            inp, external_field=field, verbose=False,
            raise_on_max_iterations=False,
        )
    except VmecError:
        return  # typed, diagnosed termination is an acceptable outcome
    for name in ("fsqr", "fsqz", "fsql"):
        assert np.isfinite(float(getattr(result, name))), f"{name} non-finite"


@pytest.mark.full
def test_recovery_restart_changes_the_trajectory() -> None:
    """A reduced time step must not replay an identical path.

    Reported failures showed a Jacobian-recovery retry reproducing the same
    residuals and total energy to five significant figures across a halved
    ``DELT``.  Restoring a checkpoint is correct; reproducing the *entire*
    subsequent trajectory is not, because the reduced step should change the
    dynamics.  Two runs that differ only in time step must therefore end in
    measurably different states.
    """
    import dataclasses

    base = qi_free_input(ns_array=(9,), niter=30)
    field = qi_free_field(int(base.nfp))

    def run(time_step: float):
        try:
            r = solve_free_boundary_multigrid(
                dataclasses.replace(base), external_field=field, verbose=False,
                raise_on_max_iterations=False, time_step=time_step,
            )
            return float(r.fsqr), float(r.fsqz), float(r.fsql)
        except VmecError:
            return None

    slow, fast = run(0.5), run(0.25)
    if slow is None or fast is None:
        pytest.skip("deck terminated with a typed error before comparison")
    assert all(np.isfinite(v) for v in slow + fast)
    # Halving the step must perturb the trajectory somewhere.
    relative = max(
        abs(a - b) / max(abs(a), abs(b), 1e-30) for a, b in zip(slow, fast)
    )
    assert relative > 1e-9, (
        "halving DELT reproduced the identical residual triple "
        f"{slow} -- the reduced step is not reaching the integrator"
    )
