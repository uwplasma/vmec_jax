"""Free-boundary stress assertions on the generated QI + coils case
(fixture built in :mod:`tests.test_qi_free_boundary_case`) — lawful
behaviour under strain, not equilibrium accuracy; the deck is deliberately
hard and not expected to reach ``ftol``.  Checked: the generated field is
usable (finite, right field-period symmetry, sane magnitude); a finite-
pressure QI free solve keeps every residual finite with a typed outcome;
and a recovery restart actually changes the trajectory instead of
replaying an identical path.
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
    """The tabulated table is finite and repeats under a 2*pi/nfp rotation —
    the periodicity NESTOR relies on when evaluating the external field."""
    field = qi_free_field(2)
    br, bp, bz = (np.asarray(a) for a in (field.br, field.bp, field.bz))

    assert int(field.nfp) == 2
    for name, arr in (("br", br), ("bp", bp), ("bz", bz)):
        assert arr.size > 0, f"{name} is empty"
        assert np.all(np.isfinite(arr)), f"{name} has non-finite entries"

    # A toroidal field dominates: |bp| should exceed the poloidal components.
    assert np.mean(np.abs(bp)) > np.mean(np.abs(br))
    assert np.mean(np.abs(bp)) > np.mean(np.abs(bz))

    # Genuine field-period symmetry of the source field: rotating an
    # evaluation point by 2*pi/nfp must rotate the Cartesian field with it.
    # (Rolling the stored table by its own length is a tautology -- this
    # compares two physically distinct evaluation points instead.)
    from tests.test_qi_free_boundary_case import (
        COIL_CURRENT, _biot_savart, _coil_filaments,
    )
    analytic = _biot_savart(_coil_filaments(2), COIL_CURRENT)
    point = np.array([[1.05, 0.10, 0.07]])
    rot = np.pi  # one full field period for nfp = 2
    c, s_ = np.cos(rot), np.sin(rot)
    rotated = np.array([[point[0, 0] * c - point[0, 1] * s_,
                         point[0, 0] * s_ + point[0, 1] * c,
                         point[0, 2]]])
    b0, b1 = analytic(point)[0], analytic(rotated)[0]
    b0_rotated = np.array([b0[0] * c - b0[1] * s_,
                           b0[0] * s_ + b0[1] * c,
                           b0[2]])
    np.testing.assert_allclose(b1, b0_rotated, rtol=1e-10, atol=1e-14)


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
    """A reduced time step must not replay an identical path (reported
    retries reproduced residuals to five significant figures across a
    halved ``DELT``): two runs differing only in time step must end in
    measurably different states."""
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
