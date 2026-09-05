"""One-definition mirror-ratio and mirror-length diagnostics (31.4-R3)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmex.mirror.metrics import (
    DEFAULT_STRAIGHT_CURVATURE_TOLERANCE,
    axis_mirror_wells,
    closed_axis_arc,
    lcfs_mirror_ratio,
    mirror_ratio_diagnostics,
    straight_axis_spans,
)


def _open_mirror_axis(count: int = 81, curvature: float = 0.02, center: float = 0.08):
    """Paraxial two-coil on-axis field ``B(z) = B0 + k z^2`` on ``|z| <= 0.8``."""

    z = np.linspace(-0.8, 0.8, count)
    return z, center + curvature * z**2


def test_open_mirror_well_uses_the_domain_ends_as_throats() -> None:
    z, field = _open_mirror_axis()
    (well,) = axis_mirror_wells(field, z)

    assert well.lower_maximum_index == 0
    assert well.upper_maximum_index == z.size - 1
    assert well.minimum_index == z.size // 2
    # R_m,axis is max/min over the closed well interval, exactly.
    np.testing.assert_allclose(well.mirror_ratio, field.max() / field.min(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(well.confining_mirror_ratio, field.max() / field.min(), rtol=1.0e-14)
    # L_mirror,B is the distance between the bounding |B| maxima.
    np.testing.assert_allclose(well.mirror_length, 1.6, rtol=1.0e-14)


def test_periodic_axis_reports_one_well_per_leg_with_wrapped_lengths() -> None:
    period = 8.0
    arc = np.linspace(0.0, period, 64, endpoint=False)
    # Two wells per period: minima at the two straight legs, maxima between.
    field = 1.0 + 0.3 * np.cos(4.0 * np.pi * arc / period)
    wells = axis_mirror_wells(field, arc, period=period)

    assert len(wells) == 2
    ratios = [well.mirror_ratio for well in wells]
    np.testing.assert_allclose(ratios, [1.3 / 0.7, 1.3 / 0.7], rtol=2.0e-2)
    # The two mirror cells tile one period.
    np.testing.assert_allclose(sum(well.mirror_length for well in wells), period, rtol=1.0e-13)
    # One of the pairs wraps through the periodic seam.
    assert any(well.upper_maximum_index < well.lower_maximum_index for well in wells)


def test_single_maximum_periodic_axis_is_one_full_period_well() -> None:
    period = 5.0
    arc = np.linspace(0.0, period, 40, endpoint=False)
    field = 1.0 + 0.4 * np.cos(2.0 * np.pi * arc / period)
    (well,) = axis_mirror_wells(field, arc, period=period)

    assert well.lower_maximum_index == well.upper_maximum_index == 0
    np.testing.assert_allclose(well.mirror_length, period, rtol=1.0e-13)
    np.testing.assert_allclose(well.mirror_ratio, field.max() / field.min(), rtol=1.0e-13)


def test_a_well_shallower_than_the_depth_threshold_is_not_reported() -> None:
    z = np.linspace(-1.0, 1.0, 401)
    # A steep ramp with fine ripple: every dimple is under 3% of the |B| swing,
    # so none of them is a mirror cell worth reporting.
    field = 1.0 + 0.5 * z + 0.02 * np.cos(40.0 * np.pi * z)
    with pytest.raises(ValueError, match="no \\|B\\| well"):
        axis_mirror_wells(field, z)
    assert len(axis_mirror_wells(field, z, minimum_relative_depth=1.0e-6)) == 40


def test_lcfs_ratio_is_reported_separately_from_the_axis_ratio() -> None:
    z, axis_field = _open_mirror_axis()
    # A shaped boundary: the LCFS sees a larger swing than the axis.
    lcfs = axis_field[None, :] * (1.0 + 0.25 * np.cos(np.linspace(0.0, 2.0 * np.pi, 9))[:, None])
    diagnostics = mirror_ratio_diagnostics(axis_field, z, lcfs_field_strength=lcfs)

    assert diagnostics.lcfs_mirror_ratio > diagnostics.axis_mirror_ratios[0]
    np.testing.assert_allclose(diagnostics.lcfs_mirror_ratio, lcfs_mirror_ratio(lcfs), rtol=0.0)
    record = diagnostics.summary()
    assert set(record) == {"R_m_axis", "L_mirror_B", "R_m_LCFS"}
    np.testing.assert_allclose(record["R_m_axis"], list(diagnostics.axis_mirror_ratios))
    np.testing.assert_allclose(record["L_mirror_B"], list(diagnostics.mirror_lengths))


def test_straight_length_measures_the_uncurved_arc_of_a_racetrack() -> None:
    # Racetrack: two straight legs of length 3 joined by two half-circle
    # returns of radius 1, sampled uniformly in arc length.
    straight, radius = 3.0, 1.0
    period = 2.0 * straight + 2.0 * np.pi * radius
    arc = np.linspace(0.0, period, 400, endpoint=False)
    curvature = np.where(
        (arc < straight) | ((arc >= straight + np.pi * radius) & (arc < 2.0 * straight + np.pi * radius)),
        0.0,
        1.0 / radius,
    )
    total, spans = straight_axis_spans(curvature, arc, period=period)

    assert len(spans) == 2
    np.testing.assert_allclose(total, 2.0 * straight, rtol=3.0e-2)
    # An exactly straight axis is straight everywhere, seam included.
    everywhere, single = straight_axis_spans(np.zeros_like(arc), arc, period=period)
    np.testing.assert_allclose(everywhere, period, rtol=1.0e-13)
    assert single == (everywhere,)
    # A constant-curvature circle has no straight span at all.
    assert straight_axis_spans(np.ones_like(arc), arc, period=period) == (0.0, ())


def test_open_straight_axis_reports_the_whole_length_as_straight() -> None:
    z, field = _open_mirror_axis(count=33)
    diagnostics = mirror_ratio_diagnostics(field, z, axis_curvature=np.zeros_like(z))

    np.testing.assert_allclose(diagnostics.straight_length, 1.6, rtol=1.0e-13)
    assert diagnostics.straight_spans == (diagnostics.straight_length,)
    assert "L_straight" in diagnostics.summary()


def test_sampling_ripple_is_merged_instead_of_reported_as_extra_legs() -> None:
    period = 8.0
    arc = np.linspace(0.0, period, 256, endpoint=False)
    clean = 1.0 + 0.3 * np.cos(4.0 * np.pi * arc / period)
    rippled = clean + 0.004 * np.cos(48.0 * np.pi * arc / period)

    assert len(axis_mirror_wells(rippled, arc, period=period, minimum_relative_depth=0.0)) > 2
    wells = axis_mirror_wells(rippled, arc, period=period)
    assert len(wells) == 2
    np.testing.assert_allclose(
        [well.mirror_ratio for well in wells],
        [well.mirror_ratio for well in axis_mirror_wells(clean, arc, period=period)],
        rtol=3.0e-2,
    )


def test_closed_racetrack_hybrid_has_throatless_legs_and_two_straight_spans() -> None:
    """The shipped hybrid's |B| wells sit on the legs but carry no throat."""

    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from vmex.mirror import MirrorResolution, build_stellarator_mirror_hybrid
    from vmex.mirror.forces import mirror_energy
    from vmex.mirror.geometry import magnetic_field_squared
    from vmex.mirror.metrics import closed_axis_arc

    setup = build_stellarator_mirror_hybrid(MirrorResolution(ns=5, mpol=2, nxi=2), coefficient_count=32)
    discretization = setup.discretization
    energy = mirror_energy(
        discretization.evaluate_state(setup.initial_state),
        discretization.grid,
        axis=setup.axis,
        axial_flux_derivative=0.02,
        current_derivative=0.002,
    )
    mod_b = np.sqrt(np.maximum(np.asarray(magnetic_field_squared(energy.field, energy.geometry)), 0.0))
    arc, period = closed_axis_arc(setup.axis)

    np.testing.assert_allclose(period, float(setup.axis.arc_length), rtol=1.0e-14)
    assert np.all(np.diff(arc) > 0.0) and arc[0] == 0.0 and arc[-1] < period

    diagnostics = mirror_ratio_diagnostics(
        mod_b[0].mean(axis=0),
        arc,
        period=period,
        lcfs_field_strength=mod_b[-1],
        axis_curvature=np.asarray(setup.axis.curvature),
    )
    # One well per straight leg, and the two mirror cells tile the circuit.
    assert len(diagnostics.wells) == 2
    np.testing.assert_allclose(sum(diagnostics.mirror_lengths), period, rtol=1.0e-12)
    # 31.4-R4: constant leg semi-axes mean the legs have no |B| throat, so
    # R_m,axis is ~1 while the shaped returns give a much larger R_m,LCFS.
    assert max(diagnostics.axis_mirror_ratios) < 1.01
    assert diagnostics.lcfs_mirror_ratio > 2.0
    # Two straight legs; the cubic spline rounds each leg-return junction, so
    # the exactly straight arc is shorter than the nominal 2 x 8 m.
    assert len(diagnostics.straight_spans) == 2
    np.testing.assert_allclose(diagnostics.straight_spans[0], diagnostics.straight_spans[1], rtol=1.0e-9)
    assert 10.0 < diagnostics.straight_length < 16.0


def test_definitions_reject_inputs_that_have_no_well_or_bad_sampling() -> None:
    z = np.linspace(-1.0, 1.0, 9)
    with pytest.raises(ValueError, match="no \\|B\\| well"):
        axis_mirror_wells(np.exp(z), z)
    with pytest.raises(ValueError, match="no maximum"):
        axis_mirror_wells(np.ones_like(z), z, period=4.0)
    with pytest.raises(ValueError, match="at least three samples"):
        axis_mirror_wells(np.ones(2), np.arange(2.0))
    with pytest.raises(ValueError, match="positive and finite"):
        axis_mirror_wells(np.zeros_like(z), z)
    with pytest.raises(ValueError, match="one value per sample"):
        axis_mirror_wells(np.ones_like(z), z[:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        axis_mirror_wells(np.ones_like(z), z[::-1])
    with pytest.raises(ValueError, match="period must exceed"):
        axis_mirror_wells(np.ones_like(z), z, period=1.0)
    with pytest.raises(ValueError, match="nonempty and finite"):
        lcfs_mirror_ratio(np.asarray([]))
    with pytest.raises(ValueError, match="must be positive"):
        lcfs_mirror_ratio(np.zeros(4))
    with pytest.raises(ValueError, match="at least two samples"):
        straight_axis_spans(np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="must be finite"):
        straight_axis_spans(np.asarray([0.0, np.nan]), np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="tolerance must lie"):
        straight_axis_spans(np.zeros(4), np.arange(4.0), tolerance=1.0)
    with pytest.raises(ValueError, match="minimum_relative_depth"):
        axis_mirror_wells(np.ones_like(z), z, minimum_relative_depth=1.0)
    with pytest.raises(ValueError, match="at least three nodes"):
        closed_axis_arc(SimpleNamespace(speed=np.ones(2), arc_length=1.0))
    with pytest.raises(ValueError, match="must be positive"):
        closed_axis_arc(SimpleNamespace(speed=np.zeros(4), arc_length=1.0))
    assert 0.0 < DEFAULT_STRAIGHT_CURVATURE_TOLERANCE < 1.0
