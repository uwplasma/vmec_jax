"""One definition of the reported mirror ratios and mirror lengths.

The mirror lanes previously reported a quantity called "mirror ratio" with
four different meanings (a prescribed boundary-shape parameter, the grid
maximum of the on-axis vacuum field, the LCFS ``max/min``, and a field-line
``max/min``), which makes numbers from different examples, tests, and doc
pages incomparable.  This module fixes the definitions once:

``R_m,axis`` (per leg)
    ``max|B| / min|B|`` on the magnetic axis over one |B| well, where the well
    is the axial interval between the two |B| maxima that bound its minimum.
    An open mirror has one well; the periodic stellarator-mirror hybrid has
    one per straight leg, hence "per leg".

``R_m,LCFS``
    ``max|B| / min|B|`` over the last closed flux surface.  It is a different
    number from ``R_m,axis`` on any shaped boundary and is reported
    separately, never as "the" mirror ratio.

``L_mirror,B`` (per leg)
    The arc-length distance between the two |B| maxima bounding a well: the
    length of the mirror cell measured by the field, not by the device.

``L_straight``
    The arc length of the axis over which its curvature is negligible.  It is
    the geometric length of the straight legs and is unrelated to
    ``L_mirror,B`` unless the |B| maxima happen to sit at the leg ends.

All four are host-side reporting diagnostics built from discrete extrema, so
they are deliberately plain NumPy and are not differentiable.
"""

from dataclasses import dataclass

import numpy as np

# A node counts as straight when its curvature falls below this fraction of
# the largest curvature on the axis.  The threshold is relative so it is
# scale-free: on an exactly straight axis every node qualifies, and on a
# constant-curvature (circular) axis none does.
DEFAULT_STRAIGHT_CURVATURE_TOLERANCE = 1.0e-3


@dataclass(frozen=True)
class MirrorWell:
    """One on-axis |B| well and the two maxima that bound it."""

    minimum_index: int
    lower_maximum_index: int
    upper_maximum_index: int
    field_minimum: float
    lower_field_maximum: float
    upper_field_maximum: float
    mirror_length: float

    @property
    def mirror_ratio(self) -> float:
        """``R_m,axis``: ``max|B| / min|B|`` over the closed well interval."""

        return max(self.lower_field_maximum, self.upper_field_maximum) / self.field_minimum

    @property
    def confining_mirror_ratio(self) -> float:
        """The smaller of the two throat ratios, which sets the loss cone."""

        return min(self.lower_field_maximum, self.upper_field_maximum) / self.field_minimum


@dataclass(frozen=True)
class MirrorRatioDiagnostics:
    """The reported mirror ratios and lengths of one equilibrium."""

    wells: tuple[MirrorWell, ...]
    lcfs_mirror_ratio: float | None
    straight_length: float | None
    straight_spans: tuple[float, ...]

    @property
    def axis_mirror_ratios(self) -> tuple[float, ...]:
        """``R_m,axis`` per leg, in increasing axial order."""

        return tuple(well.mirror_ratio for well in self.wells)

    @property
    def mirror_lengths(self) -> tuple[float, ...]:
        """``L_mirror,B`` per leg, in increasing axial order."""

        return tuple(well.mirror_length for well in self.wells)

    def summary(self) -> dict:
        """Return a JSON-serializable record with the definitions named."""

        record: dict = {
            "R_m_axis": list(self.axis_mirror_ratios),
            "L_mirror_B": list(self.mirror_lengths),
        }
        if self.lcfs_mirror_ratio is not None:
            record["R_m_LCFS"] = self.lcfs_mirror_ratio
        if self.straight_length is not None:
            record["L_straight"] = self.straight_length
            record["L_straight_spans"] = list(self.straight_spans)
        return record


def _checked_axial_coordinate(coordinate, size: int, *, period: float | None) -> np.ndarray:
    arc = np.asarray(coordinate, dtype=float)
    if arc.shape != (size,):
        raise ValueError("the axial coordinate must have one value per sample")
    if np.any(np.diff(arc) <= 0.0):
        raise ValueError("the axial coordinate must be strictly increasing")
    if period is not None and (not np.isfinite(period) or period <= float(arc[-1] - arc[0])):
        raise ValueError("period must exceed the sampled arc span of a periodic axis")
    return arc


def _local_maximum_indices(values: np.ndarray, *, periodic: bool) -> list[int]:
    """Return the candidate |B| maxima that can bound a well."""

    count = values.size
    if periodic:
        previous = np.roll(values, 1)
        following = np.roll(values, -1)
        return [index for index in range(count) if values[index] > previous[index] and values[index] >= following[index]]
    interior = [
        index
        for index in range(1, count - 1)
        if values[index] > values[index - 1] and values[index] >= values[index + 1]
    ]
    # The two ends of an open mirror bound the modelled domain, so they are
    # admissible throats even when the coils sit outside the grid.
    return [0, *interior, count - 1]


def _well_between(
    values: np.ndarray,
    arc: np.ndarray,
    lower: int,
    upper: int,
    period: float | None,
) -> tuple[int, float, float]:
    """Return ``(minimum index, well depth, arc span)`` for one maxima pair."""

    if upper > lower:
        indices = np.arange(lower, upper + 1)
        span = float(arc[upper] - arc[lower])
    else:
        indices = np.arange(lower, upper + values.size + 1) % values.size
        span = float(arc[upper] - arc[lower]) + float(period)
    interior = indices[1:-1]
    minimum_index = int(interior[np.argmin(values[interior])])
    depth = min(float(values[lower]), float(values[upper])) - float(values[minimum_index])
    return minimum_index, depth, span


def _maxima_pairs(maxima: list[int], *, periodic: bool) -> list[tuple[int, int]]:
    if not periodic:
        return list(zip(maxima, maxima[1:], strict=False))
    if len(maxima) == 1:
        return [(maxima[0], maxima[0])]
    return [(maxima[index], maxima[(index + 1) % len(maxima)]) for index in range(len(maxima))]


def _prune_shallow_maxima(
    values: np.ndarray,
    arc: np.ndarray,
    maxima: list[int],
    *,
    periodic: bool,
    period: float | None,
    threshold: float,
    protected: frozenset[int],
) -> list[int]:
    """Merge wells whose depth is below ``threshold`` (persistence pruning).

    Discrete |B| samples of a solved equilibrium carry small ripple, which
    would otherwise register as extra legs.  Removing the lower bounding
    maximum of the shallowest well merges it into its neighbour, exactly as
    topological persistence prescribes, until every surviving well is deeper
    than the threshold.
    """

    maxima = list(maxima)
    minimum_count = 1 if periodic else 2
    while len(maxima) > minimum_count:
        pairs = _maxima_pairs(maxima, periodic=periodic)
        depths = [_well_between(values, arc, lower, upper, period)[1] for lower, upper in pairs]
        index = int(np.argmin(depths))
        if depths[index] >= threshold:
            break
        # Consecutive maxima are never both protected while more than the
        # minimum remain, so there is always something to merge away.
        candidates = [node for node in set(pairs[index]) if node not in protected]
        maxima.remove(min(candidates, key=lambda node: values[node]))
    return maxima


def axis_mirror_wells(
    axis_field_strength,
    axial_coordinate,
    *,
    period: float | None = None,
    minimum_relative_depth: float = 0.05,
) -> tuple[MirrorWell, ...]:
    """Return the on-axis |B| wells with their ``R_m,axis`` and ``L_mirror,B``.

    ``axial_coordinate`` is the arc length (or ``z`` for a straight axis) of
    each sample and must be strictly increasing.  Pass ``period`` for a closed
    periodic axis whose samples cover one period without repeating the first
    point; leave it ``None`` for an open mirror, whose two domain ends are then
    admissible bounding maxima.

    ``minimum_relative_depth`` is the well depth, as a fraction of the total
    on-axis |B| swing, below which a well is merged into its neighbour.  It
    keeps sampling ripple in a solved |B| from being reported as extra legs.
    """

    values = np.asarray(axis_field_strength, dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("axis field strength must be one-dimensional with at least three samples")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("axis field strength must be positive and finite")
    if not 0.0 <= float(minimum_relative_depth) < 1.0:
        raise ValueError("minimum_relative_depth must lie in [0, 1)")
    arc = _checked_axial_coordinate(axial_coordinate, values.size, period=period)
    periodic = period is not None
    maxima = _local_maximum_indices(values, periodic=periodic)
    if not maxima:
        raise ValueError("the on-axis field strength has no maximum bounding a well")
    threshold = float(minimum_relative_depth) * float(np.max(values) - np.min(values))
    maxima = _prune_shallow_maxima(
        values,
        arc,
        maxima,
        periodic=periodic,
        period=period,
        threshold=threshold,
        protected=frozenset() if periodic else frozenset({0, values.size - 1}),
    )
    wells = []
    for lower, upper in _maxima_pairs(maxima, periodic=periodic):
        minimum_index, depth, span = _well_between(values, arc, lower, upper, period)
        if depth <= 0.0 or depth < threshold:
            continue
        wells.append(
            MirrorWell(
                minimum_index=minimum_index,
                lower_maximum_index=int(lower),
                upper_maximum_index=int(upper),
                field_minimum=float(values[minimum_index]),
                lower_field_maximum=float(values[lower]),
                upper_field_maximum=float(values[upper]),
                mirror_length=span,
            )
        )
    if not wells:
        raise ValueError("the on-axis field strength has no |B| well between two maxima")
    return tuple(wells)


def closed_axis_arc(axis) -> tuple[np.ndarray, float]:
    """Return ``(arc, period)`` for a periodic axis on its uniform node set.

    ``axis`` is a :class:`~vmex.mirror.geometry.ClosedAxisGeometry`.  The node
    parameter is uniform over one period, so trapezoidal increments of the
    speed give the arc coordinate; they are rescaled to the axis's own
    ``arc_length`` so the returned period is exactly consistent.
    """

    speed = np.asarray(axis.speed, dtype=float)
    period = float(axis.arc_length)
    if speed.ndim != 1 or speed.size < 3:
        raise ValueError("a closed axis needs at least three nodes")
    if not np.all(speed > 0.0) or period <= 0.0:
        raise ValueError("closed-axis speed and arc length must be positive")
    increments = 0.5 * (speed + np.roll(speed, -1))
    increments *= period / float(np.sum(increments))
    return np.concatenate(([0.0], np.cumsum(increments)[:-1])), period


def lcfs_mirror_ratio(field_strength) -> float:
    """Return ``R_m,LCFS``: ``max|B| / min|B|`` over the last closed surface."""

    values = np.asarray(field_strength, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("LCFS field strength must be nonempty and finite")
    minimum = float(np.min(values))
    if minimum <= 0.0:
        raise ValueError("LCFS field strength must be positive")
    return float(np.max(values)) / minimum


def straight_axis_spans(
    curvature,
    axial_coordinate,
    *,
    period: float | None = None,
    tolerance: float = DEFAULT_STRAIGHT_CURVATURE_TOLERANCE,
) -> tuple[float, tuple[float, ...]]:
    """Return ``(L_straight, per-span lengths)`` of the negligibly curved axis.

    A node is straight when its curvature is at most ``tolerance`` times the
    largest curvature on the axis, and ``L_straight`` is the arc measure of the
    intervals whose two endpoints are both straight.  The threshold is relative
    so the same tolerance works for any device size; on an exactly straight
    axis the whole length is returned.
    """

    kappa = np.abs(np.asarray(curvature, dtype=float))
    if kappa.ndim != 1 or kappa.size < 2:
        raise ValueError("curvature must be one-dimensional with at least two samples")
    if not np.all(np.isfinite(kappa)):
        raise ValueError("curvature must be finite")
    if not 0.0 < float(tolerance) < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    arc = _checked_axial_coordinate(axial_coordinate, kappa.size, period=period)
    straight = kappa <= float(tolerance) * float(np.max(kappa))
    flags = straight[:-1] & straight[1:]
    lengths = np.diff(arc)
    if period is not None:
        flags = np.append(flags, straight[-1] & straight[0])
        lengths = np.append(lengths, float(period) - float(arc[-1] - arc[0]))
        if np.all(flags):
            total = float(np.sum(lengths))
            return total, (total,)
        # Rotate so a curved edge comes first; a straight run may otherwise be
        # split by the arbitrary choice of periodic starting node.
        start = int(np.argmin(flags))
        flags = np.roll(flags, -start)
        lengths = np.roll(lengths, -start)
    spans: list[float] = []
    running = 0.0
    for is_straight, length in zip(flags, lengths, strict=True):
        if is_straight:
            running += float(length)
        elif running > 0.0:
            spans.append(running)
            running = 0.0
    if running > 0.0:
        spans.append(running)
    return float(sum(spans)), tuple(spans)


def mirror_ratio_diagnostics(
    axis_field_strength,
    axial_coordinate,
    *,
    lcfs_field_strength=None,
    axis_curvature=None,
    period: float | None = None,
    minimum_relative_depth: float = 0.05,
    straight_curvature_tolerance: float = DEFAULT_STRAIGHT_CURVATURE_TOLERANCE,
) -> MirrorRatioDiagnostics:
    """Report ``R_m,axis`` per leg, ``R_m,LCFS``, ``L_mirror,B``, ``L_straight``.

    This is the single entry point the examples, tests, and docs use, so a
    "mirror ratio" quoted anywhere in the mirror lane has exactly one meaning.
    ``lcfs_field_strength`` and ``axis_curvature`` are optional; the quantities
    they support are omitted when they are not supplied.
    """

    wells = axis_mirror_wells(
        axis_field_strength,
        axial_coordinate,
        period=period,
        minimum_relative_depth=minimum_relative_depth,
    )
    lcfs = None if lcfs_field_strength is None else lcfs_mirror_ratio(lcfs_field_strength)
    straight_length: float | None = None
    spans: tuple[float, ...] = ()
    if axis_curvature is not None:
        straight_length, spans = straight_axis_spans(
            axis_curvature,
            axial_coordinate,
            period=period,
            tolerance=straight_curvature_tolerance,
        )
    return MirrorRatioDiagnostics(
        wells=wells,
        lcfs_mirror_ratio=lcfs,
        straight_length=straight_length,
        straight_spans=spans,
    )


__all__ = [
    "DEFAULT_STRAIGHT_CURVATURE_TOLERANCE",
    "MirrorRatioDiagnostics",
    "MirrorWell",
    "axis_mirror_wells",
    "closed_axis_arc",
    "lcfs_mirror_ratio",
    "mirror_ratio_diagnostics",
    "straight_axis_spans",
]
