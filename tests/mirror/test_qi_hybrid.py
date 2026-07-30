"""Cut-and-splice construction of the QI-mirror hybrid axis and its solve setup.

The splice cuts a stellarator-symmetric closed axis at its ``2 * nfp``
low-curvature symmetry planes and inserts an exactly-straight mirror leg *along
the local axis tangent* at each; the per-cut leg lengths are chosen so the loop
closes, and one symmetric half is reflected so the racetrack is stellarator
symmetric.  The closed-spline builder wraps it with a circular section and
returns a solvable :class:`StellaratorMirrorSetup`.  These are smoke-level
checks: the construction runs, closes, is tangent-aligned and stellarator
symmetric, and the two representations reproduce the straight legs (B-spline
exactly, Fourier with residual ringing).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror import (  # noqa: E402
    MirrorResolution,
    QIMirrorSplice,
    build_qi_mirror_hybrid,
    splice_straight_legs,
)
from vmex.mirror.splines import _closed_tangent, _sample_closed_polyline  # noqa: E402
from vmex.mirror.basis import CubicBSplineBasis  # noqa: E402

_STELL_SYMMETRY = np.diag([1.0, -1.0, -1.0])


def _model_qi_axis(n: int = 256, nfp: int = 2) -> np.ndarray:
    """A closed nfp=2 stellarator-symmetric axis with low-curvature planes.

    ``phi = 0, pi/2, pi, 3pi/2`` are the symmetry planes where the axis crosses
    the midplane, mimicking the four cut locations of a real QI equilibrium.
    """

    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = 1.0 + 0.08 * np.cos(nfp * phi)
    height = 0.35 * np.sin(nfp * phi)
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), height], axis=1)


def _four_cuts(n: int = 256) -> tuple[int, int, int, int]:
    """The four symmetry planes phi = 0, pi/2, pi, 3pi/2 as sample indices."""

    return (0, n // 4, n // 2, 3 * n // 4)


def test_splice_is_tangent_aligned_stellarator_symmetric_and_closes() -> None:
    points = _model_qi_axis()
    cut = _four_cuts()
    splice = splice_straight_legs(points, cut_indices=cut, straight_length=1.2)

    assert isinstance(splice, QIMirrorSplice)
    # four legs, one per cut
    assert len(splice.leg_windows) == len(cut) == 4
    assert splice.leg_lengths.shape == (4,)
    # loop closes: the cancelling leg displacements sum to zero
    assert splice.closure_error < 1.0e-12
    # each leg extends along the local axis tangent (no shared bisector corner):
    # rigorously, the leg direction IS the axis tangent at its cut...
    for k, c in enumerate(cut):
        assert np.allclose(splice.leg_directions[k], _closed_tangent(points, c), atol=1.0e-12)
    # ...so the discrete junction tangent break is small (vs a real ~36 deg corner)
    assert splice.corner_angle < 5.0  # degrees, sampling-limited
    # every leg is an exactly-straight segment
    for start, stop in splice.leg_windows:
        samples = _sample_closed_polyline(
            splice.points, np.linspace(start + 0.05, stop - 0.05, 40)
        )
        directions = np.diff(samples, axis=0)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        assert np.allclose(np.abs(directions @ directions[0]), 1.0, atol=1.0e-10)
    # the spliced racetrack is stellarator symmetric (x, y, z) -> (x, -y, -z)
    reflected = splice.points @ _STELL_SYMMETRY
    residual = max(
        float(np.linalg.norm(splice.points - q, axis=1).min()) for q in reflected[::5]
    )
    assert residual < 1.0e-9


def test_splice_leg_lengths_form_symmetry_classes() -> None:
    points = _model_qi_axis()
    splice = splice_straight_legs(points, cut_indices=_four_cuts(), straight_length=1.2)
    lengths = np.sort(np.asarray(splice.leg_lengths))
    # two symmetry-plane classes -> two equal-length pairs, all positive
    assert np.all(lengths > 0.0)
    assert lengths[0] == pytest.approx(lengths[1], rel=1e-6)
    assert lengths[2] == pytest.approx(lengths[3], rel=1e-6)


def test_bspline_reproduces_legs_better_than_fourier() -> None:
    points = _model_qi_axis()
    splice = splice_straight_legs(points, cut_indices=_four_cuts(), straight_length=1.2)

    dense = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    arc = dense / (2.0 * np.pi) * splice.total_length
    target = _sample_closed_polyline(splice.points, arc)

    # B-spline midpoint of a leg: machine precision once backed by enough controls
    basis = CubicBSplineBasis.periodic_uniform(256)
    nodes = np.asarray(basis.collocation_nodes)
    coefficients = basis.fit(
        _sample_closed_polyline(splice.points, nodes / (2.0 * np.pi) * splice.total_length),
        axis=0,
    )
    start, stop = splice.leg_windows[0]
    midpoint = 0.5 * (start + stop)
    fitted = np.asarray(
        basis.evaluate(coefficients, np.array([midpoint / splice.total_length * 2.0 * np.pi]), axis=0)
    )[0]
    exact = _sample_closed_polyline(splice.points, np.array([midpoint]))[0]
    bspline_leg = float(np.linalg.norm(fitted - exact))

    # Fourier least-squares at comparable resolution: residual ringing on the leg
    columns = [np.ones_like(dense)]
    for order in range(1, 33):
        columns += [np.cos(order * dense), np.sin(order * dense)]
    design = np.stack(columns, axis=1)
    fourier = np.stack(
        [design @ np.linalg.lstsq(design, target[:, j], rcond=None)[0] for j in range(3)], axis=1
    )
    interior = (arc >= start + 0.20) & (arc <= stop - 0.20)
    fourier_leg = float(np.linalg.norm(fourier[interior] - target[interior], axis=1).max())

    assert bspline_leg < 1.0e-9
    assert fourier_leg > 1.0e-4
    assert bspline_leg < 1.0e-3 * fourier_leg


def test_build_qi_mirror_hybrid_returns_solvable_setup() -> None:
    points = _model_qi_axis()
    resolution = MirrorResolution(ns=5, mpol=3, nxi=4)
    setup = build_qi_mirror_hybrid(
        points, resolution, cut_indices=_four_cuts(), straight_length=1.2,
        section_radius=0.12, coefficient_count=32,
    )
    axis = setup.axis
    assert float(axis.closure_error) < 1.0e-12
    assert float(jnp.max(axis.curvature)) > 0.0
    # the reconstructed axis spans a four-leg-and-return racetrack
    assert float(axis.arc_length) > 4.0 * 1.2
    # circular section: the boundary radius is the requested constant
    np.testing.assert_allclose(np.asarray(setup.boundary.radius_coefficients), 0.12, atol=1.0e-12)
    # the nested initial state has the right shapes for a solve
    assert setup.initial_state.radius_coefficients.shape[0] == resolution.ns
    assert np.all(np.isfinite(np.asarray(setup.initial_state.lambda_coefficients)))


def test_splice_rejects_bad_inputs() -> None:
    points = _model_qi_axis(n=64)
    with pytest.raises(ValueError):  # not strictly increasing
        splice_straight_legs(points, cut_indices=(10, 5, 20, 40), straight_length=1.0)
    with pytest.raises(ValueError):  # negative length
        splice_straight_legs(points, cut_indices=_four_cuts(64), straight_length=-1.0)
    with pytest.raises(ValueError):  # not (P, 3)
        splice_straight_legs(points[:, :2], cut_indices=_four_cuts(64), straight_length=1.0)
    with pytest.raises(ValueError):  # too few cuts
        splice_straight_legs(points, cut_indices=(0,), straight_length=1.0)
