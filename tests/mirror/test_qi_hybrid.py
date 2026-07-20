"""Cut-and-splice construction of the QI-mirror hybrid axis and its solve setup.

The splice inserts exactly-straight mirror legs into a closed magnetic axis and
closes the loop; the closed-spline builder wraps it with a circular section and
returns a solvable :class:`StellaratorMirrorSetup`.  These are smoke-level
checks: the construction runs, closes, and the two representations reproduce the
straight legs (B-spline exactly, Fourier with residual ringing).
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
from vmex.mirror.splines import _sample_closed_polyline  # noqa: E402
from vmex.mirror.basis import CubicBSplineBasis  # noqa: E402


def _model_qi_axis(n: int = 256, nfp: int = 2) -> np.ndarray:
    """A closed nfp=2 stellarator-symmetric axis with low-curvature planes.

    ``phi=0`` and ``phi=pi`` are curvature minima where the axis crosses the
    midplane, mimicking the cut locations of a real QI equilibrium.
    """

    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = 1.0 + 0.08 * np.cos(nfp * phi)
    height = 0.35 * np.sin(nfp * phi)
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), height], axis=1)


def test_splice_closes_and_inserts_exact_straight_legs() -> None:
    points = _model_qi_axis()
    cut = (0, points.shape[0] // 2)
    splice = splice_straight_legs(points, cut_indices=cut, straight_length=1.2)

    assert isinstance(splice, QIMirrorSplice)
    assert splice.closure_error < 1.0e-13
    assert splice.total_length > 2.0 * 1.2  # two legs plus two returns
    # each leg window is exactly one straight-length long
    for start, stop in splice.leg_windows:
        assert stop - start == pytest.approx(1.2)
    # the two legs are anti-parallel straight segments (a racetrack)
    for start, stop in splice.leg_windows:
        samples = _sample_closed_polyline(
            splice.points, np.linspace(start + 0.05, stop - 0.05, 40)
        )
        deltas = np.diff(samples, axis=0)
        directions = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
        # every step is collinear with the first -> exactly straight
        assert np.allclose(np.abs(directions @ directions[0]), 1.0, atol=1.0e-12)


def test_bspline_reproduces_legs_better_than_fourier() -> None:
    points = _model_qi_axis()
    cut = (0, points.shape[0] // 2)
    splice = splice_straight_legs(points, cut_indices=cut, straight_length=1.2)

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
    interior = (arc >= start + 0.25) & (arc <= stop - 0.25)
    fourier_leg = float(np.linalg.norm(fourier[interior] - target[interior], axis=1).max())

    assert bspline_leg < 1.0e-9
    assert fourier_leg > 1.0e-4
    assert bspline_leg < 1.0e-3 * fourier_leg


def test_build_qi_mirror_hybrid_returns_solvable_setup() -> None:
    points = _model_qi_axis()
    cut = (0, points.shape[0] // 2)
    resolution = MirrorResolution(ns=5, mpol=3, nxi=4)
    setup = build_qi_mirror_hybrid(
        points, resolution, cut_indices=cut, straight_length=1.2,
        section_radius=0.12, coefficient_count=32,
    )
    axis = setup.axis
    assert float(axis.closure_error) < 1.0e-12
    assert float(jnp.max(axis.curvature)) > 0.0
    # the reconstructed axis spans a straight-leg-and-return racetrack
    assert float(axis.arc_length) > 2.0 * 1.2
    # circular section: the boundary radius is the requested constant
    np.testing.assert_allclose(np.asarray(setup.boundary.radius_coefficients), 0.12, atol=1.0e-12)
    # the nested initial state has the right shapes for a solve
    assert setup.initial_state.radius_coefficients.shape[0] == resolution.ns
    assert np.all(np.isfinite(np.asarray(setup.initial_state.lambda_coefficients)))


def test_splice_rejects_bad_inputs() -> None:
    points = _model_qi_axis(n=64)
    with pytest.raises(ValueError):
        splice_straight_legs(points, cut_indices=(10, 5), straight_length=1.0)
    with pytest.raises(ValueError):
        splice_straight_legs(points, cut_indices=(0, 32), straight_length=-1.0)
    with pytest.raises(ValueError):
        splice_straight_legs(points[:, :2], cut_indices=(0, 32), straight_length=1.0)
