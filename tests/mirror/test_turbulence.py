"""Closed-mirror geometry contract for local gyrokinetic solvers."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror import (  # noqa: E402
    MirrorResolution,
    build_stellarator_mirror_hybrid,
    gk_closed_fieldline_geometry,
)
from vmex.mirror.model import MirrorState  # noqa: E402


@pytest.fixture(scope="module")
def closed_mirror():
    resolution = MirrorResolution(ns=5, mpol=4, nxi=4)
    setup = build_stellarator_mirror_hybrid(
        resolution,
        coefficient_count=16,
        straight_length=4.0,
        return_radius=2.0,
        semi_major=0.4,
        semi_minor=0.3,
        section_turns=0,
        axial_flux_derivative=0.02,
        quadrature_order=3,
    )
    return setup, setup.discretization.evaluate_state(setup.initial_state)


def test_closed_mirror_contract_is_periodic_equal_arc_and_positive(closed_mirror) -> None:
    setup, state = closed_mirror
    mapping = gk_closed_fieldline_geometry(
        state,
        setup.discretization,
        setup.axis,
        axial_flux_derivative=0.02,
        ntheta=32,
        arc_oversample=8,
    )

    for name in (
        "theta",
        "gradpar",
        "bmag",
        "bgrad",
        "gds2",
        "gds21",
        "gds22",
        "cvdrift",
        "gbdrift",
        "cvdrift0",
        "gbdrift0",
    ):
        values = np.asarray(mapping[name])
        assert values.shape == (32,)
        assert np.all(np.isfinite(values))

    np.testing.assert_allclose(mapping["gradpar"], mapping["gradpar"][0], rtol=0.0, atol=0.0)
    metric_determinant = np.asarray(mapping["gds2"]) * np.asarray(mapping["gds22"]) - np.asarray(mapping["gds21"]) ** 2
    assert np.min(metric_determinant) > 0.0
    # The named field-line mirror ratio, not an anonymous max/min: R_m,axis
    # and R_m,LCFS are separate equilibrium-level quantities (31.4-R3).
    field_line_ratio = float(mapping["vmex_mirror"]["field_line_mirror_ratio"])
    np.testing.assert_allclose(
        field_line_ratio,
        np.max(mapping["bmag"]) / np.min(mapping["bmag"]),
        rtol=1.0e-14,
    )
    assert field_line_ratio > 1.5
    # "epsilon" is the field-line |B| modulation depth, the one definition
    # shared with the core lane (GKX's own bmag = 1/(1 + eps cos theta) has
    # exactly this eps); the VMEX-named diagnostic repeats it, and the
    # field-line mirror ratio is (1 + eps) / (1 - eps).
    modulation = float(mapping["vmex_mirror"]["field_line_b_modulation"])
    np.testing.assert_allclose(modulation, (field_line_ratio - 1.0) / (field_line_ratio + 1.0), rtol=1.0e-14)
    assert float(mapping["epsilon"]) == modulation
    assert 0.0 < modulation < 1.0
    # Not the former std/mean export, which is ~eps/sqrt(2) for a cosine-like
    # modulation and never equal to the depth on a non-uniform line.
    assert abs(float(np.std(mapping["bmag"]) / np.mean(mapping["bmag"])) - modulation) > 1.0e-3
    # R0 is the effective major radius L_axis / (2 pi) (= V / (2 pi^2 L_ref^2)),
    # not L_ref, so GKX's derived aminor = epsilon * R0 is a length in metres.
    meta = mapping["vmex_mirror"]
    np.testing.assert_allclose(float(mapping["R0"]), float(meta["axis_arc_length"]) / (2.0 * np.pi), rtol=1.0e-14)
    assert float(meta["R_major"]) == float(mapping["R0"])
    assert float(mapping["R0"]) > float(meta["L_ref"]) > 0.0
    assert abs(float(mapping["vmex_mirror"]["closure_residual"])) < 1.0e-12
    assert mapping["s_hat"] == 0.0

    # Independent periodic spectral derivative of the emitted field strength.
    modes = jnp.fft.fftfreq(32, d=1.0 / 32)
    d_log_b_dz = jnp.fft.ifft(1j * modes * jnp.fft.fft(jnp.log(mapping["bmag"]))).real
    reconstructed = mapping["gradpar"] * d_log_b_dz
    np.testing.assert_allclose(mapping["bgrad"], reconstructed, rtol=6.0e-2, atol=2.0e-3)


def test_geometry_directional_derivative_matches_centered_difference(closed_mirror) -> None:
    setup, state = closed_mirror

    def objective(scale):
        scaled = MirrorState(state.radius_scale * scale, state.lambda_stream)
        mapping = gk_closed_fieldline_geometry(
            scaled,
            setup.discretization,
            setup.axis,
            axial_flux_derivative=0.02,
            ntheta=8,
            arc_oversample=2,
        )
        return jnp.mean(mapping["bmag"] ** 2)

    derivative = jax.grad(objective)(1.0)
    step = 1.0e-4
    finite_difference = (objective(1.0 + step) - objective(1.0 - step)) / (2.0 * step)
    assert np.isfinite(float(derivative))
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-5, atol=2.0e-8)


def test_nonclosing_field_line_is_rejected(closed_mirror) -> None:
    setup, state = closed_mirror
    with pytest.raises(ValueError, match="does not close"):
        gk_closed_fieldline_geometry(
            state,
            setup.discretization,
            setup.axis,
            axial_flux_derivative=0.02,
            current_derivative=0.002,
            ntheta=8,
            arc_oversample=2,
        )
