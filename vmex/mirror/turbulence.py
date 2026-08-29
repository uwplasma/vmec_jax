"""Differentiable local gyrokinetic geometry for closed VMEX mirrors.

This module deliberately supports the periodic stellarator--mirror lane.  An
open mirror needs particle, sheath, source, and end-loss boundary conditions;
turning its two end cuts into a periodic flux tube would be a different model.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import (
    ClosedAxisGeometry,
    contravariant_field,
    evaluate_closed_geometry,
    magnetic_field_squared,
    magnetic_field_xyz,
    radial_derivative,
)
from .model import MirrorState
from .splines import SplineMirrorDiscretization, trace_closed_field_line

Array = Any

GK_GEOMETRY_FIELDS = (
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
)

__all__ = ["GK_GEOMETRY_FIELDS", "gk_closed_fieldline_geometry"]


def _is_traced(value: Any) -> bool:
    return isinstance(value, jax.core.Tracer)


def _surface_interpolate(
    values: Array,
    discretization: SplineMirrorDiscretization,
    theta: Array,
    axial_parameter: Array,
) -> Array:
    """Evaluate a ``(theta, u, ...)`` quadrature array at paired points."""

    values = jnp.asarray(values)
    theta = jnp.asarray(theta)
    axial_parameter = jnp.asarray(axial_parameter)
    # Recover spline coefficients once and evaluate them with JAX.  The
    # convenience ``axial_basis.interpolate`` builds a NumPy matrix and is
    # therefore intentionally not used here: field-line locations are dynamic
    # under differentiation.
    moved = jnp.moveaxis(values, 1, -1)
    coefficients_u = jnp.tensordot(
        moved,
        jnp.asarray(discretization.grid.axial_basis.recovery_matrix).T,
        axes=((-1,), (0,)),
    )
    axial = discretization.spline.evaluate(coefficients_u, axial_parameter, axis=-1)
    axial = jnp.moveaxis(axial, -1, 1)
    modes = jnp.asarray(np.fft.fftfreq(discretization.grid.ntheta, d=1.0 / discretization.grid.ntheta))
    coefficients = jnp.fft.fft(axial, axis=0) / discretization.grid.ntheta
    phase = jnp.exp(1j * modes[:, None] * theta[None, :])
    phase = phase.reshape(phase.shape + (1,) * (values.ndim - 2))
    return jnp.real(jnp.sum(coefficients * phase, axis=0))


def _closed_surface_tensors(
    state: MirrorState,
    discretization: SplineMirrorDiscretization,
    axis: ClosedAxisGeometry,
    *,
    radial_index: int,
    axial_flux_derivative: Array,
    current_derivative: Array,
) -> tuple[dict[str, Array], Any, Any]:
    """Build exact coordinate tensors before field-line interpolation."""

    grid = discretization.grid
    geometry = evaluate_closed_geometry(state, grid, axis)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    mod_b = jnp.sqrt(jnp.maximum(magnetic_field_squared(field, geometry), 0.0))
    b_xyz = magnetic_field_xyz(field, geometry)

    j = radial_index
    coordinate_basis = jnp.stack(
        (geometry.e_s_xyz[j], geometry.e_theta_xyz[j], geometry.e_xi_xyz[j]),
        axis=-1,
    )
    dual = jnp.linalg.inv(coordinate_basis)
    grad_s, grad_theta, grad_u = dual[..., 0, :], dual[..., 1, :], dual[..., 2, :]

    ds = float(grid.s[1] - grid.s[0])
    lam = jnp.asarray(state.lambda_stream)
    lam_s = radial_derivative(lam, ds)[j]
    lam_theta = grid.theta_basis.differentiate(lam, axis=1)[j]
    lam_u = grid.axial_basis.differentiate(lam, axis=2)[j]
    psi_prime = jnp.asarray(axial_flux_derivative, dtype=lam.dtype)
    current_prime = jnp.asarray(current_derivative, dtype=lam.dtype)
    grad_alpha = (
        (lam_s / psi_prime)[..., None] * grad_s
        + (1.0 + lam_theta / psi_prime)[..., None] * grad_theta
        + (-current_prime / psi_prime + lam_u / psi_prime)[..., None] * grad_u
    )

    mod_b_s = radial_derivative(mod_b, ds)[j]
    mod_b_theta = grid.theta_basis.differentiate(mod_b, axis=1)[j]
    mod_b_u = grid.axial_basis.differentiate(mod_b, axis=2)[j]
    grad_mod_b = mod_b_s[..., None] * grad_s + mod_b_theta[..., None] * grad_theta + mod_b_u[..., None] * grad_u
    b_cross_grad_b = jnp.cross(b_xyz[j], grad_mod_b)
    b_dot_grad_b = jnp.sum(b_xyz[j] * grad_mod_b, axis=-1)

    tensors = {
        "mod_b": mod_b[j],
        "b_sup_u": field.b_sup_xi[j],
        "grad_alpha_sq": jnp.sum(grad_alpha * grad_alpha, axis=-1),
        "grad_alpha_dot_grad_s": jnp.sum(grad_alpha * grad_s, axis=-1),
        "grad_s_sq": jnp.sum(grad_s * grad_s, axis=-1),
        "b_cross_grad_b_dot_grad_alpha": jnp.sum(b_cross_grad_b * grad_alpha, axis=-1),
        "b_cross_grad_b_dot_grad_s": jnp.sum(b_cross_grad_b * grad_s, axis=-1),
        "b_dot_grad_b": b_dot_grad_b,
        "xyz": geometry.xyz[j],
    }
    return tensors, field, geometry


def gk_closed_fieldline_geometry(
    state: MirrorState,
    discretization: SplineMirrorDiscretization,
    axis: ClosedAxisGeometry,
    *,
    axial_flux_derivative: Array,
    current_derivative: Array = 0.0,
    radial_index: int | None = None,
    theta0: float = 0.0,
    ntheta: int = 32,
    arc_oversample: int = 4,
    mu0_dp_ds: Array = 0.0,
    closure_tolerance: float = 2.0e-5,
) -> dict[str, Any]:
    """Return the GKX geometry contract on one closed mirror field line.

    The exact Clebsch label is

    ``alpha = theta - I'(s) u / Psi'(s) + lambda(s,theta,u) / Psi'(s)``.

    The perpendicular metric and magnetic drifts are evaluated from the
    Cartesian coordinate basis.  The parallel coordinate is remapped to equal
    arc so ``b . grad z`` is constant, as required by GKX/GX.  The returned
    ``s_hat=0`` means that ``kx`` is the direct normalized radial wavenumber;
    the three metric arrays still retain the complete local cross metric.

    Only a field line that closes after one periodic-axis circuit is accepted.
    This is exact for the zero-current closed mirror and for integer-transform
    cases.  Irrational-transform tubes need a separate twist-linked contract.
    Open mirrors are intentionally rejected by requiring a closed
    discretization.
    """

    if not discretization.closed:
        raise ValueError("GK mirror flux tubes require a closed periodic discretization")
    if int(ntheta) < 8:
        raise ValueError("ntheta must be at least 8")
    if int(arc_oversample) < 2:
        raise ValueError("arc_oversample must be at least 2")
    if np.ndim(axial_flux_derivative) or np.ndim(current_derivative):
        raise ValueError("the first closed-mirror GK contract requires scalar flux derivatives")
    if not _is_traced(axial_flux_derivative) and float(axial_flux_derivative) == 0.0:
        raise ValueError("axial_flux_derivative must be nonzero")

    grid = discretization.grid
    j = grid.ns - 2 if radial_index is None else int(radial_index)
    if not 1 <= j < grid.ns:
        raise ValueError("radial_index must select a non-axis surface")
    s_value = jnp.asarray(grid.s[j])
    sqrt_s = jnp.sqrt(s_value)

    tensors, field, geometry = _closed_surface_tensors(
        state,
        discretization,
        axis,
        radial_index=j,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )

    fine_steps = int(arc_oversample) * int(ntheta)
    line = trace_closed_field_line(
        field,
        discretization,
        radial_index=j,
        theta0=theta0,
        turns=1,
        steps_per_turn=fine_steps,
    )
    poloidal_advance = line.theta[-1] - line.theta[0]
    closure_residual = poloidal_advance - 2.0 * jnp.pi * jnp.round(poloidal_advance / (2.0 * jnp.pi))
    if not _is_traced(closure_residual) and abs(float(closure_residual)) > float(closure_tolerance):
        raise ValueError(
            "the selected mirror field line does not close after one axis circuit; "
            "use zero current/an integer transform or a future twist-linked adapter "
            f"(closure residual {float(closure_residual):.3e} rad)"
        )

    mod_b_fine = _surface_interpolate(tensors["mod_b"], discretization, line.theta, line.axial_parameter)
    b_sup_u_fine = _surface_interpolate(tensors["b_sup_u"], discretization, line.theta, line.axial_parameter)

    effective_minor_radius = jnp.sqrt(geometry.volume / (jnp.pi * jnp.asarray(axis.arc_length)))
    b_reference = 2.0 * jnp.abs(jnp.asarray(axial_flux_derivative)) / (effective_minor_radius**2)
    gradpar_fine = effective_minor_radius * jnp.abs(b_sup_u_fine) / mod_b_fine
    du = line.axial_parameter[1] - line.axial_parameter[0]
    inverse_gradpar = 1.0 / gradpar_fine
    cumulative = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=mod_b_fine.dtype),
            jnp.cumsum(0.5 * (inverse_gradpar[1:] + inverse_gradpar[:-1]) * du),
        )
    )
    equal_theta_fine = -jnp.pi + 2.0 * jnp.pi * cumulative / cumulative[-1]
    theta = jnp.linspace(-jnp.pi, jnp.pi, int(ntheta), endpoint=False)
    u_eval = jnp.interp(theta, equal_theta_fine, line.axial_parameter)
    theta_eval = jnp.interp(theta, equal_theta_fine, line.theta)
    gradpar_value = 2.0 * jnp.pi / cumulative[-1]

    sampled = {
        name: _surface_interpolate(values, discretization, theta_eval, u_eval) for name, values in tensors.items()
    }
    mod_b = sampled["mod_b"]
    sign_psi = jnp.sign(jnp.asarray(axial_flux_derivative))
    bmag = mod_b / b_reference
    gds2 = effective_minor_radius**2 * s_value * sampled["grad_alpha_sq"]
    gds21 = 0.5 * effective_minor_radius**2 * sampled["grad_alpha_dot_grad_s"]
    gds22 = effective_minor_radius**2 * sampled["grad_s_sq"] / (4.0 * s_value)
    gbdrift = (
        -2.0
        * b_reference
        * effective_minor_radius**2
        * sqrt_s
        * sign_psi
        * sampled["b_cross_grad_b_dot_grad_alpha"]
        / mod_b**3
    )
    gbdrift0 = (
        b_reference * effective_minor_radius**2 * sign_psi * sampled["b_cross_grad_b_dot_grad_s"] / (mod_b**3 * sqrt_s)
    )
    cvdrift = gbdrift - (
        2.0
        * b_reference
        * effective_minor_radius**2
        * sqrt_s
        * jnp.asarray(mu0_dp_ds)
        / (jnp.abs(jnp.asarray(axial_flux_derivative)) * mod_b**2)
    )
    bgrad = effective_minor_radius * sampled["b_dot_grad_b"] / mod_b**2
    grad_rho = effective_minor_radius * jnp.sqrt(sampled["grad_s_sq"]) / (2.0 * sqrt_s)
    iota = poloidal_advance / (2.0 * jnp.pi)
    q = jnp.where(jnp.abs(iota) > 1.0e-10, 1.0 / jnp.abs(iota), 1.0)

    return {
        "theta": theta,
        "gradpar": gradpar_value * jnp.ones_like(theta),
        "bmag": bmag,
        "bgrad": bgrad,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": cvdrift,
        "gbdrift": gbdrift,
        "cvdrift0": gbdrift0,
        "gbdrift0": gbdrift0,
        "jacobian": 1.0 / (gradpar_value * bmag),
        "grho": grad_rho,
        "q": q,
        "s_hat": 0.0,
        "epsilon": jnp.std(bmag) / jnp.mean(bmag),
        "R0": effective_minor_radius,
        "B0": b_reference,
        "alpha": float(theta0),
        "nfp": 1,
        "vmex_mirror": {
            "surface_index": j,
            "s": s_value,
            "iota": iota,
            "closure_residual": closure_residual,
            "axis_arc_length": axis.arc_length,
            "L_ref": effective_minor_radius,
            "B_ref": b_reference,
            "u": u_eval,
            "fieldline_theta": theta_eval,
            "xyz": sampled["xyz"],
            "field_line_convention": (
                "closed Clebsch alpha = theta - I'/Psi' u + lambda/Psi'; "
                "equal-arc periodic GKX coordinate; direct zero-shear radial kx"
            ),
            "scope": (
                "closed periodic mirror-hybrid equilibrium; no open-end loss, sheath, source, or loss-cone model"
            ),
        },
    }
