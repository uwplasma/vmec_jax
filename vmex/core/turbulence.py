"""Turbulence-proxy optimization objectives via SPECTRAX-GK (R26h.h4).

Wires the gyrokinetic turbulence proxies of `SPECTRAX-GK
<https://github.com/uwplasma/spectrax-gk>`_ (uwplasma; JAX-native
Hermite-Laguerre flux-tube solver) to converged ``(SpectralState,
SolverRuntime)`` pairs, in two layers:

1. **Geometry adapter** — :func:`gk_fieldline_geometry` samples one field
   line of the converged interior solution and emits the solver-ready
   flux-tube geometry contract of
   ``spectraxgk.flux_tube_geometry_from_mapping`` (``bmag``, ``gradpar``,
   ``gds2/gds21/gds22``, ``gbdrift/gbdrift0``, ``cvdrift/cvdrift0``,
   ``bgrad``, …).  The arrays follow the GS2/GX normalizations of simsopt's
   ``vmec_fieldlines`` (Landreman) — the exact conventions already used by
   the ballooning objective in :mod:`vmex.core.stability`, whose
   spectral point-evaluation machinery (``_ballooning_context``,
   ``_parabola``, ``_theta_vmec_from_pest``) is reused here, extended with
   the ``grad s``/``grad psi`` metric and drift projections that ballooning
   does not need.  Everything is exact trig sums + JAX AD: the adapter is
   jit/grad-transparent and needs no spectraxgk import.

2. **Objective wrappers** — thin ``(state, runtime)`` callables around the
   proxies SPECTRAX-GK itself promotes for VMEC-side optimization (its
   ``VMECJAXTransportObjectiveConfig`` kinds, docs
   ``stellarator_optimization.rst``):

   - :func:`turbulent_growth_rate` — kind ``"growth"``: dominant linear
     ITG/TEM-branch growth rate ``gamma`` of the spectral gyrokinetic
     operator on the sampled flux tube
     (``spectraxgk.solver_growth_rate_from_geometry``; the eigenvalue
     carries SPECTRAX-GK's implicit-eigenpair custom AD rule).
   - :func:`quasilinear_flux_proxy` — kind ``"quasilinear_flux"``: the
     mixing-length quasilinear heat-flux proxy
     ``gamma * W_Q / k_perp_eff^2`` built from the dominant eigenmode's
     heat-flux weight and effective perpendicular wavenumber.
   - :func:`nonlinear_heat_flux_proxy` — kind
     ``"nonlinear_window_heat_flux"``: SPECTRAX-GK's smooth reduced
     nonlinear-window heat-flux surrogate (saturation-rule closure
     ``csat * W_Q * 2 gamma_+ / (1 + 2.2 k_perp_eff^2 + 0.15 gamma_+)``,
     ``spectraxgk.objectives.vmec_transport_tables``).  This is the
     documented *proxy* for the nonlinear transport window; a production
     nonlinear claim still requires SPECTRAX-GK's matched long nonlinear
     audits, per its own docs.
   - :func:`turbulence_objective_vector` — the underlying ordered
     ``SOLVER_OBJECTIVE_NAMES`` vector ``(gamma, omega, kperp_eff2,
     linear_heat_flux_weight, linear_particle_flux_weight,
     mixing_length_heat_flux_proxy)``.

   Each wrapper is a two-positional ``(state, runtime)`` callable, so it
   composes with :func:`vmex.core.optimize.least_squares`.
   Traceability status (validated in ``tests/test_turbulence.py``):
   SPECTRAX-GK is JAX-native, and :func:`turbulent_growth_rate` is fully
   differentiable in both AD modes (``jac=None`` *and* ``jac="implicit"``
   — the wrapper reduces SPECTRAX-GK's explicit operator matrix with
   ``jnp.linalg.eigvals``, which carries a JVP, where SPECTRAX-GK's own
   ``dominant_real_eigenvalue`` is a reverse-only ``custom_vjp`` that the
   forward-mode implicit Jacobian cannot trace).  The quasilinear and
   nonlinear-window proxies additionally weight the dominant *eigenvector*
   (heat-flux weight, ``kperp_eff``), and JAX declines derivatives of
   non-symmetric eigenvectors — those two are value-level objectives:
   use ``jac=None`` (finite differences), exactly like the wout-engine
   terms (``d_merc``, ``l_grad_b``) in the optimization examples.

The heavy dependency is optional: only the objective wrappers import
``spectraxgk`` (``pip install spectraxgk``; its ``solvax`` pin is satisfied
API-wise by the in-house solvax's ``gmres``/``tridiagonal_solve``/
``chunked_jacfwd``).  The geometry adapter works without it.

Scope notes
-----------
- Stellarator-symmetric states only (``lasym = False``), inherited from
  :func:`vmex.core.stability._ballooning_context`.
- Surfaces need ``iota != 0`` (field-line parameterization divides by iota).
- The flux tube covers one poloidal turn ``theta in [-pi, pi)`` (the solver
  z-grid convention of ``spectraxgk.core.grid.build_spectral_grid``); the
  parallel boundary is handled by SPECTRAX-GK's twist-shift machinery from
  the emitted ``q``/``s_hat``/``nfp``.
- ``gds21``/``gbdrift0`` signs follow simsopt ``vmec_fieldlines`` with
  ``psi = s * psi_edge`` in vmex's internal (signed) edge-flux
  convention.  The default single-``kx`` proxies (``nx = 1``) are
  insensitive to this overall sign, matching SPECTRAX-GK's own VMEC bridge.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .solver import SolverRuntime, SpectralState
from .stability import (
    _ballooning_context, _parabola, _theta_vmec_from_pest,
    _validate_surface_index,
)

__all__ = [
    "GK_GEOMETRY_FIELDS",
    "TURBULENCE_OBJECTIVE_NAMES",
    "gk_fieldline_geometry",
    "flux_tube_geometry",
    "turbulence_objective_vector",
    "turbulent_growth_rate",
    "quasilinear_flux_proxy",
    "nonlinear_heat_flux_proxy",
]

Array = Any

#: Field-line array fields of the SPECTRAX-GK flux-tube geometry contract
#: (``spectraxgk.geometry.flux_tube_contract._ARRAY_FIELDS``).
GK_GEOMETRY_FIELDS = (
    "theta", "gradpar", "bmag", "bgrad", "gds2", "gds21", "gds22",
    "cvdrift", "gbdrift", "cvdrift0", "gbdrift0",
)

#: Ordered observables of :func:`turbulence_objective_vector`
#: (``spectraxgk.SOLVER_OBJECTIVE_NAMES``).
TURBULENCE_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "linear_particle_flux_weight",
    "mixing_length_heat_flux_proxy",
)


def _spectraxgk():
    """Import the optional spectraxgk dependency with a helpful error."""
    try:
        import spectraxgk
    except ImportError as err:  # pragma: no cover - exercised via message test
        raise ImportError(
            "the turbulence objectives need the optional dependency "
            "spectraxgk (github.com/uwplasma/spectrax-gk): pip install "
            "spectraxgk.  The geometry adapter gk_fieldline_geometry works "
            "without it.") from err
    return spectraxgk


# ---------------------------------------------------------------------------
# Field-line point geometry (extends stability._make_point_fn with the
# grad-s / grad-psi projections the GK metric and drift arrays need)
# ---------------------------------------------------------------------------


def _make_gk_point_fn(m: Array, xn: Array, rtab: Array, ztab: Array,
                      ltab: Array, iota: Array, diota: Array, phipf_j: Array):
    """Point-evaluation closure for one flux surface (GK geometry set).

    Same spectral machinery as :func:`vmex.core.stability._make_point_fn`
    (radial parabola tables, cylindrical position via trig sums, covariant/
    dual bases and ``nabla |B|`` from JAX AD), returning the extended tuple

    ``(|B|, B^phi, |grad alpha|^2, grad alpha . grad s, |grad s|^2,
    B x grad|B| . grad alpha, B x grad|B| . grad s, B . grad|B|)``

    at ``q = (t, theta, phi)`` with ``t = s - s_j`` (evaluated at ``t = 0``)
    and ``phi_rel = phi - zeta0`` carrying the secular shear term of
    ``grad alpha``.
    """

    def coeffs(tab: Array, t: Array) -> Array:
        return tab[0] + t * tab[1] + (t * t) * tab[2]

    def lam_fn(q: Array) -> Array:
        t, th, ph = q[0], q[1], q[2]
        return coeffs(ltab, t) @ jnp.sin(th * m - ph * xn)

    def pos_fn(q: Array) -> Array:
        t, th, ph = q[0], q[1], q[2]
        ang = th * m - ph * xn
        R = coeffs(rtab, t) @ jnp.cos(ang)
        Z = coeffs(ztab, t) @ jnp.sin(ang)
        return jnp.array([R * jnp.cos(ph), R * jnp.sin(ph), Z])

    def b_vector(q: Array) -> Array:
        J = jax.jacfwd(pos_fn)(q)                     # columns: e_s, e_th, e_ph
        sqrt_g = jnp.linalg.det(J)
        lam_g = jax.grad(lam_fn)(q)                   # (lam_s, lam_th, lam_ph)
        iota_t = iota + diota * q[0]
        return phipf_j * ((iota_t - lam_g[2]) * J[:, 1]
                          + (1.0 + lam_g[1]) * J[:, 2]) / sqrt_g

    def modb_fn(q: Array) -> Array:
        return jnp.linalg.norm(b_vector(q))

    def point(q: Array, phi_rel: Array):
        J = jax.jacfwd(pos_fn)(q)
        sqrt_g = jnp.linalg.det(J)
        dual = jnp.linalg.inv(J)                      # rows: grad s, grad th, grad ph
        lam_g = jax.grad(lam_fn)(q)
        iota_t = iota + diota * q[0]
        B = phipf_j * ((iota_t - lam_g[2]) * J[:, 1]
                       + (1.0 + lam_g[1]) * J[:, 2]) / sqrt_g
        modB = jnp.linalg.norm(B)
        dB = jax.grad(modb_fn)(q)                     # (d|B|/ds, d/dth, d/dph)
        grad_modB = dB[0] * dual[0] + dB[1] * dual[1] + dB[2] * dual[2]
        # grad alpha with the secular shear term, alpha = theta* - iota (phi - zeta0).
        alpha_cov = jnp.array([lam_g[0] - phi_rel * diota,
                               1.0 + lam_g[1],
                               lam_g[2] - iota_t])
        grad_alpha = (alpha_cov[0] * dual[0] + alpha_cov[1] * dual[1]
                      + alpha_cov[2] * dual[2])
        grad_s = dual[0]
        b_sup_phi = phipf_j * (1.0 + lam_g[1]) / sqrt_g
        b_cross_gradb = jnp.cross(B, grad_modB)
        return (modB, b_sup_phi,
                grad_alpha @ grad_alpha, grad_alpha @ grad_s, grad_s @ grad_s,
                b_cross_gradb @ grad_alpha, b_cross_gradb @ grad_s,
                B @ grad_modB)

    return point


def _resolve_surface(s_index, ns: int) -> int:
    if s_index is None:
        s_index = min(max(int(round(0.6 * (ns - 1))), 2), ns - 2)
    return _validate_surface_index(s_index, ns)


def _line_arrays(ctx: dict, j: int, alpha: float, zeta0: float, x: Array):
    """Raw point-geometry tuple along the field line at PEST angles ``alpha + x``."""
    hs = ctx["hs"]
    iotas = ctx["iotas"]
    iota = 0.5 * (iotas[j] + iotas[j + 1])
    point = _make_gk_point_fn(
        ctx["m"], ctx["xn"],
        _parabola(ctx["rmnc"], j, hs),
        _parabola(ctx["zmns"], j, hs),
        _parabola(ctx["lmns"], j, hs),
        iota, (iotas[j + 1] - iotas[j]) / hs, ctx["phipf"][j],
    )
    theta_star = alpha + x
    phi = zeta0 + x / iota                 # field line: theta* = alpha + iota (phi - zeta0)
    lmns0 = _parabola(ctx["lmns"], j, hs)[0]
    theta_v = _theta_vmec_from_pest(theta_star, phi, lmns0, ctx["m"], ctx["xn"])
    q = jnp.stack([jnp.zeros_like(theta_v), theta_v, phi], axis=-1)
    return jax.vmap(point)(q, phi - zeta0)


# ---------------------------------------------------------------------------
# Geometry adapter
# ---------------------------------------------------------------------------


def gk_fieldline_geometry(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    s_index: int | None = None,
    alpha: float = 0.0,
    zeta0: float = 0.0,
    ntheta: int = 32,
    equal_arc: bool = True,
    arc_oversample: int = 4,
) -> dict:
    """Flux-tube geometry mapping of one field line of a converged state.

    Returns the in-memory geometry contract consumed by
    ``spectraxgk.flux_tube_geometry_from_mapping`` (keys
    :data:`GK_GEOMETRY_FIELDS` plus ``grho``/``jacobian`` and the scalar
    metadata ``q``, ``s_hat``, ``epsilon``, ``R0``, ``B0``, ``alpha``,
    ``nfp``), all in the GS2/GX normalizations of simsopt
    ``vmec_fieldlines`` with ``L_ref`` the effective minor radius and
    ``B_ref = 2 |psi_edge| / L_ref^2`` (identical to
    :mod:`vmex.core.stability`).  A ``"vmex"`` sub-dict carries
    diagnostics used by the parity tests (``dp_drho``, ``gradpar_profile``,
    the sampled PEST angles, …).  Pure jnp — traceable and differentiable
    w.r.t. ``(state, runtime)``; no spectraxgk import.

    Parameters
    ----------
    s_index:
        Full-mesh surface index in ``[2, ns - 2]``; default ~60 % of the
        radius (a typical core gradient region).
    alpha, zeta0:
        Field-line label ``alpha = theta* - iota (phi - zeta0)`` and the
        toroidal angle of the tube center.
    ntheta:
        Parallel samples over one poloidal turn; the emitted ``theta`` is
        ``linspace(-pi, pi, ntheta, endpoint=False)`` — exactly the
        SPECTRAX-GK solver z grid.
    equal_arc:
        Resample the parallel coordinate so ``b . grad z`` is constant
        (``gradpar`` exactly uniform, SPECTRAX-GK's validated contract).
        The coordinate map is built from an ``arc_oversample`` x finer
        quadrature of ``1/gradpar``; geometry values are exact spectral
        evaluations at the mapped points (only the map itself is
        interpolated).  ``equal_arc=False`` samples uniformly in the PEST
        angle instead (stability.py's grid; ``gradpar`` then varies along
        the line and downstream use relies on spectraxgk's mean-``gradpar``
        reduction, as in its own VMEC bridge).
    """
    if int(ntheta) < 8:
        raise ValueError("ntheta must be >= 8")
    ctx = _ballooning_context(state, rt)
    j = _resolve_surface(s_index, ctx["ns"])
    dtype = ctx["s"].dtype

    hs = ctx["hs"]
    s_j = ctx["s"][j]
    sqrt_s = jnp.sqrt(s_j)
    iotas, pres = ctx["iotas"], ctx["pres"]
    iota = 0.5 * (iotas[j] + iotas[j + 1])
    diota = (iotas[j + 1] - iotas[j]) / hs
    dpres = (pres[j + 1] - pres[j]) / hs            # internal units: mu0 dp/ds
    shat = -2.0 * s_j * diota / iota                # (r/q) dq/dr, r = L_ref sqrt(s)
    L_ref, B_ref = ctx["L_ref"], ctx["B_ref"]
    psi_edge, sign_psi = ctx["psi_edge"], ctx["sign_psi"]
    alpha_c = jnp.asarray(alpha, dtype=dtype)
    zeta0_c = jnp.asarray(zeta0, dtype=dtype)

    def gradpar_of(modB: Array, b_sup_phi: Array) -> Array:
        return jnp.abs(L_ref * iota * b_sup_phi / modB)   # L_ref |b . grad theta*|

    theta = jnp.linspace(-jnp.pi, jnp.pi, int(ntheta), endpoint=False, dtype=dtype)
    if equal_arc:
        # Monotone map x(z) with b.grad z constant: z ~ cumulative int dx / gradpar(x).
        nfine = int(arc_oversample) * int(ntheta) + 1
        x_fine = jnp.linspace(-jnp.pi, jnp.pi, nfine, dtype=dtype)
        modB_f, b_sup_phi_f, *_ = _line_arrays(ctx, j, alpha_c, zeta0_c, x_fine)
        w = 1.0 / gradpar_of(modB_f, b_sup_phi_f)
        dx = x_fine[1] - x_fine[0]
        cum = jnp.concatenate([jnp.zeros((1,), dtype=dtype),
                               jnp.cumsum(0.5 * (w[1:] + w[:-1]) * dx)])
        z_fine = -jnp.pi + 2.0 * jnp.pi * cum / cum[-1]
        x_eval = jnp.interp(theta, z_fine, x_fine)
        gradpar_value = 2.0 * jnp.pi / cum[-1]            # = b.grad z (constant)
    else:
        x_eval = theta
        gradpar_value = None

    (modB, b_sup_phi, gaa, gas, gss,
     bxgb_dot_ga, bxgb_dot_gs, b_dot_gradb) = _line_arrays(ctx, j, alpha_c, zeta0_c, x_eval)

    bmag = modB / B_ref
    gradpar_profile = gradpar_of(modB, b_sup_phi)
    gradpar = (gradpar_value * jnp.ones_like(bmag) if equal_arc else gradpar_profile)
    gds2 = gaa * (L_ref * L_ref) * s_j
    gds21 = (psi_edge * gas) * shat / B_ref
    gds22 = (psi_edge * psi_edge * gss) * shat * shat / (L_ref * L_ref * B_ref * B_ref * s_j)
    gbdrift = (-2.0 * B_ref * L_ref * L_ref * sqrt_s * sign_psi
               * bxgb_dot_ga / (modB ** 3))
    gbdrift0 = (psi_edge * bxgb_dot_gs) * 2.0 * shat * sign_psi / (modB ** 3 * sqrt_s)
    cvdrift = gbdrift - (2.0 * B_ref * L_ref * L_ref * sqrt_s * dpres
                         / (jnp.abs(psi_edge) * modB * modB))
    bgrad = L_ref * b_dot_gradb / (modB * modB)           # b . grad ln|B|, normalized
    grho = L_ref * jnp.sqrt(gss) / (2.0 * sqrt_s)         # |grad rho| L_ref, rho = sqrt(s)

    mean_b = jnp.mean(bmag)
    return {
        "theta": theta,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": bgrad,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": cvdrift,
        "gbdrift": gbdrift,
        "cvdrift0": gbdrift0,
        "gbdrift0": gbdrift0,
        "jacobian": 1.0 / (gradpar * bmag),
        "grho": grho,
        "q": 1.0 / jnp.abs(iota),
        "s_hat": shat,
        "epsilon": jnp.std(bmag) / mean_b,
        "R0": L_ref,
        "B0": B_ref,
        "alpha": float(alpha),
        "nfp": int(rt.resolution.nfp),
        "vmex": {
            "surface_index": j,
            "s": s_j,
            "iota": iota,
            "d_iota_d_s": diota,
            "shat": shat,
            "dp_drho": 2.0 * sqrt_s * dpres / (B_ref * B_ref),
            "L_ref": L_ref,
            "B_ref": B_ref,
            "psi_edge": psi_edge,
            "sign_psi": sign_psi,
            "theta_pest": alpha_c + x_eval,
            "gradpar_profile": gradpar_profile,
            "field_line_convention":
                "PEST theta* = alpha + iota (phi - zeta0); simsopt "
                "vmec_fieldlines normalizations; internal signed psi_edge",
        },
    }


def flux_tube_geometry(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    validate: bool = False,
    **geometry_kwargs,
):
    """SPECTRAX-GK ``FluxTubeGeometryData`` for one field line (needs spectraxgk).

    Thin wrapper: :func:`gk_fieldline_geometry` ->
    ``spectraxgk.flux_tube_geometry_from_mapping``.  ``validate=True`` turns
    on spectraxgk's host-side finite/constant-``gradpar`` checks (concrete
    arrays only — leave ``False`` under jit/grad tracing).
    """
    spx = _spectraxgk()
    return spx.flux_tube_geometry_from_mapping(
        gk_fieldline_geometry(state, rt, **geometry_kwargs),
        source_model="vmex:core.turbulence",
        validate_finite=bool(validate),
    )


# ---------------------------------------------------------------------------
# Objective wrappers (SPECTRAX-GK proxies as (state, runtime) callables)
# ---------------------------------------------------------------------------

_GEOMETRY_KEYS = ("s_index", "alpha", "zeta0", "ntheta", "equal_arc", "arc_oversample")


def _split_kwargs(kwargs: dict) -> tuple[dict, dict]:
    geometry = {k: kwargs.pop(k) for k in _GEOMETRY_KEYS if k in kwargs}
    return geometry, kwargs


def _linear_params(spx, params_linear, r_over_lt, r_over_ln):
    """SPECTRAX-GK LinearParams: explicit object, or its collisionless
    optimization defaults with optionally overridden drive gradients."""
    if params_linear is not None:
        if r_over_lt is not None or r_over_ln is not None:
            raise ValueError("pass either params_linear or r_over_lt/r_over_ln, not both")
        return params_linear
    from spectraxgk.objectives.core import _default_gradient_linear_params
    params = _default_gradient_linear_params()
    import dataclasses
    updates = {}
    if r_over_lt is not None:
        updates["R_over_LTi"] = float(r_over_lt)
    if r_over_ln is not None:
        updates["R_over_Ln"] = float(r_over_ln)
    return dataclasses.replace(params, **updates) if updates else params


def turbulence_objective_vector(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    selected_ky_index: int = 1,
    n_laguerre: int = 2,
    n_hermite: int = 3,
    nx: int = 1,
    ny: int = 4,
    lx: float = 6.0,
    ly: float = 12.0,
    params_linear=None,
    terms=None,
    r_over_lt: float | None = None,
    r_over_ln: float | None = None,
    **geometry_kwargs,
) -> jnp.ndarray:
    """Ordered SPECTRAX-GK linear/quasilinear observable vector (traceable).

    Samples one flux tube (:func:`gk_fieldline_geometry` keyword arguments
    ``s_index``/``alpha``/``zeta0``/``ntheta``/``equal_arc`` pass through),
    builds SPECTRAX-GK's spectral linear gyrokinetic operator on it at the
    ``selected_ky_index`` binormal wavenumber (``ky = 2 pi k / ly`` in
    ``rho_ref`` units), selects the maximum-growth eigenbranch, and returns
    :data:`TURBULENCE_OBJECTIVE_NAMES`
    (``spectraxgk.solver_objective_vector_from_geometry``).

    The drive gradients live in SPECTRAX-GK's ``LinearParams``
    (``params_linear``; default: its collisionless optimization defaults
    ``R/L_n = 2.2``, ``R/L_Ti = 6.9`` — the Cyclone-base ITG drive —
    optionally overridden via ``r_over_lt``/``r_over_ln``).
    """
    spx = _spectraxgk()
    geom = flux_tube_geometry(state, rt, **geometry_kwargs)
    return spx.solver_objective_vector_from_geometry(
        geom,
        selected_ky_index=int(selected_ky_index),
        n_laguerre=int(n_laguerre), n_hermite=int(n_hermite),
        nx=int(nx), ny=int(ny), lx=float(lx), ly=float(ly),
        params_linear=_linear_params(spx, params_linear, r_over_lt, r_over_ln),
        terms=terms,
    )


def turbulent_growth_rate(state: SpectralState, rt: SolverRuntime, **kwargs) -> jnp.ndarray:
    """Dominant linear gyrokinetic growth rate on one flux tube (traceable).

    SPECTRAX-GK objective kind ``"growth"``: the largest real part of the
    eigenvalues of its spectral Hermite-Laguerre linear operator on the
    sampled flux tube, in ``v_th / L_ref`` units.  Positive = unstable.

    The operator matrix is SPECTRAX-GK's own
    (``spectraxgk.solver_linear_operator_matrix_from_geometry`` — the exact
    matrix behind its ``solver_growth_rate_from_geometry``); the eigenvalue
    reduction here uses ``jnp.linalg.eigvals`` so the objective carries
    *both* JVP and VJP rules — SPECTRAX-GK's ``dominant_real_eigenvalue``
    is a reverse-only ``custom_vjp``, which vmex's forward-mode
    implicit Jacobian cannot trace (values agree to roundoff; gated in
    ``tests/test_turbulence.py``).  Keyword arguments as
    :func:`turbulence_objective_vector` (minus the eigenvector-dependent
    pieces).  Two-positional ``(state, runtime)`` — drive it toward zero /
    negative in :func:`vmex.core.optimize.least_squares` with
    ``jac=None`` or ``jac="implicit"``.
    """
    geometry_kwargs, solver_kwargs = _split_kwargs(dict(kwargs))
    spx = _spectraxgk()
    params_linear = _linear_params(
        spx, solver_kwargs.pop("params_linear", None),
        solver_kwargs.pop("r_over_lt", None), solver_kwargs.pop("r_over_ln", None))
    geom = flux_tube_geometry(state, rt, **geometry_kwargs)
    matrix = spx.solver_linear_operator_matrix_from_geometry(
        geom, params_linear=params_linear, **solver_kwargs)
    eigenvalues = jnp.linalg.eigvals(matrix)
    return jnp.real(eigenvalues[jnp.argmax(jnp.real(eigenvalues))])


def quasilinear_flux_proxy(state: SpectralState, rt: SolverRuntime, **kwargs) -> jnp.ndarray:
    """Mixing-length quasilinear heat-flux proxy (value-level; ``jac=None``).

    SPECTRAX-GK objective kind ``"quasilinear_flux"``: ``gamma * W_Q /
    max(kperp_eff^2, 1e-12)`` with ``W_Q`` the dominant mode's normalized
    heat-flux weight — the mixing-length saturation rule of its quasilinear
    transport lane.  The weight depends on the dominant *eigenvector*, whose
    non-symmetric derivatives JAX declines — use finite differences
    (``jac=None``), like the wout-engine terms.  Keyword arguments as
    :func:`turbulence_objective_vector`.
    """
    spx = _spectraxgk()
    vector = turbulence_objective_vector(state, rt, **kwargs)
    return spx.solver_scalar_objective_from_vector(vector, "quasilinear_flux")


def nonlinear_heat_flux_proxy(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    csat: float = 0.85,
    saturation_floor: float = 1.0e-10,
    **kwargs,
) -> jnp.ndarray:
    """Smooth reduced nonlinear-window heat-flux surrogate (value-level; ``jac=None``).

    SPECTRAX-GK objective kind ``"nonlinear_window_heat_flux"``: its
    saturation-rule closure ``csat * max(W_Q, 0) * 2 gamma_+ /
    (1 + 2.2 kperp_eff^2 + 0.15 gamma_+)`` mapping the linear solver row to
    a nonlinear heat-flux proxy (``spectraxgk.objectives.
    vmec_transport_tables._solver_table_to_nonlinear_window_proxy`` — the
    exact objective its VMEC-JAX optimization scripts use for this kind).
    This is a smooth *surrogate* for the nonlinear transport window;
    SPECTRAX-GK's docs require matched long nonlinear audits before any
    production nonlinear claim.  Eigenvector-weighted like
    :func:`quasilinear_flux_proxy` — use ``jac=None``.  Keyword arguments
    as :func:`turbulence_objective_vector`.
    """
    _spectraxgk()
    from spectraxgk.objectives.vmec_transport_config import VMECJAXTransportObjectiveConfig
    from spectraxgk.objectives.vmec_transport_tables import (
        _solver_table_to_nonlinear_window_proxy,
    )
    config = VMECJAXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        nonlinear_csat=float(csat),
        nonlinear_saturation_floor=float(saturation_floor),
    )
    vector = turbulence_objective_vector(state, rt, **kwargs)
    return _solver_table_to_nonlinear_window_proxy(vector, config)
