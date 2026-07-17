"""Optimization objectives and least-squares driver for the new core (plan.md §5.1, §10).

Simsopt-style vocabulary for the QA/QH/QP/QI examples on the pure new core:

- :class:`QuasisymmetryRatioResidual` — the two-term quasisymmetry ratio
  residual of Landreman & Paul (simsopt ``QuasisymmetryRatioResidual``),
  evaluated from the wout-engine field tables of a converged core state;
  math ported verbatim from the parity-proven legacy
  ``vmec_jax/quasisymmetry.py`` (``quasisymmetry_ratio_residual_from_wout``).
- practical scalar targets — :func:`aspect_ratio`, :func:`mean_iota`,
  :func:`edge_iota`, :func:`mirror_ratio`, :func:`volume`,
  :func:`magnetic_well` — each a pure function of
  ``(SpectralState, SolverRuntime)`` on :mod:`vmec_jax.core.geometry` /
  :mod:`vmec_jax.core.fields`.
- a distilled Goodman-style QI residual (:func:`quasi_isodynamic_residual`)
  keeping exactly the four terms the legacy minimal-seed QI examples
  exercised (now ``examples/optimization/QI_optimization.py``): level-set
  bounce-width
  variance, branch trapped-well width variance, field-line profile
  consistency, and the branch-shuffle profile comparison.  The unused legacy
  knobs (``aligned_profile_*``, ``weighted_shuffle_*``,
  ``shuffle_profile_nphi_out``) were dropped.
- :func:`least_squares` — a thin :func:`scipy.optimize.least_squares` driver
  over boundary Fourier dofs (:func:`pack_boundary`/:func:`unpack_boundary`),
  taking simsopt-style ``(callable, target, weight)`` terms.

Helicity conventions (match legacy/simsopt exactly)
---------------------------------------------------
The QS residual keeps the ``|B|`` spectrum aligned with the single helicity
``chi = helicity_m * theta - helicity_n * nfp * phi`` — ``helicity_n`` is in
units of ``nfp`` (the internal target mode number is ``nn = helicity_n * nfp``):

- QA: ``(helicity_m, helicity_n) = (1, 0)``
- QH: ``(1, -1)`` (i.e. ``chi = theta + nfp*phi``; legacy/simsopt sign — the
  plan's "``n = -nfp``" written in physical toroidal mode numbers)
- QP: ``(0, 1)``

Gradient modes
--------------
:func:`least_squares` defaults to scipy finite differences (``jac=None`` ->
``"2-point"``).  ``jac="implicit"`` uses the Phase-6 implicit-gradient path
(:mod:`vmec_jax.core.implicit`): each trial boundary is solved once through
:func:`~vmec_jax.core.implicit.solve_implicit` (a ``jax.custom_vjp`` around
the host solver) and the exact residual Jacobian comes from *forward*
implicit differentiation of the fixed point — one preconditioned GMRES per
boundary dof (a few dozen residual linearizations each) instead of one full
equilibrium solve per dof.  In implicit mode every objective term must be a
traceable function of ``(SpectralState, SolverRuntime)``; vector-valued
terms exposing a ``residuals_state`` method
(:class:`QuasisymmetryRatioResidual`) contribute their full pointwise
residual vector, matching the finite-difference stacked-residual cost and
Gauss-Newton geometry (internal-grid sampling instead of the wout grid).
Wout-engine terms (:func:`d_merc`, :func:`l_grad_b`, the Boozer-based QI
residual) run on host NumPy and are finite-difference-only.  The implicit
parameter map does not implement the lasym ``readin.f`` delta rotation, so
``jac="implicit"`` requires ``lasym = False``.

Measured cost (2026-07-10, RTX A4000, nfp2 circular seed, QS + aspect +
iota objective, ``max_mode=2`` -> 24 dofs): warm implicit Jacobian 2.5 s
(~1.5 hot-restart equilibrium-solve equivalents, independent of the dof
count) vs the 2-point FD Jacobian's 24 hot solves ~ 39 s — **15.7x**; the
gap widens linearly with ``max_mode``.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from solvax import (
    auto_chunk_size,
    block_thomas_factor,
    block_thomas_solve,
    chunk_map,
)

from .input import VmecInput
from .multigrid import solve_multigrid
from .solver import (
    SolveResult,
    SolverRuntime,
    SpectralState,
    prepare_runtime,
    resolution_from_input,
)
from .fields import surface_currents
# Shared state-physics primitives (statephysics.py, R26a).  Re-exported here
# for backward compatibility: external user code and tests reach them as
# ``vmec_jax.core.optimize._as_1d`` etc.
from .statephysics import (
    _as_1d,
    _field_chain,
    _half_grid,
    _interp_half_grid,
    _iotas_half,
    _iotas_half_from_fields,
    _mode_matrix,
)
from .wout import WoutData, wout_from_state

__all__ = [
    "Equilibrium",
    "solve_equilibrium",
    "QuasisymmetryRatioResidual",
    "aspect_ratio",
    "mean_iota",
    "edge_iota",
    "mirror_ratio",
    "volume",
    "magnetic_well",
    "d_merc",
    "l_grad_b",
    "quasi_isodynamic_residual",
    "boozer_modes_from_wout",
    "quasi_isodynamic_residual_from_wout",
    "boundary_dof_names",
    "pack_boundary",
    "unpack_boundary",
    "least_squares",
    "RedlBootstrapMismatch",  # noqa: F822 - provided lazily by __getattr__ below
]

Array = Any


def __getattr__(name: str):  # PEP 562 lazy re-export
    # bootstrap.py lazily imports this module inside self_consistent_bootstrap,
    # so the f_boot objective is re-exported lazily to keep the two decoupled.
    if name == "RedlBootstrapMismatch":
        from .bootstrap import RedlBootstrapMismatch
        return RedlBootstrapMismatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ===========================================================================
# Converged-equilibrium bundle
# ===========================================================================


@dataclass(frozen=True)
class Equilibrium:
    """A converged fixed-boundary equilibrium plus its evaluation contexts.

    Objective callables in :func:`least_squares` receive one of these.  The
    solver-native pieces (``state``, ``runtime``) feed the differentiable
    scalar targets; ``wout`` (built lazily, host NumPy) feeds the wout-table
    objectives (QS ratio residual, Boozer-based QI residual).
    """

    inp: VmecInput
    state: SpectralState
    runtime: SolverRuntime
    result: SolveResult

    @cached_property
    def wout(self) -> WoutData:
        """Full wout dataset of this state (``vmec_jax.core.wout``, cached)."""
        r = self.result
        return wout_from_state(
            inp=self.inp, state=self.state,
            fsqr=float(r.fsqr), fsqz=float(r.fsqz), fsql=float(r.fsql),
            niter=int(r.iterations), converged=bool(r.converged),
        )


def solve_equilibrium(
    inp: VmecInput,
    *,
    initial_state: SpectralState | None = None,
    raise_on_max_iterations: bool = False,
    **solve_kwargs,
) -> Equilibrium:
    """Converge ``inp`` with the core multigrid solver -> :class:`Equilibrium`.

    ``raise_on_max_iterations=False`` by default: during optimization a
    NITER-exhausted trial state is still a usable (penalized) sample —
    VMEC2000 behaves the same way.  Extra keywords go to
    :func:`vmec_jax.core.multigrid.solve_multigrid`.
    """
    result = solve_multigrid(
        inp, verbose=False, initial_state=initial_state,
        raise_on_max_iterations=raise_on_max_iterations, **solve_kwargs,
    )
    ns = int(np.shape(result.state.R_cos)[0])
    runtime = prepare_runtime(inp, resolution_from_input(inp, ns=ns))
    return Equilibrium(inp=inp, state=result.state, runtime=runtime, result=result)


# ===========================================================================
# Quasisymmetry ratio residual (simsopt convention; legacy parity port)
# ===========================================================================


class QuasisymmetryRatioResidual:
    """Two-term quasisymmetry ratio residual (simsopt convention).

    On each requested surface the field is sampled on a uniform
    ``(theta, phi)`` grid (VMEC angles) and the pointwise residual

    ``f = [(B x grad B . grad psi)(nn - iota*m) - (B . grad B)(m*G + nn*I)] / B^3``

    (``m = helicity_m``, ``nn = helicity_n * nfp``, ``G``/``I`` the Boozer
    covariant field averages ``bvco``/``buco``) is weighted by the
    flux-surface measure ``sqrt(nfp*dtheta*dphi*|sqrt g| / V')`` so that
    ``total = sum(residuals**2)`` is simsopt's surface-averaged QS ratio.
    ``f`` vanishes identically iff ``|B|`` depends on the angles only through
    ``helicity_m*theta - nn*phi``.

    The evaluation consumes the parity-proven wout-engine tables
    (``bmnc/gmnc/bsub*/bsup*``, :mod:`vmec_jax.core.nyquist`) of a
    :class:`~vmec_jax.core.wout.WoutData` — from
    :func:`~vmec_jax.core.wout.wout_from_state` or any ``wout_*.nc`` — ported
    from legacy ``quasisymmetry_ratio_residual_from_wout`` (A/B bit-exact).
    """

    name = "qs"

    def __init__(
        self,
        surfaces,
        helicity_m: int = 1,
        helicity_n: int = 0,
        *,
        weights: Iterable[float] | None = None,
        ntheta: int = 63,
        nphi: int = 64,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.helicity_m = int(helicity_m)
        self.helicity_n = int(helicity_n)
        self.weights = None if weights is None else np.asarray(list(weights), dtype=float)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)

    # -- wout-table evaluation ------------------------------------------------

    def compute(self, wout) -> dict[str, Array]:
        """Full diagnostics dict from a wout-like object or :class:`Equilibrium`."""
        if isinstance(wout, Equilibrium):
            wout = wout.wout
        surfaces = _as_1d(self.surfaces)
        nsurf = int(surfaces.shape[0])
        weights = jnp.ones((nsurf,)) if self.weights is None else _as_1d(self.weights)
        if int(weights.shape[0]) != nsurf:
            raise ValueError("weights must have the same length as surfaces")

        nfp = int(wout.nfp)
        iotas = _as_1d(np.asarray(wout.iotas, dtype=float))
        ns = int(iotas.shape[0])
        xm = _as_1d(np.asarray(wout.xm_nyq, dtype=float))
        xn = _as_1d(np.asarray(wout.xn_nyq, dtype=float))
        mn = int(xm.shape[0])
        s_half = _half_grid(ns, iotas.dtype)

        def half(values):
            return _interp_half_grid(values[1:], surfaces, s_half)

        iota = half(iotas)
        G = half(_as_1d(np.asarray(wout.bvco, dtype=float)))
        I = half(_as_1d(np.asarray(wout.buco, dtype=float)))  # noqa: E741 - Boozer I

        tables = {}
        for name in ("gmnc", "bmnc", "bsubumnc", "bsubvmnc", "bsupumnc", "bsupvmnc"):
            tables[name] = half(_mode_matrix(wout, name, ns=ns, mn=mn))
        for name in ("gmns", "bmns", "bsubumns", "bsubvmns", "bsupumns", "bsupvmns"):
            optional = not bool(getattr(wout, "lasym", False))
            tables[name] = half(_mode_matrix(wout, name, ns=ns, mn=mn, optional=True)
                                if not optional else jnp.zeros((ns, mn)))

        theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, self.ntheta, endpoint=False)
        phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, self.nphi, endpoint=False)
        dtheta, dphi = theta1d[1] - theta1d[0], phi1d[1] - phi1d[0]
        angle = (theta1d[:, None, None] * xm[None, None, :]
                 - phi1d[None, :, None] * xn[None, None, :])
        cosangle, sinangle = jnp.cos(angle), jnp.sin(angle)

        def synth(cos_tab, sin_tab, cos_w=cosangle, sin_w=sinangle):
            return (jnp.einsum("sm,tpm->stp", cos_tab, cos_w)
                    + jnp.einsum("sm,tpm->stp", sin_tab, sin_w))

        modB = synth(tables["bmnc"], tables["bmns"])
        dB_dtheta = (jnp.einsum("sm,tpm,m->stp", tables["bmnc"], -sinangle, xm)
                     + jnp.einsum("sm,tpm,m->stp", tables["bmns"], cosangle, xm))
        dB_dphi = (jnp.einsum("sm,tpm,m->stp", tables["bmnc"], sinangle, xn)
                   + jnp.einsum("sm,tpm,m->stp", tables["bmns"], -cosangle, xn))
        sqrtg = synth(tables["gmnc"], tables["gmns"])
        bsubu = synth(tables["bsubumnc"], tables["bsubumns"])
        bsubv = synth(tables["bsubvmnc"], tables["bsubvmns"])
        bsupu = synth(tables["bsupumnc"], tables["bsupumns"])
        bsupv = synth(tables["bsupvmnc"], tables["bsupvmns"])

        d_psi_d_s = -_as_1d(np.asarray(wout.phi, dtype=float))[-1] / (2.0 * jnp.pi)
        sqrtg_safe = jnp.where(sqrtg != 0.0, sqrtg, jnp.ones_like(sqrtg))
        B_dot_grad_B = bsupu * dB_dtheta + bsupv * dB_dphi
        B_cross_grad_B_dot_grad_psi = (
            d_psi_d_s * (bsubu * dB_dphi - bsubv * dB_dtheta) / sqrtg_safe)

        tiny = jnp.asarray(jnp.finfo(sqrtg.dtype).tiny, dtype=sqrtg.dtype)
        sqrtg_abs = jnp.maximum(jnp.abs(sqrtg), tiny)
        modB_safe = jnp.maximum(jnp.abs(modB), tiny)
        V_prime = nfp * dtheta * dphi * jnp.sum(sqrtg_abs, axis=(1, 2))

        nn = self.helicity_n * nfp
        prefactor = jnp.sqrt(
            weights[:, None, None] * nfp * dtheta * dphi / V_prime[:, None, None] * sqrtg_abs)
        residuals3d = prefactor * (
            B_cross_grad_B_dot_grad_psi * (nn - iota[:, None, None] * self.helicity_m)
            - B_dot_grad_B * (self.helicity_m * G[:, None, None] + nn * I[:, None, None])
        ) / (modB_safe ** 3)

        residuals1d = jnp.ravel(residuals3d)
        return {
            "surfaces": surfaces,
            "residuals1d": residuals1d,
            "residuals3d": residuals3d,
            "profile": jnp.sum(residuals3d * residuals3d, axis=(1, 2)),
            "total": jnp.sum(residuals1d * residuals1d),
            "modB": modB,
            "iota": iota,
            "G": G,
            "I": I,
            "V_prime": V_prime,
        }

    def residuals(self, wout) -> jnp.ndarray:
        """Flat least-squares residual vector (target 0, weight applied by the driver)."""
        return self.compute(wout)["residuals1d"]

    def profile(self, wout) -> jnp.ndarray:
        """Per-surface sum of squared residuals."""
        return self.compute(wout)["profile"]

    def total(self, wout) -> Array:
        """Scalar QS ratio objective ``sum(residuals**2)``."""
        return self.compute(wout)["total"]

    def J(self, eq: Equilibrium) -> jnp.ndarray:
        """Objective-term entry point for :func:`least_squares` (residual vector)."""
        return self.residuals(eq)

    __call__ = J  # the instance itself can be an objective term

    # -- traceable (state, runtime) evaluation --------------------------------

    def _pointwise_state(self, state: SpectralState, rt: SolverRuntime):
        """Weighted pointwise QS residual on the solver's internal grid.

        Traceable core of :meth:`residuals_state` / :meth:`profile_state` /
        :meth:`total_state`.  The reduced symmetric ``[0, pi]`` theta grid is
        mirrored to the full circle with the stellarator-symmetry map
        ``X(2 pi - theta, -zeta) = X(theta, zeta)`` and the ``|B|`` angular
        derivatives come from FFT spectral differentiation on that full
        periodic grid (exact at grid resolution).  Returns ``(r3d, s_half)``
        with ``r3d`` shaped ``(ns - 1, ntheta1, nzeta)`` normalized so that
        ``sum_angles r3d[i]**2`` is the surface-averaged QS ratio ``<f^2>``
        of half-mesh surface ``i`` — the same quantity as the wout-table
        :meth:`profile`, agreeing at discretization level (solver angular
        grid vs the 63x64 wout sampling), not bitwise.  Symmetric
        configurations only (``lasym = False``).
        """
        setup = rt.setup
        if bool(setup.lasym):
            raise NotImplementedError(
                "QuasisymmetryRatioResidual traceable evaluation supports "
                "lasym = False only")
        s = jnp.asarray(setup.s_full)
        nfp = int(rt.resolution.nfp)
        _, jacobian, _, fields, _ = _field_chain(state, rt)

        # Mirror the reduced [0, pi] grid to the full theta circle.
        ntheta2 = int(np.shape(fields.total_pressure)[1])
        nzeta = int(np.shape(fields.total_pressure)[2])
        ntheta1 = max(2 * (ntheta2 - 1), 1)
        i_full = np.arange(ntheta1)
        i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
        k = np.arange(nzeta)
        k_src = np.where(i_full[:, None] < ntheta2, k[None, :],
                         (nzeta - k[None, :]) % nzeta)
        i_src = np.broadcast_to(i_src[:, None], (ntheta1, nzeta))

        def full(a):
            # Drop the zeroed axis row (js = 0) *before* the singular
            # divisions below: keeping it poisons reverse-mode AD with
            # 0 * inf = nan even though the row never enters the result.
            return jnp.asarray(a)[1:, i_src, k_src]

        # |B| on the half-mesh internal grid (bcovar.f: bsq = |B|^2/2 + p).
        bsq2 = 2.0 * (jnp.asarray(fields.total_pressure)
                      - jnp.asarray(fields.pressure)[:, None, None])
        tiny = jnp.asarray(jnp.finfo(bsq2.dtype).tiny, dtype=bsq2.dtype)
        bmag = jnp.sqrt(jnp.maximum(full(bsq2), tiny))

        # FFT spectral differentiation on the full periodic (theta, zeta) grid;
        # zeta spans one field period, so d/dphi carries the nfp factor.
        kt = jnp.asarray(np.fft.fftfreq(ntheta1) * ntheta1)
        kz = jnp.asarray(np.fft.fftfreq(nzeta) * nzeta * nfp)
        bhat = jnp.fft.fft2(bmag, axes=(1, 2))
        dB_dtheta = jnp.real(jnp.fft.ifft2(1j * kt[None, :, None] * bhat, axes=(1, 2)))
        dB_dphi = jnp.real(jnp.fft.ifft2(1j * kz[None, None, :] * bhat, axes=(1, 2)))

        # Profiles: iota (add_fluxes.f), Boozer covariant averages G/I (fbal.f).
        iota = _iotas_half_from_fields(setup, fields)
        cur = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv,
                               trig=rt.trig, s=s, signgs=setup.signgs)
        G, I = jnp.asarray(cur.bvco), jnp.asarray(cur.buco)  # noqa: E741

        # d(psi)/ds = -phi_edge / (2 pi), wout sign convention.
        hs = s[1] - s[0]
        d_psi_d_s = -float(setup.signgs) * hs * jnp.sum(jnp.asarray(setup.phipf)[1:])

        gsqrt = full(jacobian.sqrt_g)
        gsqrt_safe = jnp.where(gsqrt != 0.0, gsqrt, jnp.ones_like(gsqrt))
        B_dot_grad_B = full(fields.bsupu) * dB_dtheta + full(fields.bsupv) * dB_dphi
        B_cross_grad_B_dot_grad_psi = (
            d_psi_d_s * (full(fields.bsubu) * dB_dphi - full(fields.bsubv) * dB_dtheta)
            / gsqrt_safe)
        nn = self.helicity_n * nfp
        iota_h, G_h, I_h = iota[1:], G[1:], I[1:]      # match the sliced grid
        f = (B_cross_grad_B_dot_grad_psi * (nn - iota_h[:, None, None] * self.helicity_m)
             - B_dot_grad_B * (self.helicity_m * G_h + nn * I_h)[:, None, None]) / bmag ** 3

        # Flux-surface measure weights: sum_angles r3d^2 = <f^2> per surface.
        g_abs = jnp.abs(gsqrt)
        den = jnp.maximum(jnp.sum(g_abs, axis=(1, 2), keepdims=True), tiny)
        r3d = f * jnp.sqrt(g_abs / den)                # half-mesh js = 1..ns-1
        return r3d, 0.5 * (s[:-1] + s[1:])

    def _surface_coefficients(self, s_half: jnp.ndarray) -> jnp.ndarray:
        """Nonnegative half-mesh weights ``c`` with ``sum(c * <f^2>) = total``.

        The wout-table convention interpolates per-surface totals onto the
        requested ``surfaces`` and applies ``weights``; because linear
        interpolation is linear in the profile, that is exactly a fixed
        nonnegative combination ``c`` of the half-mesh surfaces (obtained
        here as the VJP of the interpolation).
        """
        surfaces = _as_1d(self.surfaces)
        weights = (jnp.ones((int(surfaces.shape[0]),)) if self.weights is None
                   else _as_1d(self.weights))
        probe = jnp.zeros_like(s_half)
        _, vjp = jax.vjp(
            lambda p: jnp.sum(weights * jnp.interp(surfaces, s_half, p)), probe)
        return vjp(jnp.asarray(1.0, dtype=probe.dtype))[0]

    def residuals_state(self, state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
        """Traceable flat residual vector with ``sum(r**2) = total_state``.

        The internal-grid analogue of :meth:`residuals` (wout tables): the
        pointwise weighted residual of :meth:`_pointwise_state` scaled by the
        square roots of the surface coefficients — this is the residual
        vector ``jac="implicit"`` optimizes, giving the least-squares driver
        the full pointwise Gauss-Newton geometry.
        """
        r3d, s_half = self._pointwise_state(state, rt)
        c = self._surface_coefficients(s_half)
        return jnp.ravel(jnp.sqrt(c)[:, None, None] * r3d)

    def profile_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable *weighted* per-surface QS totals at ``surfaces``.

        ``weights * interp(surfaces, <f^2> profile)`` from
        :meth:`_pointwise_state`; ``sum = total_state``.
        """
        r3d, s_half = self._pointwise_state(state, rt)
        profile = jnp.sum(r3d * r3d, axis=(1, 2))
        surfaces = _as_1d(self.surfaces)
        weights = (jnp.ones((int(surfaces.shape[0]),)) if self.weights is None
                   else _as_1d(self.weights))
        return weights * jnp.interp(surfaces, s_half, profile)

    def total_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable scalar QS objective: ``sum(profile_state)`` (see there)."""
        return jnp.sum(self.profile_state(state, rt))


# ===========================================================================
# Practical scalar targets — pure functions of (SpectralState, SolverRuntime)
# ===========================================================================


def _aspect_scalars(state: SpectralState, rt: SolverRuntime):
    """``aspectratio.f`` scalars ``(Aminor_p, Rmajor_p, aspect, volume_p)``.

    Boundary-surface quadrature identical to the wout writer
    (:func:`vmec_jax.core.postprocess.aspect_ratio_scalars`), kept in JAX.
    """
    geometry, _, _, _, _ = _field_chain(state, rt)
    sqrts_edge = jnp.asarray(rt.setup.sqrts)[-1]
    rb = jnp.asarray(geometry.R_even)[-1] + sqrts_edge * jnp.asarray(geometry.R_odd)[-1]
    zub = (jnp.asarray(geometry.dZ_dtheta_even)[-1]
           + sqrts_edge * jnp.asarray(geometry.dZ_dtheta_odd)[-1])
    wint = jnp.asarray(rt.trig.wint)
    t1 = rb * zub * wint
    volume_p = 2.0 * jnp.pi ** 2 * jnp.abs(jnp.sum(rb * t1))
    area = 2.0 * jnp.pi * jnp.abs(jnp.sum(t1))
    area_safe = jnp.where(area != 0.0, area, 1.0)
    aminor = jnp.sqrt(area_safe / jnp.pi)
    rmajor = volume_p / (2.0 * jnp.pi * area_safe)
    return aminor, rmajor, rmajor / aminor, volume_p


def aspect_ratio(state: SpectralState, rt: SolverRuntime) -> Array:
    """VMEC aspect ratio ``Rmajor_p / Aminor_p`` (``aspectratio.f`` convention).

    ``Aminor_p = sqrt(<cross-section area> / pi)``, ``Rmajor_p =
    volume_p / (2 pi <area>)`` from the boundary surface quadrature; equals
    the wout ``aspect`` scalar of the same state.
    """
    return _aspect_scalars(state, rt)[2]


def volume(state: SpectralState, rt: SolverRuntime) -> Array:
    """Plasma volume ``volume_p`` [m^3] (wout convention, boundary quadrature)."""
    return _aspect_scalars(state, rt)[3]


def mean_iota(state: SpectralState, rt: SolverRuntime) -> Array:
    """Mean rotational transform over the half-mesh surfaces (axis excluded).

    Matches the legacy optimization ``mean_iota`` convention
    (``mean(iotas[1:])``, i.e. the mean of the wout ``iotas`` profile).
    """
    iotas = _iotas_half(state, rt)
    return jnp.mean(iotas[1:])


def edge_iota(state: SpectralState, rt: SolverRuntime) -> Array:
    """Rotational transform at the boundary (wout ``iotaf[-1]`` convention:
    linear extrapolation of the half mesh, ``1.5 iotas[-1] - 0.5 iotas[-2]``)."""
    iotas = _iotas_half(state, rt)
    return 1.5 * iotas[-1] - 0.5 * iotas[-2]


def mirror_ratio(state: SpectralState, rt: SolverRuntime, *, s_index: int = -1) -> Array:
    """Mirror ratio ``(Bmax - Bmin) / (Bmax + Bmin)`` on one half-mesh surface.

    ``|B|`` is evaluated on the solver's internal angular grid from the
    half-mesh field state (``|B|^2 = 2 (bsq - p)``, ``bcovar.f``); ``s_index``
    selects the half-mesh surface (default: outermost).  Hard max/min — smooth
    almost everywhere, adequate for finite-difference least squares (the
    legacy ``VMECMirrorRatio`` softmax knobs were an optimizer nicety only).
    """
    _, _, _, fields, _ = _field_chain(state, rt)
    bsq = jnp.asarray(fields.total_pressure) - jnp.asarray(fields.pressure)[:, None, None]
    bmag = jnp.sqrt(jnp.maximum(2.0 * bsq[s_index],
                                jnp.asarray(jnp.finfo(bsq.dtype).tiny, dtype=bsq.dtype)))
    bmax, bmin = jnp.max(bmag), jnp.min(bmag)
    return (bmax - bmin) / (bmax + bmin)


def magnetic_well(state: SpectralState, rt: SolverRuntime) -> Array:
    """VMEC/simsopt magnetic-well proxy ``(V'(0) - V'(1)) / V'(0)``.

    ``V' = dV/ds`` endpoints are linear extrapolations of the half-mesh
    differential volume ``vp`` (``bcovar.f``); positive values mean a
    favorable well (``vacuum_well`` in simsopt).  Ported from legacy
    ``vmec_jax.finite_beta.magnetic_well_from_vp``.
    """
    _, _, _, _, energies = _field_chain(state, rt)
    dvol = jnp.abs(jnp.asarray(energies.vp))[1:]
    v0 = 1.5 * dvol[0] - 0.5 * dvol[1]
    v1 = 1.5 * dvol[-1] - 0.5 * dvol[-2]
    v0_safe = jnp.where(v0 != 0.0, v0, jnp.ones_like(v0))
    return jnp.where(v0 != 0.0, (v0 - v1) / v0_safe, 0.0)


def d_merc(eq) -> jnp.ndarray:
    """Mercier stability criterion profile ``DMerc(s)`` (full mesh).

    Positive interior values indicate Mercier stability.  Evaluated through
    the parity-proven wout engine (:func:`vmec_jax.core.nyquist.mercier_and_jxb`
    via :func:`~vmec_jax.core.wout.wout_from_state`) — host NumPy, so this
    objective is finite-difference-only (not jit/AD transparent; the first
    two surfaces and the edge carry the usual near-axis noise, so practical
    targets should penalize e.g. ``min(DMerc[2:-1], 0)``).  Accepts an
    :class:`Equilibrium` or any wout-like object.
    """
    wout = eq.wout if isinstance(eq, Equilibrium) else eq
    return jnp.asarray(np.asarray(wout.DMerc, dtype=float))


def l_grad_b(eq, *, s_index: int = -1, ntheta: int = 24, nphi: int = 24) -> Array:
    """Magnetic-gradient scale length ``min L_grad_B`` on one half-mesh surface.

    ``L_grad_B = |B| sqrt(2 / (grad B : grad B))`` with ``grad B : grad B``
    the squared Frobenius norm of the Cartesian field-gradient tensor —
    the Kappel/Landreman coil-complexity / compactness proxy, and the
    ``L_grad_B`` diagnostic of the legacy QI scripts
    (``vmec_jax.quasi_isodynamic.objectives.lgradb_from_state``).  Here it is
    evaluated from the wout tables of the converged state: ``B^u``/``B^v``
    from the half-mesh ``bsupumnc/bsupvmnc`` Nyquist spectra, the coordinate
    basis vectors and their derivatives spectrally from ``rmnc/zmns``, and
    radial derivatives from the native half/full-mesh finite differences
    (one-sided at the edge) — so values agree with the legacy diagnostic at
    discretization level, not bitwise.

    Returns the (hard) minimum over a uniform ``(theta, phi)`` grid on the
    selected surface (``s_index`` indexes the ``ns``-long half-mesh arrays;
    default edge).  Larger is better; a practical least-squares term is
    ``max(1/L - 1/threshold, 0)``.  Symmetric configurations only (lasym
    sine partners are ignored).  Accepts an :class:`Equilibrium` or wout-like.
    """
    wout = eq.wout if isinstance(eq, Equilibrium) else eq
    ns = int(wout.ns)
    j = max(1, min(s_index % ns, ns - 1))
    hs = 1.0 / (ns - 1)
    xm = jnp.asarray(np.asarray(wout.xm, dtype=float))
    xn = jnp.asarray(np.asarray(wout.xn, dtype=float))
    xmn = jnp.asarray(np.asarray(wout.xm_nyq, dtype=float))
    xnn = jnp.asarray(np.asarray(wout.xn_nyq, dtype=float))
    rmnc = jnp.asarray(np.asarray(wout.rmnc, dtype=float))
    zmns = jnp.asarray(np.asarray(wout.zmns, dtype=float))
    bsupu_t = jnp.asarray(np.asarray(wout.bsupumnc, dtype=float))
    bsupv_t = jnp.asarray(np.asarray(wout.bsupvmnc, dtype=float))

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / int(wout.nfp), int(nphi), endpoint=False)

    def tables(m, n):
        ang = (theta[:, None, None] * m[None, None, :]
               - phi[None, :, None] * n[None, None, :])
        return jnp.cos(ang), jnp.sin(ang)

    cosang, sinang = tables(xm, xn)

    def series(coeff, parity, second: bool = True):
        """Value + angular derivatives of a cos/sin(m theta - n phi) series."""
        base, alt = (cosang, sinang) if parity == "cos" else (sinang, cosang)
        s1 = -1.0 if parity == "cos" else 1.0
        val = jnp.einsum("m,tpm->tp", coeff, base)
        d_t = s1 * jnp.einsum("m,tpm,m->tp", coeff, alt, xm)
        d_p = -s1 * jnp.einsum("m,tpm,m->tp", coeff, alt, xn)
        if not second:
            return val, d_t, d_p, None, None, None
        d_tt = -jnp.einsum("m,tpm,m->tp", coeff, base, xm * xm)
        d_tp = jnp.einsum("m,tpm,m->tp", coeff, base, xm * xn)
        d_pp = -jnp.einsum("m,tpm,m->tp", coeff, base, xn * xn)
        return val, d_t, d_p, d_tt, d_tp, d_pp

    # Full-mesh R/Z -> half-mesh values + radial derivatives (exact on half mesh).
    R, Ru, Rv, Ruu, Ruv, Rvv = series(0.5 * (rmnc[j - 1] + rmnc[j]), "cos")
    Z, Zu, Zv, Zuu, Zuv, Zvv = series(0.5 * (zmns[j - 1] + zmns[j]), "sin")
    Rs, Rsu, Rsv, _, _, _ = series((rmnc[j] - rmnc[j - 1]) / hs, "cos", second=False)
    Zs, Zsu, Zsv, _, _, _ = series((zmns[j] - zmns[j - 1]) / hs, "sin", second=False)

    cphi, sphi = jnp.cos(phi)[None, :], jnp.sin(phi)[None, :]

    def cart(vR, vP, vZ):
        """Cylindrical (R, phi, Z) components -> Cartesian (x, y, z)."""
        return jnp.stack([vR * cphi - vP * sphi, vR * sphi + vP * cphi, vZ], axis=-1)

    zero = jnp.zeros_like(R)
    e_s, e_u, e_v = cart(Rs, zero, Zs), cart(Ru, zero, Zu), cart(Rv, R, Zv)
    # d(e_u)/du, d(e_u)/dv=d(e_v)/du, d(e_v)/dv, d(e_u)/ds, d(e_v)/ds
    deu_u = cart(Ruu, zero, Zuu)
    deu_v = cart(Ruv, Ru, Zuv)
    dev_v = cart(Rvv - R, 2.0 * Rv, Zvv)
    deu_s = cart(Rsu, zero, Zsu)
    dev_s = cart(Rsv, Rs, Zsv)

    # Half-mesh contravariant field (Nyquist modes) + radial derivative.
    cosn, sinn = tables(xmn, xnn)

    def nyq(coeff):
        return (jnp.einsum("m,tpm->tp", coeff, cosn),
                -jnp.einsum("m,tpm,m->tp", coeff, sinn, xmn),
                jnp.einsum("m,tpm,m->tp", coeff, sinn, xnn))

    bu, bu_t, bu_p = nyq(bsupu_t[j])
    bv, bv_t, bv_p = nyq(bsupv_t[j])
    lo, hi = (j - 1, j + 1) if 1 < j < ns - 1 else ((j, j + 1) if j == 1 else (j - 1, j))
    span = hs * (hi - lo)
    bu_s, _, _ = nyq((bsupu_t[hi] - bsupu_t[lo]) / span)
    bv_s, _, _ = nyq((bsupv_t[hi] - bsupv_t[lo]) / span)

    B = bu[..., None] * e_u + bv[..., None] * e_v
    dB = jnp.stack([
        bu_s[..., None] * e_u + bv_s[..., None] * e_v
        + bu[..., None] * deu_s + bv[..., None] * dev_s,
        bu_t[..., None] * e_u + bv_t[..., None] * e_v
        + bu[..., None] * deu_u + bv[..., None] * deu_v,
        bu_p[..., None] * e_u + bv_p[..., None] * e_v
        + bu[..., None] * deu_v + bv[..., None] * dev_v,
    ], axis=-2)                                        # (t, p, coord, cart)

    basis = jnp.stack([e_s, e_u, e_v], axis=-2)
    g = jnp.einsum("...ic,...jc->...ij", basis, basis)
    ginv = jnp.linalg.inv(g)
    grad_sq = jnp.einsum("...ic,...ij,...jc->...", dB, ginv, dB)
    tiny = jnp.asarray(jnp.finfo(grad_sq.dtype).tiny, dtype=grad_sq.dtype)
    bmag = jnp.sqrt(jnp.maximum(jnp.sum(B * B, axis=-1), tiny))
    return jnp.min(bmag * jnp.sqrt(2.0 / jnp.maximum(grad_sq, tiny)))


# ===========================================================================
# Quasi-isodynamic residual (Goodman-style; distilled legacy port)
# ===========================================================================


def _qi_grid(bmnc_b, xm_b, xn_b, iota_b, *, nfp: int, weights, nphi: int,
             nalpha: int, n_bounce: int, include_bounce_endpoints: bool,
             softness: float, phimin: float):
    """Normalized ``|B|`` along field lines + bounce levels (legacy `_qi_boozer_surface_grid`).

    ``theta = alpha + iota * phi`` samples ``nalpha`` field-line labels over
    one field period; ``bnorm`` rescales ``|B|`` to [0, 1] per surface.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    xm_b = jnp.asarray(xm_b, dtype=jnp.float64)
    xn_b = jnp.asarray(xn_b, dtype=jnp.float64)
    iota_b = jnp.asarray(iota_b, dtype=jnp.float64)
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if nphi < 4 or nalpha < 2 or n_bounce < 2:
        raise ValueError("QI residual requires nphi >= 4, nalpha >= 2, n_bounce >= 2")
    nsurf = int(bmnc_b.shape[0])
    dtype = bmnc_b.dtype
    weights_arr = jnp.ones((nsurf,), dtype=dtype) if weights is None else _as_1d(weights)

    phi0 = jnp.asarray(float(phimin), dtype=dtype)
    phi1 = phi0 + jnp.asarray(2.0 * np.pi / nfp, dtype=dtype)
    phi = jnp.linspace(phi0, phi1, nphi, endpoint=True, dtype=dtype)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, nalpha, endpoint=False, dtype=dtype)
    theta = alpha[None, None, :] + iota_b[:, None, None] * phi[None, :, None]
    angle = (theta[:, :, :, None] * xm_b[None, None, None, :]
             - phi[None, :, None, None] * xn_b[None, None, None, :])
    bmag = jnp.sum(bmnc_b[:, None, None, :] * jnp.cos(angle), axis=-1)

    bmin = jnp.min(bmag, axis=(1, 2), keepdims=True)
    bmax = jnp.max(bmag, axis=(1, 2), keepdims=True)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    bnorm = (bmag - bmin) / jnp.maximum(bmax - bmin, tiny)

    if include_bounce_endpoints:
        levels = jnp.linspace(0.0, 1.0, n_bounce, endpoint=True, dtype=dtype)
    else:
        levels = jnp.linspace(0.0, 1.0, n_bounce + 2, endpoint=True, dtype=dtype)[1:-1]
    eps = jnp.maximum(jnp.asarray(float(softness), dtype=dtype),
                      jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    return weights_arr, phi0, phi1, phi, alpha, bmag, bnorm, levels, eps


def quasi_isodynamic_residual(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    weights: Iterable[float] | None = None,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
    include_bounce_endpoints: bool = False,
    softness: float = 2.0e-2,
    width_weight: float = 1.0,
    branch_width_weight: float = 0.5,
    branch_width_softness: float = 1.0e-2,
    profile_weight: float = 0.1,
    shuffle_profile_weight: float = 1.0,
    shuffle_profile_softness: float = 2.0e-2,
    phimin: float = 0.0,
) -> dict[str, Array]:
    """Smooth Goodman-style quasi-isodynamic residual from Boozer ``|B|`` modes.

    A configuration is quasi-isodynamic when the ``|B|`` contours are
    poloidally closed and the trapped-particle bounce distance between the
    two branches of each magnetic well is independent of the field-line label
    ``alpha`` (omnigenity).  This residual samples the normalized ``|B|``
    along field lines ``theta = alpha + iota*phi`` over one field period and
    penalizes, per surface (weights are the legacy defaults, i.e. exactly the
    terms the minimal-seed QI examples used):

    - **level-set width variance** (``width_weight``): for each bounce level
      ``B*`` the smooth occupancy ``sigmoid((B* - bnorm)/softness)`` gives the
      fraction of the field line below ``B*``; its variance over ``alpha``
      measures misalignment of the ``|B|`` contours.
    - **branch width variance** (``branch_width_weight``): each field line is
      split at its ``|B|`` minimum, both branches are made monotone with a
      running maximum, and the (smooth) level-crossing distances of the two
      branches are summed — the trapped-well bounce width, whose variance
      over ``alpha`` is the classic omnigenity error.
    - **profile consistency** (``profile_weight``): small penalty on the
      variance of ``bnorm`` itself over ``alpha`` at fixed ``phi``, which
      keeps degenerate QH-like candidates from gaming the width terms.
    - **branch-shuffle profile** (``shuffle_profile_weight``): the "squash and
      shuffle" comparison — each well's branch crossings are shifted so every
      field line has the *mean* bounce width, the shuffled well is
      reinterpolated onto the original grid and compared pointwise to the
      original ``bnorm`` (the closest smooth analogue of Goodman et al.'s
      construction of the nearest omnigenous field).

    Ported from the legacy ``vmec_jax.quasi_isodynamic.objectives.
    quasi_isodynamic_residual_from_boozer_modes`` with the unused
    ``aligned_profile_*`` / ``weighted_shuffle_*`` / ``shuffle_profile_nphi_out``
    machinery removed (they defaulted to off in the QI examples).  ``xn_b``
    uses physical toroidal mode numbers (booz_xform convention).  Returns
    ``residuals1d`` (least-squares vector) and ``total`` (its squared norm).
    """
    (weights_arr, phi0, phi1, phi, alpha, bmag, bnorm, levels, eps) = _qi_grid(
        bmnc_b, xm_b, xn_b, iota_b, nfp=int(nfp), weights=weights, nphi=int(nphi),
        nalpha=int(nalpha), n_bounce=int(n_bounce),
        include_bounce_endpoints=bool(include_bounce_endpoints),
        softness=float(softness), phimin=float(phimin))
    dtype = bnorm.dtype
    nsurf, nphi_, nalpha_ = int(bnorm.shape[0]), int(bnorm.shape[1]), int(bnorm.shape[2])
    nlev = int(levels.shape[0])
    sqrt_w = jnp.sqrt(weights_arr)[:, None, None]
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    pieces: list[jnp.ndarray] = []

    # -- level-set occupancy width variance + profile consistency ----------
    occupancy = jax.nn.sigmoid((levels[None, None, None, :] - bnorm[:, :, :, None]) / eps)
    widths = jnp.mean(occupancy, axis=1)                      # (nsurf, nalpha, nlev)
    width_res = (widths - jnp.mean(widths, axis=1, keepdims=True)) * sqrt_w * width_weight
    pieces.append(jnp.ravel(width_res) / jnp.sqrt(jnp.asarray(nalpha_ * nlev, dtype=dtype)))

    profile_res = (bnorm - jnp.mean(bnorm, axis=2, keepdims=True)) * sqrt_w * profile_weight
    pieces.append(jnp.ravel(profile_res) / jnp.sqrt(jnp.asarray(nalpha_ * nphi_, dtype=dtype)))

    # -- branch-based trapped-well width variance --------------------------
    if float(branch_width_weight) != 0.0:
        bper = jnp.swapaxes(bnorm[:, :-1, :], 1, 2)           # periodic, (nsurf, nalpha, nper)
        nper = nphi_ - 1
        offs = jnp.arange(max(1, nper // 2) + 1, dtype=jnp.int32)
        imin = jnp.argmin(bper, axis=-1)
        left = jnp.maximum.accumulate(
            jnp.take_along_axis(bper, jnp.mod(imin[:, :, None] - offs[None, None, :], nper), axis=-1), axis=-1)
        right = jnp.maximum.accumulate(
            jnp.take_along_axis(bper, jnp.mod(imin[:, :, None] + offs[None, None, :], nper), axis=-1), axis=-1)
        left = (left - left[..., :1]) / jnp.maximum(left[..., -1:] - left[..., :1], tiny)
        right = (right - right[..., :1]) / jnp.maximum(right[..., -1:] - right[..., :1], tiny)
        distance = jnp.asarray(offs, dtype=dtype) / jnp.asarray(nper, dtype=dtype)
        beps = jnp.maximum(jnp.asarray(float(branch_width_softness), dtype=dtype),
                           jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))

        def crossing(branch):
            logits = -((branch[:, :, :, None] - levels[None, None, None, :]) / beps) ** 2
            logits = logits - jnp.max(logits, axis=2, keepdims=True)
            w = jnp.exp(logits)
            w = w / jnp.sum(w, axis=2, keepdims=True)
            return jnp.sum(w * distance[None, None, :, None], axis=2)

        bw = crossing(left) + crossing(right)                 # (nsurf, nalpha, nlev)
        bw_res = (bw - jnp.mean(bw, axis=1, keepdims=True)) * sqrt_w * branch_width_weight
        pieces.insert(1, jnp.ravel(bw_res) / jnp.sqrt(jnp.asarray(nalpha_ * nlev, dtype=dtype)))

    # -- branch-shuffle profile comparison ----------------------------------
    if float(shuffle_profile_weight) != 0.0:
        b_alpha = jnp.swapaxes(bnorm, 1, 2)                   # (nsurf, nalpha, nphi)
        offs = jnp.arange(nphi_, dtype=jnp.int32)
        offs_f = jnp.asarray(offs, dtype=dtype)
        dphi = (phi1 - phi0) / jnp.asarray(nphi_ - 1, dtype=dtype)
        period = phi1 - phi0
        imin = jnp.argmin(b_alpha, axis=-1)
        li_raw = imin[:, :, None] - offs[None, None, :]
        ri_raw = imin[:, :, None] + offs[None, None, :]
        lvalid, rvalid = li_raw >= 0, ri_raw < nphi_
        lraw = jnp.take_along_axis(b_alpha, jnp.clip(li_raw, 0, nphi_ - 1), axis=-1)
        rraw = jnp.take_along_axis(b_alpha, jnp.clip(ri_raw, 0, nphi_ - 1), axis=-1)
        one = jnp.asarray(1.0, dtype=dtype)
        left = jnp.maximum.accumulate(jnp.where(lvalid, lraw, one), axis=-1)
        right = jnp.maximum.accumulate(jnp.where(rvalid, rraw, one), axis=-1)

        seps = jnp.maximum(jnp.asarray(float(shuffle_profile_softness), dtype=dtype),
                           jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
        trapz_w = jnp.ones((nphi_,), dtype=dtype).at[0].set(0.5).at[-1].set(0.5)

        def branch_crossing(branch):
            occ = jax.nn.sigmoid((levels[None, None, None, :] - branch[:, :, :, None]) / seps)
            return jnp.sum(occ * trapz_w[None, None, :, None], axis=2) * dphi

        lcross, rcross = branch_crossing(left), branch_crossing(right)
        bw = lcross + rcross
        bw_mean = jnp.mean(bw, axis=1, keepdims=True)

        min_phi = phi0 + jnp.asarray(imin, dtype=dtype) * dphi
        lend = jnp.maximum(min_phi - phi0, 0.0)
        rend = jnp.maximum(phi1 - min_phi, 0.0)
        signed_phi = (offs_f[None, None, :] - jnp.asarray(imin[:, :, None], dtype=dtype)) * dphi

        level_full = jnp.concatenate([jnp.zeros((1,), dtype=dtype), levels,
                                      jnp.ones((1,), dtype=dtype)])
        y_target = jnp.concatenate([jnp.flip(level_full, axis=0), level_full[1:]], axis=0)

        delta = 0.5 * (bw - bw_mean)
        ltarget = jnp.clip(lcross - delta, 0.0, lend[:, :, None])
        rtarget = jnp.clip(rcross - delta, 0.0, rend[:, :, None])
        zeros = jnp.zeros((nsurf, nalpha_, 1), dtype=dtype)
        lfull = jnp.maximum.accumulate(
            jnp.concatenate([zeros, ltarget, lend[:, :, None]], axis=-1), axis=-1)
        rfull = jnp.maximum.accumulate(
            jnp.concatenate([zeros, rtarget, rend[:, :, None]], axis=-1), axis=-1)
        x_target = jnp.concatenate([-jnp.flip(lfull, axis=-1), rfull[:, :, 1:]], axis=-1)
        ramp = (jnp.arange(x_target.shape[-1], dtype=dtype)
                * jnp.asarray(1.0e-14, dtype=dtype) * period)
        x_target = x_target + ramp[None, None, :]

        def interp_one(xp, x):
            return jnp.interp(x, xp, y_target)

        shuffled = jax.vmap(jax.vmap(interp_one, in_axes=(0, 0)), in_axes=(0, 0))(
            x_target, signed_phi)
        shuffle_res = (shuffled - b_alpha) * sqrt_w * shuffle_profile_weight
        pieces.append(jnp.ravel(shuffle_res)
                      / jnp.sqrt(jnp.asarray(nalpha_ * nphi_, dtype=dtype)))

    residuals1d = jnp.concatenate(pieces)
    return {
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "bnorm": bnorm,
        "bmag": bmag,
        "levels": levels,
        "phi": phi,
        "alpha": alpha,
    }


def boozer_modes_from_wout(
    wout,
    *,
    surfaces,
    mboz: int = 18,
    nboz: int = 18,
    jit: bool = False,
) -> dict[str, Any]:
    """Boozer ``|B|`` spectrum of selected surfaces via ``booz_xform_jax``.

    ``wout`` is a :class:`~vmec_jax.core.wout.WoutData` (or any wout-like
    object accepted by ``Booz_xform.read_wout_data``); ``surfaces`` are
    normalized-flux values matched to the nearest half-mesh surfaces.
    Returns ``{bmnc_b, xm_b, xn_b, iota_b, nfp, s_b}`` with ``bmnc_b`` shaped
    ``(nsurf, nmodes)`` — the inputs of :func:`quasi_isodynamic_residual`.

    ``booz_xform_jax`` is an optional dependency (soft import).
    """
    try:
        from booz_xform_jax import Booz_xform
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Boozer-based objectives require booz_xform_jax; "
            "run `pip install booz_xform_jax`.") from exc
    if isinstance(wout, Equilibrium):
        wout = wout.wout
    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    s_in = np.asarray(bx.s_in, dtype=float)
    values = np.atleast_1d(np.asarray(list(np.ravel(surfaces)), dtype=float))
    indices = sorted({int(np.argmin(np.abs(s_in - v))) for v in values})
    bx.compute_surfs = indices
    bx.run(jit=bool(jit))
    bmnc_b = np.asarray(bx.bmnc_b, dtype=float)
    xm_b = np.asarray(bx.xm_b, dtype=float)
    if bmnc_b.shape[0] == xm_b.shape[0]:      # (nmodes, nsurf) -> (nsurf, nmodes)
        bmnc_b = bmnc_b.T
    return {
        "bmnc_b": bmnc_b,
        "xm_b": xm_b,
        "xn_b": np.asarray(bx.xn_b, dtype=float),
        "iota_b": np.asarray(bx.iota, dtype=float)[indices],
        "nfp": int(bx.nfp),
        "s_b": s_in[indices],
    }


def quasi_isodynamic_residual_from_wout(
    wout,
    *,
    surfaces,
    mboz: int = 18,
    nboz: int = 18,
    jit_booz: bool = False,
    **qi_kwargs,
) -> dict[str, Array]:
    """QI residual of a converged equilibrium: wout -> Boozer -> residual.

    Convenience composition of :func:`boozer_modes_from_wout` and
    :func:`quasi_isodynamic_residual`; ``qi_kwargs`` are the residual's
    sampling/weight knobs.  Accepts a :class:`Equilibrium` too, so it can be
    used directly as a :func:`least_squares` objective term via
    ``lambda eq: quasi_isodynamic_residual_from_wout(eq, surfaces=...)["residuals1d"]``.
    """
    booz = boozer_modes_from_wout(wout, surfaces=surfaces, mboz=mboz, nboz=nboz,
                                  jit=jit_booz)
    return quasi_isodynamic_residual(
        bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], nfp=booz["nfp"], **qi_kwargs)


# ===========================================================================
# Boundary degrees of freedom + scipy least-squares driver
# ===========================================================================


def _dof_modes(inp: VmecInput, max_mode: int) -> list[tuple[int, int]]:
    """Canonical (m, n) list for the boundary dofs at ``max_mode``.

    ``m = 0`` keeps only ``n >= 1`` (negative-``n`` m=0 cosine modes are
    redundant, the m=0 sine modes are their sign flips, and ``RBC(0, 0)`` —
    the major radius — is held fixed to remove the trivial scale direction,
    exactly like the simsopt QS examples fix the major radius).
    """
    m_max = min(int(max_mode), int(inp.mpol) - 1)
    n_max = min(int(max_mode), int(inp.ntor))
    out: list[tuple[int, int]] = []
    for m in range(0, m_max + 1):
        for n in range(-n_max, n_max + 1):
            if m == 0 and n <= 0:
                continue
            out.append((m, n))
    return out


def boundary_dof_names(inp: VmecInput, max_mode: int) -> list[str]:
    """Human-readable labels ("RBC(n,m)" / "ZBS(n,m)", INDATA index order)."""
    modes = _dof_modes(inp, max_mode)
    return ([f"RBC({n},{m})" for (m, n) in modes]
            + [f"ZBS({n},{m})" for (m, n) in modes])


def pack_boundary(inp: VmecInput, max_mode: int) -> np.ndarray:
    """Flat boundary-dof vector ``[rbc..., zbs...]`` (see :func:`_dof_modes`).

    Inverse of :func:`unpack_boundary`; ``RBC(0,0)`` is excluded (fixed major
    radius).  Only the stellarator-symmetric ``rbc``/``zbs`` families are
    packed — lasym boundary optimization is out of scope here.
    """
    modes = _dof_modes(inp, max_mode)
    ntor = int(inp.ntor)
    rbc = np.asarray(inp.rbc, dtype=float)
    zbs = np.asarray(inp.zbs, dtype=float)
    return np.asarray([rbc[n + ntor, m] for (m, n) in modes]
                      + [zbs[n + ntor, m] for (m, n) in modes], dtype=float)


def unpack_boundary(inp: VmecInput, x, max_mode: int) -> VmecInput:
    """New :class:`VmecInput` with the boundary dofs ``x`` applied."""
    modes = _dof_modes(inp, max_mode)
    x = np.asarray(x, dtype=float).ravel()
    if x.size != 2 * len(modes):
        raise ValueError(f"expected {2 * len(modes)} dofs, got {x.size}")
    ntor = int(inp.ntor)
    rbc = np.array(inp.rbc, dtype=float, copy=True)
    zbs = np.array(inp.zbs, dtype=float, copy=True)
    for k, (m, n) in enumerate(modes):
        rbc[n + ntor, m] = x[k]
        zbs[n + ntor, m] = x[len(modes) + k]
    return dataclasses.replace(inp, rbc=rbc, zbs=zbs)


#: curtor dof storage scale (dof = CURTOR/1e6, i.e. MA) — keeps the trust
#: region O(1) alongside the boundary dofs (spec notes_r26g section 6.4).
_CURTOR_SCALE = 1.0e6


def _current_dof_setup(inp: VmecInput, current_dofs: int | None) -> tuple[int, float]:
    """Validate the optional AC/CURTOR dof block of :func:`least_squares`.

    Returns ``(k, ac_scale)``: ``k`` leading ``AC`` power-series coefficients
    are freed (0 disables the block); the dof vector then gains ``k + 1``
    trailing entries ``[ac_0/ac_scale, ..., ac_{k-1}/ac_scale,
    curtor/1e6]``.  ``ac_scale = max|AC|`` frozen from the seed input (VMEC
    normalizes the AC profile by its own edge integral, so the coefficient
    magnitude — ampere-scale for the Zenodo/self_consistent_bootstrap decks,
    O(1) for shape-normalized decks — is the right trust-region unit; the
    spec's ``|curtor|`` is the fallback when the seed AC block is all zero).
    """
    if not current_dofs:
        return 0, 1.0
    k = int(current_dofs)
    if k <= 0:
        raise ValueError(f"current_dofs must be a positive int, got {current_dofs!r}")
    if int(inp.ncurr) != 1:
        raise ValueError("current_dofs requires ncurr = 1 (prescribed current)")
    kind = str(inp.pcurr_type).strip().lower()
    if "spline" in kind or "line_segment" in kind:
        raise ValueError(
            "current_dofs requires an AC-coefficient pcurr_type (e.g. "
            f"'power_series'), got {inp.pcurr_type!r}; re-parameterize the "
            "deck (e.g. with vmec_jax.core.bootstrap.self_consistent_bootstrap, "
            "whose refit emits a power_series AC) first")
    if k > int(np.asarray(inp.ac).size):
        raise ValueError(f"current_dofs = {k} exceeds the dense AC length "
                         f"{int(np.asarray(inp.ac).size)}")
    ac_scale = float(np.max(np.abs(np.asarray(inp.ac, dtype=float))))
    if ac_scale == 0.0:
        ac_scale = max(abs(float(inp.curtor)), 1.0)
    return k, ac_scale


def _pack_current(inp: VmecInput, k: int, ac_scale: float) -> np.ndarray:
    """Scaled ``[ac_0..ac_{k-1}, curtor]`` dof block (see :func:`_current_dof_setup`)."""
    return np.concatenate([np.asarray(inp.ac, dtype=float)[:k] / ac_scale,
                           [float(inp.curtor) / _CURTOR_SCALE]])


def _apply_current(inp: VmecInput, xc, k: int, ac_scale: float) -> VmecInput:
    """New :class:`VmecInput` with the scaled current dof block ``xc`` applied."""
    xc = np.asarray(xc, dtype=float).ravel()
    if xc.size != k + 1:
        raise ValueError(f"expected {k + 1} current dofs, got {xc.size}")
    ac = np.array(inp.ac, dtype=float, copy=True)
    ac[:k] = xc[:k] * ac_scale
    return dataclasses.replace(inp, ac=ac, curtor=float(xc[k]) * _CURTOR_SCALE)


def _call_term(fun: Callable, eq: Equilibrium) -> np.ndarray:
    """Evaluate an objective callable against an :class:`Equilibrium`.

    Callables with two or more positional parameters are treated as pure
    ``(state, runtime)`` functions (the scalar targets above); single-argument
    callables receive the :class:`Equilibrium` (e.g.
    ``QuasisymmetryRatioResidual.J`` or user lambdas).
    """
    try:
        params = [p for p in inspect.signature(fun).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        two_positional = len(params) >= 2 and params[1].default is inspect.Parameter.empty
    except (TypeError, ValueError):  # builtins / partials without signature
        two_positional = False
    value = fun(eq.state, eq.runtime) if two_positional else fun(eq)
    return np.atleast_1d(np.asarray(jax.device_get(value), dtype=float)).ravel()


def _ess_scale(inp: VmecInput, max_mode: int, alpha: float) -> np.ndarray:
    """Exponential Spectral Scaling (ESS) trust-region weights per dof.

    ``x_scale[i] = exp(-alpha * max(|m_i|, |n_i|)) / exp(-alpha)`` — higher
    (m, n) boundary harmonics get proportionally smaller trust-region steps,
    which stabilizes staged ``max_mode`` continuation from crude seeds.
    Ported from legacy ``optimizers.fixed_boundary.parameterization.
    create_x_scale`` (the ``use_ess``/``ess_alpha`` option of the legacy
    ``least_squares_solve``); passed to scipy as ``x_scale``.
    """
    modes = _dof_modes(inp, max_mode)
    levels = np.asarray([max(abs(m), abs(n)) for (m, n) in modes] * 2, dtype=float)
    if alpha <= 0.0:
        return np.ones_like(levels)
    return np.exp(-alpha * levels) / np.exp(-alpha)


def least_squares(
    objective_terms: Sequence[tuple[Callable, float, float]],
    inp: VmecInput,
    *,
    max_mode: int | Sequence[int] = 1,
    x0: np.ndarray | None = None,
    current_dofs: int | None = None,
    jac: str | None = None,
    jac_chunk_size: int | str | None = "auto",
    jac_solver: str = "block",
    recycle: bool = False,
    hot_restart: bool = True,
    warm_start: str | None = "perturbation",
    use_ess: bool = False,
    ess_alpha: float = 1.2,
    device: Any = None,
    solve_kwargs: dict | None = None,
    verbose: int = 0,
    **scipy_kwargs,
):
    """Boundary-shape least squares: simsopt's ``least_squares_serial_solve``.

    ``objective_terms`` is a list of ``(fun, target, weight)``: each ``fun``
    maps a converged :class:`Equilibrium` (or, for two-positional-argument
    callables, its ``(state, runtime)`` pair) to a scalar or residual vector,
    and contributes ``weight * (fun(eq) - target)`` rows to the stacked
    residual, i.e. ``cost = 1/2 sum_i w_i^2 (f_i - t_i)^2`` (scipy's 1/2
    convention).  The decision variables are the boundary Fourier
    coefficients up to ``max_mode`` (:func:`pack_boundary` — ``RBC(0,0)``
    fixed).  Trial boundaries whose solve fails return a large finite
    residual so the trust region backs off instead of crashing.  Objective
    swaps between stages of a staged campaign (e.g. the QP-basin-then-QI
    route) are just two calls with different ``objective_terms``, the second
    seeded with the first call's ``result.input``.

    ``current_dofs = k`` (plan R26.g, spec section 6.4) additionally frees
    the equilibrium current profile: the first ``k`` ``AC`` power-series
    coefficients (VMEC ``pcurr_type="power_series"``, i.e. ``I'(s)``) plus
    ``CURTOR``, appended to the dof vector as
    ``[..boundary.., ac_0/ac_scale, ..., ac_{k-1}/ac_scale, curtor/1e6]``
    (``ac_scale = max(|curtor|, 1)`` frozen from the seed) so the trust
    region sees O(1) numbers.  Requires ``ncurr = 1`` and an
    AC-parameterized ``pcurr_type`` (splines are rejected — re-fit them to a
    power series first, e.g. via
    :func:`vmec_jax.core.bootstrap.self_consistent_bootstrap`).  Both
    gradient modes support it (finite differences re-solve per current dof;
    ``jac="implicit"`` adds ``k + 1`` one-hot tangent rows through
    ``ImplicitParams.ac``/``curtor``, which ``runtime_from_params`` already
    traces).  Note that VMEC normalizes the AC profile by its own edge
    integral (only the *shape* of ``I'`` matters; ``CURTOR`` sets the
    amplitude), so the overall-AC-scale direction is objective-neutral; the
    trust-region solvers handle the resulting Jacobian null direction.
    This is the dof set of the self-consistent-bootstrap objective
    (:class:`vmec_jax.core.bootstrap.RedlBootstrapMismatch`).

    ``max_mode`` may be a single int or an increasing schedule (e.g.
    ``(1, 2, 3)``): each continuation stage optimizes the enlarged dof set
    starting from the previous stage's boundary (higher harmonics enter at
    their current — typically zero — values).  Repeated trial solves are
    cheap by construction: runtimes with the same
    :class:`~vmec_jax.core.fourier.Resolution` are structural pytrees, so the
    solver reuses one XLA executable across all boundary trials
    (``vmec_jax.core.solver`` plan.md Phase-2 cache; only the first solve of
    a stage compiles).  ``device`` is forwarded to the solver
    (:mod:`vmec_jax.core.device` policy applies when ``None``).

    ``use_ess`` enables Exponential Spectral Scaling of the trust region
    (:func:`_ess_scale`, ``ess_alpha``), the legacy ``use_ess`` option.

    ``jac=None`` (default) uses scipy ``"2-point"`` finite differences.
    ``jac="implicit"`` uses the Phase-6 implicit-gradient path
    (:mod:`vmec_jax.core.implicit`): one hot-restarted forward solve per
    trial boundary, and the exact residual Jacobian by forward implicit
    differentiation — one preconditioned GMRES per boundary dof instead of
    one full equilibrium solve per dof.  Requirements (see the module
    docstring): every term traceable in ``(state, runtime)`` (vector terms
    expose ``residuals_state``; wout-engine terms like :func:`d_merc` /
    :func:`l_grad_b` / the Boozer QI residual need ``jac=None``) and
    ``lasym = False`` (the implicit parameter map does not implement the
    lasym ``readin.f`` boundary rotation).
    ``jac_chunk_size`` (R17.1 memory knob, ``jac="implicit"`` only) chunks the
    per-dof Jacobian columns via :func:`solvax.chunk_map`: ``"auto"`` (default)
    lets :func:`solvax.auto_chunk_size` pick a memory-bounded width (the
    largest block that fits the device budget on GPU, a sqrt-balanced width on
    CPU) so peak Jacobian memory is ``m0 + m1*chunk`` instead of scaling with
    the full dof count; an ``int`` fixes that many boundary dofs at a time; and
    ``None`` forces one wide ``vmap`` over all dofs (the pre-R17.1 behavior,
    fastest but peak memory O(dofs)).  The column blocks are mathematically
    independent, so the assembled Jacobian is identical to float64 round-off
    (~1e-15) across chunk sizes.  It is inert for ``jac=None`` (scipy computes
    the finite-difference Jacobian itself).

    ``jac_solver`` (plan R25.2, ``jac="implicit"`` only) selects the linear
    solver behind the per-dof implicit-Jacobian columns.  ``"block"``
    (default) amortizes one block-tridiagonal factorization of the *raw*
    force Jacobian — whose radial coupling is exactly nearest-neighbor, so
    ns dense ``(3*mn, 3*mn)`` blocks assembled by 3-colored ``jax.jvp``
    probes capture it completely at a cost independent of the dof count —
    then backsolves every dof right-hand side directly
    (:func:`solvax.block_thomas_factor` / :func:`solvax.block_thomas_solve`)
    and certifies each column with a short warm-started GMRES pass against
    the preconditioned system (same ``adjoint_tol`` norm as the default
    path; columns already at tolerance cost one matvec).  ``"gmres"`` is the
    pre-R25.2 path: one independent preconditioned GMRES per dof column.
    Both produce the same Jacobian to solver tolerance; ``"gmres"`` is the
    fallback if the block path misbehaves on an exotic configuration.
    Inert for ``jac=None``; ``recycle=True`` takes precedence.

    ``recycle`` (plan R25.3, ``jac="implicit"`` only) carries a GCROT
    deflation pair across the per-dof implicit-Jacobian solves — a
    ``lax.scan`` over dof chunks (vmapped within a chunk) threads the
    :func:`solvax.gcrot` recycle space ``(C, U)`` between chunks and, via a
    Python-side holder, between successive trust-region Jacobian
    evaluations.  Recycled solves keep the exact ``adjoint_tol`` /
    ``adjoint_maxiter`` budget of the default path.  **Default False**:
    measured on the nfp2 minimal-seed max_mode-2 operator (2026-07-11), the
    solvax v0.1 recycle space (FIFO cycle corrections, not the harmonic
    Ritz vectors of GCRO-DR) *slows* warm-started columns — e.g. 140 (cold
    GMRES) -> 236/347/479 iterations at k = 2/5/10 — so columns that then
    exhaust ``adjoint_maxiter`` return larger residuals.  Enable only after
    benchmarking per-column iteration counts on your operator.
    ``recycle=False`` uses the independent per-column :func:`solvax.gmres`
    path (identical columns across chunk sizes to float64 round-off).
    Inert for ``jac=None``.

    ``hot_restart`` seeds each trial solve from the previous converged state
    (both modes; in implicit mode via the per-config host-solve cache).

    ``warm_start`` (plan R25.4, ``jac="implicit"`` only) refines what that
    seed is.  ``"perturbation"`` (default) seeds each *trial* solve with the
    DESC-style first-order prediction ``x_ref + sum_j (dx)_j dz_j``
    (arXiv:2203.15927 ``eq.perturb`` before ``eq.solve``): the per-dof state
    responses ``dz_j = -(dF/dz)^{-1} dF/dp t_j`` are exactly the columns the
    implicit Jacobian already solves, so the linearization is stashed at
    each ``jac(x_ref)`` call for free and evaluated per trial at the cost of
    one small tensor contraction.  ``"state"`` is the plain hot restart
    (seed = last converged state); ``None`` disables warm starting entirely
    (every trial re-solves from the interior guess).  All three converge to
    the same fixed points — only the inner iteration count changes — and
    anything missing or mismatched (first call, stage change, failed seed)
    falls back silently down the ladder perturbation -> state -> cold.
    ``recycle=True`` bypasses the perturbation stash (its Jacobian variant
    carries the GCROT pair instead); ``hot_restart=False`` forces
    ``warm_start=None``.  Inert for ``jac=None``.  Measured (2026-07-12,
    M-series CPU, nfp2 minimal seed, max_mode=2, QS+aspect, ``max_nfev=20``):
    total forward-solve iterations 23685 (``"state"``) -> 6364
    (``"perturbation"``), **3.7x fewer**, wall 156 s -> 145 s (this deck's
    wall is Jacobian-dominated at ns=35; the solve-phase win grows with
    ``ns``), and the seed additionally rescued a trial whose plain hot
    restart failed with a Jacobian-sign error.

    Remaining keywords go to :func:`scipy.optimize.least_squares` (e.g.
    ``max_nfev``, ``ftol``, ``xtol``, ``diff_step``).

    Returns the scipy ``OptimizeResult`` of the final stage with extra
    attributes: ``input`` (optimized :class:`VmecInput`), ``equilibrium``
    (last successfully solved :class:`Equilibrium`), ``stage_results``
    (per-``max_mode`` results for schedules) and, in implicit mode,
    ``solve_stats`` (``{"solves", "iterations"}`` totals of the stage's host
    forward solves — the R25.4 warm-start instrumentation).
    """
    import scipy.optimize

    modes_schedule = ([int(max_mode)] if np.isscalar(max_mode)
                      else [int(m) for m in max_mode])
    if len(modes_schedule) > 1:
        if x0 is not None:
            raise ValueError("x0 cannot be combined with a max_mode schedule")
        stage_results = []
        current = inp
        result = None
        for mm in modes_schedule:
            result = least_squares(
                objective_terms, current, max_mode=mm,
                current_dofs=current_dofs, jac=jac,
                jac_chunk_size=jac_chunk_size, jac_solver=jac_solver,
                recycle=recycle, hot_restart=hot_restart,
                warm_start=warm_start, use_ess=use_ess,
                ess_alpha=ess_alpha, device=device, solve_kwargs=solve_kwargs,
                verbose=verbose, **scipy_kwargs)
            stage_results.append(result)
            current = result.input
        result.stage_results = stage_results
        return result
    max_mode = modes_schedule[0]
    k_cur, ac_scale = _current_dof_setup(inp, current_dofs)

    def _ess_scale_full() -> np.ndarray:
        scale = _ess_scale(inp, max_mode, float(ess_alpha))
        if k_cur:  # current dofs are already O(1) by construction
            scale = np.concatenate([scale, np.ones(k_cur + 1)])
        return scale

    if jac == "implicit":
        if use_ess:
            scipy_kwargs.setdefault("x_scale", _ess_scale_full())
        return _least_squares_implicit(
            objective_terms, inp, max_mode=max_mode, x0=x0,
            current_dofs=current_dofs,
            jac_chunk_size=jac_chunk_size, jac_solver=jac_solver,
            recycle=recycle,
            warm_start=(warm_start if hot_restart else None),
            solve_kwargs=dict(solve_kwargs or {}),
            device=device, verbose=verbose, **scipy_kwargs)
    if jac is not None:
        raise ValueError(f"jac must be None or 'implicit', got {jac!r}")

    solve_kwargs = dict(solve_kwargs or {})
    if device is not None:
        solve_kwargs.setdefault("device", device)
    if use_ess:
        scipy_kwargs.setdefault("x_scale", _ess_scale_full())
    if x0 is None:
        x0 = pack_boundary(inp, max_mode)
        if k_cur:
            x0 = np.concatenate([x0, _pack_current(inp, k_cur, ac_scale)])
    nb = 2 * len(_dof_modes(inp, max_mode))

    def unpack_full(x: np.ndarray) -> VmecInput:
        trial = unpack_boundary(inp, np.asarray(x, dtype=float)[:nb], max_mode)
        if k_cur:
            trial = _apply_current(trial, np.asarray(x, dtype=float)[nb:],
                                   k_cur, ac_scale)
        return trial

    state_holder: dict[str, Any] = {"hot": None, "eq": None, "nres": None}
    single_stage = int(np.asarray(inp.ns_array).size) == 1

    def fun(x: np.ndarray) -> np.ndarray:
        trial = unpack_full(x)
        try:
            seed = state_holder["hot"] if (hot_restart and single_stage) else None
            eq = solve_equilibrium(trial, initial_state=seed, **solve_kwargs)
            parts = [w * (_call_term(f, eq) - t) for (f, t, w) in objective_terms]
            residual = np.concatenate(parts)
        except Exception as exc:  # zero-crash policy: penalize, don't die
            if state_holder["nres"] is None:
                raise  # the very first evaluation must succeed (sizes scipy's residual)
            if verbose:
                print(f"[least_squares] trial solve failed: {exc}")
            return np.full((state_holder["nres"],), 1.0e6)
        if not np.all(np.isfinite(residual)):
            residual = np.where(np.isfinite(residual), residual, 1.0e6)
        state_holder["hot"] = eq.state
        state_holder["eq"] = eq
        state_holder["nres"] = residual.size
        if verbose:
            print(f"[least_squares] cost = {0.5 * float(residual @ residual):.6e}")
        return residual

    result = scipy.optimize.least_squares(fun, np.asarray(x0, dtype=float),
                                          jac="2-point", **scipy_kwargs)
    result.input = unpack_full(result.x)
    result.equilibrium = state_holder["eq"]
    return result


# ---------------------------------------------------------------------------
# Implicit-gradient mode (vmec_jax.core.implicit wiring)
# ---------------------------------------------------------------------------


def _traceable_term(fun: Callable) -> Callable:
    """Objective callable -> traceable ``(state, runtime)`` function.

    Terms exposing ``residuals_state`` (:class:`QuasisymmetryRatioResidual`
    instances or their bound ``J``/``residuals`` methods) contribute their
    full traceable pointwise residual vector — same least-squares cost as
    the finite-difference stacked residuals (internal-grid sampling instead
    of the 63x64 wout grid), same Gauss-Newton geometry.
    Two-positional-argument callables (the scalar targets) are used as-is.
    Anything else (wout-table objectives — host NumPy) is rejected with a
    pointer to ``jac=None``.
    """
    owner = getattr(fun, "__self__", fun)
    if hasattr(owner, "residuals_state"):
        return owner.residuals_state
    try:
        params = [p for p in inspect.signature(fun).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        two_positional = len(params) >= 2 and params[1].default is inspect.Parameter.empty
    except (TypeError, ValueError):
        two_positional = False
    if two_positional:
        return fun
    raise ValueError(
        f"objective term {fun!r} is not implicit-differentiable: jac='implicit' "
        "needs traceable (state, runtime) callables or a residuals_state method. "
        "Wout-engine terms (d_merc, l_grad_b, the Boozer QI residual) run on "
        "host NumPy — use jac=None (finite differences) for those.")


def _least_squares_implicit(
    objective_terms: Sequence[tuple[Callable, float, float]],
    inp: VmecInput,
    *,
    max_mode: int,
    x0: np.ndarray | None,
    current_dofs: int | None = None,
    jac_chunk_size: int | str | None = "auto",
    jac_solver: str = "block",
    recycle: bool = False,
    warm_start: str | None = "perturbation",
    solve_kwargs: dict,
    device: Any = None,
    verbose: int = 0,
    **scipy_kwargs,
):
    """Single-stage boundary least squares with implicit-gradient Jacobians.

    ``fun`` maps the dof vector through the traceable boundary update ->
    :func:`~vmec_jax.core.implicit.solve_implicit` (host solver behind
    ``pure_callback``, warm-started per ``warm_start`` — see
    :func:`least_squares`) ->
    :func:`~vmec_jax.core.implicit.runtime_from_params` -> the stacked
    objective rows: one warm host solve per trial ``x``.  ``jac`` computes
    the exact residual Jacobian by *forward* implicit differentiation:
    by default (``jac_solver="block"``, see ``jacobian_rows_block``) one
    amortized block-tridiagonal factorization backsolves every boundary-dof
    column at once; ``jac_solver="gmres"`` (see ``jacobian_rows``) runs one
    preconditioned GMRES per boundary dof, batched.  Either way this is far
    below one full equilibrium solve per dof (finite differences) while
    keeping the full pointwise Gauss-Newton residual geometry.  Both are
    jit-compiled once per stage.

    The residual and Jacobian graphs run on the device chosen by
    :func:`vmec_jax.core.device.resolve_implicit_device` — the CPU by default,
    where the per-dof vmapped adjoint GMRES is far faster than the
    launch-bound, dof-count-scaling GPU compile (plan.md R1); an explicit
    ``device=`` overrides this.  The forward equilibrium solve is a host
    callback and always runs on the CPU regardless.
    """
    import scipy.optimize

    from . import implicit as imp
    from .device import resolve_implicit_device

    if bool(inp.lasym):
        raise NotImplementedError(
            "jac='implicit' requires lasym = False (the implicit parameter map "
            "does not implement the lasym readin.f boundary rotation)")
    terms = [(_traceable_term(f), float(t), float(w)) for (f, t, w) in objective_terms]
    modes = _dof_modes(inp, max_mode)
    nm = len(modes)
    ntor = int(inp.ntor)
    row_idx = np.asarray([n + ntor for (_, n) in modes], dtype=int)
    col_idx = np.asarray([m for (m, _) in modes], dtype=int)
    # Optional AC/CURTOR dof block (spec 6.4): k + 1 trailing dofs, one-hot
    # tangents through ImplicitParams.ac / .curtor (runtime_from_params
    # already traces both).
    k_cur, ac_scale = _current_dof_setup(inp, current_dofs)
    ndof = 2 * nm + (k_cur + 1 if k_cur else 0)
    # multigrid=True routes the host solve through solve_multigrid (even for
    # single-stage ladders) so NITER-exhausted trials are penalized instead
    # of raising, matching the finite-difference path's trial policy.
    # Loose adjoint budget: the trust-region optimizer only needs ~1e-3
    # gradient accuracy; measured row-norm deviation vs the tight
    # (1e-11, 300) diagnostics default is <~1e-4 at a fraction of the cost.
    # hot_restart seeds each trial's host solve from the stage's previous
    # converged state (same fixed points, far fewer iterations) — the
    # implicit-mode analogue of the finite-difference path's hot restart.
    # warm_start="perturbation" (R25.4) sharpens that seed to the DESC-style
    # first-order prediction; see least_squares.
    if warm_start not in ("perturbation", "state", None):
        raise ValueError(
            "warm_start must be 'perturbation', 'state' or None, "
            f"got {warm_start!r}")
    if recycle and warm_start == "perturbation":
        warm_start = "state"  # the recycled variant carries (C, U) instead
    cfg = imp.make_config(inp, multigrid=True,
                          hot_restart=(warm_start is not None),
                          adjoint_tol=1e-6, adjoint_maxiter=30)
    # Pin the residual/Jacobian graphs to the fastest device for this launch-
    # bound path (CPU by default; explicit device= honored) — committing the
    # input dof vector to it makes both jits compile and run there, and their
    # uncommitted constants follow.  ``None`` leaves placement untouched.
    jac_device = resolve_implicit_device(device, cfg.resolution)

    def _place(x: np.ndarray) -> jnp.ndarray:
        a = jnp.asarray(x, dtype=jnp.float64)
        return a if jac_device is None else jax.device_put(a, jac_device)

    params0 = imp.params_from_input(inp)
    imp._template_runtime(cfg)  # host-built template: warm the per-cfg cache
    # eagerly so runtime_from_params stays traceable under jit below

    def params_of(x: jnp.ndarray):
        rbc = params0.rbc.at[row_idx, col_idx].set(x[:nm])
        zbs = params0.zbs.at[row_idx, col_idx].set(x[nm:2 * nm])
        params = dataclasses.replace(params0, rbc=rbc, zbs=zbs)
        if k_cur:
            ac = params0.ac.at[:k_cur].set(x[2 * nm:2 * nm + k_cur] * ac_scale)
            params = dataclasses.replace(
                params, ac=ac, curtor=x[2 * nm + k_cur] * _CURTOR_SCALE)
        return params

    def term_rows(state, rt) -> jnp.ndarray:
        return jnp.concatenate([
            jnp.atleast_1d(w * (jnp.asarray(f(state, rt)) - t)).ravel()
            for (f, t, w) in terms])

    def residual_rows(x: jnp.ndarray) -> jnp.ndarray:
        params = params_of(x)
        state = imp.solve_implicit(params, cfg)
        return term_rows(state, imp.runtime_from_params(params, cfg))

    rows_jit = jax.jit(residual_rows)

    # The evolved-dof mask is a *structural* per-config constant; fetch it
    # once (first host solve, cached in implicit._MASK_CACHE) so the Jacobian
    # graph below can close over it.
    if x0 is None:
        x0 = pack_boundary(inp, max_mode)
        if k_cur:
            x0 = np.concatenate([x0, _pack_current(inp, k_cur, ac_scale)])
    params0_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64),
                              params_of(jnp.asarray(x0, dtype=jnp.float64)))
    _, mask_np = imp._host_solve_and_mask(cfg, params0_np)
    mask_const = jax.tree.map(jnp.asarray, mask_np)

    # One-hot dof tangents in ImplicitParams space, stacked over dofs
    # (leading axis ndof) so chunk_map can process them in fixed-size chunks:
    # boundary rbc/zbs rows first, then the scaled AC/CURTOR rows.
    t_rbc = np.zeros((ndof,) + np.shape(params0.rbc))
    t_zbs = np.zeros((ndof,) + np.shape(params0.zbs))
    t_ac = np.zeros((ndof,) + np.shape(params0.ac))
    t_curtor = np.zeros((ndof,))
    for j in range(nm):
        t_rbc[j, row_idx[j], col_idx[j]] = 1.0
        t_zbs[nm + j, row_idx[j], col_idx[j]] = 1.0
    for j in range(k_cur):
        t_ac[2 * nm + j, j] = ac_scale
    if k_cur:
        t_curtor[2 * nm + k_cur] = _CURTOR_SCALE
    zerop = jax.tree.map(jnp.zeros_like, params0)
    tangent_stack = (jnp.asarray(t_rbc), jnp.asarray(t_zbs),
                     jnp.asarray(t_ac), jnp.asarray(t_curtor))

    # R17.1 memory knob: chunk_size None == one wide vmap (current behavior),
    # an int / "auto" caps peak Jacobian memory at that many dofs at a time.
    if jac_chunk_size == "auto":
        chunk = int(auto_chunk_size(ndof))
    elif jac_chunk_size is None or isinstance(jac_chunk_size, int):
        chunk = jac_chunk_size
    else:
        raise ValueError(
            "jac_chunk_size must be None, a positive int, or 'auto', "
            f"got {jac_chunk_size!r}")

    def _jac_parts(x: jnp.ndarray):
        """Shared per-x setup of the implicit-Jacobian maps.

        At the fixed point, ``dz_j = -(dF/dz)^{-1} dF/dp t_j`` per boundary
        dof tangent ``t_j`` (F's linearization is plain JAX, so forward mode
        is available even though the solve itself is an opaque custom-VJP
        callback), then ``J[:, j] = G_z dz_j + G_p t_j`` with ``G`` the
        residual rows of the assembled state.  Returns the linearized
        operator ``Fz`` plus the per-dof tangent/RHS/column maps shared by
        all Jacobian variants below, and the ``(params, frozen, P, z_star)``
        linearization point (the block variant re-linearizes the *raw*
        residual formulation there).
        """
        params = params_of(x)
        frozen = jax.lax.stop_gradient(imp.solve_implicit(params, cfg))
        P = imp._dof_projector(cfg, mask_const)
        edge = imp._edge_mask(cfg)
        F = imp.residual_fn(cfg, frozen, mask_const)
        z_star = P(frozen)

        def G(z, prm):
            rt_p = imp.runtime_from_params(prm, cfg)
            return term_rows(imp._assemble(z, rt_p, frozen, P, edge), rt_p)

        def Fz(dz):
            return jax.jvp(lambda z: F(z, params), (z_star,), (dz,))[1]

        def tangent_of(tp):
            return dataclasses.replace(zerop, rbc=tp[0], zbs=tp[1],
                                       ac=tp[2], curtor=tp[3])

        def rhs_of(tp):
            b = jax.jvp(lambda prm: F(z_star, prm), (params,), (tp,))[1]
            return jax.tree.map(jnp.negative, b)

        def column_of(dz, tp):
            return jax.jvp(G, (z_star, params), (P(dz), tp))[1]

        return Fz, tangent_of, rhs_of, column_of, (params, frozen, P, z_star)

    def jacobian_rows(x: jnp.ndarray):
        """Exact residual Jacobian by *forward* implicit differentiation.

        One batched preconditioned GMRES per boundary dof (see
        ``_jac_parts``) — far below one forward solve per dof (finite
        differences) — while exposing the *full* pointwise Gauss-Newton
        geometry to scipy.  Columns are mathematically independent, so the
        result is identical across chunk sizes to float64 round-off.
        Also returns the per-dof state responses ``dz_j`` (leading axis
        ``2*nm``): they are the R25.4 perturbation warm-start linearization,
        already paid for by the column solves.
        """
        Fz, tangent_of, rhs_of, column_of, _ = _jac_parts(x)

        def column(tp_stack):
            tp = tangent_of(tp_stack)
            dz, _ = imp._adjoint_solve(Fz, rhs_of(tp), cfg)
            return column_of(dz, tp), dz

        cols, dz_cols = chunk_map(column, tangent_stack, chunk_size=chunk)
        return jnp.transpose(cols), dz_cols

    # R25.2 amortized block-tridiagonal variant.  The *raw* residual
    # formulation (un-preconditioned scalxc-scaled spectral force; see
    # implicit.residual_fn) has a Jacobian that is exactly block-tridiagonal
    # in the radial index (verified numerically: per-surface probe response
    # is 0.0 beyond |i-j| = 1 — the radial coupling is the nearest-neighbor
    # full/half-mesh FD stencil; the *preconditioned* formulation is dense
    # in radius because the 1D preconditioner applies per-mode radial
    # tridiagonal *solves*).  Both formulations share the fixed point, so
    #   dz_j = -(dF/dz)^{-1} dF/dp t_j
    # is the same solution through either: assemble the raw blocks once with
    # 3-colored jvp probes (cost ~3*(3*mn) residual linearizations,
    # independent of the dof count), factor once (solvax block Thomas), and
    # backsolve all 2*nm right-hand sides — then one short preconditioned
    # GMRES pass per column (warm-started at the direct solution) certifies
    # cfg.adjoint_tol in the same norm as the default path: solvax checks
    # the initial residual before the first Arnoldi cycle, so columns whose
    # direct solve already meets tolerance cost one matvec.
    mn_state = int(np.asarray(mask_np.R_cos).shape[1])
    ns_state = int(cfg.resolution.ns)
    active_fields = tuple(f for f in imp._STATE_FIELDS
                          if np.asarray(getattr(mask_np, f)).any())
    n_act = len(active_fields)
    m_block = n_act * mn_state
    # Probe (color, field, column) index triples, color-major so the probe
    # axis reshapes to (3, m_block, ...) below.
    probe_color = jnp.asarray(np.repeat(np.arange(3), m_block))
    probe_field = jnp.asarray(np.tile(np.repeat(np.arange(n_act), mn_state), 3))
    probe_col = jnp.asarray(np.tile(np.tile(np.arange(mn_state), n_act), 3))
    if jac_chunk_size == "auto":
        probe_chunk = int(auto_chunk_size(3 * m_block))
    else:
        probe_chunk = chunk

    def _pack(t) -> jnp.ndarray:
        """SpectralState -> (ns, m_block): active fields side by side."""
        return jnp.concatenate([getattr(t, f) for f in active_fields], axis=1)

    def _unpack(mat: jnp.ndarray) -> SpectralState:
        """(ns, m_block) -> SpectralState (structurally-zero fields zero)."""
        parts = dict(zip(active_fields, jnp.split(mat, n_act, axis=1)))
        return SpectralState(**{
            f: parts.get(f, jnp.zeros((ns_state, mn_state), mat.dtype))
            for f in imp._STATE_FIELDS})

    def jacobian_rows_block(x: jnp.ndarray):
        """``jacobian_rows`` via one block-tridiagonal factorization (R25.2).

        Same Jacobian as the default path to ``cfg.adjoint_tol`` (the GMRES
        corrector runs against the identical preconditioned system) at a
        cost that does not grow with the boundary-dof count.  Returns the
        certified per-dof responses ``dz_j`` alongside the rows (the R25.4
        perturbation warm-start linearization, same contract as
        ``jacobian_rows``).
        """
        Fz, tangent_of, rhs_of, column_of, (params, frozen, P, z_star) = \
            _jac_parts(x)
        F_raw = imp.residual_fn(cfg, frozen, mask_const, formulation="raw")

        def Fz_raw(dz):
            return jax.jvp(lambda z: F_raw(z, params), (z_star,), (dz,))[1]

        def probe_response(spec):
            c, fi, k = spec
            rows = (jnp.arange(ns_state) % 3 == c)
            mat = jnp.where(rows[:, None],
                            jax.nn.one_hot(k, mn_state, dtype=x.dtype)[None, :],
                            0.0)
            stack = (jax.nn.one_hot(fi, n_act, dtype=x.dtype)[:, None, None]
                     * mat[None])
            dz = _unpack(jnp.concatenate(
                [stack[i] for i in range(n_act)], axis=1))
            # F_raw projects both sides onto the evolved-dof subspace, so
            # its linearization is singular on the (I - P) complement; the
            # identity fill (dz - P(dz)) makes the assembled blocks
            # invertible without changing the solution for P-masked RHS.
            return _pack(jax.tree.map(lambda a, b, p: a + (b - p),
                                      Fz_raw(dz), dz, P(dz)))

        probes = chunk_map(probe_response,
                           (probe_color, probe_field, probe_col),
                           chunk_size=probe_chunk)
        # probes[c*m_block + q, i, :] = rows i of A(dz) for the color-c
        # one-hot-q tangent; for row i the unique in-stencil source surface
        # of color c is j = i + d with d = the offset satisfying
        # (i + d) % 3 == c, so gathering at color (i + d) % 3 reads off the
        # d-band blocks A[i, i+d] for every surface at once.
        probes = probes.reshape((3, m_block, ns_state, m_block))
        ii = jnp.arange(ns_state)

        def band(d):
            g = probes[(ii + d) % 3, :, ii, :]  # (ns, col q, row)
            return jnp.swapaxes(g, 1, 2)  # (ns, row, col)

        # lower[0] / upper[-1] gather out-of-stencil (zero) responses and
        # are ignored by the factorization anyway.
        factors = block_thomas_factor(band(-1), band(0), band(1))

        def raw_rhs(tp_stack):
            tp = tangent_of(tp_stack)
            b = jax.jvp(lambda prm: F_raw(z_star, prm), (params,), (tp,))[1]
            return _pack(jax.tree.map(jnp.negative, b))

        rhs = chunk_map(raw_rhs, tangent_stack, chunk_size=chunk)
        dz0 = block_thomas_solve(factors, jnp.moveaxis(rhs, 0, -1))

        def column(args):
            *tp_stack_j, dz0_mat = args
            tp = tangent_of(tuple(tp_stack_j))
            dz, _ = imp._adjoint_solve(
                Fz, rhs_of(tp), cfg, x0=_unpack(dz0_mat),
                max_restarts=min(3, cfg.adjoint_maxiter))
            return column_of(dz, tp), dz

        cols, dz_cols = chunk_map(
            column, (*tangent_stack, jnp.moveaxis(dz0, -1, 0)),
            chunk_size=chunk)
        return jnp.transpose(cols), dz_cols

    # R25.3 recycled variant: all 2*nm solves share the operator Fz (and Fz
    # drifts slowly between accepted trust-region iterates), so a GCROT
    # deflation pair (C, U) is threaded through a lax.scan over fixed-size
    # dof chunks — vmapped *within* a chunk with the incoming pair shared
    # read-only, then advanced from one representative (first) lane — and
    # returned to the caller, which stashes it between jac_jit calls.  The
    # dof axis is zero-padded to a whole number of chunks; padded columns
    # have zero RHS (gcrot converges in zero cycles) and are discarded.
    n_flat = sum(int(np.prod(s.shape))
                 for s in jax.tree.leaves(imp._state_struct(cfg)))
    csize = int(chunk) if chunk else ndof
    nchunks = -(-ndof // csize)
    pad = nchunks * csize - ndof

    def jacobian_rows_recycled(x: jnp.ndarray, C: jnp.ndarray,
                               U: jnp.ndarray):
        """``jacobian_rows`` with GCROT recycle carry (plan R25.3).

        Same ``cfg.adjoint_tol`` / ``cfg.adjoint_maxiter`` budget per solve
        as the default path; the Jacobian matches to solver tolerance *when
        the solves converge within budget*.  See the ``recycle`` note in
        :func:`least_squares` for why this is opt-in: the solvax v0.1
        recycle space measurably slows warm-started columns on the
        production operator, so budget-capped columns can come back with
        larger residuals than the GMRES path.
        """
        Fz, tangent_of, rhs_of, column_of, _ = _jac_parts(x)

        def column(tp_stack_j, rec):
            tp = tangent_of(tp_stack_j)
            dz, sol = imp._recycled_solve(Fz, rhs_of(tp), cfg, rec)
            return column_of(dz, tp), sol.recycle

        def scan_body(carry, tp_chunk):
            cols_chunk, recs = jax.vmap(
                column, in_axes=(0, None))(tp_chunk, carry)
            # Lane 0 is always a real dof (pad < csize): its updated pair
            # seeds the next chunk / the next Jacobian evaluation.
            return jax.tree.map(lambda a: a[0], recs), cols_chunk

        def pad_stack(t):
            t = jnp.concatenate(
                [t, jnp.zeros((pad,) + t.shape[1:], t.dtype)])
            return t.reshape((nchunks, csize) + t.shape[1:])

        (C, U), cols = jax.lax.scan(
            scan_body, (C, U),
            tuple(pad_stack(t) for t in tangent_stack))
        cols = cols.reshape((nchunks * csize,) + cols.shape[2:])[:ndof]
        return jnp.transpose(cols), C, U

    if jac_solver not in ("block", "gmres"):
        raise ValueError(
            f"jac_solver must be 'block' or 'gmres', got {jac_solver!r}")
    if recycle:
        jac_impl = jacobian_rows_recycled  # opt-in R25.3 experiment wins
    elif jac_solver == "block":
        jac_impl = jacobian_rows_block
    else:
        jac_impl = jacobian_rows
    jac_jit = jax.jit(jac_impl)

    holder: dict[str, Any] = {"nres": None, "lin": None}
    if recycle:
        # An all-zero pair is a cold start (gcrot's warm-start QR masks the
        # rank-deficient columns out); shapes are static so jac_jit compiles
        # once and the carried pair never triggers a re-trace.
        holder["recycle"] = (_place(np.zeros((n_flat, imp._RECYCLE_K))),
                             _place(np.zeros((n_flat, imp._RECYCLE_K))))

    # R25.4 perturbation warm start (DESC arXiv:2203.15927 ``eq.perturb``
    # before ``eq.solve``): each jac(x_ref) call stashes its linearization —
    # the converged state plus the per-dof responses dz_j its columns just
    # solved — and every subsequent trial fun(x) deposits the first-order
    # predicted state in implicit._PERTURB_SEED for the host solve to
    # consume, instead of restarting from the unmoved last converged state.
    P_seed = imp._dof_projector(cfg, mask_const)
    edge_seed = imp._edge_mask(cfg)

    @jax.jit
    def predicted_state(x_trial, x_ref, frozen, dz_cols):
        """First-order trial-state prediction around the stashed jac point.

        ``x_pred = frozen + P(sum_j (x_trial - x_ref)_j dz_j) +
        edge*(boundary(p_trial) - frozen)`` through the same dof-projector /
        assemble machinery the implicit residual uses, so the edge row lands
        exactly on the trial boundary (the solver's ``hot_restart_state``
        boundary shift becomes a no-op) and frozen directions stay frozen.
        """
        rt_p = imp.runtime_from_params(params_of(x_trial), cfg)
        dz = jax.tree.map(
            lambda d: jnp.tensordot(x_trial - x_ref, d, axes=1), dz_cols)
        z = jax.tree.map(jnp.add, P_seed(frozen), dz)
        return imp._assemble(z, rt_p, frozen, P_seed, edge_seed)

    def _stash_linearization(x: np.ndarray, dz_cols) -> None:
        """Record ``(x_ref, converged state, dz columns)`` for trial seeding."""
        hit = imp._LAST_SOLVE.get(cfg)
        params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64),
                                 params_of(jnp.asarray(x, dtype=jnp.float64)))
        if hit is not None and hit[0] == imp._params_key(params_np):
            holder["lin"] = (np.array(x, dtype=float), hit[1].state, dz_cols)
        else:  # unexpected call pattern: better no seed than a wrong one
            holder["lin"] = None

    def fun(x: np.ndarray) -> np.ndarray:
        lin = holder["lin"]
        if lin is not None and lin[0].shape == np.shape(x):
            seed = jax.tree.map(
                lambda a: np.asarray(a, dtype=np.float64),
                jax.device_get(predicted_state(
                    _place(x), _place(lin[0]), lin[1], lin[2])))
            if all(np.all(np.isfinite(a)) for a in jax.tree.leaves(seed)):
                imp._PERTURB_SEED[cfg] = seed
        try:
            residual = np.asarray(
                jax.device_get(rows_jit(_place(x))), dtype=float)
        except Exception as exc:  # zero-crash policy: penalize, don't die
            if holder["nres"] is None:
                raise
            if verbose:
                print(f"[least_squares] trial solve failed: {exc}")
            return np.full((holder["nres"],), 1.0e6)
        finally:
            imp._PERTURB_SEED.pop(cfg, None)  # one-shot: never leak a seed
        if not np.all(np.isfinite(residual)):
            residual = np.where(np.isfinite(residual), residual, 1.0e6)
        holder["nres"] = residual.size
        if verbose:
            print(f"[least_squares] cost = {0.5 * float(residual @ residual):.6e}")
        return residual

    def jac_fn(x: np.ndarray) -> np.ndarray:
        try:
            if recycle:
                rows, C, U = jac_jit(_place(x), *holder["recycle"])
                holder["recycle"] = (C, U)  # deflate the next jac evaluation
                jac = np.asarray(jax.device_get(rows), dtype=float)
            else:
                rows, dz_cols = jac_jit(_place(x))
                if warm_start == "perturbation":
                    _stash_linearization(np.asarray(x, dtype=float), dz_cols)
                jac = np.asarray(jax.device_get(rows), dtype=float)
        except Exception as exc:  # zero-crash policy (mirrors fun): a trial whose
            # equilibrium fails (e.g. VmecJacobianError from a self-intersecting
            # boundary) must be *rejected*, not crash the optimization.  Reuse
            # the last valid Jacobian so scipy's trust region steps back off the
            # bad point (fun already returns a large penalty residual there).
            if holder.get("last_jac") is None:
                raise
            if verbose:
                print(f"[least_squares] trial jacobian failed: {exc}")
            return holder["last_jac"]
        if np.all(np.isfinite(jac)):
            holder["last_jac"] = jac
        return jac

    # Pre-size the residual from the (converged) seed so a *first*-iteration
    # trial failure penalizes like any later one instead of re-raising.
    if holder["nres"] is None:
        try:
            holder["nres"] = int(np.asarray(jax.device_get(rows_jit(_place(x0)))).size)
        except Exception:  # the seed itself does not converge -> fun raises clearly
            pass

    result = scipy.optimize.least_squares(fun, np.asarray(x0, dtype=float),
                                          jac=jac_fn, **scipy_kwargs)
    result.input = unpack_boundary(inp, result.x[:2 * nm], max_mode)
    if k_cur:
        result.input = _apply_current(result.input, result.x[2 * nm:],
                                      k_cur, ac_scale)
    stats = imp._SOLVE_STATS.get(cfg)
    result.solve_stats = None if stats is None else dict(stats)
    try:
        # Hot-seed the diagnostic re-solve from the stage's last converged
        # trial state (plan R25.1): the optimizer's final x was just solved
        # by the implicit path, so this converges in ~1 sweep instead of
        # repeating a full cold solve per continuation stage.
        seed = imp._HOT_CACHE.get(cfg)
        try:
            result.equilibrium = solve_equilibrium(
                result.input, initial_state=seed, **solve_kwargs)
        except Exception:
            if seed is None:
                raise
            # ns-mismatched seed (different ladder) must not cost the
            # diagnostic: fall back to the plain cold solve.
            result.equilibrium = solve_equilibrium(result.input, **solve_kwargs)
    except Exception:  # pragma: no cover - diagnostic attribute only
        result.equilibrium = None
    return result
