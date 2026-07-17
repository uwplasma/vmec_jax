"""Traceable single-surface Boozer input tables (wout-convention spectra).

Pure-JAX bridge from a converged spectral state to the single-surface
``wout``-convention mode tables that a differentiable Boozer transform
(``booz_xform_jax``) consumes.  This is the differentiable route between
vmex and downstream kinetic codes:

    boundary dofs -> :func:`vmex.core.implicit.solve_implicit`
    -> :func:`boozer_input_tables` (this module)
    -> booz_xform_jax (Boozer |B| spectrum)
    -> kinetic solvers (e.g. sfincs_jax bootstrap-current objectives),

so ``jax.grad`` flows through the whole physics chain.  Origin: ported from
the flagship optimization example ``examples/optimize_QA_bootstrap.py`` of
sfincs_jax, where it was validated against the host wout engine and a
classic host booz_xform run.

Everything here evaluates the core field chain (``geometry``/``fields``,
pure JAX), mirrors the reduced stellarator-symmetric ``[0, pi]`` theta grid
to the full circle, and projects onto the wout ``cos(m*theta - n*zeta)`` /
``sin(...)`` tables — no host callbacks, so the function can be jitted and
differentiated.

Public API
----------
``boozer_input_tables(state, rt, j) -> dict``
    Spectral tables ``rmnc/zmns/lmns/bmnc/bsubumnc/bsubvmnc`` plus
    ``iota/G/I`` at half-mesh row ``j`` (stellarator-symmetric only).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .fields import magnetic_fields, metric_elements, surface_currents
from .geometry import half_mesh_jacobian
from .solver import SolverRuntime, SpectralState, _geometry

__all__ = ["boozer_input_tables"]


def boozer_input_tables(state: SpectralState, rt: SolverRuntime, j: int) -> dict:
    """Traceable wout-convention spectral tables at half-mesh row ``j``.

    Builds the single-surface inputs of a Boozer transform from a converged
    (stellarator-symmetric) equilibrium, entirely in JAX:

    - ``bmnc``, ``bsubumnc``, ``bsubvmnc``: |B| and the covariant field
      components, native to the half mesh (``bcovar.f``), projected on the
      grid-representable ``cos(m*theta - n*zeta)`` modes;
    - ``rmnc``, ``zmns``: R/Z tables from the full-mesh rows ``j-1``/``j``
      with the VMEC odd-m ``sqrt(s)`` parity interpolation to the half mesh;
    - ``lmns``: the wout lambda sine table, reconstructed from the
      (lamscale-scaled) angular derivatives with the wout ``1/phips`` factor;
    - ``iota`` (``add_fluxes.f`` for ``ncurr=1``, else the prescribed
      profile) and the Boozer covariant averages ``G = bvco``/``I = buco``;
    - ``xm``, ``xn``: static NumPy mode-number arrays (``xn`` carries the
      ``nfp`` factor, wout convention).

    Validation (tests/test_boozer_tables.py, and the sfincs_jax
    flagship-example tests where this function originated): ``bmnc`` and the
    parity-interpolated ``rmnc/zmns`` match the host wout engine to
    ~1e-15..1e-10 relative (identical quadrature); ``bsubumnc/bsubvmnc`` and
    ``lmns`` agree at the ~1e-3 half-mesh finite-difference level (looser at
    very small ``ns``), which is the wout engine's own grid discrepancy, not
    an error of this projection.

    Parameters
    ----------
    state:
        Converged spectral state (e.g. from
        :func:`vmex.core.implicit.solve_implicit`, which makes the whole
        chain differentiable in the boundary, or ``SolveResult.state``).
    rt:
        The matching :class:`~vmex.core.solver.SolverRuntime` (e.g. from
        :func:`vmex.core.implicit.runtime_from_params`).
    j:
        Half-mesh radial row index, ``1 <= j <= ns - 1`` (static).

    Returns
    -------
    dict
        Keys ``xm, xn`` (static ``np.ndarray``) and ``rmnc, zmns, lmns,
        bmnc, bsubumnc, bsubvmnc, iota, G, I`` (JAX arrays, traced).
    """
    setup = rt.setup
    nfp = int(rt.resolution.nfp)
    s = jnp.asarray(setup.s_full)
    sqrt_s = jnp.sqrt(s)
    s_half_j = 0.5 * (s[j] + s[j - 1])
    _, geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )

    # mirror the reduced symmetric [0, pi] grid to the full theta circle
    ntheta2 = int(np.shape(fields.total_pressure)[1])
    nzeta = int(np.shape(fields.total_pressure)[2])
    ntheta1 = max(2 * (ntheta2 - 1), 1)
    i_full = np.arange(ntheta1)
    kk = np.arange(nzeta)
    i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
    k_src = np.where(i_full[:, None] < ntheta2, kk[None, :], (nzeta - kk[None, :]) % nzeta)
    i_src2 = np.broadcast_to(i_src[:, None], (ntheta1, nzeta))
    sign_odd = np.where(i_full < ntheta2, 1.0, -1.0)[:, None]

    def mirror(a2d, parity):
        out = jnp.asarray(a2d)[i_src2, k_src]
        return out if parity == "even" else out * jnp.asarray(sign_odd)

    # uniform-grid Fourier projection onto the grid-representable modes
    theta = 2.0 * np.pi * np.arange(ntheta1) / ntheta1
    zeta = 2.0 * np.pi * np.arange(nzeta) / (nfp * nzeta)
    m_max, n_max = ntheta1 // 2 - 1, max(nzeta // 2 - 1, 0)
    ml, nl = [], []
    for m in range(0, m_max + 1):
        for n in range(-n_max, n_max + 1):
            if m == 0 and n < 0:
                continue
            ml.append(m)
            nl.append(n * nfp)
    xm, xn = np.asarray(ml), np.asarray(nl)
    ang = theta[:, None, None] * xm[None, None, :] - zeta[None, :, None] * xn[None, None, :]
    cos_t, sin_t = jnp.asarray(np.cos(ang)), jnp.asarray(np.sin(ang))
    w = 2.0 / (ntheta1 * nzeta) * np.ones(xm.shape)
    w[(xm == 0) & (xn == 0)] = 1.0 / (ntheta1 * nzeta)
    w = jnp.asarray(w)

    def project(f, parity):
        return w * jnp.einsum("tz,tzm->m", f, cos_t if parity == "even" else sin_t)

    # |B|, B_theta, B_zeta live on the half mesh natively (bcovar.f)
    bsq2 = 2.0 * (jnp.asarray(fields.total_pressure)[j] - jnp.asarray(fields.pressure)[j])
    bmnc = project(mirror(jnp.sqrt(jnp.maximum(bsq2, 1e-300)), "even"), "even")
    bsubumnc = project(mirror(jnp.asarray(fields.bsubu)[j], "even"), "even")
    bsubvmnc = project(mirror(jnp.asarray(fields.bsubv)[j], "even"), "even")

    # R, Z: full-mesh rows j-1, j -> spectral -> VMEC parity interpolation
    def phys_row(even, odd, row):
        return jnp.asarray(even)[row] + sqrt_s[row] * jnp.asarray(odd)[row]

    def spectral_half(even, odd, parity):
        a = project(mirror(phys_row(even, odd, j - 1), parity), parity)
        b = project(mirror(phys_row(even, odd, j), parity), parity)
        m_even = jnp.asarray(xm % 2 == 0)
        interp_even = 0.5 * (a + b)
        interp_odd = 0.5 * (a / jnp.maximum(sqrt_s[j - 1], 1e-30) + b / sqrt_s[j]) * jnp.sqrt(s_half_j)
        return jnp.where(m_even, interp_even, interp_odd)

    rmnc = spectral_half(geometry.R_even, geometry.R_odd, "even")
    zmns = spectral_half(geometry.Z_even, geometry.Z_odd, "odd")

    # lambda: reconstruct the wout lmns sine table from the (lamscale-scaled)
    # angular derivatives; the wout convention carries a 1/phips factor.
    lamscale = jnp.asarray(fields.lamscale)
    phips_j = jnp.asarray(setup.phips)[j]

    def half_native(even, odd):
        return 0.5 * (phys_row(even, odd, j - 1) + phys_row(even, odd, j)) * lamscale

    lth = project(mirror(half_native(geometry.dlambda_dtheta_even,
                                     geometry.dlambda_dtheta_odd), "even"), "even")
    lze = project(mirror(half_native(geometry.dlambda_dzeta_even,
                                     geometry.dlambda_dzeta_odd), "even"), "even")
    m_safe = jnp.asarray(np.where(xm != 0, xm, 1), dtype=jnp.float64)
    n_safe = jnp.asarray(np.where(xn != 0, xn, 1), dtype=jnp.float64)
    lmns = jnp.where(jnp.asarray(xm != 0), lth / m_safe,
                     jnp.where(jnp.asarray(xn != 0), -lze / n_safe, 0.0)) / phips_j

    # iota (add_fluxes.f, ncurr=1) and the Boozer covariant averages G, I
    iota = (jnp.asarray(fields.chips)[j] / jnp.asarray(setup.phips)[j]
            if int(setup.ncurr) == 1 else jnp.asarray(setup.iotas)[j])
    cur = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv, trig=rt.trig,
                           s=s, signgs=setup.signgs)
    return dict(xm=xm, xn=xn, rmnc=rmnc, zmns=zmns, lmns=lmns, bmnc=bmnc,
                bsubumnc=bsubumnc, bsubvmnc=bsubvmnc, iota=iota,
                G=jnp.asarray(cur.bvco)[j], I=jnp.asarray(cur.buco)[j])
