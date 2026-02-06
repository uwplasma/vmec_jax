"""VMEC-style real-space synthesis (Step-10 parity utilities).

This module provides a VMEC-compatible inverse transform on the VMEC internal
angle grid using the trig/weight tables from ``fixaray``. It is intended for
parity diagnostics where matching VMEC's reduced theta grid and scaling
conventions matters.
"""

from __future__ import annotations

from typing import Any

from ._compat import jnp
from .modes import ModeTable
from .vmec_tomnsp import VmecTrigTables


def _vmec_mode_scaling(*, m: Any, n: Any, trig: VmecTrigTables) -> Any:
    """Return 1/(mscale* nscale) for each (m,n)."""
    m = jnp.asarray(m)
    n = jnp.asarray(n)
    n1 = jnp.abs(n)
    mscale = jnp.asarray(trig.mscale)
    nscale = jnp.asarray(trig.nscale)
    return 1.0 / (mscale[m] * nscale[n1])


def _vmec_phase_tables(*, m: Any, n: Any, trig: VmecTrigTables):
    m = jnp.asarray(m).astype(jnp.int32)
    n = jnp.asarray(n).astype(jnp.int32)
    n1 = jnp.abs(n)
    sgn = jnp.where(n < 0, -1.0, 1.0)

    cosmu = jnp.asarray(trig.cosmu)  # (ntheta3, mmax+1)
    sinmu = jnp.asarray(trig.sinmu)
    cosnv = jnp.asarray(trig.cosnv)  # (nzeta, nmax+1)
    sinnv = jnp.asarray(trig.sinnv)

    cosmu_m = cosmu[:, m].T  # (K, ntheta3)
    sinmu_m = sinmu[:, m].T
    cosnv_n = cosnv[:, n1].T  # (K, nzeta)
    sinnv_n = sinnv[:, n1].T

    cos_phase = cosmu_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * sinnv_n[:, None, :]
    sin_phase = sinmu_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * sinnv_n[:, None, :]
    return cos_phase, sin_phase


def _vmec_phase_tables_dtheta(*, m: Any, n: Any, trig: VmecTrigTables):
    m = jnp.asarray(m).astype(jnp.int32)
    n = jnp.asarray(n).astype(jnp.int32)
    n1 = jnp.abs(n)
    sgn = jnp.where(n < 0, -1.0, 1.0)

    cosmu = jnp.asarray(trig.cosmu)  # (ntheta3, mmax+1)
    sinmu = jnp.asarray(trig.sinmu)
    cosmum = jnp.asarray(trig.cosmum)
    sinmum = jnp.asarray(trig.sinmum)
    cosnv = jnp.asarray(trig.cosnv)
    sinnv = jnp.asarray(trig.sinnv)

    cosmu_m = cosmu[:, m].T
    sinmu_m = sinmu[:, m].T
    cosmum_m = cosmum[:, m].T
    sinmum_m = sinmum[:, m].T
    cosnv_n = cosnv[:, n1].T
    sinnv_n = sinnv[:, n1].T

    dcos_phase = sinmum_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * cosmum_m[:, :, None] * sinnv_n[:, None, :]
    dsin_phase = cosmum_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * sinmum_m[:, :, None] * sinnv_n[:, None, :]
    return dcos_phase, dsin_phase


def _vmec_phase_tables_dzeta(*, m: Any, n: Any, trig: VmecTrigTables):
    m = jnp.asarray(m).astype(jnp.int32)
    n = jnp.asarray(n).astype(jnp.int32)
    n1 = jnp.abs(n)
    sgn = jnp.where(n < 0, -1.0, 1.0)

    cosmu = jnp.asarray(trig.cosmu)
    sinmu = jnp.asarray(trig.sinmu)
    cosnvn = jnp.asarray(trig.cosnvn)
    sinnvn = jnp.asarray(trig.sinnvn)

    cosmu_m = cosmu[:, m].T
    sinmu_m = sinmu[:, m].T
    cosnvn_n = cosnvn[:, n1].T
    sinnvn_n = sinnvn[:, n1].T

    dcos_phase = cosmu_m[:, :, None] * sinnvn_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * cosnvn_n[:, None, :]
    dsin_phase = sinmu_m[:, :, None] * sinnvn_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * cosnvn_n[:, None, :]
    return dcos_phase, dsin_phase


def vmec_realspace_synthesis(
    *,
    coeff_cos: Any,
    coeff_sin: Any,
    modes: ModeTable,
    trig: VmecTrigTables,
) -> Any:
    """Synthesize a real-space field on the VMEC internal grid.

    This implements the same trigonometric synthesis as VMEC's ``totzsp``
    using the precomputed ``fixaray`` tables. The input coefficients are
    assumed to be in the **wout/physical** convention, so we apply the
    VMEC scaling (divide by ``mscale*nscale``) internally.

    Parameters
    ----------
    coeff_cos, coeff_sin:
        Arrays of shape (ns, K) with Fourier coefficients for cos/sin.
    modes:
        Mode table with arrays ``m`` and ``n`` (n is *not* multiplied by nfp).
    trig:
        VMEC trig tables from ``vmec_trig_tables``.

    Returns
    -------
    f:
        Real-space field of shape (ns, ntheta3, nzeta).
    """
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    m = jnp.asarray(modes.m).astype(jnp.int32)
    n = jnp.asarray(modes.n).astype(jnp.int32)

    if coeff_cos.ndim != 2 or coeff_sin.ndim != 2:
        raise ValueError("Expected coeff arrays with shape (ns, K)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    if coeff_cos.shape[1] != m.shape[0]:
        raise ValueError("Mode count mismatch between coefficients and modes")

    # VMEC internal scaling: coefficients are stored divided by mscale*nscale.
    scale = _vmec_mode_scaling(m=m, n=n, trig=trig).astype(coeff_cos.dtype)
    coeff_cos = coeff_cos * scale[None, :]
    coeff_sin = coeff_sin * scale[None, :]

    cos_phase, sin_phase = _vmec_phase_tables(m=m, n=n, trig=trig)

    # Sum over modes.
    f = jnp.einsum("sk,kij->sij", coeff_cos, cos_phase) + jnp.einsum("sk,kij->sij", coeff_sin, sin_phase)
    return f


def vmec_realspace_synthesis_dtheta(
    *,
    coeff_cos: Any,
    coeff_sin: Any,
    modes: ModeTable,
    trig: VmecTrigTables,
) -> Any:
    """Theta derivative of the VMEC real-space synthesis."""
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    m = jnp.asarray(modes.m).astype(jnp.int32)
    n = jnp.asarray(modes.n).astype(jnp.int32)

    if coeff_cos.ndim != 2 or coeff_sin.ndim != 2:
        raise ValueError("Expected coeff arrays with shape (ns, K)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    if coeff_cos.shape[1] != m.shape[0]:
        raise ValueError("Mode count mismatch between coefficients and modes")

    scale = _vmec_mode_scaling(m=m, n=n, trig=trig).astype(coeff_cos.dtype)
    coeff_cos = coeff_cos * scale[None, :]
    coeff_sin = coeff_sin * scale[None, :]

    dcos_phase, dsin_phase = _vmec_phase_tables_dtheta(m=m, n=n, trig=trig)
    f = jnp.einsum("sk,kij->sij", coeff_cos, dcos_phase) + jnp.einsum("sk,kij->sij", coeff_sin, dsin_phase)
    return f


def vmec_realspace_synthesis_dzeta_phys(
    *,
    coeff_cos: Any,
    coeff_sin: Any,
    modes: ModeTable,
    trig: VmecTrigTables,
) -> Any:
    """Zeta(physical) derivative of the VMEC real-space synthesis."""
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    m = jnp.asarray(modes.m).astype(jnp.int32)
    n = jnp.asarray(modes.n).astype(jnp.int32)

    if coeff_cos.ndim != 2 or coeff_sin.ndim != 2:
        raise ValueError("Expected coeff arrays with shape (ns, K)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    if coeff_cos.shape[1] != m.shape[0]:
        raise ValueError("Mode count mismatch between coefficients and modes")

    scale = _vmec_mode_scaling(m=m, n=n, trig=trig).astype(coeff_cos.dtype)
    coeff_cos = coeff_cos * scale[None, :]
    coeff_sin = coeff_sin * scale[None, :]

    dcos_phase, dsin_phase = _vmec_phase_tables_dzeta(m=m, n=n, trig=trig)
    f = jnp.einsum("sk,kij->sij", coeff_cos, dcos_phase) + jnp.einsum("sk,kij->sij", coeff_sin, dsin_phase)
    return f


def vmec_realspace_geom_from_state(
    *,
    state,
    modes: ModeTable,
    trig: VmecTrigTables,
) -> dict[str, Any]:
    """Compute VMEC real-space geometry fields on the internal grid."""
    R = vmec_realspace_synthesis(coeff_cos=state.Rcos, coeff_sin=state.Rsin, modes=modes, trig=trig)
    Z = vmec_realspace_synthesis(coeff_cos=state.Zcos, coeff_sin=state.Zsin, modes=modes, trig=trig)
    Ru = vmec_realspace_synthesis_dtheta(coeff_cos=state.Rcos, coeff_sin=state.Rsin, modes=modes, trig=trig)
    Zu = vmec_realspace_synthesis_dtheta(coeff_cos=state.Zcos, coeff_sin=state.Zsin, modes=modes, trig=trig)
    Rv = vmec_realspace_synthesis_dzeta_phys(coeff_cos=state.Rcos, coeff_sin=state.Rsin, modes=modes, trig=trig)
    Zv = vmec_realspace_synthesis_dzeta_phys(coeff_cos=state.Zcos, coeff_sin=state.Zsin, modes=modes, trig=trig)
    if hasattr(state, "Lcos") and hasattr(state, "Lsin"):
        Lu = vmec_realspace_synthesis_dtheta(coeff_cos=state.Lcos, coeff_sin=state.Lsin, modes=modes, trig=trig)
        Lv = vmec_realspace_synthesis_dzeta_phys(coeff_cos=state.Lcos, coeff_sin=state.Lsin, modes=modes, trig=trig)
    else:
        Lu = None
        Lv = None
    return {
        "R": R,
        "Z": Z,
        "Ru": Ru,
        "Zu": Zu,
        "Rv": Rv,
        "Zv": Zv,
        "Lu": Lu,
        "Lv": Lv,
    }
