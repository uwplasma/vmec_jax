"""Effective-ripple diagnostics through the optional NEO_JAX backend."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "diagnostic_neo_config",
    "epsilon_effective_from_boozer",
    "epsilon_effective_from_wout",
]


def _neo_imports():
    try:
        from neo_jax import NeoConfig, run_neo
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "effective ripple requires NEO_JAX; install VMEX with "
            "`pip install vmex[neoclassical]`") from exc
    return NeoConfig, run_neo


def diagnostic_neo_config():
    """Return the bounded NEO resolution used by VMEX summary figures.

    This setting is intended for radial trends, not final transport numbers.
    Pass a ``neo_jax.NeoConfig`` with tighter tolerances to
    :func:`epsilon_effective_from_wout` for publication calculations.
    """
    NeoConfig, _ = _neo_imports()
    return NeoConfig(
        theta_n=16, phi_n=16, npart=8, multra=1, no_bins=12,
        nstep_per=4, nstep_min=20, nstep_max=40, acc_req=0.2,
        max_rational_field_periods=100000,
    )


def epsilon_effective_from_boozer(booz: Any, *, config=None):
    r"""Return ``(s, epsilon_eff**(3/2))`` from Boozer-coordinate data.

    ``booz`` may be a NEO_JAX ``BoozerData`` or a booz_xform-style mapping.
    The returned NEO quantity is the effective-ripple transport measure
    :math:`\epsilon_\mathrm{eff}^{3/2}`, conventionally named ``epstot``.
    JAX arrays remain differentiable when the mapping and selected NEO path
    are JAX-native.
    """
    NeoConfig, run_neo = _neo_imports()
    cfg = NeoConfig() if config is None else config
    result = run_neo(
        booz, config=cfg, use_jax=True, progress=False, jax_surface_scan=True)
    values = getattr(result, "eps_eff", None)
    if values is None:
        values = result.epsilon_effective
    if hasattr(booz, "es"):
        surfaces = booz.es
    elif isinstance(booz, dict) and "s_b" in booz:
        surfaces = booz["s_b"]
    else:
        surfaces = np.arange(np.asarray(values).size, dtype=float)
    return surfaces, values


def epsilon_effective_from_wout(
    wout,
    *,
    surfaces: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.95),
    mboz: int = 16,
    nboz: int = 12,
    config=None,
    clear_jax_caches: bool = True,
):
    """Compute ``epsilon_eff**(3/2)`` directly from an in-memory VMEX wout.

    VMEX performs the Boozer transform in memory and passes its arrays to
    NEO_JAX; no ``boozmn`` file is required. NEO_JAX currently represents the
    stellarator-symmetric cosine/sine convention, so ``LASYM`` wouts are
    rejected rather than silently dropping asymmetric harmonics. By default,
    completed VMEX executables are released before compiling NEO; this keeps
    end-of-run plotting responsive and memory-bounded. Set
    ``clear_jax_caches=False`` when preserving other warm JAX executables
    matters more than peak memory.
    """
    if bool(getattr(wout, "lasym", False)):
        raise NotImplementedError(
            "NEO_JAX's current BoozerData contract does not support LASYM harmonics")
    if clear_jax_caches:
        import gc

        import jax

        jax.clear_caches(); gc.collect()
    from booz_xform_jax import Booz_xform

    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    s_in = np.asarray(bx.s_in, dtype=float)
    indices = sorted({int(np.argmin(np.abs(s_in - float(s)))) for s in surfaces})
    bx.compute_surfs = indices
    bx.run()
    booz = {
        "nfp_b": int(bx.nfp), "ns_b": len(indices),
        "ixm_b": np.asarray(bx.xm_b), "ixn_b": np.asarray(bx.xn_b),
        "iota_b": np.asarray(bx.iota)[indices],
        "buco_b": np.asarray(bx.Boozer_I), "bvco_b": np.asarray(bx.Boozer_G),
        "rmnc_b": np.asarray(bx.rmnc_b), "zmns_b": np.asarray(bx.zmns_b),
        "pmns_b": -np.asarray(bx.numns_b), "bmnc_b": np.asarray(bx.bmnc_b),
        "s_b": np.asarray(bx.s_b),
    }
    return epsilon_effective_from_boozer(booz, config=config)
