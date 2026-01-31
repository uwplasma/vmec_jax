"""VMEC profile evaluation (step-3).

This module implements a small subset of VMEC2000's profile logic as found in
`profile_functions.f`, starting with the common `power_series` parameterization.

We intentionally keep this code:
- dependency-light (NumPy for parsing, JAX-compatible math via ``vmec_jax._compat.jnp``)
- pure (no I/O, no global state)
- easy to extend later (splines, pedestal, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from ._compat import jnp
from .namelist import InData


def _as_float_list(x: Any) -> list[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    return [float(x)]


def _coeff_array(
    coeffs: Sequence[float],
    *,
    nmin: int = 21,
    dtype: Any = np.float64,
) -> Any:
    """Convert a Python list of coefficients into a dense 1D array.

    VMEC's default arrays are 0:20 (21 coefficients). We pad with zeros to at
    least ``nmin`` to match typical VMEC behavior and keep shapes stable.
    """
    coeffs = list(coeffs)
    n = max(int(nmin), len(coeffs))
    out = np.zeros((n,), dtype=dtype)
    if coeffs:
        out[: len(coeffs)] = np.asarray(coeffs, dtype=dtype)
    return jnp.asarray(out)


def _lower(s: Any, default: str) -> str:
    if s is None:
        return default
    if isinstance(s, list):
        s = s[0] if s else default
    return str(s).strip().lower()


def _power_series(coeffs, x):
    """Evaluate Σ_i coeffs[i] * x**i using Horner (coeffs in ascending order)."""
    coeffs = jnp.asarray(coeffs)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x, dtype=coeffs.dtype)
    # Horner: iterate from high order to low order.
    for i in range(int(coeffs.shape[0]) - 1, -1, -1):
        y = y * x + coeffs[i]
    return y


def _pcurr_power_series_ip(ac, x):
    """VMEC `pcurr` default: parameterize I'(x) as a power series, return I(x).

    In VMEC2000 (`profile_functions.f`, case 'power_series'):
      I'(x) = Σ_i ac(i) * x**i
      I(x)  = ∫_0^x I'(t) dt = Σ_i ac(i)/(i+1) * x**(i+1)
    """
    ac = jnp.asarray(ac)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x, dtype=ac.dtype)
    for i in range(int(ac.shape[0]) - 1, -1, -1):
        y = y * x + ac[i] / (i + 1)
    return x * y


@dataclass(frozen=True)
class ProfileInputs:
    """Profile-related inputs extracted from &INDATA."""

    pmass_type: str
    piota_type: str
    pcurr_type: str

    am: Any  # (n_am,)
    ai: Any  # (n_ai,)
    ac: Any  # (n_ac,)

    pres_scale: float
    bloat: float
    spres_ped: float
    lrfp: bool
    ncurr: int


def profiles_from_indata(indata: InData) -> ProfileInputs:
    """Extract and normalize profile inputs from an :class:`~vmec_jax.namelist.InData`."""
    pmass_type = _lower(indata.get("PMASS_TYPE", "power_series"), "power_series")
    piota_type = _lower(indata.get("PIOTA_TYPE", "power_series"), "power_series")
    pcurr_type = _lower(indata.get("PCURR_TYPE", "power_series"), "power_series")

    am = _coeff_array(_as_float_list(indata.get("AM", [])))
    ai = _coeff_array(_as_float_list(indata.get("AI", [])))
    ac = _coeff_array(_as_float_list(indata.get("AC", [])))

    pres_scale = float(indata.get_float("PRES_SCALE", 1.0))
    bloat = float(indata.get_float("BLOAT", 1.0))
    spres_ped = float(abs(indata.get_float("SPRES_PED", 1.0)))
    lrfp = bool(indata.get_bool("LRFP", False))
    ncurr = int(indata.get_int("NCURR", 0))

    return ProfileInputs(
        pmass_type=pmass_type,
        piota_type=piota_type,
        pcurr_type=pcurr_type,
        am=am,
        ai=ai,
        ac=ac,
        pres_scale=pres_scale,
        bloat=bloat,
        spres_ped=spres_ped,
        lrfp=lrfp,
        ncurr=ncurr,
    )


def eval_profiles(cfg: ProfileInputs | InData, s_grid) -> Dict[str, Any]:
    """Evaluate VMEC profiles on a radial grid.

    Parameters
    ----------
    cfg:
        Either a :class:`ProfileInputs` or an :class:`~vmec_jax.namelist.InData`.
    s_grid:
        1D array of `s` values in [0, 1]. (In VMEC, s is normalized toroidal flux.)

    Returns
    -------
    dict
        Keys include:
        - ``pressure`` (Pa)
        - ``iota`` (dimensionless) if AI present
        - ``current`` (VMEC's I(s) function) if AC present
        - ``ncurr`` (0: iota-driven, 1: current-driven)

    Notes
    -----
    - For now we implement only the common ``power_series`` profile type.
    - Pressure is returned in **Pa** (VMEC stores pressure internally in mu0*Pa).
    """
    if isinstance(cfg, InData):
        cfg = profiles_from_indata(cfg)

    s = jnp.asarray(s_grid)
    x = jnp.minimum(jnp.abs(s * cfg.bloat), 1.0)

    out: Dict[str, Any] = {"ncurr": int(cfg.ncurr)}

    # --- Pressure (pmass) ---
    if cfg.pmass_type != "power_series":
        raise NotImplementedError(f"pmass_type={cfg.pmass_type!r} not implemented (only 'power_series')")
    # VMEC pmass() returns mu0 * pres_scale * poly(x). We return Pa.
    p_pa = cfg.pres_scale * _power_series(cfg.am, x)
    if cfg.spres_ped < 1.0:
        x_ped = jnp.minimum(jnp.abs(jnp.asarray(cfg.spres_ped) * cfg.bloat), 1.0)
        p_ped = cfg.pres_scale * _power_series(cfg.am, x_ped)
        p_pa = jnp.where(s > cfg.spres_ped, p_ped, p_pa)
    out["pressure"] = p_pa

    # --- Iota / q (piota) ---
    if cfg.ai is not None and int(jnp.size(cfg.ai)) > 0:
        if cfg.piota_type != "power_series":
            raise NotImplementedError(f"piota_type={cfg.piota_type!r} not implemented (only 'power_series')")
        iota = _power_series(cfg.ai, x)
        if cfg.lrfp:
            iota = jnp.where(iota != 0, 1.0 / iota, jnp.asarray(np.inf, dtype=iota.dtype))
        out["iota"] = iota

    # --- Toroidal current function (pcurr) ---
    if cfg.ac is not None and int(jnp.size(cfg.ac)) > 0:
        if cfg.pcurr_type != "power_series":
            raise NotImplementedError(f"pcurr_type={cfg.pcurr_type!r} not implemented (only 'power_series')")
        out["current"] = _pcurr_power_series_ip(cfg.ac, x)

    return out
