"""VMEC `residue/getfsq`-style scalar residuals (Step-10 parity work).

VMEC2000 reports scalar force residual measures:

  - ``fsqr`` : R-equation residual norm
  - ``fsqz`` : Z-equation residual norm
  - ``fsql`` : lambda-equation residual norm

Internally these are computed from Fourier-space force arrays produced by
``tomnsps`` and normalized by the force norms ``fnorm`` and ``fnormL``
computed in ``bcovar``.

This module implements the *scalar* pieces needed for Step-10 output-parity
tests against bundled VMEC2000 ``wout`` files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .vmec_tomnsp import TomnspsRZL, VmecTrigTables


@dataclass(frozen=True)
class VmecForceNorms:
    """VMEC normalization constants used by `getfsq` and `residue`."""

    fnorm: float
    fnormL: float
    r1: float  # 1/(2*r0scale)^2


@dataclass(frozen=True)
class VmecFsqScalars:
    fsqr: float
    fsqz: float
    fsql: float


def vmec_wint_from_trig(trig: VmecTrigTables, *, nzeta: int) -> jnp.ndarray:
    """Return VMEC's `wint` angular integration weights as a (ntheta3,nzeta) array."""
    w_theta = jnp.asarray(trig.cosmui3[:, 0]) / jnp.asarray(trig.mscale[0])
    return w_theta[:, None] * jnp.ones((int(nzeta),), dtype=w_theta.dtype)[None, :]


def vmec_force_norms_from_bcovar(*, bc, trig: VmecTrigTables, wout, s) -> VmecForceNorms:
    """Compute (fnorm, fnormL) using VMEC's bcovar normalization formulas.

    Notes
    -----
    VMEC uses:
      volume = hs * sum(vp(2:ns))
      r2     = max(wb, wp) / volume
      fnorm  = 1 / (sum(guu*r12^2*wint) * r2^2)
      fnormL = 1 / (sum((bsubu^2+bsubv^2)*wint) * lamscale^2)
    """
    s = np.asarray(s)
    if s.size < 2:
        return VmecForceNorms(fnorm=float("nan"), fnormL=float("nan"), r1=float("nan"))

    hs = float(s[1] - s[0])
    vp = np.asarray(wout.vp, dtype=float)
    volume = hs * float(np.sum(vp[1:]))  # vp(2:ns)

    wb = float(wout.wb)
    wp = float(wout.wp)
    r2 = max(wb, wp) / volume if volume != 0.0 else float("inf")

    # Angular integration weights (`wint`) as used throughout VMEC real-space routines.
    guu = jnp.asarray(bc.guu)
    wint = vmec_wint_from_trig(trig, nzeta=int(guu.shape[2]))
    wint3 = wint[None, :, :]

    # R/Z force norm: use the half-mesh metric element guu and R12 from the Jacobian.
    # (VMEC `bcovar.f` multiplies `guu` by `r12**2` just before forming `fnorm`.)
    r12 = jnp.asarray(bc.jac.r12)
    guu_r12sq = (guu * (r12 * r12)).astype(jnp.float64)

    # Exclude axis surface (js=1 in Fortran -> index 0 here). In VMEC, `wint` on
    # the axis is also zero, but we exclude explicitly for robustness.
    denom_f = float(jnp.sum((guu_r12sq[1:] * wint3).astype(jnp.float64)))
    fnorm = 1.0 / (denom_f * (r2 * r2)) if denom_f != 0.0 else float("inf")

    bsubu = jnp.asarray(bc.bsubu)
    bsubv = jnp.asarray(bc.bsubv)
    lamscale = float(np.asarray(bc.lamscale))
    denom_L = float(jnp.sum(((bsubu[1:] * bsubu[1:]) + (bsubv[1:] * bsubv[1:])) * wint3))
    fnormL = 1.0 / (denom_L * (lamscale * lamscale)) if denom_L != 0.0 else float("inf")

    r0scale = float(trig.r0scale)
    r1 = 1.0 / (2.0 * r0scale) ** 2
    return VmecForceNorms(fnorm=float(fnorm), fnormL=float(fnormL), r1=float(r1))


def vmec_fsq_from_tomnsps(*, frzl: TomnspsRZL, norms: VmecForceNorms) -> VmecFsqScalars:
    """Compute (fsqr,fsqz,fsql) from VMEC-style tomnsps outputs.

    When `lasym=True`, VMEC also computes and includes the asymmetric blocks
    produced by `tomnspa`. In this repo those blocks (if present) are carried on
    the same dataclass as optional fields.
    """
    gcr2 = jnp.sum(jnp.asarray(frzl.frcc) ** 2)
    gcz2 = jnp.sum(jnp.asarray(frzl.fzsc) ** 2)
    gcl2 = jnp.sum(jnp.asarray(frzl.flsc) ** 2)
    if frzl.frss is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frss) ** 2)
    if frzl.fzcs is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcs) ** 2)
    if frzl.flcs is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcs) ** 2)

    if getattr(frzl, "frsc", None) is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frsc) ** 2)
    if getattr(frzl, "fzcc", None) is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcc) ** 2)
    if getattr(frzl, "flcc", None) is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcc) ** 2)

    if getattr(frzl, "frcs", None) is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frcs) ** 2)
    if getattr(frzl, "fzss", None) is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzss) ** 2)
    if getattr(frzl, "flss", None) is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flss) ** 2)

    fsqr = norms.r1 * norms.fnorm * float(gcr2)
    fsqz = norms.r1 * norms.fnorm * float(gcz2)
    fsql = norms.fnormL * float(gcl2)
    return VmecFsqScalars(fsqr=float(fsqr), fsqz=float(fsqz), fsql=float(fsql))
