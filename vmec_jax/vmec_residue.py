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


@dataclass(frozen=True)
class VmecFsqSums:
    """Unnormalized sum-of-squares pieces used by `getfsq`-style scalars."""

    gcr2: float
    gcz2: float
    gcl2: float
    gcr2_blocks: dict[str, float]
    gcz2_blocks: dict[str, float]
    gcl2_blocks: dict[str, float]


def _constrain_m1_pair(*, gcr: Any, gcz: Any, lconm1: bool) -> tuple[Any, Any]:
    """VMEC's `constrain_m1` transform for the m=1 polar constraint.

    In `VMEC2000/Sources/General/residue.f90:constrain_m1_par`, VMEC optionally
    applies a polar constraint by rotating the (R,Z) force pair into
    (gcr+gcz, gcr-gcz)/sqrt(2) and then setting the second component to zero
    once close to convergence.

    For Step-10 parity work we apply the rotation when `lconm1=True` and always
    zero the constrained component (this matches the converged-equilibrium
    regime used for regression against bundled `wout_*.nc` files).
    """
    gcr = jnp.asarray(gcr)
    gcz = jnp.asarray(gcz)
    if bool(lconm1):
        osqrt2 = jnp.asarray(1.0 / np.sqrt(2.0), dtype=gcr.dtype)
        tmp = gcr
        gcr = osqrt2 * (gcr + gcz)
        gcz = osqrt2 * (tmp - gcz)
    gcz = jnp.zeros_like(gcz)
    return gcr, gcz


def vmec_apply_m1_constraints(
    *,
    frzl: TomnspsRZL,
    lconm1: bool = True,
) -> TomnspsRZL:
    """Apply VMEC's converged-iteration m=1 polar constraints to Fourier forces.

    VMEC calls `constrain_m1_par` on two block pairs prior to computing `fsqr/fsqz`:

    - 3D symmetric constraint: (R_ss, Z_cs)  -> enforce Z_cs(m=1) ≈ 0
    - asymmetric constraint:   (R_sc, Z_cc)  -> enforce Z_cc(m=1) ≈ 0

    See `VMEC2000/Sources/General/residue.f90`.
    """
    mpol = int(jnp.asarray(frzl.frcc).shape[1])
    if mpol <= 1:
        return frzl

    frss = frzl.frss
    fzcs = frzl.fzcs
    frsc = getattr(frzl, "frsc", None)
    fzcc = getattr(frzl, "fzcc", None)

    # 3D: constrain (rss,zcs) at m=1.
    if frss is not None and fzcs is not None:
        gcr, gcz = _constrain_m1_pair(gcr=frss[:, 1, :], gcz=fzcs[:, 1, :], lconm1=lconm1)
        frss = jnp.asarray(frss).at[:, 1, :].set(gcr)
        fzcs = jnp.asarray(fzcs).at[:, 1, :].set(gcz)

    # lasym: constrain (rsc,zcc) at m=1.
    if frsc is not None and fzcc is not None:
        gcr, gcz = _constrain_m1_pair(gcr=frsc[:, 1, :], gcz=fzcc[:, 1, :], lconm1=lconm1)
        frsc = jnp.asarray(frsc).at[:, 1, :].set(gcr)
        fzcc = jnp.asarray(fzcc).at[:, 1, :].set(gcz)

    return TomnspsRZL(
        frcc=frzl.frcc,
        frss=frss,
        fzsc=frzl.fzsc,
        fzcs=fzcs,
        flsc=frzl.flsc,
        flcs=frzl.flcs,
        frsc=frsc,
        frcs=getattr(frzl, "frcs", None),
        fzcc=fzcc,
        fzss=getattr(frzl, "fzss", None),
        flcc=getattr(frzl, "flcc", None),
        flss=getattr(frzl, "flss", None),
    )


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

    # VMEC's `bcovar` uses all surfaces (js=1..ns) in the `fnorm` sum.
    # Keep the axis included here to match the serial path in `bcovar.f`.
    denom_f = float(jnp.sum((guu_r12sq * wint3).astype(jnp.float64)))
    fnorm = 1.0 / (denom_f * (r2 * r2)) if denom_f != 0.0 else float("inf")

    bsubu = jnp.asarray(bc.bsubu)
    bsubv = jnp.asarray(bc.bsubv)
    lamscale = float(np.asarray(bc.lamscale))
    denom_L = float(jnp.sum(((bsubu * bsubu) + (bsubv * bsubv)) * wint3))
    fnormL = 1.0 / (denom_L * (lamscale * lamscale)) if denom_L != 0.0 else float("inf")

    r0scale = float(trig.r0scale)
    r1 = 1.0 / (2.0 * r0scale) ** 2
    return VmecForceNorms(fnorm=float(fnorm), fnormL=float(fnormL), r1=float(r1))


def vmec_fsq_from_tomnsps(
    *,
    frzl: TomnspsRZL,
    norms: VmecForceNorms,
    lconm1: bool = True,
    apply_m1_constraints: bool = True,
    include_edge: bool = False,
) -> VmecFsqScalars:
    """Compute (fsqr,fsqz,fsql) from VMEC-style tomnsps outputs.

    When `lasym=True`, VMEC also computes and includes the asymmetric blocks
    produced by `tomnspa`. In this repo those blocks (if present) are carried on
    the same dataclass as optional fields.
    """
    if bool(apply_m1_constraints):
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(lconm1))

    ns = int(jnp.asarray(frzl.frcc).shape[0])
    if ns <= 1:
        jsmax = ns
    else:
        jsmax = ns if bool(include_edge) else (ns - 1)

    gcr2 = jnp.sum(jnp.asarray(frzl.frcc)[:jsmax] ** 2)
    gcz2 = jnp.sum(jnp.asarray(frzl.fzsc)[:jsmax] ** 2)
    gcl2 = jnp.sum(jnp.asarray(frzl.flsc) ** 2)
    if frzl.frss is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frss)[:jsmax] ** 2)
    if frzl.fzcs is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcs)[:jsmax] ** 2)
    if frzl.flcs is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcs) ** 2)

    if getattr(frzl, "frsc", None) is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frsc)[:jsmax] ** 2)
    if getattr(frzl, "fzcc", None) is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcc)[:jsmax] ** 2)
    if getattr(frzl, "flcc", None) is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcc) ** 2)

    if getattr(frzl, "frcs", None) is not None:
        gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frcs)[:jsmax] ** 2)
    if getattr(frzl, "fzss", None) is not None:
        gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzss)[:jsmax] ** 2)
    if getattr(frzl, "flss", None) is not None:
        gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flss) ** 2)

    fsqr = norms.r1 * norms.fnorm * float(gcr2)
    fsqz = norms.r1 * norms.fnorm * float(gcz2)
    fsql = norms.fnormL * float(gcl2)
    return VmecFsqScalars(fsqr=float(fsqr), fsqz=float(fsqz), fsql=float(fsql))


def vmec_fsq_sums_from_tomnsps(
    *,
    frzl: TomnspsRZL,
    lconm1: bool = True,
    apply_m1_constraints: bool = True,
    include_edge: bool = False,
) -> VmecFsqSums:
    """Return the sum-of-squares components used in `vmec_fsq_from_tomnsps`.

    This is a diagnostic helper to attribute `fsqr/fsqz/fsql` differences to
    particular tomnsps/tomnspa blocks during the parity push.
    """
    if bool(apply_m1_constraints):
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(lconm1))

    ns = int(jnp.asarray(frzl.frcc).shape[0])
    if ns <= 1:
        jsmax = ns
    else:
        jsmax = ns if bool(include_edge) else (ns - 1)

    gcr2_blocks: dict[str, float] = {}
    gcz2_blocks: dict[str, float] = {}
    gcl2_blocks: dict[str, float] = {}

    def _add_block(d: dict[str, float], name: str, a: Any | None, *, slice_js: bool):
        if a is None:
            return
        arr = jnp.asarray(a)
        if slice_js:
            arr = arr[:jsmax]
        d[name] = float(jnp.sum(arr ** 2))

    # Symmetric blocks (tomnsps).
    _add_block(gcr2_blocks, "frcc", frzl.frcc, slice_js=True)
    _add_block(gcr2_blocks, "frss", frzl.frss, slice_js=True)
    _add_block(gcz2_blocks, "fzsc", frzl.fzsc, slice_js=True)
    _add_block(gcz2_blocks, "fzcs", frzl.fzcs, slice_js=True)
    _add_block(gcl2_blocks, "flsc", frzl.flsc, slice_js=False)
    _add_block(gcl2_blocks, "flcs", frzl.flcs, slice_js=False)

    # Asymmetric blocks (tomnspa, lasym=True).
    _add_block(gcr2_blocks, "frsc", getattr(frzl, "frsc", None), slice_js=True)
    _add_block(gcr2_blocks, "frcs", getattr(frzl, "frcs", None), slice_js=True)
    _add_block(gcz2_blocks, "fzcc", getattr(frzl, "fzcc", None), slice_js=True)
    _add_block(gcz2_blocks, "fzss", getattr(frzl, "fzss", None), slice_js=True)
    _add_block(gcl2_blocks, "flcc", getattr(frzl, "flcc", None), slice_js=False)
    _add_block(gcl2_blocks, "flss", getattr(frzl, "flss", None), slice_js=False)

    gcr2 = float(sum(gcr2_blocks.values()))
    gcz2 = float(sum(gcz2_blocks.values()))
    gcl2 = float(sum(gcl2_blocks.values()))
    return VmecFsqSums(
        gcr2=gcr2,
        gcz2=gcz2,
        gcl2=gcl2,
        gcr2_blocks=gcr2_blocks,
        gcz2_blocks=gcz2_blocks,
        gcl2_blocks=gcl2_blocks,
    )
