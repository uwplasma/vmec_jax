#!/usr/bin/env python3
"""Build the self-consistent public QI free-boundary mgrid (sheet-current fit).

Constructs, deterministically and from public data only, an external field
whose flux surface is (to fit accuracy) the bundled ``input.nfp2_QI`` vacuum
boundary — the self-consistent public QI free-boundary case the review asked
for.  Recipe (every step was probe-validated against both codes):

1. Fixed-boundary vacuum solve of ``input.nfp2_QI`` (AM=0, CURTOR=0).
2. Winding surface: OUTWARD uniform normal offset of the boundary by
   ``1.2 x minor radius``, Fourier-smoothed (m <= 8, |n| <= 8 per period) —
   a raw offset folds in the concavities (VMEC's ``e_theta x e_phi`` points
   INWARD; the offset must flip it).
3. NESCOIL-style fit: divergence-free sheet current from a potential
   ``Phi = phi + sum c_mn sin(m theta - n nfp phi)`` (secular toroidal term
   plus 18x18 periodic modes), Tikhonov-regularised least squares on
   ``B . n = 0`` over the boundary.  Achieved ``max|Bn|/<|B|> = 2.5e-4``
   with field-direction alignment >= 0.998 against the equilibrium.
4. Scale by the boundary ``<|B|^2>`` ratio (the edge force balance is
   pointwise; a coarse flux integral left the field ~2% high, which froze
   BOTH codes' activation kick), then measure the field's actual toroidal
   flux on a fine grid — the free-boundary deck must use THAT as PHIEDGE.
5. Tabulate onto a 64 x 64 x 36 mgrid (kp = 36 keeps NZETA = 36 compatible)
   and write ``mgrid_qi_sheet.nc`` + the matching free-boundary deck.

Measured cross-validation (2026-07-28): VMEX converges the free case to
fsqr ~ 4e-6 within 200 iterations of activation (DELT = 0.9 works);
VMEC2000 converges with DELT <= 0.55 (its activation kick at DELT 0.9
death-spirals the time step); both settle on the QI boundary (r00
0.9270 / 0.9302 vs the fixed solve's 0.9266).  The deck ships DELT = 0.50:
a four-point DELT sweep on the gate ladder showed VMEX's free-multigrid
trajectory on x86-linux hits a non-finite force at 0.55 and 0.60 while
0.45 and 0.50 converge on x86-linux AND arm64-macos with the same vacuum
activation iteration (45) as VMEC2000 — 0.50 keeps a full step of margin
from the platform-sensitive stability edge and gives the tightest
two-code agreement (wb to ~1e-5 relative).  (The 0.55/0.60 non-finite
trajectories predate the vacuum-source ordering fix "Free boundary: never
feed NESTOR a sign-changed state"; post-fix all four swept DELT values
converge on both platforms — hosted x86-linux confirmed 0.55 dense/FFT
and 0.60 dense — the 0.55 ladder is pinned as a regression by
``tests/test_qi_sheet_gate.py``, and the shipped deck stays at the
doubly-safe 0.50.)

Calibration disclosure: the sheet-current AMPLITUDE is calibrated against
the VMEX fixed-boundary solve of this same deck — the boundary-``<|B|^2>``
scale of step 4 and the measured PHIEDGE written into the free deck both
derive from VMEX outputs (``res.state`` surface fields and the scaled
field's flux integral).  The free-boundary comparison is therefore
self-consistent rather than fully independent; the independent check is
that VMEC2000 then solves the SAME deck + mgrid byte-for-byte and lands
the same equilibrium.

Usage::

    python tools/build_qi_sheet_mgrid.py --outdir DIR [--offset-factor 1.2]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp

T0 = time.perf_counter()


def _log(msg: str) -> None:
    print(f"[{time.perf_counter()-T0:7.1f}s] {msg}", flush=True)


def build(outdir: Path, *, offset_factor: float = 1.2, mmax: int = 18,
          nmax: int = 18, smooth_m: int = 8, smooth_n: int = 8,
          nt_b: int = 48, np_b: int = 48, ir: int = 64, jz: int = 64,
          kp: int = 36) -> dict:
    jax.config.update("jax_enable_x64", True)
    from vmex.core.input import VmecInput
    from vmex.core.freeboundary_diff import surface_field_data_from_state
    from vmex.core.mgrid import MgridData, write_mgrid
    from vmex.core.multigrid import solve_multigrid

    deck_path = REPO / "examples" / "data" / "input.nfp2_QI"
    inp = VmecInput.from_file(str(deck_path))
    nfp = int(inp.nfp)
    _log("fixed-boundary vacuum solve of input.nfp2_QI...")
    res = solve_multigrid(inp, verbose=False, raise_on_max_iterations=False)
    assert bool(res.converged)
    xm, xn = np.asarray(res.xm), np.asarray(res.xn)
    rmnc_b = np.asarray(res.rmnc)[-1]
    zmns_b = np.asarray(res.zmns)[-1]

    def surf(TH, PH):
        a = xm[None, None, :] * TH[..., None] - xn[None, None, :] * PH[..., None]
        R = (rmnc_b[None, None, :] * np.cos(a)).sum(-1)
        Z = (zmns_b[None, None, :] * np.sin(a)).sum(-1)
        Rt = (-rmnc_b[None, None, :] * xm * np.sin(a)).sum(-1)
        Rp = (rmnc_b[None, None, :] * xn * np.sin(a)).sum(-1)
        Zt = (zmns_b[None, None, :] * xm * np.cos(a)).sum(-1)
        Zp = (-zmns_b[None, None, :] * xn * np.cos(a)).sum(-1)
        return R, Z, Rt, Rp, Zt, Zp

    # boundary targets
    th = np.linspace(0, 2 * np.pi, nt_b, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, np_b, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    R_b, Z_b, Rt, Rp, Zt, Zp = surf(TH, PH)
    ct, st = np.cos(PH), np.sin(PH)
    e_t = np.stack([Rt * ct, Rt * st, Zt], -1)
    e_p = np.stack([Rp * ct - R_b * st, Rp * st + R_b * ct, Zp], -1)
    n_hat = np.cross(e_t, e_p)
    n_hat /= np.linalg.norm(n_hat, axis=-1, keepdims=True)
    targets = jnp.asarray(np.stack([R_b * ct, R_b * st, Z_b], -1).reshape(-1, 3))
    n_t = jnp.asarray(n_hat.reshape(-1, 3))

    # smoothed OUTWARD-offset winding surface (see module docstring)
    minor = 0.5 * (R_b.max() - R_b.min())
    offset = offset_factor * minor
    nt_f, np_f = 96, 96
    THf, PHf = np.meshgrid(np.linspace(0, 2 * np.pi, nt_f, endpoint=False),
                           np.linspace(0, 2 * np.pi / nfp, np_f, endpoint=False),
                           indexing="ij")
    Rf, Zf, Rtf, Rpf, Ztf, Zpf = surf(THf, PHf)
    cf, sf = np.cos(PHf), np.sin(PHf)
    e_tf = np.stack([Rtf * cf, Rtf * sf, Ztf], -1)
    e_pf = np.stack([Rpf * cf - Rf * sf, Rpf * sf + Rf * cf, Zpf], -1)
    n_f = np.cross(e_tf, e_pf)
    n_f /= np.linalg.norm(n_f, axis=-1, keepdims=True)
    n_f = -n_f  # VMEC angle convention: e_theta x e_phi points INWARD
    X_off = Rf * cf + offset * n_f[..., 0]
    Y_off = Rf * sf + offset * n_f[..., 1]
    Z_off = Zf + offset * n_f[..., 2]
    R_off = np.hypot(X_off, Y_off)
    wm_l, wn_l, rc_l, zs_l = [], [], [], []
    for m in range(0, smooth_m + 1):
        for n in range(-smooth_n, smooth_n + 1):
            if m == 0 and n < 0:
                continue
            a = m * THf - n * nfp * PHf
            norm = nt_f * np_f * (1.0 if (m == 0 and n == 0) else 0.5)
            rc_l.append(np.sum(R_off * np.cos(a)) / norm)
            zs_l.append(np.sum(Z_off * np.sin(a)) / norm)
            wm_l.append(m)
            wn_l.append(n * nfp)
    wm = np.asarray(wm_l, float)
    wn = np.asarray(wn_l, float)
    rc_w = np.asarray(rc_l)
    zs_w = np.asarray(zs_l)

    # full-torus quadrature of the smooth winding surface, analytic tangents
    nt_w, np_q = 64, 64
    THq, PHq = np.meshgrid(np.linspace(0, 2 * np.pi, nt_w, endpoint=False),
                           np.linspace(0, 2 * np.pi / nfp, np_q, endpoint=False),
                           indexing="ij")
    P_l, et_l, ep_l, TH_L, PH_L = [], [], [], [], []
    for kper in range(nfp):
        PHk = PHq + 2 * np.pi * kper / nfp
        a = wm[None, None, :] * THq[..., None] - wn[None, None, :] * PHk[..., None]
        Rq = (rc_w[None, None, :] * np.cos(a)).sum(-1)
        Zq = (zs_w[None, None, :] * np.sin(a)).sum(-1)
        Rtq = (-rc_w[None, None, :] * wm * np.sin(a)).sum(-1)
        Rpq = (rc_w[None, None, :] * wn * np.sin(a)).sum(-1)
        Ztq = (zs_w[None, None, :] * wm * np.cos(a)).sum(-1)
        Zpq = (-zs_w[None, None, :] * wn * np.cos(a)).sum(-1)
        cq, sq = np.cos(PHk), np.sin(PHk)
        P_l.append(np.stack([Rq * cq, Rq * sq, Zq], -1))
        et_l.append(np.stack([Rtq * cq, Rtq * sq, Ztq], -1))
        ep_l.append(np.stack([Rpq * cq - Rq * sq, Rpq * sq + Rq * cq, Zpq], -1))
        TH_L.append(THq)
        PH_L.append(PHk)
    src = jnp.asarray(np.concatenate(P_l, 1).reshape(-1, 3))
    et_w = jnp.asarray(np.concatenate(et_l, 1).reshape(-1, 3))
    ep_w = jnp.asarray(np.concatenate(ep_l, 1).reshape(-1, 3))
    THs = np.concatenate(TH_L, 1).reshape(-1)
    PHs = np.concatenate(PH_L, 1).reshape(-1)
    dA_w = (2 * np.pi) * (2 * np.pi) / (nt_w * np_q * nfp)

    @jax.jit
    def bn_of(phi_t, phi_p):
        K = phi_t[:, None] * ep_w - phi_p[:, None] * et_w
        d = targets[:, None, :] - src[None, :, :]
        r3 = jnp.sum(d * d, -1) ** 1.5
        B = jnp.sum(jnp.cross(K[None], d) / r3[..., None], 1) * dA_w / (4 * np.pi)
        return jnp.sum(B * n_t, -1)

    modes = [(m, n) for m in range(0, mmax + 1)
             for n in range(-nmax, nmax + 1) if not (m == 0 and n <= 0)]
    _log(f"assembling {len(modes)+1} sheet-mode responses...")
    cols = [np.asarray(bn_of(jnp.zeros_like(jnp.asarray(THs)),
                             jnp.ones_like(jnp.asarray(PHs))))]
    for m, n in modes:
        a = m * THs - n * nfp * PHs
        cols.append(np.asarray(bn_of(jnp.asarray(m * np.cos(a)),
                                     jnp.asarray(-n * nfp * np.cos(a)))))
    A = np.stack(cols, 1)
    rhs, Am = -A[:, 0], A[:, 1:]
    best = None
    for lam_rel in (1e-9, 1e-12, 1e-15):
        lam = lam_rel * np.linalg.norm(Am, ord="fro") ** 2 / Am.shape[1]
        ci = np.linalg.solve(Am.T @ Am + lam * np.eye(Am.shape[1]), Am.T @ rhs)
        ri = A[:, 0] + Am @ ci
        if best is None or np.abs(ri).max() < np.abs(best[1]).max():
            best = (ci, ri)
    c, bn_res = best

    # full current distribution
    phi_t = np.zeros_like(THs)
    phi_p = np.ones_like(PHs)
    for (m, n), ci in zip(modes, c):
        a = m * THs - n * nfp * PHs
        phi_t += ci * m * np.cos(a)
        phi_p += ci * (-n * nfp) * np.cos(a)
    K = jnp.asarray(phi_t)[:, None] * ep_w - jnp.asarray(phi_p)[:, None] * et_w

    @jax.jit
    def _b_chunk(points):
        d = points[:, None, :] - src[None, :, :]
        r3 = jnp.sum(d * d, -1) ** 1.5
        return jnp.sum(jnp.cross(K[None], d) / r3[..., None], 1) * dA_w / (4 * np.pi)

    # The naive single-call broadcast is n_pts x n_src x 3 doubles plus the
    # cross/r3 intermediates -- ~20 GB at the 200x200 flux grid -- which
    # evicts a 16 GB hosted CI runner.  Fixed-size chunks (padded, so XLA
    # compiles exactly one shape) bound the peak below ~1 GB; each point's
    # source sum is unchanged.
    chunk = 2048

    def b_at(points):
        pts = np.asarray(points)
        out = np.empty_like(pts)
        for i in range(0, len(pts), chunk):
            blk = pts[i:i + chunk]
            n = len(blk)
            if n < chunk:
                blk = np.concatenate([blk, np.zeros((chunk - n, 3))], 0)
            out[i:i + n] = np.asarray(_b_chunk(jnp.asarray(blk)))[:n]
        return out

    # scale to the equilibrium's boundary <|B|^2>
    sd = surface_field_data_from_state(inp, res.state, nphi=48, ntheta=48)
    Beq = np.linalg.norm(np.asarray(sd.B_total), axis=0)
    gam = np.asarray(sd.gamma)
    Bs = np.asarray(b_at(jnp.asarray(
        np.stack([gam[0], gam[1], gam[2]], -1).reshape(-1, 3))))
    Bs_mag = np.linalg.norm(Bs, axis=-1).reshape(Beq.shape)
    scale = float(np.sqrt(np.mean(Beq ** 2) / np.mean(Bs_mag ** 2)))
    fit_metric = float(np.abs(bn_res).max() / Bs_mag.mean())
    # field-direction alignment: mean boundary cosine between the sheet field
    # and the equilibrium boundary field (sd.B_total components on axis 0;
    # Bs rows follow the same gamma flattening).  Scale-invariant.
    Beq_vec = np.asarray(sd.B_total).reshape(3, -1).T
    alignment = float(np.mean(np.sum(
        (Bs / np.linalg.norm(Bs, axis=-1, keepdims=True))
        * (Beq_vec / np.linalg.norm(Beq_vec, axis=-1, keepdims=True)), -1)))
    _log(f"fit max|Bn|/<|B|> = {fit_metric:.3e}; boundary-bsq scale {scale:.6f}")
    _log(f"sheet/equilibrium boundary field alignment <cos> = {alignment:.6f}")

    # accurate toroidal flux of the SCALED field -> the deck's PHIEDGE
    th_f = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    a0 = xm[None, :] * th_f[:, None]
    R0c = (rmnc_b[None, :] * np.cos(a0)).sum(-1)
    Z0c = (zmns_b[None, :] * np.sin(a0)).sum(-1)
    from matplotlib.path import Path as MplPath
    poly = MplPath(np.stack([R0c, Z0c], -1))
    rf = np.linspace(R0c.min(), R0c.max(), 200)
    zf = np.linspace(Z0c.min(), Z0c.max(), 200)
    RRf, ZZf = np.meshgrid(rf, zf, indexing="xy")
    mask = poly.contains_points(
        np.stack([RRf.ravel(), ZZf.ravel()], -1)).reshape(RRf.shape)
    Bf = np.asarray(b_at(jnp.asarray(
        np.stack([RRf.ravel(), np.zeros(RRf.size), ZZf.ravel()], -1)))) * scale
    phiedge = float((Bf[:, 1].reshape(RRf.shape) * mask).sum()
                    * (rf[1] - rf[0]) * (zf[1] - zf[0]))
    _log(f"measured toroidal flux (deck PHIEDGE): {phiedge:.8e}")

    # tabulate the mgrid
    margin = 0.10 * (R_b.max() - R_b.min())
    rmin, rmax = R_b.min() - margin, R_b.max() + margin
    zmin, zmax = Z_b.min() - margin, Z_b.max() + margin
    rg = np.linspace(rmin, rmax, ir)
    zg = np.linspace(zmin, zmax, jz)
    pg = np.arange(kp) * (2 * np.pi / nfp) / kp
    br = np.zeros((kp, jz, ir))
    bp = np.zeros((kp, jz, ir))
    bz = np.zeros((kp, jz, ir))
    for k in range(kp):
        RR, ZZ = np.meshgrid(rg, zg, indexing="xy")
        pts = np.stack([RR * np.cos(pg[k]), RR * np.sin(pg[k]), ZZ],
                       -1).reshape(-1, 3)
        B = np.asarray(b_at(jnp.asarray(pts))).reshape(jz, ir, 3) * scale
        br[k] = B[..., 0] * np.cos(pg[k]) + B[..., 1] * np.sin(pg[k])
        bp[k] = -B[..., 0] * np.sin(pg[k]) + B[..., 1] * np.cos(pg[k])
        bz[k] = B[..., 2]
        if (k + 1) % 12 == 0:
            _log(f"  tabulated {k+1}/{kp} planes")

    outdir.mkdir(parents=True, exist_ok=True)
    mgrid_path = outdir / "mgrid_qi_sheet.nc"
    write_mgrid(mgrid_path, MgridData(
        nextcur=1, kp=kp, jz=jz, ir=ir, nfp=nfp,
        rmin=float(rmin), rmax=float(rmax),
        zmin=float(zmin), zmax=float(zmax),
        coil_groups=("qi_sheet",), raw_coil_cur=(1.0,), mgrid_mode="S",
        br=br[None], bp=bp[None], bz=bz[None],
    ))
    _log(f"wrote {mgrid_path}")

    # matching free-boundary deck (DELT 0.50: VMEC2000's activation kick at
    # the native 0.9 collapses its time step, and the VMEX ladder on
    # x86-linux was non-finite at 0.55/0.60 before the vacuum-source
    # ordering fix — see the module docstring's measured DELT sweep; 0.50
    # is green on both platforms in both codes, and 0.55 is now pinned as
    # a regression by tests/test_qi_sheet_gate.py)
    import re
    deck = deck_path.read_text()
    deck = deck.replace(
        "&INDATA",
        "&INDATA\n  LFREEB = T\n  MGRID_FILE = 'mgrid_qi_sheet.nc'\n"
        "  EXTCUR = 1.0\n  NZETA = 36", 1)
    deck = re.sub(r"PHIEDGE *= *[0-9.eE+-]+", f"PHIEDGE = {phiedge:.10e}", deck)
    deck = re.sub(r"DELT *= *[0-9.eE+-]+", "DELT = 0.50", deck)
    (outdir / "input.qi_sheet_free").write_text(deck)
    _log(f"wrote {outdir / 'input.qi_sheet_free'}")
    return {"fit_metric": fit_metric, "scale": scale, "phiedge": phiedge,
            "alignment": alignment, "mgrid": str(mgrid_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--offset-factor", type=float, default=1.2)
    args = ap.parse_args()
    meta = build(args.outdir, offset_factor=args.offset_factor)
    print("META", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
