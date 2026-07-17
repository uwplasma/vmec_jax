#!/usr/bin/env python
"""Cold-start single-stage vs sequential two-stage stellarator optimization.

A genuine COLD-START benchmark, aligned with the protocol of R. Jorge et al.,
PPCF 65, 074003 (2023), arXiv:2302.10622 (single-stage ``J = J_plasma +
w * J_coils``) and the cold-start convention of arXiv:2406.07830: both
pipelines start from the SAME crude seeds -- a near-circular torus (nfp=2,
R0=1 m, a=0.2 m) and four equally spaced circular coils -- and are compared
on the SAME final metrics.

Two pipelines
-------------
* **Two-stage** (the classical sequential approach):

  - ``--phase stage1``: fixed-boundary quasi-axisymmetry optimization of the
    plasma boundary alone (the ``examples/optimization/QA_optimization.py``
    recipe: QS ratio residual + aspect 6 + mean-iota 0.42, staged ``max_mode``
    ladder, implicit-adjoint least squares).
  - ``--phase stage2``: coil-only optimization on the *frozen* stage-1
    boundary: minimize the virtual-casing normal-field residual
    ``<(B_plasma + B_coil).n>^2`` over the coil curve dofs + currents, plus
    coil length/curvature regularization.  No equilibrium re-solves.

* **Single-stage** (``--phase single``): ONE ``jax.value_and_grad`` over the
  concatenated dof vector [boundary Fourier modes (m <= 2, |n| <= 2), coil
  curve dofs, currents[1:]] of

  ``J = sum(qs_residuals^2) + w_bn * <(B.n)^2> + 1e-2 (A - 6)^2
      + (mean_iota - 0.42)^2 + coil penalties``

  threading the implicit-differentiation adjoint of the equilibrium AND the
  differentiable virtual casing AND Biot-Savart off the ESSOS coil filaments.

* **Evaluate** (``--phase evaluate``): the honest comparison table -- for both
  results, a fixed-boundary re-solve of the final boundary, the wout-based QS
  ratio residual, the coil-field normal-field errors ``<|B.n|>/<B>`` and
  ``max|B.n|/<B>`` on that boundary, and per-coil length / max curvature.
  Writes ``comparison.json`` + a README-ready markdown table.

Cases: ``--case vacuum`` and ``--case beta`` (parabolic pressure calibrated to
``<beta> ~ 1.5%`` -- per arXiv:2302.10622 the published single-stage results
are vacuum-only, so the finite-beta column is the novel claim).

All outputs land in ``output_single_stage_vs_two_stage/{vacuum,beta}/``; the
phases are resumable (each loads its inputs from the previous phase's files),
so long runs can be split across sessions/machines.  Requires the optional
``essos`` and ``virtual_casing_jax`` dependencies.  Honors
``VMEC_JAX_EXAMPLES_CI=1`` (tiny grids and budgets) for the smoke test.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.optimize

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # exact gradients need float64

import vmec_jax as vj
from vmec_jax.core import freeboundary_diff as FBD
from vmec_jax.core import implicit as im
from vmec_jax.core import optimize as opt

# --------------------------- parameters ------------------------------------
DATA = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA / "input.minimal_seed_nfp2"   # circular torus, nfp=2
OUT_ROOT = Path("output_single_stage_vs_two_stage")

SEED_PERTURBATION = 0.05     # rotating-ellipse kick amplitude (see build_seed_input)
ASPECT_TARGET = 6.0
IOTA_TARGET = 0.42
QS_SURFACES = np.linspace(0.1, 1.0, 6)     # optimization surfaces (QA helicity)
EVAL_SURFACES = np.linspace(0.1, 1.0, 10)  # reporting metric (house convention)
HELICITY_M, HELICITY_N = 1, 0

# coils: the simsopt-canonical cold start (4 base coils, order 5, r=0.5)
N_COILS, ORDER, R_MAJOR, R_COIL, NSEG = 4, 8, 1.0, 0.5, 60
CURRENT0 = 2.7e5             # uniform seed currents [A]; current[0] frozen
L_MAX = 2.0 * np.pi * R_COIL * 1.4          # per-coil length budget [m]
KAPPA_MAX = 5.0                             # per-coil curvature budget [1/m]
W_LEN, W_CURV = 1.0e-1, 1.0e-3              # coil regularization weights

# single-stage objective weights (w_bn overridable via --w-bn)
W_QS, W_ASP, W_IOTA = 1.0, 1.0e-2, 1.0
BETA_TARGET_PCT = 1.5                       # --case beta: target <beta> [%]

CI = os.environ.get("VMEC_JAX_EXAMPLES_CI") == "1"
if CI:  # smoke budget: coarse everything, a handful of iterations
    NS, NPHI, NTHETA = 12, 8, 8
    MODE_LADDER, MAX_NFEV = (1,), 10
    MAXITER_STAGE2, MAXITER_SINGLE, MAXITER_POLISH = 3, 3, 3
    SOLVE = dict(ftol=1e-9, max_iterations=4000)
    LS_FTOL = 1e-4
else:
    # Fast-showcase budget: ONE vmec grid (ns=31), converged-looking force
    # residual (ftol 1e-12), iteration cap 3000 (finite beta gets more via
    # solve_kwargs), ESS single-call stage 1 at max_mode 3, and a hard 50-
    # iteration cap on the single-stage descent with a 1e-3 relative stop --
    # the point is a minutes-scale reproducible comparison, not a polish run.
    NS, NPHI, NTHETA = 31, 16, 16
    MODE_LADDER, MAX_NFEV = (3,), 300          # single ESS call, max_mode 3
    MAXITER_STAGE2, MAXITER_SINGLE, MAXITER_POLISH = 150, 50, 25
    SOLVE = dict(ftol=1e-12, max_iterations=3000)
    LS_FTOL = 1e-6
# scipy L-BFGS-B "ftol" tests the decrease against max(|f|, 1); with J < 1
# that makes 1e-3 an ABSOLUTE dJ bar which fired after 6 iterations (iota
# still at 0.02 of the 0.42 target).  1e-6 keeps the 50-iteration cap as
# the binding constraint, which is the intent of the fast budget.
SINGLE_REL_FTOL = 1e-6

def solve_kwargs(case: str) -> dict:
    """Per-case solver settings: vacuum = one grid; finite beta = multigrid.

    The pressure-loaded near-circular seed limit-cycles at fsq ~ 1e-6 on ANY
    single grid tried (ns 25/31, delt 0.4-0.9, beta 0.75-1.5%) -- the hunting
    Shafranov axis never lets the force residual reach 1e-12 -- while the
    multigrid ns-ladder converges it in seconds within the same 3000-iteration
    cap.  Multigrid is AD-safe (coarse stages are stop-gradient initializers).
    """
    kw = dict(SOLVE)
    if case == "beta":
        kw["multigrid"] = True
    return kw


# single-stage boundary dof set: m <= 2, |n| <= 2 (same convention as
# optimize._dof_modes -- m=0 keeps n >= 1 only, RBC(0,0) fixed)
SS_MAX_MODE = 3
SS_MODES = [(m, n) for m in range(0, SS_MAX_MODE + 1)
            for n in range(-SS_MAX_MODE, SS_MAX_MODE + 1)
            if not (m == 0 and n <= 0)]

# per-dof scaling (L-BFGS-B works in u, x = x0 + D*u, bounded steps -- unscaled
# steps produce self-intersecting trial boundaries, VmecJacobianError)
D_BOUNDARY, D_COIL, D_CURRENT = 0.02, 0.05, 2.0e4
U_BOUND = 3.0


# --------------------------- seeds -----------------------------------------
def build_seed_input(pres_scale: float) -> vj.VmecInput:
    """The perturbed circular-torus seed at this script's solve budget."""
    inp = vj.VmecInput.from_file(str(INPUT_FILE))
    # The exact circular torus is a saddle point of the QS+iota objective
    # (see QA_optimization.py) -- a small helical kick breaks the tie.
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    # rotating-ellipse kick (OPPOSITE signs): a same-sign kick is a helical
    # shift with iota ~ 0, and an iota ~ 0 seed cannot confine the finite-beta
    # pressure (measured: fsq limit-cycles at ~1e-6 forever).  +/-0.05 gives
    # iota_edge ~ 0.12 -- still an obviously crude seed.
    rbc[inp.ntor + 1, 1] += SEED_PERTURBATION
    zbs[inp.ntor + 1, 1] -= SEED_PERTURBATION
    kw = dict(rbc=rbc, zbs=zbs, ns_array=[NS],
              niter_array=[SOLVE["max_iterations"]],
              ftol_array=[SOLVE["ftol"]], lfreeb=False)
    if pres_scale > 0:  # p(s) = pres_scale * (1 - s)
        kw.update(pmass_type="power_series",
                  am=[1.0, -1.0] + [0.0] * 19, pres_scale=pres_scale)
    return dataclasses.replace(inp, **kw)


def get_seed_pres_scale(case: str, out: Path) -> float:
    """0 for vacuum; for beta, calibrate pres_scale to <beta> ~ 1.5% (cached)."""
    if case == "vacuum":
        return 0.0
    cache = out / "seed.json"
    if cache.exists():
        return float(json.loads(cache.read_text())["pres_scale"])
    ps = 1000.0
    for it in range(2):  # one calibration re-solve (beta ~ linear in pres_scale)
        inp = build_seed_input(ps)
        eq = opt.solve_equilibrium(inp)
        beta = 100.0 * float(eq.wout.betatotal)
        print(f"[seed] calibration {it}: pres_scale={ps:.1f} -> <beta>={beta:.3f}%")
        if abs(beta - BETA_TARGET_PCT) < 0.05 * BETA_TARGET_PCT:
            break
        ps *= BETA_TARGET_PCT / max(beta, 1e-12)
    cache.write_text(json.dumps({"pres_scale": ps, "beta_pct": beta}, indent=2))
    return ps


def seed_coils() -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Equally spaced circular seed coils -> (curve dofs, currents, nfp)."""
    from essos.coils import CreateEquallySpacedCurves

    inp = vj.VmecInput.from_file(str(INPUT_FILE))
    nfp = int(inp.nfp)
    curves = CreateEquallySpacedCurves(N_COILS, ORDER, R_MAJOR, R_COIL,
                                       n_segments=NSEG, nfp=nfp, stellsym=True)
    return jnp.asarray(curves.dofs), jnp.full((N_COILS,), CURRENT0), nfp


# ---------------------- differentiable coil field --------------------------
def coil_field(cdofs, currents, nfp: int):
    """xyz(...,3) -> B(...,3) callable, differentiable in curve dofs + currents.

    Builds the ESSOS ``Curves``/``Coils`` INSIDE the trace and hand-rolls the
    filamentary Biot-Savart (validated == ``essos.fields.BiotSavart`` to 1e-16).
    """
    from essos.coils import Coils, Curves

    coils = Coils(Curves(cdofs, NSEG, nfp, True), currents)
    gamma, gdash = jnp.asarray(coils.gamma), jnp.asarray(coils.gamma_dash)
    cur = jnp.asarray(coils.currents)

    def B(pts):
        def one(pt):
            dR = pt - gamma                                       # (nc, nseg, 3)
            norm = jnp.linalg.norm(dR, axis=-1, keepdims=True)
            integrand = jnp.cross(gdash, dR) / norm ** 3
            per_coil = jnp.mean(integrand, axis=1)                # (nc, 3)
            return 1e-7 * jnp.sum(cur[:, None] * per_coil, axis=0)
        return jax.vmap(one)(pts.reshape(-1, 3)).reshape(pts.shape)

    return B


def coil_penalties(cdofs, nfp: int):
    """(length, curvature) penalties on the BASE coils.

    Length: quadratic penalty above the per-coil budget L_MAX (one-sided --
    the essos ``loss_coil_length`` is two-sided ``(L/Lmax - 1)^2``, which would
    *inflate* short coils, so it is hand-rolled here).  Curvature: the essos
    ``loss_coil_curvature`` form, arclength-weighted quadratic excess above
    KAPPA_MAX.  Both use ``Curves.gamma_dash``/``curvature`` (parametrized
    over [0,1], so length = mean |gamma_dash|; verified == ``Curves.length``).
    """
    from essos.coils import Curves

    curves = Curves(cdofs, NSEG, nfp, True)
    gdash = jnp.asarray(curves.gamma_dash)[:N_COILS]        # base coils only
    speed = jnp.linalg.norm(gdash, axis=-1)                 # (nc, nseg)
    length = jnp.mean(speed, axis=-1)                       # per-coil length
    j_len = jnp.sum(jnp.maximum(length - L_MAX, 0.0) ** 2)
    kappa = jnp.asarray(curves.curvature)[:N_COILS]         # (nc, nseg)
    j_curv = jnp.sum(jnp.mean(jnp.maximum(kappa - KAPPA_MAX, 0.0) ** 2 * speed,
                              axis=-1))
    return j_len, j_curv


def coil_metrics(cdofs, nfp: int) -> dict:
    """Per-coil length and max curvature of the base coils (reporting)."""
    from essos.coils import Curves

    curves = Curves(jnp.asarray(cdofs), NSEG, nfp, True)
    length = np.asarray(curves.length)[:N_COILS]
    kappa = np.asarray(curves.curvature)[:N_COILS]
    return dict(lengths=[float(v) for v in length],
                max_curvatures=[float(v) for v in kappa.max(axis=-1)],
                L_max=float(L_MAX), kappa_max=float(KAPPA_MAX))


# ------------------------------ stage 1 ------------------------------------
def phase_stage1(case: str, out: Path, args) -> None:
    """Fixed-boundary QA optimization from the circular seed (two-stage step 1)."""
    ps = get_seed_pres_scale(case, out)
    inp = build_seed_input(ps)
    qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, HELICITY_M, HELICITY_N)
    terms = [(qs, 0.0, 1.0),
             (opt.aspect_ratio, ASPECT_TARGET, 1.0),
             (opt.mean_iota, IOTA_TARGET, 10.0)]

    eq = opt.solve_equilibrium(inp)
    print(f"[stage1:{case}] seed: QS={float(qs.total(eq)):.4e} "
          f"aspect={float(opt.aspect_ratio(eq.state, eq.runtime)):.3f} "
          f"iota={float(opt.mean_iota(eq.state, eq.runtime)):.4f}")

    t0 = time.time()
    for max_mode in MODE_LADDER:
        print(f"\n===== stage1 max_mode = {max_mode} =====")
        result = opt.least_squares(terms, inp, max_mode=max_mode,
                                   jac="implicit", use_ess=True, verbose=1,
                                   max_nfev=MAX_NFEV, ftol=LS_FTOL, xtol=1e-10)
        inp = result.input
    eq = result.equilibrium or opt.solve_equilibrium(inp)
    wall = time.time() - t0

    inp.to_indata(out / "input.stage1")
    vj.write_wout(out / "wout_stage1.nc", eq.wout)
    metrics = dict(
        qs_total=float(qs.total(eq)),
        aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
        mean_iota=float(opt.mean_iota(eq.state, eq.runtime)),
        ladder=list(MODE_LADDER), max_nfev=MAX_NFEV, wall_s=wall)
    (out / "stage1.json").write_text(json.dumps(metrics, indent=2))
    print(f"[stage1:{case}] done in {wall:.0f}s: QS={metrics['qs_total']:.4e} "
          f"aspect={metrics['aspect']:.3f} iota={metrics['mean_iota']:.4f}")
    print(f"wrote {out / 'input.stage1'}, {out / 'wout_stage1.nc'}")


# ------------------------------ stage 2 ------------------------------------
def phase_stage2(case: str, out: Path, args) -> None:
    """Coil-only optimization on the frozen stage-1 boundary (two-stage step 2)."""
    stage1_deck = out / "input.stage1"
    if not stage1_deck.exists():
        raise SystemExit(f"missing {stage1_deck} -- run --phase stage1 first")
    inp = load_boundary_input(stage1_deck, case, out)
    p = im.params_from_input(inp)
    sol = im.run(inp, p, **solve_kwargs(case))

    # Build the differentiable free-boundary problem ONCE from the solved
    # stage-1 state: the boundary (and its virtual-casing plasma field) is a
    # constant of this phase -- no equilibrium re-solves, only Biot-Savart.
    sd = FBD.surface_field_data_from_state(inp, sol.state, nphi=NPHI, ntheta=NTHETA)
    prob = FBD.FreeBoundaryDiffProblem.from_surface_data(sd, digits=4)

    cdofs0, cur0, nfp = seed_coils()
    ncd = cdofs0.size

    def unpack(x):
        cdofs = jnp.asarray(x[:ncd]).reshape(cdofs0.shape)
        cur = jnp.concatenate([cur0[:1], jnp.asarray(x[ncd:])])  # current[0] frozen
        return cdofs, cur

    def objective(x):
        cdofs, cur = unpack(x)
        j_bn = prob.bnormal_objective(coil_field(cdofs, cur, nfp))
        j_len, j_curv = coil_penalties(cdofs, nfp)
        J = j_bn + W_LEN * j_len + W_CURV * j_curv
        return J, (j_bn, j_len, j_curv)

    x0 = np.concatenate([np.asarray(cdofs0).ravel(), np.asarray(cur0[1:])])
    D = np.concatenate([np.full(ncd, D_COIL), np.full(N_COILS - 1, D_CURRENT)])
    vg = jax.value_and_grad(objective, has_aux=True)
    hist: list[float] = []

    def fun(u):
        (J, aux), g = vg(jnp.asarray(x0 + D * u))
        hist.append(float(J))
        if len(hist) % 10 == 1:
            print(f"  eval {len(hist):4d}: J={float(J):.6e} "
                  f"bn={float(aux[0]):.3e} len={float(aux[1]):.3e} "
                  f"curv={float(aux[2]):.3e}")
        return float(J), np.asarray(g, dtype=float) * D

    t0 = time.time()
    res = scipy.optimize.minimize(
        fun, np.zeros_like(x0), jac=True, method="L-BFGS-B",
        bounds=[(-U_BOUND, U_BOUND)] * x0.size,
        options={"maxiter": args.maxiter_stage2, "ftol": 1e-14, "gtol": 1e-12})
    wall = time.time() - t0

    xf = x0 + D * res.x
    (Jf, (jbnf, jlenf, jcurvf)), _ = vg(jnp.asarray(xf))
    cdofs_f, cur_f = unpack(xf)
    np.savez(out / "coils_stage2.npz",
             cdofs=np.asarray(cdofs_f), currents=np.asarray(cur_f),
             nfp=nfp, n_segments=NSEG)
    metrics = dict(J0=hist[0], Jf=float(Jf), j_bn=float(jbnf),
                   j_len=float(jlenf), j_curv=float(jcurvf),
                   nit=int(res.nit), nev=len(hist), wall_s=wall,
                   **coil_metrics(cdofs_f, nfp))
    (out / "stage2.json").write_text(json.dumps(metrics, indent=2))
    print(f"[stage2:{case}] J {hist[0]:.4e} -> {float(Jf):.4e} "
          f"({hist[0] / max(float(Jf), 1e-30):.1f}x) in {res.nit} iters, "
          f"{len(hist)} evals, {wall:.0f}s")
    print(f"wrote {out / 'coils_stage2.npz'}")


# ---------------------------- single stage ----------------------------------
def phase_single(case: str, out: Path, args, *, warm: bool = False) -> None:
    """Joint boundary+coil optimization by one exact gradient.

    ``warm=False`` (``--phase single``): the cold start -- circular-torus
    boundary + circular coils.  ``warm=True`` (``--phase polish``): the
    literature's canonical use (arXiv:2302.10622 runs single-stage as a
    "stage 3"): start FROM the two-stage result (stage-1 boundary + stage-2
    coils) and let boundary and coils co-adapt -- fixing the coil<->plasma
    inconsistency that frozen-boundary stage 2 cannot, at held quasisymmetry.
    """
    label = "polish" if warm else "single"
    ps = get_seed_pres_scale(case, out)
    if warm:
        for f in (out / "input.stage1", out / "coils_stage2.npz"):
            if not f.exists():
                raise SystemExit(f"missing {f} -- run stage1/stage2 first")
        inp = load_boundary_input(out / "input.stage1", case, out)
        dat = np.load(out / "coils_stage2.npz")
        cdofs0 = jnp.asarray(dat["cdofs"])
        cur0 = jnp.asarray(dat["currents"])
        nfp = int(dat["nfp"])
    else:
        inp = build_seed_input(ps)
        cdofs0, cur0, nfp = seed_coils()
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)
    nb = len(SS_MODES)
    qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, HELICITY_M, HELICITY_N)
    ncd = cdofs0.size

    # Freeze the virtual-casing precision plan at the seed so the plasma field
    # stays differentiable in the (moving) boundary.
    sol_seed = im.run(inp, p0, **solve_kwargs(case))
    sd_seed = FBD.surface_field_data_from_state(inp, sol_seed.state,
                                                nphi=NPHI, ntheta=NTHETA)
    plan = FBD.plan_vc_precision(sd_seed, digits=4)

    def unpack(x):
        rbc, zbs = p0.rbc, p0.zbs
        for i, (m, n) in enumerate(SS_MODES):
            rbc = rbc.at[ntor + n, m].set(x[i])
            zbs = zbs.at[ntor + n, m].set(x[nb + i])
        params = dataclasses.replace(p0, rbc=rbc, zbs=zbs)
        k = 2 * nb
        cdofs = jnp.asarray(x[k:k + ncd]).reshape(cdofs0.shape)
        cur = jnp.concatenate([cur0[:1], jnp.asarray(x[k + ncd:])])
        return params, cdofs, cur

    # The adjoint GMRES at its 1e-11 default dominates each gradient (audit:
    # one warm gradient ~ 10-15 forward solves); 1e-8 is ample for L-BFGS-B.
    run_kw = dict(solve_kwargs(case), adjoint_tol=1e-8)

    def objective(x):
        params, cdofs, cur = unpack(x)
        sol = im.run(inp, params, **run_kw)
        rt = sol.runtime          # the SolverRuntime im.run built (Item I.6)
        j_qs = jnp.sum(qs.residuals_state(sol.state, rt) ** 2)
        sd = FBD.surface_field_data_from_state(inp, sol.state,
                                               nphi=NPHI, ntheta=NTHETA)
        prob = FBD.FreeBoundaryDiffProblem.from_surface_data(
            sd, digits=4, precision=plan)
        j_bn = prob.bnormal_objective(coil_field(cdofs, cur, nfp))
        j_asp = (opt.aspect_ratio(sol.state, rt) - ASPECT_TARGET) ** 2
        j_iota = (opt.mean_iota(sol.state, rt) - IOTA_TARGET) ** 2
        j_len, j_curv = coil_penalties(cdofs, nfp)
        J = (W_QS * j_qs + args.w_bn * j_bn + W_ASP * j_asp + W_IOTA * j_iota
             + W_LEN * j_len + W_CURV * j_curv)
        return J, (j_qs, j_bn, j_asp, j_iota, j_len, j_curv)

    r0 = np.array([float(np.asarray(p0.rbc)[ntor + n, m]) for m, n in SS_MODES])
    z0 = np.array([float(np.asarray(p0.zbs)[ntor + n, m]) for m, n in SS_MODES])
    x0 = np.concatenate([r0, z0, np.asarray(cdofs0).ravel(), np.asarray(cur0[1:])])
    # ESS on the boundary block: high harmonics move on exponentially shorter
    # leashes (level = max(m, |n|), alpha = 0.7 -- the examples' convention),
    # so releasing max_mode 3 all at once stays as stable as the mode ladder.
    ess = np.array([np.exp(-0.7 * (max(m, abs(n)) - 1)) for m, n in SS_MODES])
    D = np.concatenate([D_BOUNDARY * ess, D_BOUNDARY * ess, np.full(ncd, D_COIL),
                        np.full(N_COILS - 1, D_CURRENT)])
    print(f"[{label}:{case}] dofs: {x0.size} (boundary {2 * nb} + "
          f"coil curves {ncd} + currents {N_COILS - 1}), w_bn={args.w_bn}")

    # NOTE: jax.jit around this value_and_grad fails today with a
    # TracerArrayConversionError inside the FBD/im.run stack (host-side
    # conversions that pass under bare grad) -- jit-closing that path is
    # plan_pre_vmex Item F; the adjoint_tol relaxation above is the
    # dominant per-eval win meanwhile.
    vg = jax.value_and_grad(objective, has_aux=True)
    hist: list[float] = []

    def fun(u):
        try:
            (J, aux), g = vg(jnp.asarray(x0 + D * u))
        except Exception as exc:  # e.g. VmecJacobianError on a wild trial
            print(f"  eval {len(hist) + 1:4d}: invalid trial ({type(exc).__name__})"
                  " -- penalized")
            hist.append(np.inf)
            return 1e10, np.zeros_like(u)
        hist.append(float(J))
        print(f"  eval {len(hist):4d}: J={float(J):.6e}  qs={float(aux[0]):.3e} "
              f"bn={float(aux[1]):.3e} asp={float(aux[2]):.3e} "
              f"iota={float(aux[3]):.3e} len={float(aux[4]):.3e} "
              f"curv={float(aux[5]):.3e}")
        return float(J), np.asarray(g, dtype=float) * D

    t0 = time.time()
    res = scipy.optimize.minimize(
        fun, np.zeros_like(x0), jac=True, method="L-BFGS-B",
        bounds=[(-U_BOUND, U_BOUND)] * x0.size,
        options={"maxiter": args.maxiter_polish if warm else args.maxiter_single,
                 "ftol": SINGLE_REL_FTOL, "gtol": 1e-12})
    wall = time.time() - t0

    xf = x0 + D * res.x
    (Jf, auxf), _ = vg(jnp.asarray(xf))
    params_f, cdofs_f, cur_f = unpack(xf)

    # Save the final boundary as an input deck (evaluate re-solves from it).
    rbc = np.array(inp.rbc, copy=True)
    zbs = np.array(inp.zbs, copy=True)
    for i, (m, n) in enumerate(SS_MODES):
        rbc[ntor + n, m] = float(xf[i])
        zbs[ntor + n, m] = float(xf[nb + i])
    inp_f = dataclasses.replace(inp, rbc=rbc, zbs=zbs)
    inp_f.to_indata(out / f"input.{label}")
    np.savez(out / f"coils_{label}.npz",
             cdofs=np.asarray(cdofs_f), currents=np.asarray(cur_f),
             nfp=nfp, n_segments=NSEG)
    eq_f = opt.solve_equilibrium(inp_f)
    vj.write_wout(out / f"wout_{label}.nc", eq_f.wout)

    J0 = next(v for v in hist if np.isfinite(v))
    summary = dict(
        J0=J0, Jf=float(Jf), ratio=J0 / max(float(Jf), 1e-30),
        terms_final=dict(zip(("qs", "bn", "aspect", "iota", "len", "curv"),
                             (float(v) for v in auxf))),
        w_bn=args.w_bn, nit=int(res.nit), nev=len(hist), wall_s=wall,
        **coil_metrics(cdofs_f, nfp))
    (out / f"{label}.json").write_text(json.dumps(summary, indent=2))
    print(f"[{label}:{case}] J {J0:.4e} -> {float(Jf):.4e} "
          f"({summary['ratio']:.1f}x) in {res.nit} iters, {len(hist)} evals, "
          f"{wall:.0f}s")
    print(f"wrote {out / f'input.{label}'}, {out / f'coils_{label}.npz'}, "
          f"{out / f'wout_{label}.nc'}")


# ------------------------------ evaluate ------------------------------------
def load_boundary_input(deck: Path, case: str, out: Path) -> vj.VmecInput:
    """Load a phase-written input deck at this script's solve budget."""
    inp = vj.VmecInput.from_file(str(deck))
    kw = dict(ns_array=[NS], niter_array=[SOLVE["max_iterations"]],
              ftol_array=[SOLVE["ftol"]], lfreeb=False)
    ps = get_seed_pres_scale(case, out)
    if ps > 0:  # decks round-trip the pressure, but re-assert the budget knobs
        kw.update(pmass_type="power_series",
                  am=[1.0, -1.0] + [0.0] * 19, pres_scale=ps)
    return dataclasses.replace(inp, **kw)


def evaluate_one(label: str, deck: Path, coils_npz: Path, case: str,
                 out: Path) -> dict:
    """Honest metrics for one (boundary, coils) result.

    QS is the wout-based reporting metric (``QuasisymmetryRatioResidual.total``
    on a fixed-boundary re-solve of the final boundary); the normal-field
    errors are computed directly from the coil field and the surface normals:
    ``Bn = (B_plasma + B_coil) . n`` (the ``bnormal_residual`` of the
    ``FreeBoundaryDiffProblem`` -- ``bnormal_objective`` is its area-weighted
    mean SQUARE, so the table's ``<|B.n|>/<B>`` and ``max|B.n|/<B>`` are
    derived here from the residual field itself, normalized by the
    area-weighted mean of ``|B|`` on the boundary).
    """
    inp = load_boundary_input(deck, case, out)
    eq = opt.solve_equilibrium(inp)
    qs = opt.QuasisymmetryRatioResidual(EVAL_SURFACES, HELICITY_M, HELICITY_N)
    qs_total = float(qs.total(eq))

    dat = np.load(coils_npz)
    cdofs, cur, nfp = jnp.asarray(dat["cdofs"]), jnp.asarray(dat["currents"]), int(dat["nfp"])

    sd = FBD.surface_field_data_from_state(inp, eq.state, nphi=NPHI, ntheta=NTHETA)
    prob = FBD.FreeBoundaryDiffProblem.from_surface_data(sd, digits=4)
    Bfn = coil_field(cdofs, cur, nfp)
    bn = np.asarray(prob.bnormal_residual(Bfn))          # (B_plasma + B_coil).n
    Bmag = np.asarray(jnp.linalg.norm(prob.total_B_out(Bfn), axis=0))
    w = np.asarray(prob.weights)                          # area weights, sum 1
    B_avg = float(np.sum(w * Bmag))
    metrics = dict(
        label=label,
        qs_total=qs_total,
        aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
        mean_iota=float(opt.mean_iota(eq.state, eq.runtime)),
        avg_Bn_over_B=float(np.sum(w * np.abs(bn)) / B_avg),
        max_Bn_over_B=float(np.abs(bn).max() / B_avg),
        B_avg=B_avg,
        currents=[float(v) for v in np.asarray(dat["currents"])],
        **coil_metrics(cdofs, nfp))
    return metrics


def phase_evaluate(case: str, out: Path, args) -> None:
    """Compare two-stage vs single-stage on identical final metrics."""
    needed = {"two_stage": (out / "input.stage1", out / "coils_stage2.npz"),
              "single_stage": (out / "input.single", out / "coils_single.npz")}
    for label, (deck, npz) in needed.items():
        for f in (deck, npz):
            if not f.exists():
                raise SystemExit(f"missing {f} -- run the earlier phases first")
    # the single-stage POLISH of the two-stage result is optional but is the
    # headline column (see phase_single warm=True)
    if (out / "input.polish").exists() and (out / "coils_polish.npz").exists():
        needed["single_stage_polish"] = (out / "input.polish",
                                         out / "coils_polish.npz")

    results = {label: evaluate_one(label, deck, npz, case, out)
               for label, (deck, npz) in needed.items()}
    (out / "comparison.json").write_text(json.dumps(results, indent=2))

    rows = [
        ("QS ratio residual (10 surf.)", "qs_total", "{:.3e}"),
        ("aspect ratio", "aspect", "{:.3f}"),
        ("mean iota", "mean_iota", "{:.4f}"),
        ("<|B.n|>/<B>", "avg_Bn_over_B", "{:.3e}"),
        ("max|B.n|/<B>", "max_Bn_over_B", "{:.3e}"),
    ]
    order = [k for k in ("two_stage", "single_stage_polish", "single_stage")
             if k in results]
    heads = {"two_stage": "two-stage",
             "single_stage_polish": "two-stage + single-stage polish",
             "single_stage": "single-stage (cold start)"}
    lines = ["| metric (" + case + ") | " +
             " | ".join(heads[k] for k in order) + " |",
             "|---|" + "---|" * len(order)]
    for name, key, fmt in rows:
        lines.append("| " + name + " | " +
                     " | ".join(fmt.format(results[k][key]) for k in order) + " |")
    lines.append("| coil lengths [m] (budget {:.2f}) | {} |".format(
        L_MAX, " | ".join(", ".join(f"{v:.2f}" for v in results[k]["lengths"])
                          for k in order)))
    lines.append("| coil max curvature [1/m] (budget {:.1f}) | {} |".format(
        KAPPA_MAX,
        " | ".join(", ".join(f"{v:.2f}" for v in results[k]["max_curvatures"])
                   for k in order)))
    table = "\n".join(lines)
    (out / "comparison.md").write_text(table + "\n")
    print(f"\n{'=' * 72}\nCOMPARISON ({case})\n{'=' * 72}\n{table}")
    print(f"\nwrote {out / 'comparison.json'}, {out / 'comparison.md'}")


# -------------------------------- main --------------------------------------
def phase_polish(case: str, out: Path, args) -> None:
    """Single-stage polish of the two-stage result (phase_single warm=True)."""
    phase_single(case, out, args, warm=True)


PHASES = dict(stage1=phase_stage1, stage2=phase_stage2, single=phase_single,
              polish=phase_polish, evaluate=phase_evaluate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--case", choices=("vacuum", "beta"), default="vacuum")
    parser.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    parser.add_argument("--w-bn", type=float, default=300.0,
                        help="single-stage normal-field weight (default 300)")
    parser.add_argument("--maxiter-single", type=int, default=MAXITER_SINGLE)
    parser.add_argument("--maxiter-stage2", type=int, default=MAXITER_STAGE2)
    parser.add_argument("--maxiter-polish", type=int, default=MAXITER_POLISH)
    args = parser.parse_args()

    if not FBD.have_virtual_casing_jax():
        raise SystemExit("needs virtual_casing_jax (pip install -e /path/to/virtual_casing_jax)")
    try:
        import essos.coils  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional heavy dependency
        raise SystemExit("needs essos (pip install -e /path/to/ESSOS)") from exc

    if CI and args.case == "beta":
        # The 12-point CI grid cannot push the pressure-loaded circular seed
        # below fsqr ~ 1.2e-6 (measured discretization floor: 60k iterations
        # plateau there), so the strict ``im.run`` solves of the single-stage
        # phase would always raise at 1e-9.  Keep the strict smoke ftol for
        # vacuum; relax the beta smoke to sit above its floor.
        SOLVE["ftol"] = 1e-5

    out = OUT_ROOT / args.case
    out.mkdir(parents=True, exist_ok=True)
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    for name in phases:
        t0 = time.time()
        print(f"\n{'#' * 72}\nPHASE {name}  (case={args.case})\n{'#' * 72}")
        PHASES[name](args.case, out, args)
        print(f"[{name}:{args.case}] phase wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
