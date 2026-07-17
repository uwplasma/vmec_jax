#!/usr/bin/env python
"""Five one-objective refinement campaigns off the precise-QA deck.

The point of this script is a single sentence: **the equilibrium is a
differentiable building block — pick any physics objective and the exact
gradients drive it.**  Starting from the shipped precise quasi-axisymmetric
optimum (``benchmarks/opt_decks/input.qa_optimized``: QS ratio residual
~1e-6, aspect 6.00, mean iota 0.42), each campaign pushes ONE new objective
while HOLDING quasisymmetry (the QA :class:`QuasisymmetryRatioResidual` term
stays in every campaign at a stiff weight) and re-reports the full metric row
before/after:

``lgradb``
    RAISE the coil-simplicity proxy ``min L_grad_B`` (Kappel/Landreman
    magnetic-gradient scale length) to ~1.3x its seed value via the traceable
    :func:`vmec_jax.optimize.l_grad_b_state` — smooth soft-min for the
    optimizer, hard min for reporting.  Implicit-adjoint lane.
``well``
    Deepen the vacuum magnetic well: :func:`~vmec_jax.optimize.magnetic_well`
    is ``(V'(0) - V'(1))/V'(0)`` with POSITIVE = favorable (simsopt
    ``vacuum_well``); the seed sits on a slight hill (-0.037), the target is
    a true well modestly beyond it.  Implicit-adjoint lane.
``iota_up``
    Mean iota 0.42 -> 0.55 at held aspect 6.  Implicit-adjoint lane.
``aspect_down``
    Aspect 6.00 -> 4.8 at held mean iota 0.42.  Implicit-adjoint lane.
``dmerc``
    Finite-beta Mercier stability: add a parabolic pressure profile
    (``pres_scale`` calibrated once to ``<beta> ~ 1.25%``, the
    ``single_stage_vs_two_stage.py`` recipe) and push the interior
    :func:`~vmec_jax.optimize.d_merc` profile toward POSITIVE (= Mercier
    stable) through a hinge penalty on its negative part.  ``d_merc`` is a
    wout-engine objective (host-NumPy Mercier tables) with NO traceable lane,
    so this one campaign runs honest finite differences (``jac=None``) at
    ``max_mode 2`` — few dofs keep the FD cost affordable, and the cost gap
    vs the implicit campaigns is part of the story.

Each campaign: measure the seed metrics -> optimize (short budget, ESS
trust-region scaling) -> re-solve the final deck -> print a before/after row
-> save the deck (``to_indata``) + ``metrics.json`` into
``output_objectives_showcase/<name>/``.  ``--only lgradb,dmerc`` runs a
subset.  ``VMEC_JAX_EXAMPLES_CI=1`` shrinks every campaign to a smoke budget
(ns=12, max_nfev=5) so CI covers both gradient lanes.

The self-consistent bootstrap objective (Redl) is NOT rerun here — that
campaign already exists as ``examples/optimization/{QA,QH}_bootstrap_
selfconsistent.py`` and the README's ``readme_bootstrap.png`` figure; the
README objectives row cites it directly.

Figure: ``python benchmarks/make_readme_figures.py --only objectives`` reads
the saved ``metrics.json`` files and renders
``docs/_static/figures/readme_objectives.png``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import time
from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax import optimize as opt

# --------------------------- parameters ------------------------------------
DECK = (Path(__file__).resolve().parents[2] / "benchmarks" / "opt_decks"
        / "input.qa_optimized")
OUT_ROOT = Path("output_objectives_showcase")

QS_SURFACES = np.linspace(0.1, 1.0, 10)   # house reporting convention
HELICITY_M, HELICITY_N = 1, 0             # QA: |B| = |B|(s, theta)
QS_WEIGHT = 100.0        # stiff hold: QS drifting 1e-6 -> 1e-4 costs ~0.05
W_ASPECT_HOLD = 1.0
W_IOTA_HOLD = 10.0
ASPECT_SEED, IOTA_SEED = 6.0, 0.42        # the deck's own targets, re-held

SOFTMIN_K = 50.0         # [1/m] soft-min sharpness; bias <= log(24*24)/k ~ 0.13 m
LGRADB_FACTOR = 1.3      # target = 1.3x the measured seed min L_grad_B
W_LGRADB = 5.0
WELL_STEP = 0.05         # target = seed + 0.05 (positive = favorable well)
W_WELL = 20.0
IOTA_TARGET_UP = 0.55
W_IOTA_PUSH = 10.0
ASPECT_TARGET_DOWN = 4.8
W_ASPECT_PUSH = 1.0
BETA_TARGET_PCT = 1.25   # dmerc campaign <beta> [%] (parabolic pressure)
DMERC_S_MIN = 0.25       # hinge on DMerc over s >= 0.25 (skips near-axis noise)
W_DMERC = 0.05

CI = os.environ.get("VMEC_JAX_EXAMPLES_CI") == "1"
if CI:
    # Smoke budget: every campaign (both gradient lanes) at tiny cost.  The
    # precise-QA deck converges below 1e-12 at ns=12 in BOTH vacuum and the
    # calibrated finite-beta case (measured 2026-07-17: 622 / 393 iterations),
    # so unlike single_stage_vs_two_stage.py no beta-floor ftol relaxation is
    # needed — 1e-11 just keeps headroom for wandering CI trial boundaries.
    NS, SOLVER_FTOL, SOLVER_NITER = 12, 1e-11, 3000
    MAX_MODE_IMPLICIT, MAX_MODE_FD = 2, 1
    MAX_NFEV_IMPLICIT, MAX_NFEV_FD = 5, 5
    LS_FTOL = 1e-4
else:
    # Fast-showcase budget (repo convention): ONE vmec grid at ns=31,
    # converged-looking force residual, short independent campaigns.
    NS, SOLVER_FTOL, SOLVER_NITER = 31, 1e-12, 3000
    MAX_MODE_IMPLICIT, MAX_MODE_FD = 3, 2
    MAX_NFEV_IMPLICIT, MAX_NFEV_FD = 200, 150
    LS_FTOL = 1e-8

QS = opt.QuasisymmetryRatioResidual(QS_SURFACES, HELICITY_M, HELICITY_N)


# --------------------------- seed decks ------------------------------------
def base_input(**extra) -> vj.VmecInput:
    """The precise-QA deck at this script's solve budget (vacuum by default)."""
    inp = vj.VmecInput.from_file(str(DECK))
    return dataclasses.replace(
        inp, ns_array=[NS], ftol_array=[SOLVER_FTOL],
        niter_array=[SOLVER_NITER], lfreeb=False, **extra)


def beta_input(out: Path) -> tuple[vj.VmecInput, float]:
    """Parabolic-pressure deck calibrated to ``<beta> ~ BETA_TARGET_PCT`` (cached).

    Same one-shot calibration as ``single_stage_vs_two_stage.py``:
    ``p(s) = pres_scale * (1 - s)`` and beta is ~linear in ``pres_scale``,
    so one or two re-solves land within 5% of the target.
    """
    cache = out / "pres_scale.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        ps, beta = float(cached["pres_scale"]), float(cached["beta_pct"])
        return base_input(pmass_type="power_series",
                          am=[1.0, -1.0] + [0.0] * 19, pres_scale=ps), beta
    ps = 10000.0   # measured 2026-07-17: <beta> = 1.30% on this deck at ns=31
    beta = 0.0
    for it in range(3):
        inp = base_input(pmass_type="power_series",
                         am=[1.0, -1.0] + [0.0] * 19, pres_scale=ps)
        eq = opt.solve_equilibrium(inp)
        beta = 100.0 * float(eq.wout.betatotal)
        print(f"[dmerc] beta calibration {it}: pres_scale={ps:.0f} -> "
              f"<beta>={beta:.3f}%")
        if abs(beta - BETA_TARGET_PCT) < 0.05 * BETA_TARGET_PCT:
            break
        ps *= BETA_TARGET_PCT / max(beta, 1e-12)
    cache.write_text(json.dumps({"pres_scale": ps, "beta_pct": beta}, indent=2))
    return inp, beta


# --------------------------- metrics ---------------------------------------
def held_metrics(eq: opt.Equilibrium) -> dict:
    """The metrics every campaign holds/reports: QS, aspect, mean iota."""
    return dict(qs_total=float(QS.total(eq)),
                aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
                mean_iota=float(opt.mean_iota(eq.state, eq.runtime)))


def lgradb_hard(eq: opt.Equilibrium) -> float:
    """Reported metric: HARD min L_grad_B (the soft-min is optimizer-only)."""
    return float(opt.l_grad_b_state(eq.state, eq.runtime))


def dmerc_interior(eq: opt.Equilibrium) -> np.ndarray:
    """Interior DMerc profile (s >= DMERC_S_MIN, edge excluded).

    The first two surfaces carry the documented near-axis noise and this
    compact QA additionally has a genuine (vacuum) near-axis hill that a
    short campaign cannot fix, so the hinge targets the mid/outer profile.
    """
    dm = np.asarray(opt.d_merc(eq))
    i0 = max(2, int(round(DMERC_S_MIN * (dm.size - 1))))
    return dm[i0:-1]


def dmerc_hinge(eq: opt.Equilibrium) -> np.ndarray:
    """Hinge rows: the negative part of the interior DMerc profile."""
    return np.minimum(dmerc_interior(eq), 0.0)


# --------------------------- campaign registry ------------------------------
def _hold_terms() -> list:
    """The metrics every campaign keeps in the objective."""
    return [(QS, 0.0, QS_WEIGHT),
            (opt.aspect_ratio, ASPECT_SEED, W_ASPECT_HOLD),
            (opt.mean_iota, IOTA_SEED, W_IOTA_HOLD)]


def build_lgradb(seed_eq: opt.Equilibrium, out: Path) -> dict:
    seed = lgradb_hard(seed_eq)
    target = LGRADB_FACTOR * seed
    soft = functools.partial(opt.l_grad_b_state, softmin_k=SOFTMIN_K)
    return dict(
        goal=f"raise min L_grad_B (coil-simplicity proxy) to {LGRADB_FACTOR}x seed",
        label="min L_grad_B [m]", target=target,
        terms=_hold_terms() + [(soft, target, W_LGRADB)],
        metric=lgradb_hard, fmt="{:.3f}",
        jac="implicit", max_mode=MAX_MODE_IMPLICIT, max_nfev=MAX_NFEV_IMPLICIT)


def build_well(seed_eq: opt.Equilibrium, out: Path) -> dict:
    seed = float(opt.magnetic_well(seed_eq.state, seed_eq.runtime))
    target = seed + WELL_STEP   # positive = favorable (magnetic_well docstring)
    return dict(
        goal=f"deepen the vacuum magnetic well (positive = stable) to {target:+.3f}",
        label="magnetic well W", target=target,
        terms=_hold_terms() + [(opt.magnetic_well, target, W_WELL)],
        metric=lambda eq: float(opt.magnetic_well(eq.state, eq.runtime)),
        fmt="{:+.4f}",
        jac="implicit", max_mode=MAX_MODE_IMPLICIT, max_nfev=MAX_NFEV_IMPLICIT)


def build_iota_up(seed_eq: opt.Equilibrium, out: Path) -> dict:
    return dict(
        goal=f"raise mean iota {IOTA_SEED} -> {IOTA_TARGET_UP} at held aspect {ASPECT_SEED}",
        label="mean iota", target=IOTA_TARGET_UP,
        terms=[(QS, 0.0, QS_WEIGHT),
               (opt.aspect_ratio, ASPECT_SEED, W_ASPECT_HOLD),
               (opt.mean_iota, IOTA_TARGET_UP, W_IOTA_PUSH)],
        metric=lambda eq: float(opt.mean_iota(eq.state, eq.runtime)),
        fmt="{:.4f}",
        jac="implicit", max_mode=MAX_MODE_IMPLICIT, max_nfev=MAX_NFEV_IMPLICIT)


def build_aspect_down(seed_eq: opt.Equilibrium, out: Path) -> dict:
    return dict(
        goal=f"lower aspect {ASPECT_SEED} -> {ASPECT_TARGET_DOWN} at held mean iota {IOTA_SEED}",
        label="aspect ratio", target=ASPECT_TARGET_DOWN,
        terms=[(QS, 0.0, QS_WEIGHT),
               (opt.aspect_ratio, ASPECT_TARGET_DOWN, W_ASPECT_PUSH),
               (opt.mean_iota, IOTA_SEED, W_IOTA_HOLD)],
        metric=lambda eq: float(opt.aspect_ratio(eq.state, eq.runtime)),
        fmt="{:.3f}",
        jac="implicit", max_mode=MAX_MODE_IMPLICIT, max_nfev=MAX_NFEV_IMPLICIT)


def build_dmerc(seed_eq: opt.Equilibrium, out: Path) -> dict:
    return dict(
        goal=(f"push interior DMerc (s >= {DMERC_S_MIN}) toward positive at "
              f"<beta> ~ {BETA_TARGET_PCT}%"),
        label="min interior DMerc", target=0.0,
        terms=_hold_terms() + [(dmerc_hinge, 0.0, W_DMERC)],
        metric=lambda eq: float(dmerc_interior(eq).min()),
        fmt="{:+.2f}",
        jac=None, max_mode=MAX_MODE_FD, max_nfev=MAX_NFEV_FD)


CAMPAIGNS = {
    "lgradb": build_lgradb,
    "well": build_well,
    "iota_up": build_iota_up,
    "aspect_down": build_aspect_down,
    "dmerc": build_dmerc,
}


# --------------------------- driver -----------------------------------------
def run_campaign(name: str) -> dict:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\ncampaign {name}\n{'=' * 72}")

    beta_pct = 0.0
    if name == "dmerc":
        inp, beta_pct = beta_input(out)
    else:
        inp = base_input()

    seed_eq = opt.solve_equilibrium(inp)
    spec = CAMPAIGNS[name](seed_eq, out)
    metric = spec["metric"]
    fmt = spec["fmt"]

    ndofs = len(opt.boundary_dof_names(inp, spec["max_mode"]))
    lane = ("jac='implicit' — exact gradients through the implicit adjoint"
            if spec["jac"] == "implicit" else
            "jac=None — scipy 2-point finite differences")
    print(f"goal: {spec['goal']}")
    print(f"lane: {lane}; max_mode {spec['max_mode']} ({ndofs} dofs), "
          f"max_nfev {spec['max_nfev']}, use_ess=True")
    if name == "dmerc":
        print("note: d_merc is a wout-engine objective (host-NumPy Mercier "
              "tables, vmec_jax.core.nyquist) with NO traceable lane — this "
              "campaign pays one full equilibrium re-solve PER DOF per "
              "Jacobian, which is exactly why it runs at max_mode "
              f"{spec['max_mode']}.  Honest cost is part of the story.")

    seed = dict(metric=metric(seed_eq), **held_metrics(seed_eq))
    print(f"[seed]  {spec['label']} = {fmt.format(seed['metric'])} | "
          f"QS {seed['qs_total']:.3e} aspect {seed['aspect']:.3f} "
          f"iota {seed['mean_iota']:.4f}")

    t0 = time.time()
    result = opt.least_squares(
        spec["terms"], inp, max_mode=spec["max_mode"], jac=spec["jac"],
        use_ess=True, verbose=1,
        max_nfev=spec["max_nfev"], ftol=LS_FTOL, xtol=1e-10)
    wall = time.time() - t0

    # Honest reporting: re-solve the final deck from scratch and re-measure.
    final_inp = result.input
    final_eq = opt.solve_equilibrium(final_inp)
    final = dict(metric=metric(final_eq), **held_metrics(final_eq))
    print(f"[final] {spec['label']} = {fmt.format(final['metric'])} | "
          f"QS {final['qs_total']:.3e} aspect {final['aspect']:.3f} "
          f"iota {final['mean_iota']:.4f}")
    print(f"[{name}] {spec['label']}: {fmt.format(seed['metric'])} -> "
          f"{fmt.format(final['metric'])} (target {fmt.format(spec['target'])}) | "
          f"QS {seed['qs_total']:.2e} -> {final['qs_total']:.2e} | "
          f"{result.nfev} nfev, {wall:.0f}s")

    final_inp.to_indata(out / f"input.{name}")
    record = dict(
        campaign=name, goal=spec["goal"], label=spec["label"], fmt=fmt,
        target=float(spec["target"]), jac=spec["jac"] or "fd",
        max_mode=spec["max_mode"], max_nfev=spec["max_nfev"],
        nfev=int(result.nfev), wall_s=wall, ci=CI, ns=NS, beta_pct=beta_pct,
        seed=seed, final=final)
    (out / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {out / f'input.{name}'}, {out / 'metrics.json'}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--only", default=",".join(CAMPAIGNS),
                        help="comma-separated subset of campaigns "
                             f"({','.join(CAMPAIGNS)})")
    args = parser.parse_args()
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in CAMPAIGNS]
    if unknown:
        raise SystemExit(f"unknown campaign(s) {unknown}; "
                         f"choose from {list(CAMPAIGNS)}")

    records = [run_campaign(name) for name in names]

    print(f"\n{'=' * 72}\nsummary (seed -> final, one campaign per row)\n{'=' * 72}")
    for r in records:
        fmt = r["fmt"]
        print(f"{r['campaign']:12s} {r['label']:22s} "
              f"{fmt.format(r['seed']['metric'])} -> "
              f"{fmt.format(r['final']['metric'])}  "
              f"(QS {r['seed']['qs_total']:.1e} -> {r['final']['qs_total']:.1e}, "
              f"{r['jac']}, {r['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
