#!/usr/bin/env python3
"""Staged force oracle: per-stage VMEX internals along the funct3d chain.

Replays the production fixed-boundary iteration body
(:func:`vmex.core.solver._make_body` — the exact traced update both solver
lanes run) on the host, one iteration at a time, and records STAGED
quantities of the ``funct3d.f`` force chain at chosen iterations (default
1, 2, 25, 26 — straddling the ``ns4 = 25`` preconditioner refresh — plus
the first iteration after an in-loop Jacobian retry, i.e. after a
``TimeStepControl`` ``irst = 2`` restart incremented ``ijacob``).  The
replay includes the ``eqsolve.f`` first-pass transfers (bad-Jacobian /
``LMOVE_AXIS`` axis re-guess), so recorded iteration numbers match the
printed VMEC2000 iteration rows.

Stage map (chain order; VMEC2000 counterpart in parentheses)
------------------------------------------------------------
Recorded per iteration, evaluated at the state ENTERING that iteration —
the first ``funct3d`` pass of the iteration, before any within-iteration
restart re-evaluation:

  S01_STATE_AXIS       xc / recovered axis summary (``profil3d.f`` state,
                       ``guess_axis.f`` axis): channel norms of the internal
                       spectral state, axis-coefficient norms, r00/z00.
  S02_GEOMETRY         real-space synthesis + Jacobian (``totzsps`` +
                       ``jacobian.f``): R/Z field norms, interior
                       ``sqrt(g)`` min/max, sign-change flag.
  S03_BCOVAR_FIELDS    half-mesh fields and energies (``bcovar.f``):
                       ``bsupu/bsupv/bsubu/bsubv`` norms, ``wb/wp``,
                       ``lamscale``.
  S04_LAMBDA_REAL      real-space lambda force kernels (``forces.f``
                       ``blmn``/``clmn``): even/odd du and dv kernel norms.
  S05_LAMBDA_SPECTRAL  projected spectral lambda force (``tomnsps`` output,
                       BEFORE the ``scalxc`` scaling): block norm.
  S06_SCALXC           odd-m ``1/sqrt(s)`` force scaling (``profil3d.f``
                       ``scalxc``; ``funct3d.f`` ``gc = gc*scalxc``): table
                       norm and the scaled lambda-force norm.
  S07_FNORML           lambda residual normalization (``bcovar.f``
                       ``fnormL``): the cache value in effect this iteration.
  S08_FSQL             ``getfsq.f``: the UNNORMALIZED lambda sum ``gcl2``
                       and the normalized ``fsql = fnormL * gcl2``.
  S09_FACLAM           diagonal lambda preconditioner (``lamcal.f90``
                       ``faclam``): norm of the cache array in effect.
  S10_UPDATE           final update direction (``residue.f90``
                       ``scalfor``/``faclam``-preconditioned ``gc``):
                       R/Z/lambda channel norms and ``fsqr1/fsqz1/fsql1``.

Cross-code scope (documented contract)
--------------------------------------
The VMEC2000 executable prints ONLY per-iteration screen rows
(``FSQR/FSQZ/FSQL/RAX/DELT/WMHD``); none of the staged internals above are
observable from the binary.  Therefore:

* stages ``S01``-``S10`` are **VMEX-regression-only**: they are pinned by
  RECORDED goldens (``record`` then ``check`` against a committed JSON), and
* the row stages ``R01_FSQR / R02_FSQZ / R03_FSQL / R04_RAX / R05_WMHD``
  are the **cross-code** stages: ``cross`` runs the local ``xvmec2000`` with
  ``NSTEP`` forced to 1 on the same deck and compares the shared iteration
  rows.

Comparison policy: stages are checked in chain order per iteration,
iterations ascending; the comparison FAILS AT THE FIRST DIFFERING STAGE and
reports that stage code (the leading suspect in the funct3d/bcovar/lamcal
chain).

Privacy: like ``tools/first_divergence.py``, the default console output is
VALUES-FREE — stage codes plus PASS/FAIL only — so it can be reported for a
confidential deck.  ``--details`` prints numbers and must never be shared
for such a deck.  Harness failures are reported as the exception CLASS NAME
only.  Note the golden JSON itself contains values: record goldens only for
public decks.

Usage::

    python tools/force_oracle.py record input.CASE --out golden.json
    python tools/force_oracle.py check  input.CASE --golden golden.json
    python tools/force_oracle.py cross  input.CASE --xvmec2000 /path/to/xvmec2000
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ITERATIONS = (1, 2, 25, 26)

#: Stage codes in comparison (chain) order.
STAGE_ORDER = (
    "S01_STATE_AXIS", "S02_GEOMETRY", "S03_BCOVAR_FIELDS", "S04_LAMBDA_REAL",
    "S05_LAMBDA_SPECTRAL", "S06_SCALXC", "S07_FNORML", "S08_FSQL",
    "S09_FACLAM", "S10_UPDATE",
)

#: Cross-code row stages (the only VMEC2000-observable channel).
ROW_STAGE_ORDER = ("R01_FSQR", "R02_FSQZ", "R03_FSQL", "R04_RAX", "R05_WMHD")

ROW = re.compile(
    r"^\s*(\d+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)"
    r"\s+([0-9.E+-]+)\s+([0-9.E+-]+)", re.M)


def _norm(x) -> float:
    return float(np.sqrt(np.sum(np.square(np.asarray(x, dtype=float)))))


def _stage_values(carry, rt) -> dict[str, dict[str, float]]:
    """Compute every staged quantity for the state ENTERING ``carry``.

    Reuses the exact production kernels (``vmex.core`` geometry/fields/
    forces/residuals) in the ``funct3d.f`` order; the effective
    preconditioner cache (``ns4`` refresh included) is taken from
    :func:`vmex.core.solver._evaluate`, so the staged values describe the
    same evaluation the iteration row reports.
    """
    import jax.numpy as jnp

    from vmex.core import solver
    from vmex.core.fields import (
        energies_and_force_norms, magnetic_fields, metric_elements,
        radial_force_balance_error,
    )
    from vmex.core.forces import (
        apply_m1_force_balance, mhd_forces, spectral_mhd_forces,
    )
    from vmex.core.geometry import half_mesh_jacobian
    from vmex.core.residuals import (
        force_residuals, m1_residue_rotation, m1_zero_condition,
        scalxc_scale_force, zero_m1_z_force,
    )
    from vmex.core.transforms import odd_m_sqrt_s_scaling

    state = carry.state
    setup = rt.setup
    s = setup.s_full
    res = rt.resolution

    # The production evaluation (identical to the loop body's first pass):
    # supplies the effective (possibly ns4-refreshed) cache and the final
    # residual/update quantities.
    e = solver._evaluate(
        state, carry.cache, carry.iteration, carry.iter1, carry.fsqz, rt,
        carry.fsqr + carry.fsqz,
    )
    cache = e.cache

    (R_cos, R_sin, Z_cos, Z_sin), geometry = solver._geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    energies = energies_and_force_norms(
        jacobian=jacobian, metrics=metrics, fields=fields, trig=rt.trig,
        s=s, signgs=setup.signgs,
    )
    forces = mhd_forces(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields,
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        modes=rt.modes, trig=rt.trig, s=s, phipf=setup.phipf,
        tcon=cache.tcon, signgs=setup.signgs, rcon0=rt.rcon0, zcon0=rt.zcon0,
    )
    spectral = spectral_mhd_forces(
        forces, mpol=res.mpol, ntor=res.ntor, trig=rt.trig, include_edge=False,
    )
    if rt.lforbal:
        equif = radial_force_balance_error(
            fields=fields, phipf=setup.phipf, trig=rt.trig, s=s,
            signgs=setup.signgs,
        )
        spectral = apply_m1_force_balance(
            spectral, equif=equif, factor_R=cache.force_balance_R,
            factor_Z=cache.force_balance_Z,
        )
    rotated = m1_residue_rotation(spectral, lconm1=setup.lconm1)
    gate = m1_zero_condition(
        fsqz_previous=carry.fsqz,
        iterations_since_restart=carry.iteration,
    )
    released = zero_m1_z_force(rotated, gate)
    scaled = scalxc_scale_force(released, s=s)
    residuals = force_residuals(
        scaled, fnorm=cache.fnorm, fnormL=cache.fnormL, r1=energies.r1,
        include_edge=False,
    )
    scalxc = odd_m_sqrt_s_scaling(jnp.asarray(s), res.mpol)

    def lam_blocks(force) -> float:
        total = 0.0
        for name in ("force_lambda_sc", "force_lambda_cs",
                     "force_lambda_cc", "force_lambda_ss"):
            block = getattr(force, name)
            if block is not None:
                total += float(np.sum(np.square(np.asarray(block, float))))
        return math.sqrt(total)

    sqrt_g_inner = np.asarray(jacobian.sqrt_g, float)[1:]
    stages = {
        "S01_STATE_AXIS": {
            "r00": float(e.r00), "z00": float(e.z00),
            "raxis_c_norm": _norm(setup.raxis_c),
            "zaxis_s_norm": _norm(setup.zaxis_s),
            "R_cos_norm": _norm(state.R_cos),
            "Z_sin_norm": _norm(state.Z_sin),
            "L_sin_norm": _norm(state.L_sin),
        },
        "S02_GEOMETRY": {
            "R_norm": _norm(np.asarray(geometry.R_even)
                            + np.asarray(geometry.R_odd)),
            "Z_norm": _norm(np.asarray(geometry.Z_even)
                            + np.asarray(geometry.Z_odd)),
            "sqrt_g_min": float(np.min(sqrt_g_inner)),
            "sqrt_g_max": float(np.max(sqrt_g_inner)),
            "jacobian_sign_changed": float(bool(e.jacobian_sign_changed)),
        },
        "S03_BCOVAR_FIELDS": {
            "bsupu_norm": _norm(fields.bsupu), "bsupv_norm": _norm(fields.bsupv),
            "bsubu_norm": _norm(fields.bsubu), "bsubv_norm": _norm(fields.bsubv),
            "wb": float(energies.wb), "wp": float(energies.wp),
            "lamscale": float(fields.lamscale),
        },
        "S04_LAMBDA_REAL": {
            "blmn_even_norm": _norm(forces.force_lambda_du_even),
            "blmn_odd_norm": _norm(forces.force_lambda_du_odd),
            "clmn_even_norm": _norm(forces.force_lambda_dv_even),
            "clmn_odd_norm": _norm(forces.force_lambda_dv_odd),
        },
        "S05_LAMBDA_SPECTRAL": {
            "flmn_norm": lam_blocks(spectral),
        },
        "S06_SCALXC": {
            "scalxc_norm": _norm(scalxc),
            "scaled_lambda_norm": lam_blocks(scaled),
        },
        "S07_FNORML": {
            "fnormL": float(cache.fnormL),
        },
        "S08_FSQL": {
            "gcl2": float(residuals.gcl2),
            "fsql": float(residuals.fsql),
        },
        "S09_FACLAM": {
            "faclam_norm": _norm(cache.faclam),
            "faclam_00": float(np.asarray(cache.faclam)[1, 0, 0]),
        },
        "S10_UPDATE": {
            "gc_R_norm": _norm(e.gc.R_cos) + _norm(e.gc.R_sin),
            "gc_Z_norm": _norm(e.gc.Z_sin) + _norm(e.gc.Z_cos),
            "gc_L_norm": _norm(e.gc.L_sin) + _norm(e.gc.L_cos),
            "fsqr1": float(e.pre.fsqr1),
            "fsqz1": float(e.pre.fsqz1),
            "fsql1": float(e.pre.fsql1),
        },
    }
    # Self-consistency guard: the staged chain must land on the very
    # residuals the production evaluation reports (drift detector between
    # this recomputation and solver._force_pipeline).
    stages["S08_FSQL"]["fsql_production"] = float(e.residuals.fsql)
    return stages


def replay(inp, *, ns: int | None = None, niter: int = 30,
           time_step: float | None = None,
           iterations=DEFAULT_ITERATIONS) -> dict:
    """Replay the production iteration body and record staged quantities.

    Returns ``{"iterations": {it: {stage: {name: value}}},
    "post_retry_iteration": int | None, "rows": {it: [fsqr, fsqz, fsql,
    rax, wmhd, delt]}, "termination": str}``.  The replay mirrors
    ``vmex.core.solver._solve_stage``'s first-pass axis transfer
    (``eqsolve.f``): on a first-iteration bad Jacobian or ``LMOVE_AXIS``
    high-force pass the axis is re-guessed once and recording restarts, so
    iteration numbers match the printed rows of both codes.
    """
    from vmex.core import solver
    from vmex.core.errors import (
        AXIS_REGUESS_FLAG, BAD_JACOBIAN_FLAG, JAC75_FLAG, MORE_ITER_FLAG,
        SUCCESSFUL_TERM_FLAG,
    )

    resolution = solver.resolution_from_input(inp, ns=ns)
    rt = solver.prepare_runtime(
        inp, resolution, max_iterations=niter, time_step=time_step,
    )
    state = solver._initial_state(rt.setup)
    wanted = set(int(i) for i in iterations)

    def attempt(rt, state, *, ijacob, xcdot, residuals):
        carry = solver._initial_carry(
            state, rt, ijacob=ijacob, xcdot=xcdot, residuals=residuals,
        )
        body = solver._make_body(rt)
        records: dict[int, dict] = {}
        post_retry: int | None = None
        for _ in range(niter + 400):
            it = int(carry.iteration)
            if bool(carry.done):
                break
            if it in wanted or (post_retry is not None and it == post_retry
                                and it not in records):
                records[it] = _stage_values(carry, rt)
            prev_ijacob = int(carry.ijacob)
            carry = body(carry)
            if post_retry is None and int(carry.ijacob) > prev_ijacob \
                    and not bool(carry.done):
                post_retry = int(carry.iteration)
        return carry, records, post_retry

    carry, records, post_retry = attempt(
        rt, state, ijacob=0, xcdot=None, residuals=None)
    ier = int(carry.ier)
    if ier in (BAD_JACOBIAN_FLAG, AXIS_REGUESS_FLAG) and int(carry.ijacob) == 0 \
            and rt.resolution.ns >= 3:
        rt, state, _axis = solver.reguess_initial_axis(rt, state)
        carry, records, post_retry = attempt(
            rt, state, ijacob=1,
            xcdot=carry.xcdot if ier == AXIS_REGUESS_FLAG else None,
            residuals=(carry.fsqr, carry.fsqz, carry.fsql),
        )
        ier = int(carry.ier)

    upto = int(carry.iteration) if bool(carry.done) else int(carry.iteration) - 1
    trajectory = np.asarray(carry.trajectory)
    rows: dict[int, list[float]] = {}
    for it in range(1, min(upto, trajectory.shape[0]) + 1):
        row = trajectory[it - 1]
        if int(row[0]) != it:
            continue
        rows[it] = [float(row[1]), float(row[2]), float(row[3]),
                    float(row[7]), float(row[9]), float(row[10])]

    termination = {
        SUCCESSFUL_TERM_FLAG: "CONVERGED",
        MORE_ITER_FLAG: "ITERATION_BUDGET",
        JAC75_FLAG: "JACOBIAN_75",
    }.get(ier, f"IER_{ier}")
    return {
        "iterations": {str(k): v for k, v in sorted(records.items())},
        "post_retry_iteration": post_retry,
        "rows": {str(k): v for k, v in sorted(rows.items())},
        "termination": termination,
    }


# ---------------------------------------------------------------------------
# Comparison (first differing stage wins)
# ---------------------------------------------------------------------------


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), np.finfo(float).tiny)
    return abs(a - b) / scale


def compare_to_golden(current: dict, golden: dict, *, rtol: float,
                      details: bool, emit=print) -> int:
    """Stage-ordered comparison; FAILS AT THE FIRST DIFFERING STAGE.

    Output is values-free (stage codes + PASS/FAIL) unless ``details``.
    """
    iters = sorted(int(k) for k in golden.get("iterations", {}))
    for it in iters:
        got_it = current.get("iterations", {}).get(str(it))
        want_it = golden["iterations"][str(it)]
        if got_it is None:
            emit(f"S00_COVERAGE@iter{it}: FAIL (iteration not recorded)")
            emit("assessment: ORACLE_FAIL")
            return 1
        for stage in STAGE_ORDER:
            want = want_it.get(stage, {})
            got = got_it.get(stage, {})
            for name in sorted(want):
                a, b = float(got.get(name, np.nan)), float(want[name])
                ok = (math.isfinite(a) and _rel(a, b) <= rtol) or (a == b)
                if not ok:
                    if details:
                        emit(f"{stage}@iter{it}: FAIL {name} "
                             f"(got={a:.9e} golden={b:.9e} "
                             f"rel={_rel(a, b):.3e})")
                    else:
                        emit(f"{stage}@iter{it}: FAIL ({name})")
                    emit("assessment: ORACLE_FAIL")
                    return 1
            emit(f"{stage}@iter{it}: PASS")
    if golden.get("post_retry_iteration") is not None:
        got_pr = current.get("post_retry_iteration")
        if got_pr != golden["post_retry_iteration"]:
            suffix = (f" (got={got_pr} golden={golden['post_retry_iteration']})"
                      if details else "")
            emit(f"S00_RETRY_ITERATION: FAIL{suffix}")
            emit("assessment: ORACLE_FAIL")
            return 1
        emit("S00_RETRY_ITERATION: PASS")
    emit("assessment: ORACLE_PASS")
    return 0


def _prepare_deck(src: Path, workdir: Path, niter: int | None) -> Path:
    """Copy the deck with NSTEP forced to 1 (and NITER capped) for the binary."""
    text = src.read_text()
    text, n = re.subn(r"(NSTEP\s*=\s*)\d+", r"\g<1>1", text, flags=re.I)
    if n == 0:
        text = re.sub(r"&INDATA", "&INDATA\n  NSTEP = 1,", text, count=1,
                      flags=re.I)
    if niter is not None:
        caps = ", ".join([str(niter)] * 5)
        text = re.sub(r"NITER_ARRAY\s*=\s*\d+(?:[\s,]+\d+)*",
                      f"NITER_ARRAY = {caps}", text, flags=re.I)
    dst = workdir / src.name
    dst.write_text(text)
    return dst


def vmec2000_rows(deck: Path, xvmec2000: Path, *, niter: int,
                  timeout: int = 1800) -> dict[int, list[float]]:
    """Run the binary with NSTEP=1; return ``it -> [fsqr, fsqz, fsql, rax,
    wmhd, delt]`` for the FIRST radial stage (multigrid restarts numbering)."""
    with tempfile.TemporaryDirectory() as td:
        prepared = _prepare_deck(deck, Path(td), niter)
        proc = subprocess.run(
            [str(xvmec2000), prepared.name], cwd=prepared.parent,
            capture_output=True, text=True, timeout=timeout)
        text = proc.stdout + "\n" + proc.stderr
    rows: dict[int, list[float]] = {}
    last = 0
    for m in ROW.finditer(text):
        it = int(m.group(1))
        if it < last:
            break  # a new radial rung restarted the counter
        last = it
        vals = [float(m.group(i)) for i in range(2, 8)]
        # printed order: fsqr fsqz fsql rax delt wmhd -> normalize to
        # fsqr fsqz fsql rax wmhd delt (replay row order)
        rows[it] = [vals[0], vals[1], vals[2], vals[3], vals[5], vals[4]]
    return rows


def compare_rows(current: dict, ref_rows: dict[int, list[float]], *,
                 iterations, band: float, wmhd_rtol: float, rax_rtol: float,
                 details: bool, emit=print) -> int:
    """Cross-code row stages, first differing stage wins.

    Residual channels use a log-space band (``|log10 ratio| <= band``);
    ``RAX``/``WMHD`` use relative tolerances.  Values-free by default.
    """
    checked = False
    for it in sorted(int(i) for i in iterations):
        got = current.get("rows", {}).get(str(it))
        ref = ref_rows.get(it)
        if got is None or ref is None:
            continue
        checked = True
        pairs = list(zip(ROW_STAGE_ORDER, got[:5], ref[:5]))
        for stage, a, b in pairs:
            if stage in ("R04_RAX", "R05_WMHD"):
                tol = rax_rtol if stage == "R04_RAX" else wmhd_rtol
                ok = _rel(a, b) <= tol
            else:
                ok = (a > 0 and b > 0
                      and abs(math.log10(a) - math.log10(b)) <= band)
            if not ok:
                if details:
                    emit(f"{stage}@iter{it}: FAIL "
                         f"(vmex={a:.6e} vmec2000={b:.6e})")
                else:
                    emit(f"{stage}@iter{it}: FAIL")
                emit("assessment: CROSS_CODE_FAIL")
                return 1
            emit(f"{stage}@iter{it}: PASS")
    if not checked:
        emit("R00_COVERAGE: FAIL (no shared iteration rows)")
        emit("assessment: CROSS_CODE_FAIL")
        return 2
    emit("assessment: CROSS_CODE_PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("record", "check", "cross"))
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, help="record: golden JSON output path")
    ap.add_argument("--golden", type=Path, help="check: golden JSON to compare")
    ap.add_argument("--xvmec2000", type=Path,
                    help="cross: path to a local xvmec2000 executable")
    ap.add_argument("--ns", type=int, default=None,
                    help="radial surfaces (default: first NS_ARRAY stage)")
    ap.add_argument("--niter", type=int, default=30)
    ap.add_argument("--time-step", type=float, default=None)
    ap.add_argument("--iterations", type=str, default="1,2,25,26")
    ap.add_argument("--rtol", type=float, default=5e-6)
    ap.add_argument("--band", type=float, default=math.log10(2.0),
                    help="cross: log10 band for residual rows")
    ap.add_argument("--wmhd-rtol", type=float, default=1e-4)
    ap.add_argument("--rax-rtol", type=float, default=1e-3)
    ap.add_argument("--details", action="store_true",
                    help="print values; NEVER share for a confidential deck")
    args = ap.parse_args()

    try:
        from vmex.core.input import VmecInput

        inp = VmecInput.from_file(str(args.input))
        iterations = tuple(int(t) for t in args.iterations.split(",") if t)
        current = replay(inp, ns=args.ns, niter=args.niter,
                         time_step=args.time_step, iterations=iterations)
        if args.mode == "record":
            if args.out is None:
                raise SystemExit("record requires --out")
            args.out.write_text(json.dumps(current, indent=1, sort_keys=True))
            print(f"recorded {len(current['iterations'])} iterations "
                  f"(post_retry={current['post_retry_iteration']}) "
                  f"termination={current['termination']}")
            return 0
        if args.mode == "check":
            if args.golden is None:
                raise SystemExit("check requires --golden")
            golden = json.loads(args.golden.read_text())
            return compare_to_golden(current, golden, rtol=args.rtol,
                                     details=args.details)
        if args.xvmec2000 is None or not (shutil.which(str(args.xvmec2000))
                                          or args.xvmec2000.is_file()):
            raise SystemExit("cross requires --xvmec2000 PATH")
        ref_rows = vmec2000_rows(args.input, args.xvmec2000, niter=args.niter)
        return compare_rows(current, ref_rows, iterations=iterations,
                            band=args.band, wmhd_rtol=args.wmhd_rtol,
                            rax_rtol=args.rax_rtol, details=args.details)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - privacy: class name only
        print(f"S00_HARNESS_ERROR {type(exc).__name__}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
