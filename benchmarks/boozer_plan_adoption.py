#!/usr/bin/env python
"""Cost of the Boozer transform before and after the ``BoozerPlan`` adoption.

``vmex.core.omnigenity`` used to call ``booz_xform_jax_impl`` with no plan:
the kernel then rebuilt the per-resolution trig/mode tables *inside* the
trace, on every trace, and read its only knob from the environment.  It now
builds one ``booz_xform_jax.BoozerPlan`` per distinct resolution
(:func:`vmex.core.omnigenity._boozer_plan`) and hands the kernel the tables.

This script measures both, in fresh processes:

``--mode plan``
    ``vmex.core.omnigenity.boozer_spectrum_state`` as shipped.
``--mode inline``
    :func:`_spectrum_inline`, a verbatim reconstruction of the pre-adoption
    call (``vmex/core/omnigenity.py`` at commit ``2d3be2c0``).  Everything
    outside the kernel call — surface snapping, ``boozer_input_tables``, the
    ``oversample`` grid refinement, the returned dictionary — is the shipped
    code, so the two lanes differ only in the plan.
``--mode equivalence``
    Both lanes in one process; reports the per-key difference.
``--mode plan_build``
    The once-per-process plan build alone, first thing in a fresh process,
    so its first figure is a true empty-compilation-cache cost.
``--mode interleaved``
    Both lanes in one process, warm, alternating call by call.  The
    fresh-process lanes above are the protocol of record, but their wall
    time is dominated by the solve and the cold compile; a 10-30% warm
    difference does not survive that, and this lane is where it is legible.

The parent process (the default) runs the children.  Each child gets its own
empty JAX persistent-compilation-cache directory, so ``cold`` is a real cold
compile and the caller's ``~/.cache/vmex`` is never read or written; lanes
alternate ``inline, plan, inline, plan`` so machine drift hits both equally.

::

    python benchmarks/boozer_plan_adoption.py \
        --decks input.li383_low_res input.nfp1_QI \
        --output benchmarks/boozer_plan_adoption_m4.json
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

DATA = REPO / "examples" / "data"
SCHEMA = "vmex.boozer-plan-adoption/1"
OUTPUT_KEYS = ("bmnc_b", "bmns_b", "iota_b", "G_b", "I_b", "xm_b", "xn_b")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decks", nargs="+",
        default=["input.li383_low_res", "input.nfp1_QI"],
        help="deck file names under examples/data")
    parser.add_argument("--mboz", type=int, default=16)
    parser.add_argument("--nboz", type=int, default=16)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--surfaces", type=int, default=8,
                        help="number of half-mesh surfaces to transform")
    parser.add_argument("--repeats", type=int, default=2,
                        help="fresh-process repetitions per lane")
    parser.add_argument("--warm-repeats", type=int, default=9,
                        help="in-process warm calls timed per lane")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("plan", "inline", "equivalence", "interleaved", "plan_build"),
        default=None, help="child lane (internal)")
    parser.add_argument("--deck", default=None, help="child deck (internal)")
    return parser


def _surface_values(count: int) -> tuple[float, ...]:
    """Evenly spaced normalized-flux values inside the half mesh."""
    return tuple(0.15 + 0.75 * index / max(count - 1, 1) for index in range(count))


# ---------------------------------------------------------------------------
# child: one lane in one fresh process
# ---------------------------------------------------------------------------


def _spectrum_inline(state, rt, *, surfaces, mboz, nboz, oversample):
    """The pre-adoption call: no plan, tables rebuilt inside every trace.

    Reconstructed verbatim from ``vmex/core/omnigenity.py`` at commit
    ``2d3be2c0`` (``_boozer_kernel_state``), including the axisymmetric
    ``nboz = 0`` short circuit and the ``ensure_compile_time_eval`` block
    around the shape constants.  It reuses today's ``boozer_input_tables``
    and ``_refine_booz_grids`` so the measured difference is the plan alone.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from booz_xform_jax.jax_api import (
        booz_xform_jax_impl,
        prepare_booz_xform_constants,
    )

    from vmex.core.boozer_tables import boozer_input_tables
    from vmex.core.omnigenity import _nearest_half_mesh_rows, _refine_booz_grids

    setup = rt.setup
    ns = int(jnp.asarray(setup.s_full).shape[0])
    s_half, rows = _nearest_half_mesh_rows(ns, surfaces)
    lasym = bool(setup.lasym)
    tables = [boozer_input_tables(state, rt, int(row)) for row in rows]
    stack = lambda name: jnp.stack([table[name] for table in tables])  # noqa: E731
    first = tables[0]
    xm, xn = np.asarray(first["xm"]), np.asarray(first["xn"])
    if not np.any(xn):
        nboz = 0
    with jax.ensure_compile_time_eval():
        constants, grids = prepare_booz_xform_constants(
            nfp=int(rt.resolution.nfp), mboz=int(mboz), nboz=int(nboz),
            asym=lasym, xm=xm, xn=xn, xm_nyq=xm, xn_nyq=xn)
        constants, grids = _refine_booz_grids(
            constants, grids, oversample, rt.resolution.nfp)
        xm_b = np.asarray(grids.xm_b, dtype=float)
        xn_b = np.asarray(grids.xn_b, dtype=float)
    out = booz_xform_jax_impl(
        rmnc=stack("rmnc"), zmns=stack("zmns"), lmns=stack("lmns"),
        bmnc=stack("bmnc"), bsubumnc=stack("bsubumnc"),
        bsubvmnc=stack("bsubvmnc"), iota=stack("iota"),
        xm=jnp.asarray(xm), xn=jnp.asarray(xn), xm_nyq=jnp.asarray(xm),
        xn_nyq=jnp.asarray(xn), constants=constants, grids=grids,
        **(dict(rmns=stack("rmns"), zmnc=stack("zmnc"), lmnc=stack("lmnc"),
                bmns=stack("bmns"), bsubumns=stack("bsubumns"),
                bsubvmns=stack("bsubvmns")) if lasym else {}),
    )
    return {
        "bmnc_b": out["bmnc_b"], "bmns_b": out["bmns_b"],
        "xm_b": xm_b, "xn_b": xn_b,
        "iota_b": stack("iota"), "G_b": stack("G"), "I_b": stack("I"),
        "nfp": int(rt.resolution.nfp),
        "s_b": jnp.asarray(s_half, dtype=jnp.asarray(setup.s_full).dtype)[rows - 1],
        "psi_b": jnp.asarray(setup.psi_half)[rows],
        "psi_edge": jnp.asarray(setup.psi_edge),
    }


def _solve(deck: str):
    from vmex.core.input import VmecInput
    from vmex.core.optimize import solve_equilibrium

    started = time.perf_counter()
    equilibrium = solve_equilibrium(VmecInput.from_file(DATA / deck), verbose=False)
    if not equilibrium.result.converged:
        raise RuntimeError(f"{deck} did not converge; the timing would be meaningless")
    return equilibrium, time.perf_counter() - started


def _lane(mode: str, rt, *, surfaces, mboz, nboz, oversample):
    """The callable under test: one lane, everything else identical."""
    from vmex.core.omnigenity import boozer_spectrum_state

    call = boozer_spectrum_state if mode == "plan" else _spectrum_inline
    keywords = dict(surfaces=surfaces, mboz=mboz, nboz=nboz, oversample=oversample)
    return lambda state: call(state, rt, **keywords)


def _digest(values) -> str:
    """A content hash of one output block, exact to the last bit."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _timed(callable_, argument, *, warm_repeats: int) -> dict[str, float]:
    """Cold (trace + compile + run) and warm statistics of one callable."""
    import jax

    started = time.perf_counter()
    jax.block_until_ready(callable_(argument))
    cold = time.perf_counter() - started
    jax.block_until_ready(callable_(argument))
    samples = []
    for _ in range(warm_repeats):
        started = time.perf_counter()
        jax.block_until_ready(callable_(argument))
        samples.append(time.perf_counter() - started)
    return {
        "cold_s": cold,
        "warm_median_s": statistics.median(samples),
        "warm_min_s": min(samples),
    }


def _plan_build(state, rt, *, mboz, nboz, oversample) -> dict[str, Any]:
    """The once-per-process cost the plan lane pays, on its own.

    Runs first thing in its own child, before anything has traced the
    transform, so ``first_build_s`` really is what an empty compilation
    cache costs: almost all of it is XLA compiling the ~20 small programs
    of the op-by-op table build.  ``rebuild_s`` clears the registry and
    builds again with those programs already compiled — what a long-lived
    process, a warm ``~/.cache/vmex``, or a registry eviction pays.
    """
    import jax
    import numpy as np

    from vmex.core import omnigenity as omn
    from vmex.core.boozer_tables import boozer_input_tables

    tables = boozer_input_tables(state, rt, 1)
    xm, xn = np.asarray(tables["xm"]), np.asarray(tables["xn"])
    keywords = dict(
        nfp=int(rt.resolution.nfp), asym=bool(rt.setup.lasym), xm=xm, xn=xn,
        xm_nyq=xm, xn_nyq=xn, mboz=mboz, nboz=nboz if np.any(xn) else 0,
        oversample=oversample)
    seconds = []
    plan = None
    for _ in range(2):
        omn._PLAN_CACHE.clear()
        started = time.perf_counter()
        plan = omn._boozer_plan(**keywords)
        jax.block_until_ready(plan.tables)
        seconds.append(time.perf_counter() - started)
    return {
        "first_build_s": seconds[0],
        "rebuild_s": seconds[1],
        "tables_mib": sum(int(np.asarray(table).nbytes)
                          for table in plan.tables.values()) / 2**20,
        "quadrature": [int(plan.constants.ntheta), int(plan.constants.nzeta)],
    }


def _interleaved(lanes: dict[str, Any], state, *, rounds: int) -> dict[str, Any]:
    """Warm timings with the two lanes alternating call by call.

    Both lanes are compiled and warmed first, then each round times them in
    an order that flips every round, so a drifting machine cannot bias one
    lane.  Only warm figures: the plan build is a once-per-process cost and
    belongs to the fresh-process lanes, not here.
    """
    import jax
    import jax.numpy as jnp

    timers: dict[str, Any] = {}
    builders = {
        "jit_value": lambda call: jax.jit(lambda st: call(st)["bmnc_b"]),
        "jit_gradient": lambda call: jax.jit(jax.grad(
            lambda st: jnp.sum(call(st)["bmnc_b"] ** 2))),
        "eager_value": lambda call: (lambda st: call(st)["bmnc_b"]),
    }
    for timer, build in builders.items():
        compiled = {mode: build(call) for mode, call in lanes.items()}
        for callable_ in compiled.values():
            for _ in range(2):
                jax.block_until_ready(callable_(state))
        samples: dict[str, list[float]] = {mode: [] for mode in compiled}
        count = rounds if timer != "eager_value" else max(5, rounds // 2)
        for round_index in range(count):
            order = ("inline", "plan") if round_index % 2 == 0 else ("plan", "inline")
            for mode in order:
                started = time.perf_counter()
                jax.block_until_ready(compiled[mode](state))
                samples[mode].append(time.perf_counter() - started)
        timers[timer] = {
            mode: {"warm_median_s": statistics.median(values),
                   "warm_min_s": min(values),
                   "calls": len(values)}
            for mode, values in samples.items()
        }
    return timers


def _child(arguments: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    import vmex
    from _provenance import assert_repo_vmex, git_state

    surfaces = _surface_values(arguments.surfaces)
    equilibrium, solve_s = _solve(arguments.deck)
    state, rt = equilibrium.state, equilibrium.runtime
    shared = dict(surfaces=surfaces, mboz=arguments.mboz, nboz=arguments.nboz,
                  oversample=arguments.oversample)
    report: dict[str, Any] = {
        "deck": arguments.deck,
        "mode": arguments.mode,
        "solve_s": solve_s,
        "resolution": {
            "nfp": int(rt.resolution.nfp),
            "mpol": int(rt.resolution.mpol),
            "ntor": int(rt.resolution.ntor),
            "ns": int(np.asarray(rt.setup.s_full).shape[0]),
            "lasym": bool(rt.setup.lasym),
        },
        "versions": {
            "booz_xform_jax": metadata.version("booz_xform_jax"),
            "jax": jax.__version__,
            "jaxlib": metadata.version("jaxlib"),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "vmex": vmex.__version__,
        },
        "float64": bool(jax.config.jax_enable_x64),
        "devices": [str(device) for device in jax.devices()],
        **git_state(REPO),
        "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
    }

    if arguments.mode == "equivalence":
        outputs = {
            mode: _lane(mode, rt, **shared)(state)
            for mode in ("inline", "plan")
        }
        report["digests"] = {
            mode: {key: _digest(out[key]) for key in OUTPUT_KEYS}
            for mode, out in outputs.items()
        }
        differences = {}
        for key in OUTPUT_KEYS:
            left = np.asarray(outputs["inline"][key], dtype=np.float64)
            right = np.asarray(outputs["plan"][key], dtype=np.float64)
            scale = float(np.max(np.abs(left))) or 1.0
            differences[key] = {
                "max_abs": float(np.max(np.abs(left - right))),
                "max_abs_over_max_abs_value": float(
                    np.max(np.abs(left - right))) / scale,
            }
        report["differences"] = differences
        report["bit_identical"] = all(
            report["digests"]["inline"][key] == report["digests"]["plan"][key]
            for key in OUTPUT_KEYS)
        return report

    if arguments.mode == "plan_build":
        report["plan_build"] = _plan_build(
            state, rt, mboz=arguments.mboz, nboz=arguments.nboz,
            oversample=arguments.oversample)
        return report

    if arguments.mode == "interleaved":
        report["timers"] = _interleaved(
            {mode: _lane(mode, rt, **shared) for mode in ("inline", "plan")},
            state, rounds=arguments.warm_repeats)
        report["speedup_inline_over_plan"] = {
            f"{timer}_{statistic}": (
                values["inline"][statistic] / values["plan"][statistic])
            for timer, values in report["timers"].items()
            for statistic in ("warm_median_s", "warm_min_s")
        }
        return report

    lane = _lane(arguments.mode, rt, **shared)
    value = jax.jit(lambda st: lane(st)["bmnc_b"])
    gradient = jax.jit(jax.grad(lambda st: jnp.sum(lane(st)["bmnc_b"] ** 2)))
    report["jit_value"] = _timed(value, state, warm_repeats=arguments.warm_repeats)
    report["jit_gradient"] = _timed(gradient, state,
                                    warm_repeats=arguments.warm_repeats)
    report["eager_value"] = _timed(
        lambda st: lane(st)["bmnc_b"], state,
        warm_repeats=max(5, arguments.warm_repeats // 2))
    report["digests"] = {key: _digest(lane(state)[key]) for key in OUTPUT_KEYS}
    return report


# ---------------------------------------------------------------------------
# parent: fresh processes, empty caches, alternating lanes
# ---------------------------------------------------------------------------


def _run_child(arguments: argparse.Namespace, *, deck: str, mode: str
               ) -> tuple[dict[str, Any], float]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--deck", deck, "--mode", mode,
        "--mboz", str(arguments.mboz), "--nboz", str(arguments.nboz),
        "--oversample", str(arguments.oversample),
        "--surfaces", str(arguments.surfaces),
        "--warm-repeats", str(arguments.warm_repeats),
    ]
    cache = Path(tempfile.mkdtemp(prefix="vmex-booz-cache-"))
    environment = {
        **os.environ,
        "JAX_ENABLE_X64": "1",
        "JAX_COMPILATION_CACHE_DIR": str(cache),
        "VMEX_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "VMEX_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    }
    environment.pop("BOOZ_XFORM_JAX_TRIG_F32", None)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command, env=environment, capture_output=True, text=True,
            timeout=arguments.timeout, check=False)
    finally:
        wall = time.perf_counter() - started
        shutil.rmtree(cache, ignore_errors=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"{deck}/{mode} child failed ({process.returncode}):\n{process.stderr[-4000:]}")
    return json.loads(process.stdout), wall


def _sha256_prefix(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _lane_summary(runs: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
    """Best-of-repetitions per timer; the noise floor is one-sided upward."""
    reports = [report for report, _ in runs]
    return {
        "subprocess_wall_s": min(wall for _, wall in runs),
        "solve_s": min(report["solve_s"] for report in reports),
        **{
            timer: {
                statistic: min(report[timer][statistic] for report in reports)
                for statistic in ("cold_s", "warm_median_s", "warm_min_s")
            }
            for timer in ("jit_value", "jit_gradient", "eager_value")
        },
        "repetitions": [
            {"subprocess_wall_s": wall,
             **{timer: report[timer]
                for timer in ("jit_value", "jit_gradient", "eager_value")}}
            for report, wall in runs
        ],
    }


def _host() -> str:
    machine = platform.machine()
    release = platform.release()
    return f"{platform.system()} {release} ({machine}), CPU"


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if (arguments.mode is None) != (arguments.deck is None):
        raise SystemExit("--mode and --deck select a child lane; pass both or neither")
    if arguments.mode is not None:
        print(json.dumps(_child(arguments), indent=2, sort_keys=True))
        return 0

    from _provenance import git_state

    decks: list[dict[str, Any]] = []
    child_versions: dict[str, str] = {}
    for deck in arguments.decks:
        runs: dict[str, list[tuple[dict[str, Any], float]]] = {"inline": [], "plan": []}
        for _ in range(arguments.repeats):
            for mode in ("inline", "plan"):  # alternate, never batch a lane
                report, wall = _run_child(arguments, deck=deck, mode=mode)
                runs[mode].append((report, wall))
                child_versions = report["versions"]
        builds = [_run_child(arguments, deck=deck, mode="plan_build")[0]
                  for _ in range(arguments.repeats)]
        built = min(builds, key=lambda report: report["plan_build"]["first_build_s"])
        equivalence, _ = _run_child(arguments, deck=deck, mode="equivalence")
        interleaved, _ = _run_child(arguments, deck=deck, mode="interleaved")
        summary = {mode: _lane_summary(runs[mode]) for mode in runs}
        speedup = {
            f"{timer}_{statistic}": (
                summary["inline"][timer][statistic] / summary["plan"][timer][statistic])
            for timer in ("jit_value", "jit_gradient", "eager_value")
            for statistic in ("cold_s", "warm_median_s", "warm_min_s")
        }
        speedup["subprocess_wall"] = (
            summary["inline"]["subprocess_wall_s"] / summary["plan"]["subprocess_wall_s"])
        decks.append({
            "deck": deck,
            "sha256_prefix": _sha256_prefix(DATA / deck),
            "resolution": runs["plan"][0][0]["resolution"],
            "plan_build": built["plan_build"],
            "fresh_process": summary,
            "speedup_inline_over_plan": speedup,
            "interleaved": {
                "timers": interleaved["timers"],
                "speedup_inline_over_plan": interleaved["speedup_inline_over_plan"],
            },
            "equivalence": {
                "bit_identical": equivalence["bit_identical"],
                "digests": equivalence["digests"],
                "differences": equivalence["differences"],
            },
        })

    record = {
        "schema": SCHEMA,
        "provenance": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": _host(),
            "float64": True,
            "versions": child_versions,
            "command": (
                "python benchmarks/boozer_plan_adoption.py --decks "
                + " ".join(arguments.decks)),
            "protocol": (
                "One lane at a time, foreground, one fresh process per lane per "
                "repetition, each with its own empty JAX persistent-compilation-cache "
                "directory (the caller's cache is never read or written).  Lanes "
                "alternate inline, plan, inline, plan so machine drift hits both.  "
                "cold_s is the first jitted call (trace + compile + execute); warm "
                "statistics follow one untimed repeat.  Reported figures are the "
                "best over repetitions, the noise floor being one-sided upward.  "
                "inline is the pre-adoption call reconstructed verbatim from "
                "vmex/core/omnigenity.py at commit 2d3be2c0; plan is the shipped "
                "boozer_spectrum_state.  The fresh_process block is the protocol "
                "of record; its subprocess wall is dominated by the solve and the "
                "cold compile, so the interleaved block repeats the warm "
                "comparison with the two lanes alternating call by call in one "
                "process, which is where a 10-30% difference is legible."),
            "input_data_embedded": False,
            **git_state(REPO),
        },
        "configuration": {
            "mboz": arguments.mboz,
            "nboz": arguments.nboz,
            "oversample": arguments.oversample,
            "surfaces": list(_surface_values(arguments.surfaces)),
            "repeats": arguments.repeats,
            "warm_repeats": arguments.warm_repeats,
        },
        "decks": decks,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
