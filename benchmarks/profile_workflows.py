"""One driver for VMEX workflow performance, memory, and compile observability.

Every flagship workflow runs under the same measurement contract (plan
section 9): stages are timed separately, asynchronous device work is fenced
with ``block_until_ready``, compile and cache activity is counted from JAX's
own logs, and each run emits one machine-readable JSON record.

Timing regimes are process-level where they must be:

- ``cold``            new process, empty persistent compilation cache;
- ``cache_reload``    new process, populated persistent cache;
- ``warm``            same process, same shapes and static arguments;
- ``warm_newparams``  same process, changed physical parameters, same shapes;
- ``reshape``         same process, changed resolution/shape.

The driver re-executes itself in a subprocess for the two cold regimes, so a
"cold" number can never accidentally include this process's warm state.  Warm
regimes run in-process with explicit warm-up separated from timed repeats.

Usage::

    python benchmarks/profile_workflows.py --list
    python benchmarks/profile_workflows.py F1 F4 --regimes cold warm
    python benchmarks/profile_workflows.py --all --out benchmarks/results/

Every record carries provenance (commit, dirty flag, platform, JAX versions,
x64 flag, case hash) and the compile/trace counters for the measured stage.
No number in this file is edited by hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
SCHEMA = 1

_COLD_REGIMES = ("cold", "cache_reload")
_WARM_REGIMES = ("warm", "warm_newparams", "reshape")
REGIMES = _COLD_REGIMES + _WARM_REGIMES


# ---------------------------------------------------------------------------
# Compile/trace counters from JAX's own logging
# ---------------------------------------------------------------------------


class _CompileCounter(logging.Handler):
    """Count traces and XLA compilations from ``jax_log_compiles`` records.

    Importing vmex sets ``jax_logging_level = "ERROR"``, which filters the
    WARNING-level "Compiling ..." records this reads — the same trap that
    silently zeroed the multigrid compile tests.  ``install`` therefore
    forces the logger back to WARNING after the vmex import.
    """

    def __init__(self) -> None:
        super().__init__()
        self.compiles = 0
        self.traces = 0
        self.cache_misses: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Compiling "):
            self.compiles += 1
        elif message.startswith("Finished tracing"):
            self.traces += 1
        elif "cache miss" in message.lower():
            self.cache_misses.append(message[:200])

    @classmethod
    def install(cls, *, explain_cache_misses: bool = False) -> "_CompileCounter":
        import jax

        jax.config.update("jax_log_compiles", True)
        if explain_cache_misses:
            # Opt-in only: on jax 0.9.2 this flag breaks
            # jax.lax.platform_dependent ("not enough values to unpack" inside
            # the cache-key explainer), which SOLVAX's tridiagonal solve uses
            # -- the debug flag would crash the very solve being measured.
            jax.config.update("jax_explain_cache_misses", True)
        counter = cls()
        logger = logging.getLogger("jax")
        logger.addHandler(counter)
        logger.setLevel(logging.WARNING)
        return counter

    def snapshot(self) -> dict[str, Any]:
        return {
            "compiles": self.compiles,
            "traces": self.traces,
            "cache_miss_reasons": self.cache_misses[:20],
        }

    def reset(self) -> None:
        self.compiles = 0
        self.traces = 0
        self.cache_misses = []


def _provenance(case_paths: tuple[Path, ...]) -> dict[str, Any]:
    import hashlib

    import jax
    import jaxlib

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return "unknown"

    case_sha = hashlib.sha256()
    for path in sorted(case_paths):
        case_sha.update(path.read_bytes())
    return {
        "schema": SCHEMA,
        "repo": "uwplasma/vmex",
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "case_sha256": case_sha.hexdigest(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "node": platform.node(),
        },
        "jax": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(),
            "x64": bool(jax.config.jax_enable_x64),
        },
    }


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return int(peak) * (1024 if sys.platform.startswith("linux") else 1)


# ---------------------------------------------------------------------------
# Workflow registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Workflow:
    """One measured workflow: a builder returning staged callables.

    ``build()`` runs untimed setup that is not part of any claim (path
    resolution, config construction) and returns ``(stages, variants)``:
    ``stages`` maps stage name -> zero-argument callable executed and timed
    in order; ``variants`` optionally maps the warm-regime names to callables
    that re-run the *measured* stage with changed parameters or shapes.
    Every callable must fence its own device work with
    ``block_until_ready`` on what it returns.
    """

    ident: str
    title: str
    build: Callable[[], tuple[dict[str, Callable[[], Any]], dict[str, Callable[[], Any]]]]
    cases: tuple[str, ...]


def _block(value: Any) -> Any:
    import jax

    return jax.block_until_ready(value)


def _read_input(name: str):
    from vmex.core.input import VmecInput

    return VmecInput.from_file(DATA / name)


def _wf_fixed_single() -> tuple[dict, dict]:
    import dataclasses as dc

    import numpy as np

    from vmex.core import solver

    inp = _read_input("input.li383_low_res")
    state = {}

    def solve():
        state["result"] = solver.solve(inp)
        return _block(state["result"].state.R_cos)

    def solve_newparams():
        rbc = np.array(inp.rbc)
        rbc[inp.ntor, 1] *= 1.01
        state["result"] = solver.solve(dc.replace(inp, rbc=rbc))
        return _block(state["result"].state.R_cos)

    def solve_reshape():
        state["result"] = solver.solve(
            inp.change_resolution(mpol=int(inp.mpol) + 1,
                                  ntor=int(inp.ntor),
                                  ntheta=2 * (int(inp.mpol) + 1) + 6,
                                  nzeta=int(inp.nzeta)))
        return _block(state["result"].state.R_cos)

    return ({"solve": solve},
            {"warm_newparams": solve_newparams, "reshape": solve_reshape})


def _wf_fixed_multigrid() -> tuple[dict, dict]:
    from vmex.core.multigrid import solve_multigrid

    inp = _read_input("input.cth_like_fixed_bdy")

    def solve():
        result = solve_multigrid(inp)
        return _block(result.state.R_cos)

    return ({"solve": solve}, {})


def _wf_fixed_polished() -> tuple[dict, dict]:
    from vmex.core.multigrid import solve_multigrid

    inp = _read_input("input.shaped_tokamak_pressure_polished")

    def solve():
        result = solve_multigrid(inp, polish_force_balance=True)
        return _block(result.polished_state.R_cos)

    return ({"solve": solve}, {})


def _wf_scalar_gradient() -> tuple[dict, dict]:
    import jax

    from vmex.core import implicit as im

    inp = _read_input("input.li383_low_res")
    params = im.params_from_input(inp, device=None)
    held = {}

    def objective(p):
        solution = im.run(inp, p, ns=13, ftol=1.0e-11, max_iterations=8000,
                          device=None)
        from vmex.core.statephysics import aspect_ratio

        return aspect_ratio(solution.state, solution.runtime)

    def value():
        held["value"] = objective(params)
        return _block(held["value"])

    def gradient():
        held["grad"] = jax.grad(objective)(params)
        return _block(held["grad"].rbc)

    return ({"value": value, "gradient": gradient}, {})


def _wf_hot_restart_scan() -> tuple[dict, dict]:
    import dataclasses as dc

    import numpy as np

    from vmex.core.multigrid import solve_multigrid

    inp = _read_input("input.solovev")
    steps = 4

    def scan():
        result = solve_multigrid(inp)
        for step in range(steps):
            rbc = np.array(inp.rbc)
            rbc[inp.ntor, 1] *= 1.0 + 0.002 * (step + 1)
            result = solve_multigrid(
                dc.replace(inp, rbc=rbc), restart_from=result)
        return _block(result.state.R_cos)

    return ({"scan": scan}, {})


def _wf_boozer_one_surface() -> tuple[dict, dict]:
    from vmex.core import optimize as opt
    from vmex.core.omnigenity import boozer_bmnc_state

    inp = _read_input("input.li383_low_res")
    held = {}

    def solve():
        held["eq"] = opt.solve_equilibrium(inp, verbose=False)
        return _block(held["eq"].state.R_cos)

    def transform():
        held["bmnc"] = boozer_bmnc_state(
            held["eq"].state, held["eq"].runtime, surfaces=(0.5,))
        return _block(next(iter(held["bmnc"].values())))

    return ({"solve": solve, "transform": transform}, {})


WORKFLOWS: dict[str, Workflow] = {
    "F1": Workflow("F1", "fixed-boundary single-grid value",
                   _wf_fixed_single, ("input.li383_low_res",)),
    "F2": Workflow("F2", "fixed-boundary multigrid value",
                   _wf_fixed_multigrid, ("input.cth_like_fixed_bdy",)),
    "F3": Workflow("F3", "fixed-boundary polished value",
                   _wf_fixed_polished,
                   ("input.shaped_tokamak_pressure_polished",)),
    "F4": Workflow("F4", "implicit scalar value + gradient",
                   _wf_scalar_gradient, ("input.li383_low_res",)),
    "F6": Workflow("F6", "hot-restart parameter scan",
                   _wf_hot_restart_scan, ("input.solovev",)),
    "B1": Workflow("B1", "in-process Boozer transform, one surface",
                   _wf_boozer_one_surface, ("input.li383_low_res",)),
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _run_in_process(ident: str, regime: str) -> dict[str, Any]:
    """Measure one workflow in this process (warm regimes, or a cold child)."""
    # Import vmex BEFORE installing the counter: its _configure_jax_logging
    # sets jax_logging_level = "ERROR" at package import, which would silence
    # the "Compiling ..." records the counter reads if it ran afterwards.
    # Workflow builders import vmex lazily, so without this line the first
    # build would re-silence the logger and every compile count would read 0.
    import vmex  # noqa: F401

    counter = _CompileCounter.install()
    workflow = WORKFLOWS[ident]
    build_started = time.perf_counter()
    stages, variants = workflow.build()
    build_seconds = time.perf_counter() - build_started

    timings: dict[str, float] = {"build": build_seconds}
    counters: dict[str, Any] = {}
    for name, stage in stages.items():
        counter.reset()
        started = time.perf_counter()
        stage()
        timings[name] = time.perf_counter() - started
        counters[name] = counter.snapshot()

    if regime in _WARM_REGIMES:
        # Warm-up already happened above; time the regime-specific repeat.
        repeat = variants.get(regime) or next(iter(stages.values()))
        counter.reset()
        samples = []
        repeats = 3 if regime == "warm" else 1
        for _ in range(repeats):
            started = time.perf_counter()
            repeat()
            samples.append(time.perf_counter() - started)
        timings[regime] = sorted(samples)[len(samples) // 2]
        counters[regime] = counter.snapshot()

    return {
        "workflow": ident,
        "title": workflow.title,
        "regime": regime,
        "timing_s": timings,
        "compile": counters,
        "memory_bytes": {"peak_host_rss": _peak_rss_bytes()},
        **_provenance(tuple(DATA / c for c in workflow.cases)),
    }


def _run_cold(ident: str, regime: str, cache_dir: Path) -> dict[str, Any]:
    """Run one workflow in a fresh process with a controlled persistent cache.

    ``cold`` empties the cache first; ``cache_reload`` reuses whatever the
    matching ``cold`` run left behind, so a reload claim always follows a
    logged population of the same directory.
    """
    if regime == "cold":
        for stale in cache_dir.glob("*"):
            stale.unlink()
    elif regime == "cache_reload" and not any(cache_dir.glob("*")):
        # A reload claim needs a logged population of this same directory:
        # run one unrecorded cold child to fill it.
        _run_cold(ident, "cold", cache_dir)
    env = dict(
        os.environ,
        VMEX_COMPILATION_CACHE="1",
        VMEX_COMPILATION_CACHE_DIR=str(cache_dir),
        VMEX_PROFILE_CHILD="1",
    )
    entries_before = len(list(cache_dir.glob("*")))
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), ident,
         "--regimes", regime, "--child"],
        capture_output=True, text=True, env=env, timeout=3600, cwd=ROOT,
    )
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"{ident}/{regime} child failed:\n{proc.stderr[-4000:]}")
    record = json.loads(proc.stdout.strip().splitlines()[-1])
    record["timing_s"]["process_wall"] = wall
    record["cache"] = {
        "directory": str(cache_dir),
        "entries_before": entries_before,
        "entries_after": len(list(cache_dir.glob("*"))),
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("idents", nargs="*", help="workflow ids (see --list)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--regimes", nargs="+", default=["warm"],
                        choices=list(REGIMES))
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for one JSON file per record")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--child", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list:
        for ident, workflow in sorted(WORKFLOWS.items()):
            print(f"{ident:4s} {workflow.title}")
        return 0

    idents = sorted(WORKFLOWS) if args.all else args.idents
    unknown = sorted(set(idents) - set(WORKFLOWS))
    if unknown:
        parser.error(f"unknown workflows: {unknown} (see --list)")
    if not idents:
        parser.error("give workflow ids or --all")

    if args.child:
        # Child mode: measure in-process; the parent controls the cache.
        import jax

        jax.config.update("jax_enable_x64", True)
        record = _run_in_process(idents[0], args.regimes[0])
        print(json.dumps(record))
        return 0

    import jax

    jax.config.update("jax_enable_x64", True)
    records = []
    cache_dir = args.cache_dir or (ROOT / "benchmarks" / ".profile_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ident in idents:
        for regime in args.regimes:
            if regime in _COLD_REGIMES:
                record = _run_cold(ident, regime, cache_dir)
            else:
                record = _run_in_process(ident, regime)
            records.append(record)
            summary = {k: round(v, 3) for k, v in record["timing_s"].items()}
            print(f"[{ident}/{regime}] {summary}", file=sys.stderr)
            if args.out is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                path = args.out / f"{ident}_{regime}.json"
                path.write_text(json.dumps(record, indent=1, sort_keys=True)
                                + "\n", encoding="utf-8")
    print(json.dumps(records, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
