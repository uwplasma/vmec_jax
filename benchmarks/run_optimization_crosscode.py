#!/usr/bin/env python
"""Run the QI/QA cross-code matrix in isolated processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from qi_simsopt_vmex import (
    DEFAULT_INPUT,
    DEFAULT_RESULTS,
    MAX_NFEV,
    _case_key,
    _input_resolution,
    _load_cases,
    _provenance,
    _store_case,
)


REPO = Path(__file__).resolve().parents[1]


def _run_case(command, *, environment, timeout: float, cwd: Path) -> None:
    """Run one isolated case and stop its whole MPI process group on timeout."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _timed_out_payload(
    payload, *, backend, objective, schedule, ess, timeout, workers
):
    """Complete the common schema for a right-censored case."""
    payload.update({
        "backend": backend,
        "objective": objective,
        "schedule": list(schedule),
        "max_mode": schedule[-1],
        "ess": ess,
        "status": "timed_out",
        "censored": True,
        "time_limit_seconds": timeout,
        "total_seconds": timeout,
        "max_nfev": MAX_NFEV,
        "ns": 31,
        "dofs": 4 * schedule[-1] * (schedule[-1] + 1),
        "resolution": dict(zip(
            ("mpol", "ntor", "ntheta", "nzeta"),
            _input_resolution(schedule[-1]),
        )),
        "resolutions": [dict(zip(
            ("mpol", "ntor", "ntheta", "nzeta"),
            _input_resolution(mode),
        )) for mode in schedule],
        "provenance": _provenance(DEFAULT_INPUT),
        "workers": workers if backend == "simsopt" else None,
    })
    if payload.get("accepted_costs"):
        payload["final_cost"] = payload["accepted_costs"][-1]
        if backend == "vmex":
            payload["initial_cost"] = payload["accepted_costs"][0]
    return payload


def cases():
    """Yield the timing sweep and the three requested history schedules."""
    for mode in range(1, 9):
        for ess in (False, True):
            yield "qi", (mode,), ess
    for objective in ("qi", "qa"):
        for schedule in ((1, 2, 3, 4, 5), (2,), (5,)):
            if objective == "qi" and schedule != (1, 2, 3, 4, 5):
                continue  # direct QI modes 2 and 5 are in the timing sweep
            for ess in (False, True):
                yield objective, schedule, ess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("vmex", "simsopt", "both"), default="both"
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="right-censor a case that exceeds this cold wall time",
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    backends = ("vmex", "simsopt") if args.backend == "both" else (args.backend,)
    args.result_dir = args.result_dir.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO)
    script = Path(__file__).with_name("qi_simsopt_vmex.py")
    for backend in backends:
        for objective, schedule, ess in cases():
            key = _case_key(backend, schedule, ess)
            stored = _load_cases(args.result_dir, objective)
            if args.skip_existing and key in stored:
                payload = stored[key]
                if payload.get("status") == "timed_out":
                    payload = _timed_out_payload(
                        payload,
                        backend=backend,
                        objective=objective,
                        schedule=schedule,
                        ess=ess,
                        timeout=payload.get(
                            "time_limit_seconds", args.timeout_seconds
                        ),
                        workers=args.workers,
                    )
                    _store_case(args.result_dir, objective, key, payload)
                elif "status" not in payload:
                    payload["status"] = "complete"
                    _store_case(args.result_dir, objective, key, payload)
                continue
            checkpoint = args.result_dir / f"{objective}_{key}.partial.json"
            command = [
                sys.executable,
                str(script),
                "--backend",
                backend,
                "--objective",
                objective,
                "--schedule",
                ",".join(map(str, schedule)),
                "--result-dir",
                str(args.result_dir),
                "--checkpoint",
                str(checkpoint),
                "--ess" if ess else "--no-ess",
            ]
            checkpoint.unlink(missing_ok=True)
            if backend == "simsopt":
                command = ["mpiexec", "-n", str(args.workers), *command]
                # One finite-difference group already occupies each logical
                # CPU.  Keep BLAS/XLA work inside a rank serial so 14 ranks do
                # not each create another 14-thread pool.
                for variable in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                ):
                    environment[variable] = "1"
                environment["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
            print(" ".join(command), flush=True)
            try:
                with tempfile.TemporaryDirectory(
                    prefix="vmex-crosscode-"
                ) as case_directory:
                    _run_case(
                        command,
                        environment=environment,
                        timeout=args.timeout_seconds,
                        cwd=Path(case_directory),
                    )
            except subprocess.TimeoutExpired:
                payload = (
                    json.loads(checkpoint.read_text())
                    if checkpoint.exists() else {
                        "backend": backend,
                        "objective": objective,
                        "schedule": list(schedule),
                        "max_mode": schedule[-1],
                        "ess": ess,
                        "accepted_costs": [],
                        "accepted_cost_stages": [],
                    }
                )
                payload = _timed_out_payload(
                    payload,
                    backend=backend,
                    objective=objective,
                    schedule=schedule,
                    ess=ess,
                    timeout=args.timeout_seconds,
                    workers=args.workers,
                )
                _store_case(args.result_dir, objective, key, payload)
                print(
                    f"right-censored after {args.timeout_seconds:g} s: "
                    f"{objective} {key}",
                    flush=True,
                )
            finally:
                checkpoint.unlink(missing_ok=True)

    if args.backend == "both":
        subprocess.run(
            [sys.executable, str(script), "--plot", str(args.result_dir)],
            cwd=REPO,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
