#!/usr/bin/env python3
"""Cold/warm runtime and peak-RSS evidence for FFT vs dense at 537 modes.

The case is the public CTH-like free-boundary fixture raised to
``MPOL=19/NTOR=14`` (``mnmax = 15 + 18*29 = 537``) — the exact deck of
``tests/test_high_mode_free_boundary_parity.py::
test_free_boundary_537_modes_converges_with_fft``, which sits above the
512-spectral-mode automatic-FFT threshold that the 238-mode suite cannot
reach.  Both transform kernels are measured the same way:

* **cold**: a fresh subprocess per transform (JAX compilation included),
  wrapped in ``/usr/bin/time -l`` (darwin) or ``/usr/bin/time -v`` (linux)
  so the OS reports the true process peak RSS;
* **warm**: an in-process repeat inside the same subprocess, so the second
  solve reuses the compiled executables.

The subprocess-level peak RSS therefore covers cold + warm together; the
in-child ``ru_maxrss`` snapshots after each solve attribute it more finely.
Run from the repository root (the script re-execs itself per transform)::

    python benchmarks/run_high_mode_fft.py \
      --out benchmarks/high_mode_fft.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy"
DEFAULT_MGRID = REPO / "examples" / "data" / "mgrid_cth_like.nc"

#: (ntor+1) + (mpol-1)*(2*ntor+1) = 15 + 18*29 = 537 active Fourier modes.
HIGH_MPOL, HIGH_NTOR = 19, 14
EXPECTED_MNMAX = 537


def high_mode_deck_text(deck: Path) -> str:
    """The 537-mode deck, built exactly as the guarded test builds it."""
    text = deck.read_text().split("&END")[0]
    text = text.replace("  MPOL = 5,", "  MPOL = 19,")
    text = text.replace("  NTOR = 4,", "  NTOR = 14,")
    text = text.replace("  NZETA = 36,", "  NZETA = 36,")  # 36 % 36 == 0
    text = text.replace("  FTOL_ARRAY  = 1.0E-10,", "  FTOL_ARRAY  = 1.0E-8,")
    return text + "&END\n"


def _ru_maxrss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _cpu_brand() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def child(transform: str, niter: int | None, deck: Path, mgrid: Path, report: Path) -> None:
    """One transform kernel: cold solve (compile included) + warm repeat."""
    import jax
    import vmex

    from benchmarks._provenance import assert_repo_vmex
    from vmex.core.freeboundary import solve_free_boundary
    from vmex.core.input import VmecInput
    from vmex.core.solver import resolution_from_input

    assert_repo_vmex(vmex.__file__, REPO)
    use_fft = transform == "fft"
    with tempfile.TemporaryDirectory(prefix="vmex-hm-fft-") as td:
        path = Path(td) / "input.cth_537"
        path.write_text(high_mode_deck_text(deck))
        inp = VmecInput.from_file(str(path))
    if niter is not None:
        from dataclasses import replace

        inp = replace(inp, niter_array=[int(niter)])
    resolution = resolution_from_input(inp, ns=int(inp.ns_array[0]))
    assert int(resolution.mnmax) == EXPECTED_MNMAX

    def run() -> dict:
        start = time.perf_counter()
        result = solve_free_boundary(inp, mgrid_path=str(mgrid), use_fft=use_fft, error_on_no_convergence=False)
        jax.block_until_ready(result.state.R_cos)
        wall = time.perf_counter() - start
        return {
            "wall_s": wall,
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "fsqr": float(result.fsqr),
            "fsqz": float(result.fsqz),
            "fsql": float(result.fsql),
            "ru_maxrss_bytes": _ru_maxrss_bytes(),
        }

    cold = run()
    warm = run()
    report.write_text(
        json.dumps(
            {
                "transform": transform,
                "use_fft": use_fft,
                "mnmax": int(resolution.mnmax),
                "ns": int(resolution.ns),
                "niter_cap": int(inp.niter_array[0]),
                "cold": cold,
                "warm": warm,
            },
            indent=2,
        )
        + "\n"
    )


def _time_command() -> list[str]:
    system = platform.system()
    if system == "Darwin":
        return ["/usr/bin/time", "-l"]
    if system == "Linux":
        return ["/usr/bin/time", "-v"]
    raise RuntimeError(f"no peak-RSS wrapper for {system}")


def _peak_rss_bytes(time_stderr: str) -> int:
    match = re.search(r"(\d+)\s+maximum resident set size", time_stderr)
    if match:  # darwin /usr/bin/time -l reports bytes
        return int(match.group(1))
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_stderr)
    if match:  # GNU time -v reports kbytes
        return int(match.group(1)) * 1024
    raise RuntimeError(f"peak RSS not found in time output:\n{time_stderr[-2000:]}")


def measure(transform: str, args: argparse.Namespace) -> dict:
    """Fresh subprocess for one transform, wrapped for OS peak-RSS."""
    with tempfile.TemporaryDirectory(prefix="vmex-hm-fft-") as td:
        report = Path(td) / f"{transform}.json"
        command = _time_command() + [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--transform",
            transform,
            "--deck",
            str(args.deck.resolve()),
            "--mgrid",
            str(args.mgrid.resolve()),
            "--child-report",
            str(report),
        ]
        if args.niter is not None:
            command += ["--niter", str(args.niter)]
        env = dict(os.environ)
        env["JAX_ENABLE_X64"] = "1"
        env["PYTHONPATH"] = str(REPO)
        start = time.perf_counter()
        proc = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(
                f"{transform} child failed with {proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        payload = json.loads(report.read_text())
    payload["subprocess"] = {
        "wall_s": wall,
        "peak_rss_bytes": _peak_rss_bytes(proc.stderr),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--mgrid", type=Path, default=DEFAULT_MGRID)
    parser.add_argument(
        "--niter",
        type=int,
        default=None,
        help="fixed iteration cap; default: the deck's own convergent budget, identical to the guarded 537-mode test",
    )
    parser.add_argument("--out", type=Path, default=REPO / "benchmarks" / "high_mode_fft.json")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transform", choices=("fft", "dense"), help=argparse.SUPPRESS)
    parser.add_argument("--child-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    for path in (args.deck, args.mgrid):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.child:
        child(args.transform, args.niter, args.deck, args.mgrid, args.child_report)
        return

    modes = {name: measure(name, args) for name in ("fft", "dense")}

    import jax
    import vmex

    from benchmarks._provenance import assert_repo_vmex, git_state

    report = {
        "schema": 2,
        "case": args.deck.name,
        "modifications": {
            "mpol": HIGH_MPOL,
            "ntor": HIGH_NTOR,
            "ftol": 1.0e-8,
            "mnmax": EXPECTED_MNMAX,
        },
        "input_data_embedded": False,
        "provenance": {
            **git_state(REPO),
            "vmex_version": vmex.__version__,
            "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
        },
        "protocol": {
            "cold": "fresh subprocess per transform; compile included; "
            "wall_s measured around the solve, subprocess.wall_s "
            "around the whole process",
            "warm": "in-process repeat inside the same subprocess",
            "peak_rss": "subprocess.peak_rss_bytes is the OS-reported "
            "process peak (cold + warm together); "
            "ru_maxrss_bytes snapshots attribute it per solve",
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": _cpu_brand(),
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "x64": True,  # exported to every child
        },
        "modes": modes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
