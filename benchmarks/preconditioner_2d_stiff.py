#!/usr/bin/env python3
"""Re-measure the 2D block preconditioner's iteration reduction on stiff decks.

The README/docs figure ``docs/_static/figures/readme_precond.webp`` compares
iterations to the deck FTOL with the default 1D radial preconditioner against
the opt-in matrix-free 2D block preconditioner
(:mod:`vmex.core.preconditioner_2d`).  Until this script existed the counts
were typed into ``benchmarks/make_readme_figures.py`` as literals with no
committed run behind them.  Running this writes
``benchmarks/preconditioner_2d_stiff_cases.json`` with the counts, the
converged ``wb`` from both paths, and full provenance (commit, versions,
host), and ``make_readme_figures.py`` plots that artifact instead of literals.

Usage::

    python benchmarks/preconditioner_2d_stiff.py            # every case
    python benchmarks/preconditioner_2d_stiff.py --only aspect100_ns51
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks._provenance import assert_repo_vmex, git_state  # noqa: E402

OUT = REPO / "benchmarks" / "preconditioner_2d_stiff_cases.json"
SCHEMA = "vmex.preconditioner-2d-stiff-cases/2"

# label, deck stem, ns, ftol, in the README figure
CASES: dict[str, dict[str, object]] = {
    "aspect100_ns51": {
        "label": "aspect-100 tokamak, ns=51",
        "deck": "input.circular_tokamak_aspect_100",
        "ns": 51,
        "ftol": 1.0e-11,
        "in_readme_figure": True,
    },
    "aspect100_ns101": {
        "label": "aspect-100 tokamak, ns=101",
        "deck": "input.circular_tokamak_aspect_100",
        "ns": 101,
        "ftol": 1.0e-11,
        "in_readme_figure": True,
    },
    "nfp4_QH_finite_beta_ns51": {
        "label": "nfp4 QH, finite beta, ns=51",
        "deck": "input.nfp4_QH_finite_beta",
        "ns": 51,
        "ftol": 1.0e-11,
        "in_readme_figure": True,
    },
}

# The 2D Newton step activates on the finest grid once the residual falls
# below ``threshold`` and the iteration index passes ``start_iteration``
# (VMEC2000 ``prec2d_threshold`` / ``evolve.f``).  These are the settings the
# opt-in guard ``tests/test_preconditioner_2d.py`` uses.
PREC2D_KWARGS = {
    "threshold": 1.0e-6,
    "gmres_restart": 60,
    "gmres_max_restarts": 3,
    "gmres_rtol": 3.0e-3,
}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _host() -> str:
    machine = platform.machine()
    release = platform.release()
    return f"{platform.system()} {release} ({machine}), CPU"


def _cpu_brand() -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_case(key: str) -> dict[str, object]:
    import vmex
    from vmex.core.fourier import Resolution
    from vmex.core.input import VmecInput
    from vmex.core.preconditioner_2d import Prec2DConfig
    from vmex.core.solver import resolution_from_input, solve

    assert_repo_vmex(vmex.__file__, REPO)
    spec = CASES[key]
    deck = REPO / "examples" / "data" / str(spec["deck"])
    inp = VmecInput.from_file(str(deck))
    base = resolution_from_input(inp)
    res = Resolution(
        mpol=base.mpol,
        ntor=base.ntor,
        ntheta=base.ntheta,
        nzeta=base.nzeta,
        nfp=base.nfp,
        lasym=base.lasym,
        ns=int(spec["ns"]),
    )
    ftol = float(spec["ftol"])

    t0 = time.perf_counter()
    one_d = solve(inp, res, mode="jit", ftol=ftol, max_iterations=20000)
    wall_1d = time.perf_counter() - t0

    cfg = Prec2DConfig(**PREC2D_KWARGS)
    t0 = time.perf_counter()
    two_d = solve(inp, res, mode="jit", ftol=ftol, max_iterations=20000, prec2d=cfg)
    wall_2d = time.perf_counter() - t0

    if not (one_d.converged and two_d.converged):
        raise RuntimeError(f"{key}: a path failed to converge; refusing to record it")

    wb_1d = float(one_d.wb)
    wb_2d = float(two_d.wb)
    return {
        "key": key,
        "label": spec["label"],
        "deck": f"examples/data/{spec['deck']}",
        "ns": int(spec["ns"]),
        "ftol": ftol,
        "iterations_1d": int(one_d.iterations),
        "iterations_2d": int(two_d.iterations),
        "wb_1d": wb_1d,
        "wb_2d": wb_2d,
        "wb_relative_difference": abs(wb_1d - wb_2d) / abs(wb_1d),
        # Wall time is recorded for honesty, not as a speed claim: these are
        # single cold runs including compilation, on whatever host ran them.
        "wall_s_1d_including_compile": round(wall_1d, 2),
        "wall_s_2d_including_compile": round(wall_2d, 2),
        "in_readme_figure": bool(spec["in_readme_figure"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=sorted(CASES))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    keys = args.only or list(CASES)
    cases = []
    for key in keys:
        print(f"[precond2d] {key} ...", flush=True)
        record = run_case(key)
        ratio = record["iterations_1d"] / record["iterations_2d"]
        print(
            f"[precond2d] {key}: {record['iterations_1d']} -> "
            f"{record['iterations_2d']} ({ratio:.1f}x fewer), "
            f"wb rel {record['wb_relative_difference']:.1e}",
            flush=True,
        )
        cases.append(record)

    provenance = dict(git_state(REPO))
    provenance.update(
        {
            "schema": 2,
            "measurement_date": time.strftime("%Y-%m-%d"),
            "host": _host(),
            "cpu": _cpu_brand(),
            "python": platform.python_version(),
            "versions": {
                "vmex": _package_version("vmex"),
                "jax": _package_version("jax"),
                "jaxlib": _package_version("jaxlib"),
                "solvax": _package_version("solvax"),
            },
            "prec2d_config": dict(PREC2D_KWARGS),
            "protocol": (
                "One process per invocation, both paths solved back to back "
                "from the same deck and resolution, mode='jit', float64. "
                "Iteration counts are deterministic for a given build; the "
                "wall times include compilation and are not a speed claim."
            ),
        }
    )

    payload = {
        "schema": SCHEMA,
        "provenance": provenance,
        "what_is_measured": (
            "Iterations to reach the stated FTOL with the default 1D radial "
            "preconditioner versus the opt-in matrix-free 2D block "
            "preconditioner (vmex.core.preconditioner_2d). Both paths "
            "converge to the same equilibrium; the recorded wb relative "
            "difference is the parity witness. The default 1D path is "
            "unaffected by the presence of the 2D path."
        ),
        "generator": "benchmarks/preconditioner_2d_stiff.py",
        "figure": "docs/_static/figures/readme_precond.webp",
        "cases": cases,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[precond2d] wrote {args.out.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
