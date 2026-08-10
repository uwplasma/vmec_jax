#!/usr/bin/env python
"""Apples-to-apples QI and QA optimization benchmark for SIMSOPT and VMEX.

Both backends use the same input, VMEC resolution, boundary variables,
least-squares residual, SciPy tolerances, and 15-evaluation budget.  SIMSOPT
uses centered MPI finite differences; VMEX uses exact implicit derivatives.
Run one case per fresh process; VMEX's persistent compilation cache is disabled
so the reported end-to-end time is cold.  Every case updates its entry in the
per-objective consolidated ``qi_results.json`` / ``qa_results.json`` artifact
(host platform and versions are recorded in each entry's provenance).  Use
``--plot`` afterwards.  The independent VMEX state/wout parity audit runs
after the wall timer.

Examples
--------
VMEX::

    python benchmarks/qi_simsopt_vmex.py --backend vmex --objective qi \
        --schedule 1 --ess

SIMSOPT on all 14 logical CPUs::

    mpiexec -n 14 python benchmarks/qi_simsopt_vmex.py \
        --backend simsopt --objective qi --schedule 1 --ess

Plot the consolidated results in a result directory::

    python benchmarks/qi_simsopt_vmex.py \
        --plot benchmarks/optimization_crosscode
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "examples" / "data" / "input.nfp2_QI_seed"
DEFAULT_RESULTS = REPO / "benchmarks" / "optimization_crosscode"
SURFACES = np.linspace(0.1, 1.0, 6)
QI_SETTINGS = {
    "mboz": 16,
    "nboz": 16,
    "nphi": 97,
    "nalpha": 25,
    "n_levels": 16,
}
MAX_NFEV = 15
QI_ASPECT_TARGET = 5.0
IOTA_MIN = 0.33
MAXIMUM_MIRROR = 0.21
MAXIMUM_ELONGATION = 8.0
QA_ASPECT_TARGET = 6.0
QA_IOTA_TARGET = 0.41


def _schedule(text: str) -> tuple[int, ...]:
    """Parse a comma-separated continuation schedule."""
    values = tuple(int(value) for value in text.split(",") if value.strip())
    if not values or any(value < 1 or value > 8 for value in values):
        raise argparse.ArgumentTypeError("schedule modes must be between 1 and 8")
    return values


def _ess_scale(names: list[str], alpha: float = 1.2) -> np.ndarray:
    """Return the same exponential spectral scale for either naming scheme."""
    modes = []
    for name in names:
        match = re.search(r"(?:\(|,)(-?\d+),(-?\d+)\)?$", name)
        if match is None:
            raise ValueError(f"cannot read Fourier mode from {name!r}")
        modes.append(max(abs(int(match.group(1))), abs(int(match.group(2)))))
    levels = np.asarray(modes, dtype=float)
    return np.exp(-alpha * levels) / np.exp(-alpha)


def _mirror_ratio_from_wout(wout: Any) -> float:
    """Mirror ratio on VMEC's outer half-mesh and internal angular grid."""
    ntheta = max(int(wout.mpol) * 2 + 6, 1)
    nzeta = max(int(wout.ntor) * 2 + 4, 1)
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2.0 * np.pi * np.arange(nzeta) / (nzeta * int(wout.nfp))
    angle = (
        theta[:, None, None] * np.asarray(wout.xm_nyq)[None, None, :]
        - zeta[None, :, None] * np.asarray(wout.xn_nyq)[None, None, :]
    )
    bmag = np.einsum("tzk,k->tz", np.cos(angle), np.asarray(wout.bmnc)[-1])
    if getattr(wout, "bmns", None) is not None:
        bmag += np.einsum(
            "tzk,k->tz", np.sin(angle), np.asarray(wout.bmns)[-1]
        )
    return float((np.max(bmag) - np.min(bmag)) / (np.max(bmag) + np.min(bmag)))


def _max_elongation_from_wout(wout: Any) -> float:
    """VMEX equivalent-ellipse elongation reconstructed from wout boundary modes."""
    ntheta = max(4 * int(wout.mpol), 32)
    nphi = max(4 * int(wout.ntor) + 1, 24)
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2.0 * np.pi * np.arange(nphi) / (nphi * int(wout.nfp))
    xm = np.asarray(wout.xm, dtype=float)
    angle = (
        theta[:, None, None] * xm[None, None, :]
        - zeta[None, :, None] * np.asarray(wout.xn)[None, None, :]
    )
    cosine, sine = np.cos(angle), np.sin(angle)
    rc = np.asarray(wout.rmnc)[-1]
    zs = np.asarray(wout.zmns)[-1]
    rs = np.zeros_like(rc) if getattr(wout, "rmns", None) is None else np.asarray(wout.rmns)[-1]
    zc = np.zeros_like(rc) if getattr(wout, "zmnc", None) is None else np.asarray(wout.zmnc)[-1]
    radius = np.einsum("k,tpk->tp", rc, cosine) + np.einsum("k,tpk->tp", rs, sine)
    height = np.einsum("k,tpk->tp", zc, cosine) + np.einsum("k,tpk->tp", zs, sine)
    dr = np.einsum("k,tpk->tp", -rc * xm, sine) + np.einsum("k,tpk->tp", rs * xm, cosine)
    dz = np.einsum("k,tpk->tp", -zc * xm, sine) + np.einsum("k,tpk->tp", zs * xm, cosine)
    dtheta = 2.0 * np.pi / ntheta
    perimeter = dtheta * np.sum(np.sqrt(dr * dr + dz * dz), axis=0)
    area = 0.5 * dtheta * np.abs(np.sum(radius * dz - height * dr, axis=0))
    tiny = np.finfo(float).tiny
    area = np.maximum(area, tiny)
    root = np.sqrt(8.0 * np.pi * area + perimeter * perimeter)
    discriminant = 2.0 * np.sqrt(3.0) * perimeter * root - 40.0 * np.pi * area + 4.0 * perimeter**2
    regularization = 32.0 * np.finfo(float).eps * np.maximum(perimeter**2, tiny)
    sqrt_discriminant = np.sqrt(np.sqrt(discriminant**2 + regularization**2))
    semi_major = (np.sqrt(3.0) * (root + sqrt_discriminant) + 3.0 * perimeter) / (12.0 * np.pi)
    semi_minor = area / (np.pi * np.maximum(semi_major, tiny))
    return float(np.max(semi_major / np.maximum(semi_minor, tiny)))


def shared_residual_from_wout(wout: Any, objective: str = "qi") -> np.ndarray:
    """The backend-neutral residual used for every timed case."""
    from vmex import optimize as opt
    from vmex.core.omnigenity import omnigenity_residual

    mean_iota = float(np.mean(np.asarray(wout.iotas)[1:]))
    if objective == "qa":
        qa = opt.QuasisymmetryRatioResidual(SURFACES, 1, 0)
        scalars = np.asarray([
            float(wout.aspect) - QA_ASPECT_TARGET,
            mean_iota - QA_IOTA_TARGET,
        ])
        return np.concatenate([np.asarray(qa.residuals(wout)), scalars])
    if objective != "qi":
        raise ValueError(f"unknown objective {objective!r}")

    booz = opt.boozer_modes_from_wout(
        wout,
        surfaces=SURFACES,
        mboz=QI_SETTINGS["mboz"],
        nboz=QI_SETTINGS["nboz"],
        jit=True,
    )
    qi = omnigenity_residual(
        bmnc_b=booz["bmnc_b"],
        xm_b=booz["xm_b"],
        xn_b=booz["xn_b"],
        iota_b=booz["iota_b"],
        nfp=booz["nfp"],
        nphi=QI_SETTINGS["nphi"],
        nalpha=QI_SETTINGS["nalpha"],
        n_levels=QI_SETTINGS["n_levels"],
    )
    mirror_excess = max(_mirror_ratio_from_wout(wout) - MAXIMUM_MIRROR, 0.0)
    elongation_excess = max(
        _max_elongation_from_wout(wout) - MAXIMUM_ELONGATION, 0.0
    )
    scalars = np.asarray([
        np.sqrt(0.005) * (float(wout.aspect) - QI_ASPECT_TARGET),
        np.sqrt(10.0) * max(IOTA_MIN - abs(mean_iota), 0.0),
        np.sqrt(10.0) * mirror_excess,
        np.sqrt(10.0) * elongation_excess,
    ])
    return np.concatenate([np.asarray(qi["residuals1d"]), scalars])


def _input_resolution(max_mode: int) -> tuple[int, int, int, int]:
    mpol = max(max_mode + 2, 5)
    ntor = mpol
    return mpol, ntor, 2 * mpol + 6, 2 * ntor + 4


def _provenance(input_path: Path) -> dict[str, Any]:
    import jax
    import scipy
    import vmex

    result = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "vmex": getattr(vmex, "__version__", "unknown"),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    try:
        import simsopt
        result["simsopt"] = simsopt.__version__
    except ImportError:
        result["simsopt"] = None
    return result


def _checkpoint(args: argparse.Namespace, **updates: Any) -> None:
    """Persist enough progress for an externally time-limited benchmark."""
    if args.checkpoint is None:
        return
    payload = {
        "backend": args.backend,
        "objective": args.objective,
        "schedule": list(args.schedule),
        "max_mode": args.schedule[-1],
        "ess": args.ess,
        "status": "running",
    }
    payload.update(updates)
    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.checkpoint)


def _vmex_case(args: argparse.Namespace) -> dict[str, Any]:
    # This benchmark measures cold end-to-end latency, including compilation.
    # Set this before the first VMEX/JAX import in the process.
    os.environ["VMEX_COMPILATION_CACHE"] = "disabled"
    import jax.numpy as jnp
    from scipy.optimize import least_squares
    from vmex import optimize as opt
    from vmex.core.input import VmecInput
    from vmex.core.omnigenity import QIResidual

    inp = VmecInput.from_file(args.input)
    qi = QIResidual(SURFACES, **QI_SETTINGS)
    qa = opt.QuasisymmetryRatioResidual(SURFACES, 1, 0)

    def iota_floor(state, runtime):
        return jnp.maximum(IOTA_MIN - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

    def mirror_excess(state, runtime):
        return jnp.maximum(opt.mirror_ratio(state, runtime) - MAXIMUM_MIRROR, 0.0)

    def elongation_excess(state, runtime):
        return jnp.maximum(opt.max_elongation(state, runtime) - MAXIMUM_ELONGATION, 0.0)

    qi_terms = [
        (qi, 0.0, 1.0),
        (opt.aspect_ratio, QI_ASPECT_TARGET, 0.005),
        (iota_floor, 0.0, 10.0),
        (mirror_excess, 0.0, 10.0),
        (elongation_excess, 0.0, 10.0),
    ]
    qa_terms = [
        (qa, 0.0, 1.0),
        (opt.aspect_ratio, QA_ASPECT_TARGET, 1.0),
        (opt.mean_iota, QA_IOTA_TARGET, 1.0),
    ]
    terms = qi_terms if args.objective == "qi" else qa_terms
    setup_seconds = 0.0
    compile_seconds = 0.0
    optimize_seconds = 0.0
    costs: list[float] = []
    cost_stages: list[int] = []
    stage_records = []
    failed_trials = 0
    total_nfev = 0
    total_njev = 0
    initial_cost = None
    wout_initial_cost = None
    first_problem = None
    problem = result = None
    wall_started = time.perf_counter()

    _checkpoint(
        args,
        phase="setup",
        history_includes_initial=True,
        accepted_costs=[],
        accepted_cost_stages=[],
    )

    for stage_index, max_mode in enumerate(args.schedule):
        mpol, ntor, ntheta, nzeta = _input_resolution(max_mode)
        inp = replace(inp, delt=0.5).change_resolution(
            mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nzeta
        )
        started = time.perf_counter()
        problem = opt.VmecProblem.from_tuples(
            inp, terms, max_mode=max_mode, use_ess=args.ess
        )
        stage_setup = time.perf_counter() - started
        setup_seconds += stage_setup
        started = time.perf_counter()
        initial = problem.compile_residual_and_jacobian(progress=False)
        stage_compile = time.perf_counter() - started
        compile_seconds += stage_compile
        stage_costs = [float(initial.value)]
        _checkpoint(
            args,
            phase="optimization",
            accepted_costs=[*costs, *stage_costs],
            accepted_cost_stages=[*cost_stages, stage_index],
            elapsed_seconds=time.perf_counter() - wall_started,
        )

        def callback(intermediate_result):
            stage_costs.append(float(intermediate_result.cost))
            _checkpoint(
                args,
                phase="optimization",
                accepted_costs=[*costs, *stage_costs],
                accepted_cost_stages=[
                    *cost_stages, *([stage_index] * len(stage_costs))
                ],
                elapsed_seconds=time.perf_counter() - wall_started,
            )

        kwargs = {} if problem.scales is None else {"x_scale": problem.scales}
        started = time.perf_counter()
        result = least_squares(
            problem.residual,
            problem.x0,
            jac=problem.residual_jac,
            callback=callback,
            max_nfev=MAX_NFEV,
            ftol=1.0e-4,
            gtol=1.0e-4,
            xtol=1.0e-10,
            **kwargs,
        )
        stage_optimize = time.perf_counter() - started
        optimize_seconds += stage_optimize
        inp = problem.input_from_x(result.x)
        costs.extend(stage_costs)
        cost_stages.extend([stage_index] * len(stage_costs))
        total_nfev += int(result.nfev)
        total_njev += int(result.njev)
        failed = int(problem.metadata["holder"]["failed_trials"])
        failed_trials += failed
        if initial_cost is None:
            initial_cost = float(initial.value)
            first_problem = problem
        stage_records.append({
            "max_mode": max_mode,
            "dofs": int(problem.x0.size),
            "setup_seconds": stage_setup,
            "compile_seconds": stage_compile,
            "optimize_seconds": stage_optimize,
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "initial_cost": float(initial.value),
            "final_cost": float(result.cost),
            "failed_trials": failed,
        })

    assert problem is not None and result is not None and first_problem is not None
    total_seconds = time.perf_counter() - wall_started

    # This independent wout reconstruction validates backend parity but is not
    # part of the optimization workflow.  Keep it outside the cold wall timer
    # so only VMEX is not charged for an extra diagnostic or warmed by it.
    audit_input = first_problem.input_from_x(first_problem.x0)
    audit_equilibrium = opt.solve_equilibrium(audit_input)
    parity = shared_residual_from_wout(audit_equilibrium.wout, args.objective)
    wout_initial_cost = 0.5 * float(parity @ parity)
    return {
        "backend": "vmex",
        "objective": args.objective,
        "schedule": list(args.schedule),
        "max_mode": args.schedule[-1],
        "ess": args.ess,
        "workers": None,
        "parallelism": "XLA-managed CPU threading within each solve",
        "compilation_cache": "disabled",
        "dofs": int(problem.x0.size),
        "nfev": total_nfev,
        "njev": total_njev,
        "setup_seconds": setup_seconds,
        "compile_seconds": compile_seconds,
        "optimize_seconds": optimize_seconds,
        "total_seconds": total_seconds,
        "initial_cost": initial_cost,
        "wout_initial_cost": wout_initial_cost,
        "final_cost": float(result.cost),
        "accepted_costs": costs,
        "accepted_cost_stages": cost_stages,
        "history_includes_initial": True,
        "failed_trials": failed_trials,
        "stages": stage_records,
    }


def _configure_simsopt_vmec(vmec: Any, max_mode: int) -> Any:
    mpol, ntor, ntheta, nzeta = _input_resolution(max_mode)
    vmec.indata.delt = 0.5
    vmec.indata.mpol = mpol
    vmec.indata.ntor = ntor
    vmec.indata.ntheta = ntheta
    vmec.indata.nzeta = nzeta
    surface = vmec.boundary.change_resolution(
        max(vmec.boundary.mpol, max_mode),
        max(vmec.boundary.ntor, max_mode),
    )
    vmec.boundary = surface
    surface.fix_all()
    surface.fixed_range(
        mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False
    )
    surface.fix("rc(0,0)")
    return surface


def _simsopt_case(args: argparse.Namespace) -> dict[str, Any] | None:
    # The shared wout objective imports VMEX/JAX even though equilibrium
    # derivatives come from SIMSOPT.  Disable its persistent cache too so every
    # fresh-process comparison is cold under the same policy.
    os.environ["VMEX_COMPILATION_CACHE"] = "disabled"
    from simsopt._core.optimizable import Optimizable
    from simsopt.mhd import Vmec
    from simsopt.objectives import LeastSquaresProblem
    from simsopt.solve import least_squares_mpi_solve
    from simsopt.util import MpiPartition
    from vmex.core.wout import read_wout

    mpi = MpiPartition()
    wall_started = time.perf_counter()
    vmec = Vmec(
        str(args.input), mpi=mpi, keep_all_files=True, verbose=False
    )

    class SharedResidual(Optimizable):
        def __init__(self):
            super().__init__(depends_on=[vmec])

        def J(self):
            vmec.run()
            output = Path(vmec.output_file)
            try:
                return shared_residual_from_wout(
                    read_wout(output), args.objective
                )
            finally:
                output.unlink(missing_ok=True)
                input_name = (
                    "input." + output.name.removeprefix("wout_").removesuffix(".nc")
                )
                output.with_name(input_name).unlink(missing_ok=True)

    objective = SharedResidual()
    setup_seconds = 0.0
    optimize_seconds = 0.0
    costs: list[float] = []
    cost_stages: list[int] = []
    stage_records = []
    total_nfev = 0
    total_njev = 0
    problem = None
    if mpi.proc0_world:
        _checkpoint(
            args,
            phase="setup",
            history_includes_initial=False,
            accepted_costs=[],
            accepted_cost_stages=[],
        )

    for stage_index, max_mode in enumerate(args.schedule):
        setup_started = time.perf_counter()
        _configure_simsopt_vmec(vmec, max_mode)
        problem = LeastSquaresProblem.from_tuples([(objective.J, 0.0, 1.0)])
        stage_setup = time.perf_counter() - setup_started
        setup_seconds += stage_setup
        accepted_costs = []

        def callback(intermediate_result):
            accepted_costs.append(float(intermediate_result.cost))
            if mpi.proc0_world:
                partial_costs = [*costs, *accepted_costs]
                _checkpoint(
                    args,
                    phase="optimization",
                    accepted_costs=partial_costs,
                    accepted_cost_stages=[
                        *cost_stages,
                        *([stage_index] * len(accepted_costs)),
                    ],
                    elapsed_seconds=time.perf_counter() - wall_started,
                )

        x_scale = _ess_scale(problem.dof_names) if args.ess else 1.0
        mpi.comm_world.Barrier()
        started = time.perf_counter()
        least_squares_mpi_solve(
            problem,
            mpi,
            grad=True,
            diff_method="centered",
            rel_step=1.0e-5,
            abs_step=1.0e-7,
            max_nfev=MAX_NFEV,
            ftol=1.0e-4,
            gtol=1.0e-4,
            xtol=1.0e-10,
            x_scale=x_scale,
            callback=callback,
        )
        stage_optimize = time.perf_counter() - started
        optimize_seconds += stage_optimize

        if mpi.proc0_world:
            objective_log = max(
                Path.cwd().glob("objective_*.dat"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            table = np.loadtxt(
                objective_log, delimiter=",", skiprows=5, ndmin=2
            )
            objective_log.unlink()
            jacobian_log = max(
                Path.cwd().glob("jac_log_*.dat"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            jacobian_log.unlink()
            evaluated_cost = 0.5 * table[:, -1]
            stage_costs = [float(evaluated_cost[0]), *accepted_costs]
            costs.extend(stage_costs)
            cost_stages.extend([stage_index] * len(stage_costs))
            total_nfev += int(table.shape[0])
            stage_njev = len(accepted_costs) + 1  # initial point + accepted updates
            total_njev += stage_njev
            stage_records.append({
                "max_mode": max_mode,
                "dofs": int(problem.dof_size),
                "setup_seconds": stage_setup,
                "compile_seconds": 0.0,
                "optimize_seconds": stage_optimize,
                "nfev": int(table.shape[0]),
                "njev": stage_njev,
                "initial_cost": stage_costs[0],
                "final_cost": stage_costs[-1],
                "failed_trials": None,
            })
        mpi.comm_world.Barrier()

    assert problem is not None
    input_stem = args.input.name.removeprefix("input.")
    for path in Path.cwd().glob(f"input.{input_stem}_{mpi.group:03d}_*"):
        path.unlink()
    for path in Path.cwd().glob(f"wout_{input_stem}_{mpi.group:03d}_*.nc"):
        path.unlink()
    payload = None
    if mpi.proc0_world:
        payload = {
            "backend": "simsopt",
            "objective": args.objective,
            "schedule": list(args.schedule),
            "max_mode": args.schedule[-1],
            "ess": args.ess,
            "workers": mpi.nprocs_world,
            "parallelism": "one SIMSOPT MPI finite-difference group per rank",
            "worker_thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "XLA_FLAGS",
                )
            },
            "compilation_cache": (
                "disabled for shared JAX wout residual; not applicable to VMEC2000"
            ),
            "dofs": int(problem.dof_size),
            "nfev": total_nfev,
            "njev": total_njev,
            "setup_seconds": setup_seconds,
            "compile_seconds": 0.0,
            "optimize_seconds": optimize_seconds,
            "total_seconds": time.perf_counter() - wall_started,
            "initial_cost": costs[0],
            "wout_initial_cost": costs[0],
            "final_cost": costs[-1],
            "accepted_costs": costs,
            "accepted_cost_stages": cost_stages,
            "history_includes_initial": True,
            "failed_trials": None,
            "stages": stage_records,
        }
    mpi.comm_world.Barrier()
    return payload


def _schedule_label(schedule: tuple[int, ...]) -> str:
    if len(schedule) == 1:
        return f"mode{schedule[0]}"
    if schedule == (1, 2, 3, 4, 5):
        return "ladder1-5"
    return "modes-" + "-".join(str(mode) for mode in schedule)


def _case_key(
    backend: str, schedule: tuple[int, ...], ess: bool
) -> str:
    return f"{backend}_{_schedule_label(schedule)}_ess-{str(ess).lower()}"


def _results_path(result_dir: Path, objective: str) -> Path:
    return result_dir / f"{objective}_results.json"


def _load_cases(result_dir: Path, objective: str) -> dict[str, Any]:
    path = _results_path(result_dir, objective)
    if not path.exists():
        return {}
    return json.loads(path.read_text())["cases"]


def _store_case(
    result_dir: Path, objective: str, key: str, row: dict[str, Any]
) -> Path:
    """Update one entry of the per-objective consolidated artifact."""
    path = _results_path(result_dir, objective)
    cases = _load_cases(result_dir, objective)
    cases[key] = row
    path.write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n"
    )
    return path


def _plot(result_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for objective in ("qi", "qa")
        for row in _load_cases(result_dir, objective).values()
        if "backend" in row
    ]
    expected = {
        (objective, backend, schedule, ess)
        for objective, schedules in (
            ("qi", [(mode,) for mode in range(1, 9)] + [(1, 2, 3, 4, 5)]),
            ("qa", [(1, 2, 3, 4, 5), (2,), (5,)]),
        )
        for backend in ("simsopt", "vmex")
        for schedule in schedules
        for ess in (False, True)
    }
    found = {
        (row["objective"], row["backend"], tuple(row["schedule"]), row["ess"])
        for row in rows
    }
    missing = sorted(expected - found)
    if missing:
        raise SystemExit(f"missing benchmark cases: {missing}")

    colors = {"simsopt": "#e07a1f", "vmex": "#2878b5"}
    markers = {"simsopt": "o", "vmex": "s"}
    linestyles = {False: "--", True: "-"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for backend in ("simsopt", "vmex"):
        for ess in (False, True):
            selected = sorted(
                (
                    row for row in rows
                    if row["objective"] == "qi"
                    and row["backend"] == backend
                    and row["ess"] == ess
                    and len(row["schedule"]) == 1
                ),
                key=lambda r: r["max_mode"],
            )
            ax.plot(
                [r["max_mode"] for r in selected],
                [r["total_seconds"] for r in selected],
                color=colors[backend],
                linestyle=linestyles[ess],
                marker=markers[backend],
                label=f"{backend.upper()}, ESS {'on' if ess else 'off'}",
            )
            censored = [r for r in selected if r.get("censored", False)]
            if censored:
                ax.plot(
                    [r["max_mode"] for r in censored],
                    [r["total_seconds"] for r in censored],
                    linestyle="none",
                    marker=markers[backend],
                    markerfacecolor="white",
                    markeredgecolor=colors[backend],
                    markersize=8,
                )
    ax.set(
        xlabel="maximum optimized Fourier mode",
        ylabel="cold end-to-end wall time (s)",
        xticks=list(range(1, 9)),
        yscale="log",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    time_path = result_dir / "qi_optimization_time.png"
    fig.savefig(time_path)
    plt.close(fig)

    history_cases = [
        ((1, 2, 3, 4, 5), "mode ladder [1, 2, 3, 4, 5]"),
        ((2,), "direct max_mode = 2"),
        ((5,), "direct max_mode = 5"),
    ]
    objective_paths = []
    for objective in ("qi", "qa"):
        fig, axes = plt.subplots(
            3, 1, figsize=(8.2, 8.4), dpi=180, sharex=False
        )
        for (schedule, title), ax in zip(history_cases, axes):
            for backend in ("simsopt", "vmex"):
                for ess in (False, True):
                    row = next(
                        item for item in rows
                        if item["objective"] == objective
                        and item["backend"] == backend
                        and tuple(item["schedule"]) == schedule
                        and item["ess"] == ess
                    )
                    if not row.get("accepted_costs"):
                        continue
                    ax.plot(
                        range(len(row["accepted_costs"])),
                        row["accepted_costs"],
                        color=colors[backend],
                        linestyle=linestyles[ess],
                        marker=markers[backend],
                        markersize=3,
                        label=(
                            f"{backend.upper()}, ESS "
                            f"{'on' if ess else 'off'}"
                        ),
                    )
            ax.set_title(title)
            ax.set_yscale("log")
            ax.set_ylabel("least-squares cost")
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("recorded optimization point")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        objective_path = result_dir / f"{objective}_objective_history.png"
        fig.savefig(objective_path)
        plt.close(fig)
        objective_paths.append(objective_path)
    print(time_path)
    for objective_path in objective_paths:
        print(objective_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("simsopt", "vmex"))
    parser.add_argument("--objective", choices=("qi", "qa"), default="qi")
    parser.add_argument("--schedule", type=_schedule)
    parser.add_argument(
        "--max-mode", type=int, choices=range(1, 9),
        help="backward-compatible alias for a one-stage --schedule",
    )
    parser.add_argument("--ess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="internal progress file used by the isolated-case runner",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.plot is not None:
        _plot(args.plot.resolve())
        return
    if args.schedule is not None and args.max_mode is not None:
        parser.error("use either --schedule or --max-mode, not both")
    if args.schedule is None and args.max_mode is not None:
        args.schedule = (args.max_mode,)
    if args.backend is None or args.schedule is None:
        parser.error("--backend and --schedule are required unless --plot is used")
    args.input = args.input.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    result = _vmex_case(args) if args.backend == "vmex" else _simsopt_case(args)
    if result is None:
        return
    result["max_nfev"] = MAX_NFEV
    result["resolution"] = dict(zip(
        ("mpol", "ntor", "ntheta", "nzeta"),
        _input_resolution(args.schedule[-1]),
    ))
    result["resolutions"] = [dict(zip(
        ("mpol", "ntor", "ntheta", "nzeta"), _input_resolution(mode)
    )) for mode in args.schedule]
    result["ns"] = 31
    result["provenance"] = _provenance(args.input)
    result["status"] = "complete"
    key = _case_key(args.backend, args.schedule, args.ess)
    print(_store_case(args.result_dir, args.objective, key, result))


if __name__ == "__main__":
    main()
