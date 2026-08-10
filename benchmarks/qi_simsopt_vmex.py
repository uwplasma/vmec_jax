#!/usr/bin/env python
"""Apples-to-apples QI optimization benchmark for SIMSOPT and VMEX.

Both backends use the same input, VMEC resolution, boundary variables,
least-squares residual, SciPy tolerances, and 15-evaluation budget.  SIMSOPT
uses centered MPI finite differences; VMEX uses exact implicit derivatives.
Run one case per fresh process; every case updates its entry in the single
consolidated ``qi_results.json`` artifact (host platform and versions are
recorded in each entry's provenance).  Use ``--plot`` afterwards.

Examples
--------
VMEX::

    python benchmarks/qi_simsopt_vmex.py --backend vmex --max-mode 1 --ess

SIMSOPT on all 14 logical CPUs::

    mpiexec -n 14 python benchmarks/qi_simsopt_vmex.py \
        --backend simsopt --max-mode 1 --ess

Plot the consolidated results in a result directory::

    python benchmarks/qi_simsopt_vmex.py --plot benchmarks/optimization_crosscode
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
RESULTS_NAME = "qi_results.json"
SURFACES = np.linspace(0.1, 1.0, 6)
QI_SETTINGS = {
    "mboz": 16,
    "nboz": 16,
    "nphi": 97,
    "nalpha": 25,
    "n_levels": 16,
}
MAX_NFEV = 15
ASPECT_TARGET = 4.0
IOTA_MIN = 0.33
MAXIMUM_MIRROR = 0.21
MAXIMUM_ELONGATION = 8.0


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


def shared_residual_from_wout(wout: Any) -> np.ndarray:
    """The backend-neutral residual used for every timed case."""
    from vmex import optimize as opt
    from vmex.core.omnigenity import omnigenity_residual

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
    mean_iota = float(np.mean(np.asarray(wout.iotas)[1:]))
    mirror_excess = max(_mirror_ratio_from_wout(wout) - MAXIMUM_MIRROR, 0.0)
    elongation_excess = max(
        _max_elongation_from_wout(wout) - MAXIMUM_ELONGATION, 0.0
    )
    scalars = np.asarray([
        np.sqrt(0.005) * (float(wout.aspect) - ASPECT_TARGET),
        np.sqrt(10.0) * max(IOTA_MIN - abs(mean_iota), 0.0),
        mirror_excess,
        elongation_excess,
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


def _vmex_case(args: argparse.Namespace) -> dict[str, Any]:
    import jax.numpy as jnp
    from scipy.optimize import least_squares
    from vmex import optimize as opt
    from vmex.core.input import VmecInput
    from vmex.core.omnigenity import QIResidual

    mpol, ntor, ntheta, nzeta = _input_resolution(args.max_mode)
    inp = replace(VmecInput.from_file(args.input), delt=0.5).change_resolution(
        mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nzeta
    )
    qi = QIResidual(SURFACES, **QI_SETTINGS)

    def iota_floor(state, runtime):
        return jnp.maximum(IOTA_MIN - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

    def mirror_excess(state, runtime):
        return jnp.maximum(opt.mirror_ratio(state, runtime) - MAXIMUM_MIRROR, 0.0)

    def elongation_excess(state, runtime):
        return jnp.maximum(opt.max_elongation(state, runtime) - MAXIMUM_ELONGATION, 0.0)

    terms = [
        (qi, 0.0, 1.0),
        (opt.aspect_ratio, ASPECT_TARGET, 0.005),
        (iota_floor, 0.0, 10.0),
        (mirror_excess, 0.0, 1.0),
        (elongation_excess, 0.0, 1.0),
    ]
    started = time.perf_counter()
    problem = opt.VmecProblem.from_tuples(
        inp, terms, max_mode=args.max_mode, use_ess=args.ess
    )
    setup_seconds = time.perf_counter() - started
    parity = shared_residual_from_wout(
        problem.equilibrium_from_x(problem.x0).wout
    )
    started = time.perf_counter()
    initial = problem.compile_residual_and_jacobian(progress=False)
    compile_seconds = time.perf_counter() - started
    costs = [float(initial.value)]

    def callback(x, *unused):
        rows = problem.residual(x)
        costs.append(0.5 * float(rows @ rows))

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
    optimize_seconds = time.perf_counter() - started
    return {
        "backend": "vmex",
        "max_mode": args.max_mode,
        "ess": args.ess,
        "workers": None,
        "parallelism": "XLA-managed CPU threading within each solve",
        "dofs": int(problem.x0.size),
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "setup_seconds": setup_seconds,
        "compile_seconds": compile_seconds,
        "optimize_seconds": optimize_seconds,
        "total_seconds": setup_seconds + compile_seconds + optimize_seconds,
        "initial_cost": float(initial.value),
        "wout_initial_cost": 0.5 * float(parity @ parity),
        "final_cost": float(result.cost),
        "accepted_costs": costs,
        "failed_trials": int(problem.metadata["holder"]["failed_trials"]),
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
    from simsopt._core.optimizable import Optimizable
    from simsopt.mhd import Vmec
    from simsopt.objectives import LeastSquaresProblem
    from simsopt.solve import least_squares_mpi_solve
    from simsopt.util import MpiPartition
    from vmex.core.wout import read_wout

    mpi = MpiPartition()
    setup_started = time.perf_counter()
    vmec = Vmec(
        str(args.input), mpi=mpi, keep_all_files=True, verbose=False
    )
    _configure_simsopt_vmec(vmec, args.max_mode)

    class SharedResidual(Optimizable):
        def __init__(self):
            super().__init__(depends_on=[vmec])

        def J(self):
            vmec.run()
            output = Path(vmec.output_file)
            try:
                return shared_residual_from_wout(read_wout(output))
            finally:
                output.unlink(missing_ok=True)
                input_name = (
                    "input." + output.name.removeprefix("wout_").removesuffix(".nc")
                )
                output.with_name(input_name).unlink(missing_ok=True)

    objective = SharedResidual()
    problem = LeastSquaresProblem.from_tuples([(objective.J, 0.0, 1.0)])
    setup_seconds = time.perf_counter() - setup_started
    accepted = [np.asarray(problem.x, dtype=float).copy()]

    def callback(x, *unused):
        accepted.append(np.asarray(x, dtype=float).copy())

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
    optimize_seconds = time.perf_counter() - started
    input_stem = args.input.name.removeprefix("input.")
    for path in Path.cwd().glob(f"input.{input_stem}_{mpi.group:03d}_*"):
        path.unlink()
    payload = None
    if mpi.proc0_world:
        objective_log = max(
            Path.cwd().glob("objective_*.dat"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        table = np.loadtxt(objective_log, delimiter=",", skiprows=5, ndmin=2)
        objective_log.unlink()
        jacobian_log = max(
            Path.cwd().glob("jac_log_*.dat"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        jacobian_log.unlink()
        evaluated_x = table[:, 2 : 2 + problem.dof_size]
        evaluated_cost = 0.5 * table[:, -1]
        costs = []
        for x in accepted:
            distances = np.linalg.norm(evaluated_x - x[None, :], axis=1)
            index = int(np.argmin(distances))
            if distances[index] > 1.0e-10:
                raise RuntimeError("accepted iterate is absent from SIMSOPT log")
            costs.append(float(evaluated_cost[index]))
        payload = {
            "backend": "simsopt",
            "max_mode": args.max_mode,
            "ess": args.ess,
            "workers": mpi.nprocs_world,
            "parallelism": "one SIMSOPT MPI finite-difference group per rank",
            "dofs": int(problem.dof_size),
            "nfev": int(table.shape[0]),
            "njev": max(len(accepted) - 1, 0),
            "setup_seconds": setup_seconds,
            "compile_seconds": 0.0,
            "optimize_seconds": optimize_seconds,
            "total_seconds": setup_seconds + optimize_seconds,
            "initial_cost": costs[0],
            "wout_initial_cost": costs[0],
            "final_cost": costs[-1],
            "accepted_costs": costs,
            "failed_trials": None,
        }
    mpi.comm_world.Barrier()
    return payload


def _case_key(backend: str, max_mode: int, ess: bool) -> str:
    return f"{backend}_mode{max_mode}_ess-{str(ess).lower()}"


def _load_cases(result_dir: Path) -> dict[str, Any]:
    path = result_dir / RESULTS_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())["cases"]


def _store_case(result_dir: Path, key: str, row: dict[str, Any]) -> Path:
    """Update one entry of the consolidated benchmark artifact."""
    path = result_dir / RESULTS_NAME
    cases = _load_cases(result_dir)
    cases[key] = row
    path.write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n"
    )
    return path


def _plot(result_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in _load_cases(result_dir).values() if "backend" in row]
    expected = {(b, m, e) for b in ("simsopt", "vmex") for m in range(1, 5) for e in (False, True)}
    found = {(row["backend"], row["max_mode"], row["ess"]) for row in rows}
    missing = sorted(expected - found)
    if missing:
        raise SystemExit(f"missing benchmark cases: {missing}")

    colors = {"simsopt": "#e07a1f", "vmex": "#2878b5"}
    markers = {False: "o", True: "s"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for backend in ("simsopt", "vmex"):
        for ess in (False, True):
            selected = sorted(
                (r for r in rows if r["backend"] == backend and r["ess"] == ess),
                key=lambda r: r["max_mode"],
            )
            ax.plot(
                [r["max_mode"] for r in selected],
                [r["total_seconds"] for r in selected],
                color=colors[backend],
                marker=markers[ess],
                label=f"{backend.upper()}, ESS {'on' if ess else 'off'}",
            )
    ax.set(
        xlabel="maximum optimized Fourier mode",
        ylabel="cold end-to-end wall time (s)",
        xticks=[1, 2, 3, 4],
        yscale="log",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    time_path = result_dir / "qi_optimization_time.png"
    fig.savefig(time_path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), dpi=180, sharex=False)
    for max_mode, ax in zip(range(1, 5), axes.flat):
        for backend in ("simsopt", "vmex"):
            for ess in (False, True):
                row = next(
                    r for r in rows
                    if r["backend"] == backend and r["max_mode"] == max_mode and r["ess"] == ess
                )
                ax.plot(
                    range(len(row["accepted_costs"])),
                    row["accepted_costs"],
                    color=colors[backend],
                    linestyle="-" if ess else "--",
                    marker=markers[ess],
                    markersize=3,
                    label=f"{backend.upper()}, ESS {'on' if ess else 'off'}",
                )
        ax.set_title(f"max_mode = {max_mode}")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("accepted optimization iteration")
    axes[1, 1].set_xlabel("accepted optimization iteration")
    axes[0, 0].set_ylabel("least-squares cost")
    axes[1, 0].set_ylabel("least-squares cost")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    objective_path = result_dir / "qi_objective_history.png"
    fig.savefig(objective_path)
    plt.close(fig)
    print(time_path)
    print(objective_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("simsopt", "vmex"))
    parser.add_argument("--max-mode", type=int, choices=range(1, 5))
    parser.add_argument("--ess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.plot is not None:
        _plot(args.plot.resolve())
        return
    if args.backend is None or args.max_mode is None:
        parser.error("--backend and --max-mode are required unless --plot is used")
    args.input = args.input.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    result = _vmex_case(args) if args.backend == "vmex" else _simsopt_case(args)
    if result is None:
        return
    result["max_nfev"] = MAX_NFEV
    result["resolution"] = dict(zip(("mpol", "ntor", "ntheta", "nzeta"), _input_resolution(args.max_mode)))
    result["ns"] = 25
    result["provenance"] = _provenance(args.input)
    key = _case_key(args.backend, args.max_mode, args.ess)
    print(_store_case(args.result_dir, key, result))


if __name__ == "__main__":
    main()
