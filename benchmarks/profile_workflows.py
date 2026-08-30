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
# Measurements must bind to this repository: running the script directly puts
# benchmarks/ (not the repo root) on sys.path, so a bare ``import vmex`` would
# silently measure whatever older copy is installed while the record stamps
# this repo's commit.
sys.path.insert(0, str(ROOT))
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
        # Untracked files are excluded: committed baselines are written into
        # the tree by the tool itself.
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "case_sha256": case_sha.hexdigest(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
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


def _qa_seed_input(max_mode: int = 1):
    """The examples' rotating-ellipse QA seed at optimization resolution."""
    import dataclasses as dc

    import numpy as np

    inp = _read_input("input.minimal_seed_nfp2")
    rbc, zbs = np.array(inp.rbc), np.array(inp.zbs)
    # The exactly circular torus has zero first-order iota sensitivity; the
    # explicit rotating-ellipse perturbation gives the optimizer a QA basin.
    rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.02, 0.02
    inp = dc.replace(inp, rbc=rbc, zbs=zbs, delt=0.5)
    mpol = max_mode + 2
    return inp.change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)


def _qa_terms(opt):
    qs = opt.QuasisymmetryRatioResidual(
        (0.25, 0.5, 0.75), helicity_m=1, helicity_n=0)
    return [(qs.residuals_state, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0)]


def _wf_vector_residual_jacobian() -> tuple[dict, dict]:
    from vmex.core import optimize as opt

    problem = opt.VmecProblem.from_tuples(
        _qa_seed_input(), _qa_terms(opt), max_mode=1)
    x0 = problem.x0
    held = {}

    def residual():
        held["rows"] = problem.residual(x0)
        return _block(held["rows"])

    def jacobian():
        held["jac"] = problem.residual_jac(x0)
        return _block(held["jac"])

    return ({"residual": residual, "jacobian": jacobian}, {})


def _wf_scalar_optimization() -> tuple[dict, dict]:
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize

    from vmex.core import optimize as opt

    inp = _qa_seed_input()

    def loss(equilibrium_state, solver_context):
        rows = opt.residuals_from_tuples(
            equilibrium_state, solver_context, _qa_terms(opt))
        return 0.5 * jnp.vdot(rows, rows)

    problem = opt.VmecProblem.from_loss(inp, loss, max_mode=1)

    def optimize():
        # gtol=ftol=0: the campaign always runs the full ten accepted steps.
        result = minimize(
            problem.value_and_grad, np.asarray(problem.x0), jac=True,
            method="L-BFGS-B",
            options={"maxiter": 10, "gtol": 0.0, "ftol": 0.0})
        return float(result.fun)

    return ({"optimize": optimize}, {})


def _wf_least_squares_campaign() -> tuple[dict, dict]:
    from vmex.core import optimize as opt

    inp = _qa_seed_input()

    def campaign():
        result = opt.least_squares(
            _qa_terms(opt), inp, max_mode=1, jac="implicit", max_nfev=5,
            verbose=0)
        return float(result.cost)

    return ({"campaign": campaign}, {})


def _wf_single_stage_coils() -> tuple[dict, dict]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart

    try:
        from essos.surfaces import surfacerzfourier_from_boundary
    except ImportError as error:
        raise ImportError(
            "workflow F9 needs the ESSOS branch rj/vmex-optimization-"
            "interfaces (the same gate as the single-stage examples)"
        ) from error

    from vmex.core import optimize as opt

    inp = _qa_seed_input()
    problem = opt.VmecProblem.from_tuples(inp, _qa_terms(opt), max_mode=1)
    curves = CreateEquallySpacedCurves(
        4, 2, 1.7, 0.9, n_segments=24, nfp=int(inp.nfp), stellsym=True)
    coils0 = Coils(curves, np.full(4, 1.0e5))
    x_boundary0 = np.asarray(problem.x0)
    held = {}

    def coil_objective(x):
        x_boundary, x_coils = x[: x_boundary0.size], x[x_boundary0.size:]
        rbc, zbs = problem.boundary_from_x(x_boundary)
        surface = surfacerzfourier_from_boundary(
            rbc, zbs, inp.nfp, nphi=8, ntheta=8)
        coils = coils0.with_dofs(
            jnp.concatenate((x_coils, coils0.dofs_currents)))
        field = BiotSavart(coils)
        b = jax.vmap(field.B)(
            surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
        bn = jnp.sum(b * surface.unitnormal, axis=2) / jnp.linalg.norm(
            b, axis=2)
        weights = surface.area_element / jnp.sum(surface.area_element)
        rows = jnp.sqrt(weights) * bn
        return 0.5 * 1.0e3 * jnp.vdot(rows, rows)

    x0 = jnp.concatenate(
        [jnp.asarray(x_boundary0), jnp.asarray(curves.dofs).ravel()])
    plasma_vg = jax.jit(problem.jax_value_and_grad)
    coil_vg = jax.jit(jax.value_and_grad(coil_objective))

    def value_and_gradient():
        # VMEX supplies the exact equilibrium derivative; JAX differentiates
        # the coil objective; they add directly (the examples' contract).
        plasma_value, plasma_gradient = plasma_vg(x0[: x_boundary0.size])
        coil_value, coil_gradient = coil_vg(x0)
        held["value"] = plasma_value + coil_value
        held["gradient"] = coil_gradient.at[: x_boundary0.size].add(
            plasma_gradient)
        return _block(held["gradient"])

    return ({"value_and_gradient": value_and_gradient}, {})


def _wf_free_boundary_adjoint() -> tuple[dict, dict]:
    import dataclasses as dc

    import jax
    import jax.numpy as jnp
    import numpy as np

    from vmex.core import implicit as im
    from vmex.core.freeboundary_implicit import (
        make_free_boundary_config,
        solve_free_boundary_implicit_status,
    )
    from vmex.core.mgrid import MgridField

    inp = _read_input("input.cth_like_free_bdy_lasym_small")
    from vmex.core.mgrid import read_mgrid

    mgrid_path = DATA / "mgrid_cth_like_lasym_small.nc"
    # The deck carries more EXTCUR entries than the compact mgrid has current
    # groups; truncate to nextcur exactly as the free-boundary tests do.
    extcur = np.asarray(inp.extcur, dtype=float)[: read_mgrid(mgrid_path).nextcur]
    field = MgridField.from_file(mgrid_path, extcur=extcur)
    params = im.params_from_input(inp)
    # The deck's native ns=15 and a tight forward tolerance: the coupled
    # GCROT adjoint inherits the forward state's conditioning and does not
    # converge from a loose ftol=1e-6 solve on this LASYM case.
    cfg = make_free_boundary_config(
        inp, field, ns=15, ftol=1.0e-9, max_iterations=4000,
        adjoint_tol=1.0e-6, adjoint_maxiter=200,
        field_from_parameters=lambda current: dc.replace(
            field, extcur=current))
    held = {}

    def objective(current):
        state, _status, _fsq, _ratio = solve_free_boundary_implicit_status(
            params, current, cfg)
        return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

    def value():
        held["value"] = objective(field.extcur)
        return _block(held["value"])

    def adjoint():
        held["gradient"] = jax.grad(objective)(field.extcur)
        return _block(held["gradient"])

    return ({"value": value, "adjoint": adjoint}, {})


def _wf_lasym_versus_symmetric() -> tuple[dict, dict]:
    import dataclasses as dc

    import jax
    import numpy as np

    from vmex.core import implicit as im
    from vmex.core.statephysics import aspect_ratio

    lasym_inp = _read_input("input.up_down_asymmetric_tokamak")
    # The symmetric twin zeroes the asymmetric boundary blocks at identical
    # resolution, so the pair isolates the cost of the LASYM lane itself.
    sym_inp = dc.replace(
        lasym_inp, lasym=False,
        rbs=np.zeros_like(np.asarray(lasym_inp.rbs)),
        zbc=np.zeros_like(np.asarray(lasym_inp.zbc)))
    held = {}

    def stages_for(tag, inp):
        params = im.params_from_input(inp, device=None)

        def objective(p):
            solution = im.run(inp, p, ns=13, ftol=1.0e-11,
                              max_iterations=8000, device=None)
            return aspect_ratio(solution.state, solution.runtime)

        def value():
            held[f"{tag}_value"] = objective(params)
            return _block(held[f"{tag}_value"])

        def gradient():
            held[f"{tag}_grad"] = jax.grad(objective)(params)
            return _block(held[f"{tag}_grad"].rbc)

        return value, gradient

    sym_value, sym_gradient = stages_for("symmetric", sym_inp)
    lasym_value, lasym_gradient = stages_for("lasym", lasym_inp)
    return ({"symmetric_value": sym_value,
             "symmetric_gradient": sym_gradient,
             "lasym_value": lasym_value,
             "lasym_gradient": lasym_gradient}, {})


def _wf_mirror_fixed() -> tuple[dict, dict]:
    import jax.numpy as jnp

    from vmex.mirror import (
        MirrorConfig,
        MirrorResolution,
        solve_fixed_boundary_from_radius,
    )

    config = MirrorConfig(
        resolution=MirrorResolution(ns=9, mpol=0, nxi=11),
        z_min=-0.8, z_max=0.8, ftol=1.0e-10, max_iterations=2000)

    def solve():
        result = solve_fixed_boundary_from_radius(
            jnp.asarray(0.25), config, elements=4)
        return _block(result.evaluated.force.normalized_rms)

    return ({"solve": solve}, {})


def _wf_mirror_free_boundary() -> tuple[dict, dict]:
    import jax.numpy as jnp
    import numpy as np

    from vmex.mirror import (
        MirrorBoundary,
        MirrorConfig,
        MirrorResolution,
        SplineMirrorDiscretization,
        solve_free_boundary,
    )

    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, nxi=9),
        z_min=-0.8, z_max=0.8, ftol=1.0e-10, max_iterations=400)
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build_cgl(config, elements=4)
    grid = discretization.grid
    z = jnp.asarray(grid.z)
    coil_radius, coil_z, coil_current = 0.5, 1.0, 3.72e5
    vacuum_axis_field = sum(
        4.0e-7 * jnp.pi * coil_current * coil_radius**2
        / (2.0 * (coil_radius**2 + (z - position) ** 2) ** 1.5)
        for position in (-coil_z, coil_z))
    center = int(np.argmin(np.abs(np.asarray(grid.z))))
    axial_flux_derivative = 0.5 * vacuum_axis_field[center] * 0.25**2
    initial_boundary = discretization.fit_boundary(
        MirrorBoundary.from_axis_field(
            axial_flux_derivative, vacuum_axis_field, grid),
        source_grid)

    def external_field(points):
        # The analytic two-loop field the vacuum axis profile above belongs to.
        points = jnp.asarray(points)
        x, y, height = jnp.moveaxis(points, -1, 0)
        del x, y
        field = sum(
            4.0e-7 * jnp.pi * coil_current * coil_radius**2
            / (2.0 * (coil_radius**2 + (height - position) ** 2) ** 1.5)
            for position in (-coil_z, coil_z))
        return jnp.stack(
            [jnp.zeros_like(field), jnp.zeros_like(field), field], axis=-1)

    def solve():
        result = solve_free_boundary(
            initial_boundary, discretization, config, external_field,
            axial_flux_derivative=axial_flux_derivative)
        return _block(result.plasma_force.normalized_rms)

    return ({"solve": solve}, {})


def _wf_hybrid_gk_geometry() -> tuple[dict, dict]:
    from vmex.mirror import (
        MirrorConfig,
        MirrorResolution,
        build_stellarator_mirror_hybrid,
        gk_closed_fieldline_geometry,
        solve_fixed_boundary,
    )

    # Zero current and section_turns=0: the GK export requires a field line
    # that closes after one axis circuit (the turbulence tests' contract).
    resolution = MirrorResolution(ns=5, mpol=4, nxi=4)
    config = MirrorConfig(
        resolution=resolution, ftol=1.0e-10, max_iterations=600)
    setup = build_stellarator_mirror_hybrid(
        resolution, coefficient_count=16, straight_length=4.0,
        return_radius=2.0, semi_major=0.4, semi_minor=0.3, section_turns=0,
        axial_flux_derivative=0.02, quadrature_order=3)
    held = {}

    def solve():
        held["result"] = solve_fixed_boundary(
            setup.initial_state, setup.boundary, setup.discretization,
            config, axial_flux_derivative=0.02,
            solve_lambda=True, axis=setup.axis, require_convergence=True)
        return _block(held["result"].evaluated.force.normalized_rms)

    def geometry():
        state = setup.discretization.evaluate_state(
            held["result"].coefficient_state)
        held["gk"] = gk_closed_fieldline_geometry(
            state, setup.discretization, setup.axis,
            axial_flux_derivative=0.02, ntheta=32, arc_oversample=8)
        return _block(held["gk"]["bmag"])

    return ({"solve": solve, "geometry": geometry}, {})


def _wf_boozer_many_surfaces() -> tuple[dict, dict]:
    from vmex.core import optimize as opt
    from vmex.core.omnigenity import boozer_bmnc_state

    inp = _read_input("input.li383_low_res")
    surfaces = tuple(0.15 + 0.75 * i / 7.0 for i in range(8))
    held = {}

    def solve():
        held["eq"] = opt.solve_equilibrium(inp, verbose=False)
        return _block(held["eq"].state.R_cos)

    def transform():
        held["bmnc"] = boozer_bmnc_state(
            held["eq"].state, held["eq"].runtime, surfaces=surfaces)
        return _block(next(iter(held["bmnc"].values())))

    return ({"solve": solve, "transform": transform}, {})


def _wf_epsilon_effective() -> tuple[dict, dict]:
    import numpy as np
    from neo_jax import NeoConfig

    import vmex as vj
    from vmex.core import optimize as opt

    inp = _read_input("input.LandremanPaul2021_QA_lowres")
    # The epsilon_effective example's compact NEO controls.
    config = NeoConfig(
        theta_n=24, phi_n=24, npart=12, multra=1, no_bins=20, nstep_per=6,
        nstep_min=30, nstep_max=60, acc_req=0.1,
        max_rational_field_periods=100000)
    held = {}

    def solve():
        held["eq"] = opt.solve_equilibrium(inp, verbose=False)
        return _block(held["eq"].state.R_cos)

    def epsilon():
        _s, held["eps"] = vj.epsilon_effective_from_wout(
            held["eq"].wout, surfaces=np.linspace(0.25, 0.75, 3),
            config=config)
        return _block(held["eps"])

    return ({"solve": solve, "epsilon": epsilon}, {})


def _wf_gamma_c() -> tuple[dict, dict]:
    import jax

    from vmex.core import gammac
    from vmex.core import implicit as im

    inp = _read_input("input.li383_low_res")
    params = im.params_from_input(inp, device=None)
    term = gammac.GammaC(
        [0.5], nalpha=7, num_transit=3, points_per_transit=64, num_pitch=24,
        quadrature_order=32)
    held = {}

    def objective(p):
        solution = im.run(inp, p, ns=13, ftol=1.0e-11, max_iterations=8000,
                          device=None)
        return term.total_state(solution.state, solution.runtime)

    def value():
        held["value"] = objective(params)
        return _block(held["value"])

    def gradient():
        # Exact for the discretized objective at fixed resolution; the
        # continuum Gamma_c gradient is documented non-convergent, so this
        # measures the derivative-safe fixed-resolution objective only.
        held["gradient"] = jax.grad(objective)(params)
        return _block(held["gradient"].rbc)

    return ({"value": value, "gradient": gradient}, {})


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
    "F5": Workflow("F5", "vector residual + full Jacobian",
                   _wf_vector_residual_jacobian, ("input.minimal_seed_nfp2",)),
    "F7": Workflow("F7", "scalar boundary optimization, 10 accepted steps",
                   _wf_scalar_optimization, ("input.minimal_seed_nfp2",)),
    "F8": Workflow("F8", "residual least-squares campaign, 5 evaluations",
                   _wf_least_squares_campaign, ("input.minimal_seed_nfp2",)),
    "F9": Workflow("F9", "fixed-boundary single-stage plasma + ESSOS coils",
                   _wf_single_stage_coils, ("input.minimal_seed_nfp2",)),
    "F10": Workflow("F10", "free-boundary value and adjoint",
                    _wf_free_boundary_adjoint,
                    ("input.cth_like_free_bdy_lasym_small",
                     "mgrid_cth_like_lasym_small.nc")),
    "F11": Workflow("F11", "symmetric versus LASYM value and gradient",
                    _wf_lasym_versus_symmetric,
                    ("input.up_down_asymmetric_tokamak",)),
    "M1": Workflow("M1", "isotropic fixed-boundary mirror",
                   _wf_mirror_fixed, ()),
    "M2": Workflow("M2", "axisymmetric free-boundary mirror",
                   _wf_mirror_free_boundary, ()),
    "M3": Workflow("M3", "periodic hybrid equilibrium and GK geometry",
                   _wf_hybrid_gk_geometry, ()),
    "B1": Workflow("B1", "in-process Boozer transform, one surface",
                   _wf_boozer_one_surface, ("input.li383_low_res",)),
    "B2": Workflow("B2", "in-process Boozer transform, eight surfaces",
                   _wf_boozer_many_surfaces, ("input.li383_low_res",)),
    "C1": Workflow("C1", "epsilon-effective summary diagnostic",
                   _wf_epsilon_effective,
                   ("input.LandremanPaul2021_QA_lowres",)),
    "C2": Workflow("C2", "Gamma-c value and derivative-safe objective",
                   _wf_gamma_c, ("input.li383_low_res",)),
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _run_in_process(ident: str, regime: str,
                    trace_dir: Path | None = None) -> dict[str, Any]:
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
        if regime != "warm" and regime not in variants:
            raise ValueError(
                f"workflow {ident} defines no {regime} variant; a plain "
                "warm repeat must not be reported under that label")
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

    if trace_dir is not None:
        # One XProf trace per stage, captured on a warm repeat so the trace
        # shows execution rather than compilation.
        import jax

        for name, stage in stages.items():
            with jax.profiler.trace(str(trace_dir / ident / name)):
                stage()

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
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="capture one XProf trace per stage (warm runs)")
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
                record = _run_in_process(ident, regime,
                                         trace_dir=args.trace_dir)
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
