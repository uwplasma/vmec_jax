# Examples

All runnable examples live under this single `examples/` tree. Examples marked
*preview* need ESSOS branch
[`rj/vmex-optimization-interfaces`](https://github.com/uwplasma/ESSOS/tree/rj/vmex-optimization-interfaces)
(PR #58).

- Top-level scripts demonstrate common workflows (start with
  `fixed_boundary_run.py`):
  - `fixed_boundary_run.py` — read `&INDATA`, converge, write/plot the wout.
  - `plot_and_boozer.py` — every built-in `plot_wout` figure plus the Boozer
    transform (`run_booz_xform` + `plot_boozmn`) on one converged equilibrium.
  - `plot_optimized_families.py` — README composites for optimized QA/QH/QP
    outputs and the bundled NFP=1--4 QI inputs: four toroidal cuts, 3-D LCFS,
    and LCFS `|B|` in Boozer coordinates.
  - `profiles_power_and_spline.py` — the same equilibrium from power-series and
    cubic-spline pressure/iota profiles (they agree); `NCURR=0` vs `NCURR=1`.
  - `run_from_json.py` — read/convert structured JSON (`to_json` /
    `from_file`); the JSON and `&INDATA` forms describe one equilibrium.
  - `hot_restart_scan.py` — seed each scan point from the previous converged
    state; warm restarts converge in ~1 iteration and recompile nothing.
  - `finite_beta_scan.py` — ramp the pressure (hot-restarted) and read beta,
    the Shafranov shift (magnetic-axis motion), and Mercier `DMerc` stability.
  - `parallel_ensemble_scan.py` — solve an ensemble of independent equilibria
    concurrently on CPU (`vmex.parallel.solve_ensemble`); prints the measured
    strong-scaling curve and checks the results are bit-identical to serial.
  - `take_gradients.py` — exact fixed-boundary gradients of wout scalars
    (aspect, magnetic energy, ...) by implicit differentiation, checked against
    finite differences; O(1) memory, no step size to tune.
  - `free_boundary_mgrid.py` — free-boundary equilibrium from coil currents and
    an mgrid vacuum field (NESTOR); the LCFS is solved for, not prescribed.
  - `free_boundary_beta_scan.py` — ramp the pressure of the free-boundary case
    (coil currents fixed); the LCFS is re-solved by NESTOR at each beta.
  - `free_boundary_essos_coils.py` — free-boundary beta scan directly from
    ESSOS coils (tabulated to a temporary mgrid); `PRES_SCALE` is calibrated per point so the
    *actual* wout `betatotal` hits 0/1/2/3 %.
  - `take_free_boundary_gradients.py` *(preview)* — compare the coupled
    NESTOR--VMEX adjoint with independent coil-field re-solves.
  - `vmex_get_B_gradB.py` queries the stable finite-beta interior API.
    `vmex_get_B_outside_plasma.py` *(preview)* adds coils, virtual casing, and
    named VMEX/ESSOS VJPs.
  - `vmex_fieldline_tracing_vacuum.py` and
    `vmex_fieldline_tracing_finite_beta.py` *(preview)* — compare VMEX,
    coil-only, and self-consistent exterior traces in 3-D and Poincare plots.
    Seeds form one line from the regularized magnetic axis through the
    selected exterior offset; VMEX uses toroidal angle while Cartesian traces use arclength and
    stop after leaving the LCFS neighborhood.
    The finite-beta coil fixture is reproduced by ESSOS
    `examples/coil_optimization/optimize_coils_finite_beta_vmex.py`.
  - `vmex_fixed_free_boundary_comparison.py` *(preview)* — compare a parent
    free boundary with an `s=0.5` fixed-boundary solve and its exterior field.
- `optimization/`: compact QA/QH/QP/QI scripts using `(function, target,
  weight)` terms with SciPy least-squares, BFGS, or L-BFGS-B.
  `single_stage_optimization.py` *(preview)* varies a prescribed boundary and
  coil Fourier coefficients; it does not call a free-boundary solve.
  `QA_optimization_bootstrap.py` and `QH_optimization_bootstrap.py` also vary
  a stage-refined current spline against self-consistent Redl, DMerc, and DR
  targets. `single_stage_optimization_finite_beta.py` *(preview)* adds virtual
  casing and coil derivatives. The free-boundary single-stage previews leave
  the LCFS implicit and vary only coil shape and current through the coupled
  NESTOR adjoint.
  `QA_optimization_DMerc_vacuum.py` screens a vacuum candidate with the
  frozen-geometry pressure proxies before re-solving at finite pressure, and
  `QA_optimization_global.py` explores basins with SciPy basin hopping before
  the exact least-squares finish.
  All read `VMEX_EXAMPLES_CI=1` for short CI smoke tests.
- `optimization/stellarator_asymmetry/`: the same four families with
  `lasym = True`, seeding `RBS(1,1)`/`ZBC(1,1)` so the optimizer starts off the
  stellarator-symmetric stationary subspace. Each has a finite-beta companion
  that adds Mercier and resistive-interchange rows on a radially graded weight.
  The asymmetric boundary doubles the decision variables, so a stage costs
  roughly twice its symmetric equivalent.
- `optimization/QA_maxJ_continuation.py` and `QI_maxJ_continuation.py` walk the
  constructed maximum-J target into the resolved certificate; the QA script
  states where maximum-J and quasisymmetry conflict near the axis.
- `epsilon_effective.py` computes the NEO_JAX effective ripple from a solved
  equilibrium without writing a `boozmn` file; raise its `NeoConfig` controls
  for anything beyond a radial trend.
- `mirror/mirror_fixed_boundary_nonaxisymmetric.py` compares axisymmetric and
  rotating-ellipse fixed-boundary mirrors; `mirror/mirror_free_boundary_beta_scan.py`
  continues a solved ESSOS-coil free boundary through 80% central beta and
  compares its on-axis field with `sqrt(1-beta)`.
- `data/`: bundled input decks and small checked-in fixtures.
- `data/single_grid/`: fixed-boundary single-grid benchmark inputs and optional
  fetched reference assets.

Generated outputs should go to ignored `results/`, `outputs/`, or a user-chosen
directory.  Do not commit generated WOUT, mgrid, Boozer, PDF, or plot files
unless they are compact reviewed documentation artifacts.

Published-equilibrium comparisons and reproducibility studies belong in
`../benchmarks/`, not among the user-facing optimization examples.
