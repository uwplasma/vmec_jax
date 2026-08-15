# Examples

All runnable examples live under this single `examples/` tree.

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
    ESSOS coils (tabulated to a temporary mgrid; requires ESSOS branch
    `feature/mgrid-from-coils`); `PRES_SCALE` is calibrated per point so the
    *actual* wout `betatotal` hits 0/1/2/3 %.
  - `take_free_boundary_gradients.py` — differentiate a free-boundary field
    diagnostic through the virtual-casing vacuum field.
  - `vmex_get_B_gradB.py` and `vmex_get_B_outside_plasma.py` — query a
    finite-beta field inside the LCFS or an actual ESSOS coil plus
    virtual-casing field outside it, including three spatial derivative orders
    and exact VJPs in named VMEX/ESSOS variables.
  - `vmex_fieldline_tracing_vacuum.py` and
    `vmex_fieldline_tracing_finite_beta.py` — compare VMEX, coil-only, and
    self-consistent exterior traces in 3-D and toroidal Poincare plots.
    Seeds form one line from just off-axis through the selected exterior
    offset; VMEX uses toroidal angle while Cartesian traces use arclength and
    stop after leaving the LCFS neighborhood.
    The finite-beta coil fixture is reproduced by ESSOS
    `examples/coil_optimization/optimize_coils_finite_beta_vmex.py`.
- `optimization/`: compact QA/QH/QP/QI scripts using `(function, target,
  weight)` terms with SciPy least-squares, BFGS, or L-BFGS-B. The fixed-boundary
  `single_stage_optimization.py` jointly varies VMEX boundary coefficients and
  ESSOS coil Fourier coefficients; no free-boundary solve is involved.
  `QA_optimization_bootstrap.py` and `QH_optimization_bootstrap.py` also vary
  a stage-refined current spline against self-consistent Redl, DMerc, and DR
  targets. `single_stage_optimization_finite_beta.py` combines that finite-beta
  plasma problem with exact virtual-casing and ESSOS coil derivatives.
  All read `VMEX_EXAMPLES_CI=1` for short CI smoke tests.
- `mirror_fixed_boundary_nonaxisymmetric.py` compares axisymmetric and
  rotating-ellipse fixed-boundary mirrors; `mirror_free_boundary_beta_scan.py`
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
