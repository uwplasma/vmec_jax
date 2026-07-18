# Examples

All runnable examples live under this single `examples/` tree.

- Top-level scripts demonstrate common workflows (start with
  `fixed_boundary_run.py`):
  - `fixed_boundary_run.py` — read `&INDATA`, converge, write/plot the wout.
  - `plot_and_boozer.py` — every built-in `plot_wout` figure plus the Boozer
    transform (`run_booz_xform` + `plot_boozmn`) on one converged equilibrium.
  - `profiles_power_and_spline.py` — the same equilibrium from power-series and
    cubic-spline pressure/iota profiles (they agree); `NCURR=0` vs `NCURR=1`.
  - `run_from_json.py` — read/convert VMEC++-style JSON (`to_json` /
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
    ESSOS coils (direct JAX Biot-Savart, no mgrid file); `PRES_SCALE` is
    calibrated per point so the *actual* wout `betatotal` hits 0/1/2/3 %.
  - `take_free_boundary_gradients.py` — differentiate a free-boundary field
    diagnostic through the virtual-casing vacuum field.
  - `single_stage_free_boundary_opt.py` — optimize coil currents to confine a
    target plasma by minimizing <(B.n)^2> with the exact virtual-casing gradient.
  - `single_stage_simultaneous_opt.py` — TRUE single-stage: one exact gradient
    over BOTH plasma-boundary Fourier modes and coil-group currents (implicit
    adjoint + virtual casing threaded through one `jax.value_and_grad`).
  - `single_stage_essos_coils_opt.py` — single-stage with ESSOS coils (vacuum
    and finite-beta cases); vmex stays coil-agnostic, coils enter as a
    differentiable `xyz -> B` callable. Source of `readme_single_stage.png`.
- `optimization/`: precise QA/QH/QP/QI from a circular torus — one file each,
  plus `QA_optimization_ess.py` / `QI_optimization_ess.py`: the SINGLE-call
  variants — all large-max_mode harmonics at once, Exponential Spectral
  Scaling (`use_ess`) replacing the continuation ladder,
  plus `QA_bootstrap_selfconsistent.py` / `QH_bootstrap_selfconsistent.py`:
  self-consistent Redl bootstrap current reproducing arXiv:2205.02914,
  simsopt-style (`(function, target, weight)` terms + one least-squares call
  per `max_mode` continuation stage, implicit adjoint gradients).  All read
  `VMEX_EXAMPLES_CI=1` to shrink budgets for the CI smoke tests
  (`tests/test_examples.py`).
- `data/`: bundled input decks and small checked-in fixtures.
- `data/single_grid/`: fixed-boundary single-grid benchmark inputs and optional
  fetched reference assets.

Generated outputs should go to ignored `results/`, `outputs/`, or a user-chosen
directory.  Do not commit generated WOUT, mgrid, Boozer, PDF, or plot files
unless they are compact reviewed documentation artifacts.
