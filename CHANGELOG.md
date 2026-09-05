# Changelog

Release notes are mirrored from the GitHub releases.  Where a number has a
committed artifact it is named; the 0.8.x cold-start timings were measured
in the pull-request and release bodies and are being backfilled as
`benchmarks/` artifacts (see the *Performance and validation* page).

## Unreleased

- Polish memory at production stellarator resolution: the independent
  force sweep now sizes its point batches from the problem's own mode
  table and radial basis instead of a constant tuned on one deck, and
  checkpoints the per-point kernel so reverse-mode passes stay per-batch.
  On the W7-X standard configuration (`MPOL = NTOR = 10`, `ns = 51`) the
  initial certificate falls from a single 34 GB allocation to 3.0 GiB
  peak RSS, and the full polish setup completes where it was previously
  killed by the OS. Measured in `benchmarks/polish_memory.py` with the
  pre-fix sweep as one of its arms.
- Polish observability, part two: the Gauss-Newton phase now prints while
  it runs. The per-iteration rows added in #243 are read out of the solver
  history after the jitted loop returns, which at production resolution
  meant ten hours of silence; a device callback on the inner linear
  products now emits a throttled progress line, verified bit-identical to
  the unreported solve.
- `POLISH = AUTO` prices the solve before committing to it: it times one
  Gauss-Newton linear product on the problem at hand, and if the iteration
  limits allow more than `POLISH_BUDGET` (new directive, `--polish-budget`,
  default 3600 s) it reports what it measured, returns the equilibrium
  unpolished, and names the knobs that override the decision. `POLISH = ON`
  never measures and never declines.
- README states where polishing is effective: every certified case is
  axisymmetric, and the two 3-D measurements (a QA deck at `MPOL = NTOR = 5`
  and the W7-X standard configuration) are quoted with their budgets and
  their certificate outcomes in `benchmarks/polish3d_tuning.md`.
- Polish observability: the CLI announces every polish phase (state
  refinement, initial certificate, preconditioner and chart build, compile
  notice), prints one row per Gauss-Newton iteration, and closes with a
  certificate verdict naming any failed check (#243, #244).
- Input-file polish control: `!@VMEX POLISH_TOL`, `POLISH_DEGREE`,
  `POLISH_SPANS`, `POLISH_MAX_ITER`, `POLISH_FAIL`, with mirrored CLI flags
  and `solve_file` keywords; precedence CLI flag > Python keyword > file
  directive > default (#243).
- Optimization startup: the seed refinement is staged as one per-config
  executable and deferred out of problem construction; time to first output
  62 -> 6.5 s cold on the QA example, per-trial-point evaluation 12.5 -> 8.0 s
  (#240). The staging idioms are applied to the free-boundary implicit and
  mirror Newton-Krylov paths, and CI pins the optimization cold start (#241).
- README: the polish summary plots the independent oracle's force-error
  profile (unpolished vs certified polished export, 26-fold) with both
  flux-surface sets overlaid; the extender section shows the exterior
  island chain resolved by the total field (#237, #239).
- CI: the Codecov upload is best-effort so the changed-line coverage gate
  always runs; parity lanes get budget headroom (#236, #242).

## 0.8.1 - 2026-09-02

The cold-start performance release. Cold CLI, python, and optimization runs
are faster than every previous VMEX release, including v0.3.0, on every
deck measured, on x86 and arm64: 36-core x86 QA_lowres 43.1 s (v0.3.0) ->
32.4 s, solovev 12.1 -> 7.0 s; Apple M4 QA 22.3 s (v0.8.0) -> 12.4 s, li383
-> 4.0 s, solovev -> 3.1 s. At QA resolution the cold start compiles 343 XLA
programs (v0.8.0: 773; v0.3.0: 523).

- Eager setup, WOUT-export, stage-interpolation, and printout passes became
  module-level jitted lanes (#227, #230).
- The iteration body traces the funct3d chain once (halving every lane's
  compile), and the ns4 preconditioner refresh runs only on its VMEC2000
  cadence (#229).
- Print cadence, initial DELT, and ftol no longer key lane recompilation
  (#231); non-finite iterations are detected from the residual scalars
  with full classification on trip (#232).
- 3-D polishing no longer stalls in XLA constant folding: the Ruiz/probe
  jits stopped baking linearization residuals as constants (#234).
- CI pins compile budgets (lane HLO size, cold-solve program count) and
  disables the persistent compilation cache on ephemeral runners, whose
  eviction lock had been timing every lane out (#228, #233).
- Numerical statement: per-iteration physics unchanged; graph
  restructuring shifts XLA fusion, so trajectories can differ from v0.8.0 at
  1 ULP per iteration with identical iteration counts and converged
  geometry agreeing at 1e-12.

## 0.8.0 - 2026-08-30

- Certified force-balance polishing: `--polish`, `!@VMEX POLISH = AUTO`, or
  `solve_file(..., polish="auto")` lift a converged fixed-boundary state to
  axis-regular cubic B-splines and drive both physical force channels to
  zero on an overdetermined collocation grid with matrix-free SOLVAX
  Gauss-Newton steps; acceptance is an independent certificate (volume L2
  force error below 1e-2, radial-refinement stability within 1e-3, positive
  signed Jacobian) with tangent/adjoint derivatives through the polished
  root.
- Execution directives (`!@VMEX KEY = VALUE`, JSON `_vmex`, keywords, CLI)
  separate how to run from what to solve; VMEC2000 reads the same files.
- Implicit lane recompilation per optimization trial removed (steady
  campaign step 16.6 -> 10.8 s, warm value+gradient repeat 28.8 -> 3.0 s,
  bit-identical); polished solves reuse compiled programs (warm 52.9 ->
  22.8 s).
- `benchmarks/profile_workflows.py`: seventeen flagship workflows in five
  timing regimes with compile counts and committed M4 baselines.
- CI attempt wall clock roughly halved by resharding; `tools/preflight.py`
  runs the static gates, guard tests, and diff-affected tests locally.
