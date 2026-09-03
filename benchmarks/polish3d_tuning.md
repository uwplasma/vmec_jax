# 3-D strong-force polish tuning log

Goal: a certified, multiple-fold polish improvement on a bundled 3-D
stellarator deck (README stellarator row). Baseline behavior on
`input.nfp2_QA_smooth_beta` (`solve_file(..., polish=True)`, driver
defaults): 0.377 -> 0.324 independent normalized L2 (-14%), 2 Gauss-Newton
steps, certificate correctly declines. VMEC2000's unpolished export for the
same deck certifies at ~0.394. Axisymmetric reference
(`input.shaped_tokamak_pressure_polished`): 5.05e-2 -> 1.91e-3 (26x),
certified at the 1e-2 validation tolerance.

All runs: office box, 36-core CPU, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`,
`~/.cache/vmex` and `~/.cache/jax` cleared before each run,
`benchmarks/strong_polish_3d.py` (this branch) — the instrumented mirror of
the production `polish_legacy_solution` path through the production jitted
lanes.

## Reading the driver first (what bounded the baseline)

- The production 3-D path is `polish_collocation_least_squares`:
  SOLVAX damped Gauss-Newton on the rectangular two-channel collocation
  residual, `LeastSquaresConfig(rtol=PolishConfig.tolerance, ...)`.
  GN terminates when the stationarity gradient norm falls under
  `rtol * max(initial_gradient_norm, 1)`. `PolishConfig.tolerance`
  defaults to 1e-3, so the baseline's 2-step exit is the solver declaring
  1e-3 relative stationarity — not an iteration cap
  (`max_nonlinear_iterations` defaults to 80).
- The certificate requires all three of: `normalized_l2 <=
  validation_tolerance` (default 1e-2), `radial_refinement_difference <=
  1e-3`, `minimum_signed_jacobian > 0`, on shifted overintegrated nodes.
- The correction space carries the deck's own Fourier table (MPOL=NTOR=5
  -> max m = 4) with a spline radial basis (default: degree 3,
  `min(32, (ns - 3 + 1) // 2)` = 11 spans at ns=25). Angular content of
  the force error beyond the deck modes is invisible to the solve and
  fully visible to the certificate; `strong_projection_diagnostics`
  measures that split (`angular_unresolved_fraction`).

## Code findings

0. `evaluate_strong_force` ran one flat `vmap` over every evaluation
   point, materializing per-point spectral intermediates for the whole
   grid at once. On the W7-X standard deck (MPOL=NTOR=10, ns 51) the
   very first `certify_strong_force` sweep (~2.5e5 overintegrated
   points) requested a single 34 GB allocation
   (`RESOURCE_EXHAUSTED: Out of memory allocating 34457702816 bytes`)
   and the production `POLISH = AUTO` path died before one Gauss-Newton
   step — reproducing the user-reported Mac mini OOM on the 62 GB office
   box, on current main (the #234 capture fix does NOT cover this).
   Fixed on this branch: `lax.map` in 4096-point batches beyond that
   size, flat `vmap` below it; bit-level parity test for values and
   reverse-mode gradients including a remainder chunk. After the fix the
   same certificate completes in ~109 s.

1. `strong_projection_diagnostics` crashed on every `nzeta > 1` runtime
   (broadcasting the raw theta grid against the raw zeta grid when
   rebuilding the retained-mode phase). Diagnostic-path only, but it means
   no 3-D polish attempt had ever been dissected with the projection
   split. Fixed on this branch (mesh product, `ntor=1` vs `ntor=0`
   equal-content regression test); the `ntor=0` path is bit-identical.

## Results

| # | deck | config | wall | initial L2 | final L2 | certificate | notes |
|---|------|--------|------|-----------|----------|-------------|-------|
| 1 | nfp2_QA_smooth_beta | tol=1e-6, deg 3, spans 11 (default), max 40 GN, linear 30x20 | 5h22m (GN 5h15m) | 3.773e-1 | 2.867e-1 (1.32x) | declined (L2 >> 1e-2; refine 7.3e-3 > 1e-3) | all 40 steps accepted, ratio ~1.0, damping floored 1e-12; EVERY step after the first exhausted the full 600-iteration unpreconditioned PCG budget (23,736 linear iterations total); cost fell linearly ~0.5%/step |
| 2 | W7X_standard (mpol=ntor=10, ns 51) | production tol 1e-3, max 6 GN, linear 30x5, pre-fix code | 10m05s | n/a | n/a | n/a | OOM-KILLED (signal 9) at 46.3 GiB peak RSS; reproduces the user's Mac mini OOM on current main |
| 3 | W7X_standard | same, phase-stamped, ulimit -v 50 GiB | 2m52s to kill | n/a | n/a | n/a | died INSIDE the first `certify_strong_force`: one 34.5 GB allocation (flat vmap over ~2.5e5 certificate points at mnmax=200) |
| 4 | W7X_standard | same + batched force sweep (4096-pt lax.map) | 10m55s to kill | 2.0000 (saturated) | n/a | n/a | certificate now completes in 109 s, peak RSS 7.4 GiB through that phase; next cliff: `_ruiz_probe_lane` chart probes request 40 GB (whole-grid `jax.linearize` residuals) |
| 5 | W7X_standard | same + remat boundary in batched sweep | 11h09m (setup 32m, GN 10h35m) | 4.327e6 abs | 4.279e6 abs (-1.1%) | declined | COMPLETED, exit 0, peak RSS 17.1 GiB. Chart build 22.5 min (was: 40 GB OOM). All 6 GN steps accepted, cost 67900 -> 41570 (-39%), gradient 334 -> 24; every step exhausted its 150-iteration linear ceiling |

### W7-X standard note

The deck is beta ~ 0 (AM = 1e-6) and current-free, so |JxB| and
|grad p| both vanish and the certificate's pointwise normalization
2|F|/(|JxB|+|grad p|+floor) SATURATES at its ceiling of 2.0 everywhere.
Certification against a normalized-L2 tolerance is unreachable for any
vacuum deck regardless of polish quality; the meaningful polish metric
here is the absolute L2 (N/m^3). A vacuum-appropriate normalization
(e.g. B^2/(2 mu0 a)) is a certificate-design question, out of scope for
this branch.

### Run 1 reading (nfp2 QA, mpol=ntor=5)

- Legacy solve is not the problem: 31 s, fsq ~ 1e-13 (deck ftol reached).
- The new projection diagnostics say the case is REPRESENTATION-LIMITED:
  `angular_unresolved_fraction = 0.622` at the initial state — 62% of the
  collocation force signal lies outside the deck's retained angular modes
  (total unresolved 0.734, radial-fit part 0.389). The observed final
  L2 2.87e-1 ~= 0.734 * 3.77e-1: forty GN steps removed essentially all
  the resolvable content the solve grid can see, and the certificate's
  shifted overintegrated grid still sees the unrepresentable remainder
  (`angular_spectral_tail` 0.35 before, 0.34 after). No Gauss-Newton
  tuning at MPOL=NTOR=5 can certify this deck.
- The GN inner solver is starved: `_gauss_newton_polish_lane` passes no
  preconditioner to `gauss_newton_least_squares`, so unpreconditioned
  CGNR on the damped normal equations burns its full
  `linear_restart * linear_max_restarts = 600` budget every step (~8
  min/step wall at this size). The trust ratio stays ~1.0 (locally
  near-linear), so progress per step is tiny but always accepted. A
  normal-equation preconditioner (e.g. the existing mode-block factors)
  is the first-order performance fix for production 3-D polish.
- The radial refinement gate fails on its own: the default 11-span
  degree-3 lift reports refinement difference 7.2e-3 (> 1e-3) already at
  the initial state, and the polish does not repair it (7.3e-3 after).

## Run 5 reading (W7-X standard, the decisive run)

The run completed rather than being killed, which is itself the headline:
the same deck on 0.8.0 was killed by the OS, and on this branch before the
remat boundary it requested 40 GB during the chart build.

Cost, measured rather than estimated:

| phase | wall |
|---|---|
| legacy multigrid solve (ns 13/25/51) | 40 s |
| refine + native lift + initial certificate | ~4 min |
| low-order preconditioner factors | 3 min |
| strong-root runtime | 2 min |
| structured chart | 22 min |
| **Gauss-Newton, 6 iterations x 150 linear** | **10 h 35 m** |
| final certificate | 1 min |

That is about **1.75 h per Gauss-Newton iteration** at `MPOL = NTOR = 10`,
`ns = 51`, on 36 CPU cores: 900 linear products in 38115 s, ~42 s per
product. Setup is 5% of the run and the inner linear solve is essentially
all of the rest.

### What this run does and does not show

It does **not** show that polishing cannot work at this resolution. The
budget was six Gauss-Newton iterations at 150 linear iterations each. The
one 3-D case that improved substantially, `nfp2_QA_smooth_beta` (run 1),
was given forty iterations at 600, and its own first six iterations only
reached cost 2717 of an eventual 2170 - most of its certificate gain
accumulated late. Reading run 5's -1.1% as futility would be reading a
starved budget as a physical result.

It does show what the run **costs**. A QA-like iteration count at W7-X
resolution is roughly 40 x 1.75 h, i.e. several days of a 36-core machine
for one equilibrium. That is the number `POLISH = AUTO` now measures and
reports before committing, and it is why the AUTO gate is priced on cost
rather than on any effectiveness proxy.

The initial projection diagnostics were considered as that proxy and
rejected on the evidence: W7-X starts at `unresolved_fraction` 0.694, which
is *lower* than the 0.734 of the QA case that did improve. The quantity that
does separate the two runs sharply is `sampled_rms` (QA 0.0018 vs W7-X
0.442) while `projected_residual_rms` is nearly equal (0.0086 vs 0.0094),
but two runs are not evidence for a threshold and nothing here turns that
into one.

## The inner linear solve is unpreconditioned

`build_low_order_preconditioner` factors the legacy raw-force blocks at
every polish call - 3 minutes of the W7-X setup. Those factors *are*
applied, once, inside `make_strong_root_runtime`, where a power iteration on
the low-order-solved tangent sets the equation and coordinate scales. They
are **not** applied inside the Gauss-Newton solve:
`_gauss_newton_polish_lane` calls `gauss_newton_least_squares(residual,
value, config=config)` and SOLVAX takes `precond` as a separate keyword, so
the CG on `(J^T J + mu I) p = -J^T r` runs on the Ruiz-style diagonal column
equilibration alone. Every W7-X Gauss-Newton step exhausted its linear
ceiling, which is what an unpreconditioned CG on 10573 coordinates does.

Wiring those factors in as a normal-equation preconditioner is the obvious
next lever, so it was built and measured on this branch. CG needs a
symmetric positive definite preconditioner, so the construction is `B B^T`
with `B = D^-1 M`: `M` the low-order inverse in chart coordinates
(`_solve_low_inverse`) and `D` the Ruiz column equilibration that defines
the Gauss-Newton variables. `B B^T` is then SPD by construction and is the
normal-equation companion of `M` as a right preconditioner for the square
root.

**It does not work, twice over.**

1. On 3-D decks it cannot even be applied. `B^T` needs
   `jax.linear_transpose` through the high/low transfer, which raises
   `NotImplementedError: scatter transpose is only implemented where
   unique_indices=True`. The adjoint succeeds on `ntor = 0` decks and fails
   on `nfp2_QA_smooth_beta`, so the lever is unavailable at exactly the
   resolutions that need it until the transfer's scatter is made
   transposable.

2. Where it can be applied it makes the solve worse. On
   `input.shaped_tokamak_pressure_polished` (`MPOL 5`, `ns 31`, 6
   Gauss-Newton steps, 150 linear each), same build, same deck, one arm
   each:

   | inner preconditioner | linear iterations | final cost | gradient | independent absolute L2 |
   |---|---|---|---|---|
   | none (shipped) | 887 | 345.8 | 0.313 | 211.7 |
   | low-order `B B^T` | 900 (ceiling every step) | 489.2 | 12.26 | 332.5 |

   The low-order operator carries the legacy radial physics and none of the
   high-order angular coupling, and as an approximate inverse of a
   *different* operator it steers CG away from the normal equations it is
   supposed to accelerate.

So "the factors are already paid for, just pass them to CG" is measured and
rejected. The inner solve is still the dominant cost and still the right
place to attack; a preconditioner built for the collocation normal
equations, rather than borrowed from the square root, is the open work. The
knob is therefore not shipped: an option whose only measured effect is a
worse answer, and which raises on 3-D decks, is a latent bug rather than a
research affordance.
