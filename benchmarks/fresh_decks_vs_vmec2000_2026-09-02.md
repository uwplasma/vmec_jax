# vmex vs xvmec2000 — fresh-deck cold/warm A/B (2026-09-02)

Setup: vmex 0.8.1 editable @ current main, JAX 0.11.1 x64,
reference ~/bin/xvmec2000 (VMEC2000, serial). One run at a time, foreground, fresh dir per run.
Cold = `rm -rf ~/.cache/vmex ~/.cache/jax` before the run; warm = persistent cache kept, fresh dir.
Decks never previously benchmarked on vmex; all fixed-boundary; nfp coverage 1,2,3,4,5,6.
Per-run logs were kept in the session scratch area and are not committed; every row is reproducible with the commands above.

| deck | resolution summary | xvmec2000 wall | vmex cold wall | vmex warm wall | vmex python cold (import+solve) | converged (both) | physics agreement |
|---|---|---|---|---|---|---|---|
| ITERModel (tokamak) | nfp=1, mpol=12, ntor=0, ns 13→201 (6 levels), ftol 1e-18 | 6.7 s | 21.0 s (3.13x) | 9.0 s (1.33x) | 0.28 + 20.19 s | yes/yes — 1469 vs 1470 iters, 4 Jacobian resets both | machine precision (iotaf rel 1.5e-16, volume 1.2e-16, boundary exact) |
| ESTELL | nfp=2, mpol=6, ntor=5, nzeta=44, ns 9→65, ftol 1e-12 | 23.1 s | 24.0 s (1.04x) | 14.2 s (0.61x) | 0.29 + 22.75 s | yes/yes — 2301 iters identical, 0 resets | machine precision (iotaf rel 3.5e-12, boundary 1.4e-17) |
| ARIES-CS n3are_R7.75B5.7 | nfp=3, mpol=9, ntor=5, ns 16/49, ftol 1e-11, finite beta + current (NCURR=1) | 5.1 s | 10.6 s (2.09x) | 5.8 s (1.14x) | 0.28 + 10.13 s | yes/yes — 1496 iters identical, 4 resets both | machine precision (beta rel 2.2e-13, iotaf rel 1.4e-10) |
| HSX QHS vacuum | nfp=4, mpol=10, ntor=10, ns 11→201 (7 levels), ftol 1e-12 | 162.3 s | 97.2 s (0.60x) | 81.7 s (0.50x) | 0.39 + 86.75 s | yes/yes — 1575 iters identical, 7 resets both | machine precision (iotaf rel 2.5e-10, boundary 2.2e-19) |
| W7-X standard fixed-bdy | nfp=5, mpol=10, ntor=10, ns 13/25/51, ftol 1e-12, vacuum-ish (AM~1e-6) | 9.2 s | 14.9 s (1.63x) | 8.0 s (0.87x) | 0.28 + 14.36 s | yes/yes — 1105 iters identical, 3 resets both | machine precision (iotaf rel 7.6e-12, boundary 2.8e-17) |
| NuhrenbergZille 1988 QHS | nfp=6, mpol=9, ntor=5, ns 16/51, ftol 1e-11, net current | 6.4 s | 11.5 s (1.79x) | 6.5 s (1.01x) | 0.29 + 10.96 s | yes/yes — 1843 iters identical | machine precision (iotaf rel 3.8e-11, boundary 6.9e-18) |

## Flags (cold > ~1.5x xvmec2000)

First-contact (cold) only — no deck exceeds 1.5x warm:
- ITERModel: 3.13x cold. Cold−warm delta 12.0 s.
- ARIES-CS: 2.09x cold (delta 4.8 s).
- NuhrenbergZille: 1.79x cold (delta 5.0 s).
- W7-X: 1.63x cold (delta 6.9 s).
Warm ratios: 0.50x–1.34x across all six decks. Steady-state: no flags.

## Compile census on the cold outlier (ITERModel, caches cleared)

COMPILE total_s=12.65 across 650 programs; top: jit(_block_lane) 5.68 s / 6 compiles (one per ns
ladder level), _constraint_baselines 1.01 s / 11, then a long tail of tiny op-level programs
(multiply 0.81 s/81, broadcast_in_dim 0.80 s/94, ...). The 12.65 s compile total matches the
12.0 s cold−warm delta: the entire cold gap is XLA compilation; execute time (~8–9 s) is the
same 1.3x-of-Fortran story the warm run tells. Decks with more ladder levels pay ~1 _block_lane
compile per level.

## Convergence / physics notes

- Every deck: vmex reproduces the Fortran iteration trajectory essentially exactly — same
  per-level iteration counts (ITERModel differs by 1 iter at the 1e-18/roundoff floor), same
  number of Jacobian resets, same final FSQR/FSQZ/FSQL to 3 significant figures.
- wout agreement is at or near machine precision on all compared quantities (volume_p, aspect,
  betatotal, b0, iotaf, presf, phi, boundary rmnc/zmns). No compatibility findings; no errors,
  no wrong physics, rc=0 everywhere.
- ARIES-CS deck carries STELLOPT &OPTIMUM leftovers and duplicated RAXIS/ZAXIS lines; both codes
  parsed it identically.
