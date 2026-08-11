# Diagnose a failing run

Work outward from the exit code: VMEX never crashes with a raw traceback —
every failure maps to a typed {class}`vmex.core.errors.VmecError`, printed
as the VMEC2000 `werror` message plus a one-line hint, with the matching
`ier_flag` as the exit code.

## Check the environment first

```console
vmex --doctor
```

prints the active interpreter, pip location, package versions, JAX backend
and devices, the active default device, and VMEX's placement policies. The
two classic environment failures it catches: `pip` and `python` pointing at
different interpreters, and a CPU-only jaxlib where a GPU was expected.

## Read the exit code

| exit | meaning | typical fix |
|------|---------|-------------|
| 0 | converged | — |
| 2 | `MORE ITERATIONS REQUIRED` (NITER exhausted; unconverged wout is still written, `fileout.f` semantics) | raise `NITER_ARRAY` / `--max-iter`, or loosen `--ftol` |
| other | typed fatal error, no wout | read the werror message + hint; the classes are in {mod}`vmex.core.errors` |

## Jacobian resets

The summary line `NUMBER OF JACOBIAN RESETS = N` counts how often the solver
caught a sign-changing Jacobian (a self-intersecting surface guess), restored
its checkpoint, and backed off `DELT` ({doc}`/explanation/iteration`). A few
resets are routine on hard decks. VMEC2000 aborts at 75; VMEX then retries
from the best finite checkpoint with halved initial `DELT` — bounded by
`--jacobian-retries N` (default 2, Python `jacobian_retries=`). Set it to 0
for VMEC2000's exact fatal-stop behavior. If retries also exhaust:

- check the axis guess (`RAXIS_C`/`ZAXIS_S`) sits inside the boundary;
- lower `DELT` (0.9 → 0.5) or start the ladder at smaller `NS_ARRAY[0]`;
- for perturbed-boundary restarts, seed from a converged neighbor
  ({doc}`restart-from-previous-run`) instead of cold-starting.

## First force evaluation without iterating

```console
python tools/diagnose_input.py path/to/input.case
```

runs the first VMEC force evaluation without entering the nonlinear
iteration and reports pass/fail checks for field assembly, force
normalization, Fourier projection, and preconditioning, plus a diagnostic
code. The default report omits the input path and all input-derived values,
so it is safe to share; `--details` prints values for local use with
non-confidential decks only.

## Compare against a reference run

When a deck converges but disagrees with a VMEC2000 run,
`tools/force_oracle.py` replays the production iteration body and compares
staged internals along the `funct3d.f` chain (`record`/`check` against
goldens; `cross` against a local `xvmec2000`), failing at the FIRST
differing stage. `tools/first_divergence.py` narrows down where two
iteration traces part ways. Both are described in `tools/README.md`.

## Free-boundary specifics

- A missing `MGRID_FILE` silently falls back to a fixed-boundary solve with
  a warning (VMEC2000 behavior) — check for the `In VACUUM` block if the
  boundary did not move.
- Free-boundary Jacobian recovery rebuilds the axis-current filament and the
  NESTOR structures before continuing; a failure before the
  `VACUUM PRESSURE TURNED ON` banner is a plasma-side problem, not a vacuum
  one ({doc}`free-boundary`).

## Performance surprises

A GPU run slower than CPU is usually the policy being right: small decks and
high mode counts are measured CPU winners ({doc}`run-on-gpu`). For memory or
compile-time profiles, `tools/profile_hotpaths.py` records cold-vs-warm wall
time and peak RSS per hot path.
