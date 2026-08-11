# Restart from a previous run

Seed a solve from an earlier result instead of starting cold: from an
in-memory state (`initial_state=`), or from any `wout_*.nc` file — VMEX-,
VMEC2000-, or PARVMEC-written — via `restart_from=` / `vmex --restart`.
A converged same-deck restart re-converges in 1 iteration instead of 435
(cth `ns=15` at `FTOL 1e-14`, including from the VMEC2000-written golden
wout).

## Restart from a wout file (CLI)

```console
vmex input.x --restart wout_y.nc
```

or put it in the deck with the VMEX extension key (resolved relative to the
input file; the CLI flag wins):

```text
&INDATA
  RESTART_WOUT = 'wout_y.nc'
  ...
```

Both work for fixed- and free-boundary decks. Multigrid rungs whose
resolution the seed already meets or exceeds are skipped: the free-boundary
cth ladder restart takes 99 final-stage iterations instead of 340 and skips
its coarse rung.

## Restart from a wout file (Python)

Every solve entry point accepts `restart_from` — a `wout_*.nc` path, a
{class}`~vmex.core.wout.WoutData`, a previous
{class}`~vmex.core.solver.SolveResult`, or a bare
{class}`~vmex.core.solver.SpectralState`:

```python
import vmex as vj

inp = vj.VmecInput.from_file("input.x")
result = vj.solve_multigrid(inp, restart_from="wout_y.nc")
```

The reconstruction ({mod}`vmex.core.restart`) inverts the `wrout.f` output
maps exactly — `rmnc/zmns` (and LASYM partners) remapped by `(m, n)` onto
the target `MPOL/NTOR` table, rescaled to internal `mscale*nscale`
normalization, rotated into the evolved m=1-constrained basis, and the
half-mesh `lmns` inverted surface-by-surface to the full-mesh internal
lambda. R/Z are exact at machine precision and lambda on every interior
surface, so a converged wout restarts at its converged residual — unlike
VMEC++, which zeroes lambda when restarting from a Fortran wout and never
seeds asymmetric geometry. Radial differences pass through the same
`interp.f` transfer as the multigrid ladder (up- and down-sampling), and a
seed finer than the whole ladder runs only the final rung.

Fixed-boundary restarts adapt the seed to the deck's (possibly perturbed)
boundary through {func}`~vmex.core.solver.hot_restart_state` and rebind the
`rcon0/zcon0` baselines; free-boundary restarts keep the wout's evolved free
edge and repeat vacuum activation (reset-file semantics).

## Restart from an in-memory state

Inside one process, skip the file entirely:

```python
result0 = vj.solve_multigrid(inp)
inp2 = ...                                   # perturbed deck
result1 = vj.solve_multigrid(inp2, initial_state=result0.state)
```

{func}`~vmex.core.multigrid.interpolate_state` moves a state between radial
resolutions explicitly when you need control over the transfer.

## When each pays off

- **`initial_state=`** — parameter scans in one process: no I/O, and
  successive points converge in a handful of iterations
  ({doc}`parameter-scans`).
- **`restart_from=` / `--restart`** — resume across processes or machines,
  refine a converged run at higher resolution, or seed from a VMEC2000
  archive: anything that starts from a file.
- **Cold start** — a structurally different problem (changed `NFP`,
  topology, or a far-away boundary); the multigrid ladder is the designed
  cold-start path.

The mechanism (exact output-map inversion, rung skipping, carried module
state) is {doc}`/explanation/multigrid`.
