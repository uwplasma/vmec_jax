# Scan a parameter fast

Solve the first point cold, then seed every successive point from the
previous converged state: warm restarts converge in about one iteration
instead of hundreds, and because VMEX caches one compiled executable per
solver structure, the whole scan recompiles nothing.

## The hot-restart scan pattern

```python
import dataclasses
import vmex as vj

base = vj.VmecInput.from_file("input.case")
state = None
for phiedge in phiedge_values:
    inp = dataclasses.replace(base, phiedge=phiedge)
    result = vj.solve_multigrid(inp, initial_state=state)
    state = result.state                     # seed the next point
```

`examples/hot_restart_scan.py` is the complete measured version — it prints
per-point iteration counts so the cold-vs-warm difference is visible:

```{literalinclude} ../../examples/hot_restart_scan.py
:language: python
```

## Rules that keep a scan fast

- **One executable per structure.** A compiled executable is keyed by the
  solver structure (`ns`, mode tables, angular grid, lane). Keep the
  resolution fixed across the scan and only point 1 compiles; change `ns`
  or `mpol/ntor` mid-scan and you pay a fresh compile.
- **Warm-start from the neighbor.** `initial_state=` accepts any
  {class}`~vmex.core.solver.SpectralState`; for scans the previous point is
  the right seed. Across processes, seed from the previous point's wout
  file with `restart_from=` ({doc}`restart-from-previous-run`).
- **Skip the ladder on warm points.** A converged neighbor state already
  has the final resolution, so warm points need no coarse rungs;
  `solve_multigrid` with `restart_from=` drops them automatically.

## Finite-beta scan example

`examples/finite_beta_scan.py` ramps the pressure and reads three
diagnostics straight from each wout — volume-averaged beta, the Shafranov
shift, and the Mercier `DMerc` profile — hot-restarting each step:

```{literalinclude} ../../examples/finite_beta_scan.py
:language: python
```

## Independent points: thread them

When scan points do not build on each other, solve them concurrently
instead: {doc}`parallel-ensembles` (measured 3.29x at 8 workers).
