# Solve many equilibria at once

`vj.parallel.solve_ensemble` threads independent solves over a
`ThreadPoolExecutor`; each host solve releases the GIL while XLA executes,
so the solves overlap for real wall-clock speedup — measured 1.79x at 2
workers and 3.29x at 8 on a 10-logical-CPU box.

## Run an ensemble

```python
import vmex as vj

inputs = [vj.VmecInput.from_file(f) for f in deck_files]   # N independent decks
results = vj.parallel.solve_ensemble(inputs, workers=4)    # list[SolveResult]
```

`solve_ensemble` threads {func}`vmex.core.multigrid.solve_multigrid`
(default) or {func}`vmex.core.solver.solve` (`multigrid=False`) over the
ensemble and returns the results in input order. `workers=None` uses
{func}`vmex.core.parallel.default_workers`.

For anything that is not a plain solve, use the general primitive
{func}`vmex.core.parallel.map_ensemble`, which threads any independent
per-item function — e.g. a `jax.value_and_grad` of
{func}`vmex.core.implicit.run` for an ensemble of differentiable objectives.

The full pattern (a `phiedge` scan solved serially, then with 2/4/8 workers,
with timing output) is `examples/parallel_ensemble_scan.py`.

## What to expect

Every ensemble result is *byte-identical* to solving that input alone: the
solves share no mutable state, and the concurrency only overlaps their
GIL-releasing XLA windows. `tests/test_parallel.py` asserts exactly zero
state difference (and identical iteration counts) against the serial solve
on a solovev / circular-tokamak / li383 ensemble and on a `phiedge` scan.

Measured on a balanced `nfp2_QA` `phiedge` scan (`mpol=5, ntor=5, ns=35`,
8 solves ~0.68 s each), reproduced by `examples/parallel_ensemble_scan.py`,
on a 10-logical-CPU box (best-of-3):

| workers | wall (s) | speedup | efficiency |
|---------|----------|---------|------------|
| serial  | 5.46     | 1.00x   | 100 %      |
| 2       | 3.05     | 1.79x   | 89 %       |
| 4       | 2.15     | 2.54x   | 63 %       |
| 8       | 1.66     | 3.29x   | 41 %       |

## When it does not help

- **Unbalanced ensembles.** The ensemble finishes no sooner than its slowest
  member: a heterogeneous mix of very different-sized decks
  (solovev + circular + li383 + nfp2_QA) gained only ~1.1x measured. The
  sweet spot is a balanced parameter scan at fixed resolution, where the
  members share a compiled executable and take similar iteration counts.
- **Gradient ensembles.** The implicit adjoint's backward pass is
  launch-bound (its Python-side dispatch holds the GIL), so a threaded
  `value_and_grad` ensemble overlaps the forward solves well but the reverse
  passes barely (~1.05x measured on a 2-member ensemble). Values and
  gradients stay bit-identical.
- **GPUs.** The host solver runs behind `jax.pure_callback`, which cannot
  execute on a GPU, so the ensemble helper is CPU-only today.

Why a thread pool (and not `pmap` or `vmap`) is the mechanism, and the
multi-GPU design sketch, are in {doc}`/explanation/parallelization`.
