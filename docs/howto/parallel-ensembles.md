# Solve many equilibria at once

`vj.parallel.solve_ensemble` threads independent solves over a
`ThreadPoolExecutor`; each host solve releases the GIL while XLA executes,
so the solves overlap for real wall-clock speedup.

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

How much wall-clock you get back depends on your host — core count, how
many of them XLA is already using inside a single solve, and what else is
running — so this page quotes no speedup table. Measure your own with
`examples/parallel_ensemble_scan.py`, which runs a balanced `nfp2_QA`
`phiedge` scan serially and then at 2, 4, and 8 workers and prints the
wall times. Expect efficiency to fall as workers rise: a single solve
already uses XLA's internal threading, so workers and intra-solve threads
compete for the same cores.

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
