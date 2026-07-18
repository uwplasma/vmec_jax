Parallelization
===============

This page documents what runs in parallel in vmex today, the measured
strong-scaling of concurrent **ensemble** solves, and the design sketch for
multi-GPU work. It covers the CPU capability that ships now
(:mod:`vmex.core.parallel`, exposed as ``vmex.parallel``) and scopes the GPU
work honestly as future work — the development box is CPU-only, so no GPU
numbers are fabricated here.

What already parallelizes today
-------------------------------

**Within one solve (XLA threading).**
A single forward solve is a Python loop over jitted 10-iteration blocks
(``mode="cli"``) or one ``lax.while_loop`` (``mode="jit"``). XLA:CPU already
multithreads the batched ``totzsps``/``tomnsps`` transforms and the tridiagonal
preconditioner solves that dominate each ``funct3d`` pass, so a lone solve
already uses several cores. This is automatic and needs no user action.

**Multi-cotangent Jacobian batching (implicit adjoint).**
:func:`vmex.core.implicit.implicit_state_pullback_multi_rhs` uses
``jax.vmap`` over the adjoint right-hand sides, so several state cotangents for
the same fixed point share one implicit linearization and one set of GMRES
operators. This is the batching behind the multi-RHS Jacobian assembly.

**Across independent solves (this module).**
An ensemble of independent equilibria — a parameter scan, an ensemble
optimization — is embarrassingly parallel. Each vmex forward solve runs on the
host and **releases the Python GIL while XLA executes** its compiled iteration
lanes, so a plain :class:`concurrent.futures.ThreadPoolExecutor` over the
solves overlaps their execution and gives real wall-clock speedup. This is what
:func:`vmex.core.parallel.solve_ensemble` provides.

Concurrent ensemble solves
--------------------------

.. code-block:: python

   import vmex as vj

   inputs = [vj.VmecInput.from_file(f) for f in deck_files]   # N independent decks
   results = vj.parallel.solve_ensemble(inputs, workers=4)    # list[SolveResult]

``solve_ensemble`` threads :func:`vmex.core.multigrid.solve_multigrid`
(default) or :func:`vmex.core.solver.solve` (``multigrid=False``) over the
ensemble and returns the results in input order. The general primitive is
:func:`vmex.core.parallel.map_ensemble`, which threads any independent
per-item function — e.g. a ``jax.value_and_grad`` of :func:`vmex.core.implicit.run`
for an ensemble of differentiable objectives.

**Correctness contract.** Every ensemble result is *byte-identical* to solving
that input alone: the solves share no mutable state, and the concurrency only
overlaps their GIL-releasing XLA windows. ``tests/test_parallel.py`` asserts
exactly zero state difference (and identical iteration counts) against the
serial solve on a solovev / circular-tokamak / li383 ensemble and on a
``phiedge`` scan.

Measured strong scaling
-----------------------

A balanced ``nfp2_QA`` ``phiedge`` scan (``mpol=5, ntor=5, ns=35``, 8 solves
~0.68 s each), reproduced by ``examples/parallel_ensemble_scan.py``, on a
10-logical-CPU box (best-of-3):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - workers
     - wall (s)
     - speedup
     - efficiency
   * - serial
     - 5.46
     - 1.00x
     - 100 %
   * - 2
     - 3.05
     - 1.79x
     - 89 %
   * - 4
     - 2.15
     - 2.54x
     - 63 %
   * - 8
     - 1.66
     - 3.29x
     - 41 %

The scaling is deliberately sub-linear. XLA already multithreads *within* each
solve, so as the worker count approaches the core count the ensemble workers
and the intra-solve XLA threads draw from the same pool of cores — the falling
efficiency is that contention, not a defect. The absolute speedup therefore
tracks the number of otherwise-idle cores on the box.

Honest limits
-------------

**Load balance (Amdahl).** The ensemble finishes no sooner than its slowest
member. A heterogeneous ensemble of very different-sized decks
(solovev + circular + li383 + nfp2_QA) is dominated by the largest solve and
gains little (~1.1x measured). The sweet spot is a *balanced* ensemble — a
parameter scan at fixed resolution where the members share a compiled
executable and take a similar iteration count.

**The reverse (gradient) pass overlaps far less.** The implicit adjoint
(:func:`vmex.core.implicit.solve_implicit`'s backward pass) is *launch-bound*:
it dispatches many small eager JAX ops whose Python-side dispatch holds the
GIL, so threading a ``value_and_grad`` ensemble overlaps the forward solve well
but the reverse pass barely (~1.05x measured on a 2-member ensemble). Values
and gradients remain bit-identical; the speedup is simply smaller than for a
pure forward-solve ensemble.

Mechanisms considered and rejected
----------------------------------

The threaded ensemble was chosen after comparing three CPU mechanisms on the
same ``nfp2_QA`` scan:

- **Thread pool over independent host solves** (chosen): 3.29x at 8 workers,
  bit-identical, and the only option that handles a *heterogeneous* ensemble
  (different deck shapes) as well as a same-structure scan.
- **pmap across forced host CPU devices**
  (``XLA_FLAGS=--xla_force_host_platform_device_count=N``): **measured 19 s for
  4 solves that cost ~1.5 s each serially — more than 10x slower.** Splitting
  the cores into ``N`` "devices" starves each solve's XLA threading and
  serializes the host callbacks. Rejected by measurement.
- **vmap over the** ``pure_callback``: does not apply to a heterogeneous
  ensemble (different shapes), and for a same-structure ensemble it degenerates
  to a vectorized host *loop* with no true concurrency. Rejected by design.

Multi-GPU (design sketch, future work)
--------------------------------------

The development box is CPU-only, so the following is design, not measurement.

The host solver runs behind :func:`jax.pure_callback`, which cannot execute on
a GPU, so the current ensemble helper is CPU-only. Two complementary GPU paths
are natural extensions:

1. **One equilibrium per device.** Place each ensemble member's traced solve on
   a distinct GPU with ``jax.device_put`` / an explicit ``device=`` argument
   (the solver already threads ``device`` through
   :func:`vmex.core.solver.solve`), and drive the per-device solves from the
   same thread pool. This shards an ensemble across GPUs with no new numerics.

2. **Sharded single large solve.** The fully-traced ``mode="jit"`` lane is a
   pure ``lax.while_loop`` with no host callback, so its per-iteration work (the
   radial × spectral batched transforms) can be sharded across devices with
   ``jax.sharding`` / ``shard_map`` for a single very high-resolution
   equilibrium — the multi-GPU per-*solve* target, distinct from the
   per-*ensemble* target above.

Both are correct-by-construction extensions of existing lanes; neither is
implemented or measured yet. The per-solve GPU *policy* (when a single solve is
faster on GPU vs CPU) is already characterized in :mod:`vmex.core.device`.
