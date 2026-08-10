# QI/QA optimization comparison

These files are the fresh-process, apples-to-apples SIMSOPT/VMEX comparison.
Every case uses the same bundled nfp=2 seed
(`examples/data/input.nfp2_QI_seed`), boundary variables, VMEC resolution,
objective rows, SciPy tolerances, and 15-function-evaluation budget.

VMEX uses exact implicit derivatives with its persistent compilation cache
disabled, so construction, cold JIT compilation, and optimization are all in
`total_seconds`. SIMSOPT uses centered finite differences with 14 MPI ranks.
The consolidated `qi_results.json` and `qa_results.json` artifacts record one
entry per case: package versions, the host platform, the input hash, the
stage schedule, ESS setting, timing split, and accepted-cost history.

A case that exceeds the runner's explicit wall limit is recorded as
right-censored rather than discarded.  Its entry has ``status="timed_out"``
and ``total_seconds`` is the lower bound, not a claimed completion time.  Open
markers identify these lower bounds in the timing figure; any accepted costs
checkpointed before the limit remain visible in the history figure.

Each SIMSOPT MPI rank is limited to one BLAS/XLA worker thread.  Parallelism
comes from its 14 finite-difference groups, avoiding a nested 14-by-14 thread
pool while still occupying all logical CPUs.

The VMEX state/wout seed-cost parity audit is performed after the cold wall
timer stops.  The measured relative difference is at most `1.97e-8`; it
validates that both backends evaluate the same residual
without charging VMEX for an extra diagnostic that SIMSOPT's wout path already
performs as part of every objective evaluation.

Reproduce the complete matrix and figures from the repository root:

```console
python benchmarks/run_optimization_crosscode.py --workers 14 \
    --timeout-seconds 600
```

The benchmark is deliberately not run in CI. Fast tests validate the committed
matrix structure, provenance, backend parity, and within-stage monotonicity.

## Recorded comparison

The cold QI direct-mode wall times are summarized in the repository README and
plotted through mode 8 in `qi_optimization_time.png`. The three history cases
have the following final recorded costs; `>=600` denotes a right-censored wall
time and the cost is the last accepted checkpoint, not a completed result.

| objective / schedule | backend | ESS | wall (s) | final recorded cost |
| --- | --- | :---: | ---: | ---: |
| QI ladder 1--5 | SIMSOPT | off | >=600 | 4.658e-2 |
| QI ladder 1--5 | SIMSOPT | on  | >=600 | 4.882e-2 |
| QI ladder 1--5 | VMEX    | off | 362.3 | 3.932e-2 |
| QI ladder 1--5 | VMEX    | on  | 343.7 | 4.191e-2 |
| QI direct 2 | SIMSOPT | off / on | 111.1 / 87.3 | 1.292e-1 / 4.994e-2 |
| QI direct 2 | VMEX | off / on | 330.2 / 105.7 | 8.178e-2 / 6.940e-2 |
| QI direct 5 | SIMSOPT | off / on | >=600 / >=600 | 2.814e-1 / 4.847e-2 |
| QI direct 5 | VMEX | off / on | >=600 / 129.8 | 6.431e-1 / 4.722e-2 |
| QA ladder 1--5 | SIMSOPT | off / on | >=600 / >=600 | 8.401e-7 / 5.954e-7 |
| QA ladder 1--5 | VMEX | off / on | 316.9 / 316.6 | 4.422e-7 / 3.317e-7 |
| QA direct 2 | SIMSOPT | off / on | 100.4 / 105.6 | 4.278e-5 / 4.297e-5 |
| QA direct 2 | VMEX | off / on | 168.9 / 397.7 | 1.153e-4 / 1.113e-4 |
| QA direct 5 | SIMSOPT | off / on | >=600 / 508.7 | 2.167e-3 / 7.201e-7 |
| QA direct 5 | VMEX | off / on | 134.5 / 108.5 | 2.977e-5 / 1.124e-5 |

These records support three limited conclusions. Cold JIT makes tiny problems
unfavorable to VMEX; exact implicit derivatives become advantageous as the
number of boundary variables grows; and ESS is particularly important for a
direct high-mode start, but can land above an unscaled ladder. The data do not
claim that one optimizer setting dominates every seed or objective.
