Performance and validation
==========================

This page summarizes the measured performance and parity status of the core
solver. All numbers come from checked-in benchmark artifacts —
``benchmarks/baseline.json`` (CPU suite, regenerated with
``benchmarks/run_baseline.py``) and ``benchmarks/gpu_baseline.json`` (GPU
matrix, ``benchmarks/run_gpu_matrix.py``; 2x NVIDIA RTX A4000, jax 0.6.2
cuda12) — and from the end-to-end parity suite in
``tests/test_parity_breadth.py``.

Benchmark suite (CPU, ns = 201)
-------------------------------

Wall times in seconds; "cold" is a fresh process including JIT compilation,
"warm" is a second in-process solve reusing the compiled executable (the
number that matters inside optimization loops, where the structural
executable cache makes every solve after the first warm). Every deck's
final ``NS_ARRAY`` stage is ramped to **ns = 201** — production radial
resolution, where the physics dominates the compile overhead and the warm
comparison is fairest.

.. list-table::
   :header-rows: 1
   :widths: 34 14 14 14 14

   * - case
     - VMEC2000
     - vmex cold
     - vmex warm
     - VMEC++
   * - solovev
     - 1.41
     - 11.5
     - **0.62**
     - 1.45
   * - li383_low_res (NCSX)
     - 1.06
     - 8.3
     - **0.69**
     - 0.53
   * - nfp4_QH_warm_start
     - 1.91
     - 10.9
     - **1.32**
     - 1.35
   * - nfp4_QH_warm_start (multigrid)
     - 1.89
     - 29.1
     - **1.44**
     - 1.56
   * - circular_tokamak
     - 2.02
     - 22.6
     - **1.63**
     - 3.70
   * - DSHAPE
     - 2.31
     - 32.7
     - 2.37
     - 5.45
   * - cth_like_fixed_bdy (multigrid)
     - 10.2
     - 38.5
     - **7.76**
     - failed
   * - cth_like_fixed_bdy
     - 13.2
     - 28.0
     - **9.51**
     - failed
   * - cth_like_free_bdy (free boundary)
     - 26.7
     - 71.3
     - **24.9**
     - 6.9
   * - LandremanPaul2021_QA_lowres
     - 45.0
     - 72.3
     - **42.7**
     - 24.7
   * - LandremanPaul2021_QA_lowres (multigrid)
     - 73.7
     - 103.4
     - **68.1**
     - 30.9
   * - LandremanPaul2021_QH_reactorScale_lowres
     - 64.8
     - 76.8
     - 65.6
     - failed
   * - NuhrenbergZille_1988_QHS
     - 137
     - **108**
     - **76.8**
     - 72.7
   * - cth_like_free_bdy_lasym_small (free bdy, lasym)
     - 228
     - --
     - **196**
     - n/a

Bold marks vmex beating VMEC2000. These are wall-clock seconds on a
shared Apple-Silicon CPU (``benchmarks/baseline.json``), so the
warm/Fortran *ratio* is the comparable quantity, not the absolute numbers.

Reading the table:

- **Warm** solves beat VMEC2000 on 12 of the 14 rows — typically 1.2–2.3x —
  and tie on the other two (DSHAPE, the reactor-scale QH). This includes
  the converged symmetric **free-boundary** row: the NESTOR path reaches
  VMEC2000 parity *and* edges out the Fortran wall clock. The LASYM row is a
  deliberately bounded 10,000-iteration stress case that reaches ``NITER``
  in both codes; its wall time compares equal work, not convergence.
- **Cold** runs pay a one-time 7–30 s XLA compile, so a single
  fire-and-forget run is slower than Fortran — except on the biggest deck
  (NuhrenbergZille at ns=201), where even the cold run, compile included,
  beats VMEC2000. The persistent compilation cache removes most of the
  compile cost on subsequent processes.
- **VMEC++** is faster on some converged large decks (free
  boundary, LandremanPaul QA) but *failed* rows aborted during the first
  iterations; ``vmex`` completes every supported convergent row and the
  deliberately NITER-bounded LASYM stress row (zero-crash policy).
  ``n/a`` marks a configuration VMEC++ does not support (``lasym`` free
  boundary).

Production workflows: CPU vs GPU
--------------------------------

``benchmarks/profile_production.py`` times the five workflows a design
loop actually runs, at production resolution. Warm wall-clock, measured
2026-07-12 (CPU: local Apple-Silicon, idle; GPU: office 2x NVIDIA RTX
A4000, jax cuda12 — different hosts, so read each column on its own terms):

.. list-table::
   :header-rows: 1
   :widths: 46 18 18

   * - workflow (warm)
     - M-series CPU
     - A4000 GPU
   * - fixed-boundary solve, ns = 201
     - **5.5 s** (4.3 ms/iter)
     - 6.9 s (5.4 ms/iter)
   * - multigrid ladder 51/101/201
     - **7.8 s**
     - 9.3 s
   * - implicit ``value_and_grad`` (boundary dofs)
     - **17.3 s**
     - 27.8 s
   * - ``least_squares`` opt step (2 nfev)
     - **88.8 s**
     - 151 s

The headline: **a fast desktop CPU beats the A4000 GPU on every production
workflow, even at ns = 201.** Forward solves are close (the GPU is
iteration-competitive at this size — free-boundary NESTOR runs 13.9 ms/iter
on the GPU), but the gradient pipeline is launch-bound on an accelerator,
which is why :func:`vmex.core.device.resolve_implicit_device` pins
implicit-gradient work to the CPU by default (an earlier placement leak
here cost 2x on GPU boxes — fixed, and the pin is now automatic). The
GPU's wins come against slower server cores and larger-than-production
problem sizes (see the GPU guidance below).

Optimization wall time
~~~~~~~~~~~~~~~~~~~~~~

Whole-campaign numbers, from a near-circular torus to a precise
configuration on a 36-core office CPU (details and scripts in
:doc:`optimization`): QA to QS 7.2e-6 in **14.5 min** with a single
ESS-scaled ``least_squares`` call (the staged ``max_mode`` 1–5 ladder
reaches 3.7e-7 in 25.5 min), and QI to a 25x omnigenity-residual reduction
in **17.3 min**. Two measured gradient-stack optimizations make that
possible — the block-tridiagonal implicit Jacobian (33x on the Jacobian
phase) and the perturbation warm start (3.7x fewer trial-solve iterations)
— both on by default and documented in :doc:`optimization`.

Parity with VMEC2000
--------------------

Free-boundary multigrid has a dedicated reproducible artifact,
``benchmarks/freeboundary_multigrid.json``.  On the public converged CTH-like
``NS_ARRAY = 7, 15`` ladder (Apple Silicon CPU, 2026-07-21), VMEC2000 takes
239 + 340 iterations in 0.98 s; vmex takes 250 + 340 iterations, 10.07 s cold
and 1.98 s warm.  Both activate vacuum exactly once.  Against an ns=15
VMEC2000 wout, vmex's final scale-relative maximum errors are
``6.10e-5`` (R), ``3.59e-4`` (Z), ``1.52e-6`` (iota), and ``5.94e-8``
(relative ``wb``).  The first fine-grid raw residual remains a transient
ordering difference (``FSQR=2.01e-3`` versus VMEC2000's ``1.73``), but both
then take exactly 340 fine-grid iterations to the same fixed point.  Warm
execution is within 2.1x of Fortran on this small case; the one-time XLA
compile dominates the cold result.

Per-iteration algorithmic parity (same step control, preconditioner cadence,
constants) means the solver does not just reach the same answer — it takes
the *same number of iterations* as VMEC2000 on the benchmark decks:

.. list-table::
   :header-rows: 1
   :widths: 36 16 16 32

   * - case
     - VMEC2000 iters
     - vmex iters
     - notes
   * - solovev
     - 215
     - 215
     - exact match
   * - DSHAPE (multigrid 16/32/64/128)
     - 908
     - 903
     -
   * - circular_tokamak (multigrid 10/17)
     - 368
     - 368
     - exact match
   * - cth_like_fixed_bdy
     - 434
     - 434
     - exact match
   * - nfp4_QH_warm_start (ns=35)
     - 450
     - 450
     - exact match
   * - LandremanPaul2021_QA_lowres
     - 1000
     - 1000
     - golden run is NITER-capped at its FTOL 1e-13
   * - LandremanPaul2021_QH_reactorScale_lowres
     - 2408
     - 2406
     -
   * - up_down_asymmetric_tokamak (lasym)
     - 2000 (capped)
     - 1951
     - both stopped at the matched residual 1.5e-13; a fully converged
       VMEC2000 rerun (fsq ~1e-16) matches the core to <= 7.3e-7 on every
       checked harmonic, in 3197 vs 3118 iterations
   * - li383_low_res (single grid, ns=16)
     - 123
     - within the ±25% parity gate
     -

Parity holds not just at the converged endpoint but along the whole
trajectory.  The trace below runs the quick-start QH case
(``nfp4_QH_warm_start``, single grid at ``ns=51``) through all three codes
and plots the total force residual ``fsqr + fsqz + fsql`` per iteration:
the vmex curve lies exactly on top of VMEC2000's (both converge in 502
iterations), and VMEC++ follows a near-identical path (501 iterations).
The vmex trace comes from ``SolveResult.fsq_history``, the VMEC2000
trace from its stdout iteration table run with ``NSTEP = 1``, and the
VMEC++ trace from the ``fsqt`` array of its wout payload.

.. figure:: _static/figures/readme_convergence.png
   :alt: force residual vs iteration for vmex, VMEC2000, and VMEC++
   :align: center
   :width: 95%

   Force residual vs iteration on ``nfp4_QH_warm_start`` at ``ns=51``
   (``benchmarks/make_readme_figures.py --only convergence``; traces cached
   in ``benchmarks/convergence_nfp4_ns51.json``).

The parity suite additionally asserts, per case: convergence at the deck's
``ftol``; ``wb`` within 1e-7 relative of the golden wout; boundary/interior
``rmnc/zmns`` harmonics at rtol 1e-5; and ``iotaf`` at rtol 1e-5. Where the
golden VMEC2000 run is itself NITER-capped (LandremanPaul QA, the lasym
tokamak), both codes are stopped at a matched residual and the documented
absolute tolerances cover the golden run's own remaining non-convergence.
wout files are compared per-variable with CompareWOut-style combined
rel+abs tolerances.

2D block preconditioner
-----------------------

The default 1D radial preconditioner is what reproduces VMEC2000
iteration-for-iteration. For *stiff* decks — very high aspect ratio or strong
finite-β coupling — an opt-in 2D block preconditioner
(:mod:`vmex.core.preconditioner_2d`) replaces the radial-only approximation
with a matrix-free Newton step: a Jacobian-vector-product Hessian applied
through GMRES (SOLVAX's ``block_thomas_truncated`` / Krylov layer). It cuts the
iteration count 2.5–11x on the stiff cases below, and is a strict add-on — the
default 1D path stays byte-identical, so parity is untouched.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - stiff case
     - 1D radial
     - 2D block
     - reduction
   * - aspect-100 tokamak (a)
     - 97
     - 18
     - 5.4x
   * - aspect-100 tokamak (b)
     - 163
     - 15
     - 10.9x
   * - nfp4 QH, finite beta
     - 1885
     - 204
     - 9.2x

.. figure:: _static/figures/readme_precond.png
   :alt: 2D vs 1D preconditioner iteration counts on stiff cases
   :align: center
   :width: 90%

   Iterations to converge, 2D block vs 1D radial preconditioner
   (``benchmarks/make_readme_figures.py --only precond``).

It is opt-in, not the default, on purpose. Fewer iterations is not fewer
seconds: each 2D Newton step (a GMRES solve over Hessian-vector products) costs
far more than a 1D radial sweep, so the measured wall-clock ranges 0.55–1.16x
across easy and stiff decks — a wash to *slower* (≈2x slower on a plain circular
tokamak, a tie even on the aspect-100 case) — and peak memory is ≈30% higher
(the extra GMRES/HVP compile graph). The converged ``wb`` matches the 1D result
to ~1e-10, so it changes the path, not the fixed point. Reach for it when the
1D iteration count is the bottleneck or stalls, not as a blanket default.

Memory
------

Peak resident memory is 0.6–1.5 GB on most bundled rows and about 3.3 GB on
the largest bundled multigrid deck, but those figures are not a
high-resolution upper bound. The spectral state is small; compiled transform
graphs and implicit block factors are not. On the supplied HSX
``ns=101, mpol=18, ntor=24`` deck, replacing the full mode-stacked synthesis
with a separable toroidal FFT reduced a fresh Apple-M4 CPU VMEX run from
676.46 s / 3.896 GiB to 206.56 s / 1.634 GiB. The final 2737-iteration path,
residuals, and energy are unchanged.

VMEC++ with ten threads took 92.03 s and 380.0 MiB on the same host;
one-radial-process VMEC2000 took 1154.82 s and 265.0 MiB. A one-thread
VMEC++ control took 449.79 s and 445.1 MiB. VMEX is therefore now 2.18x
faster than one-thread VMEC++ and 5.61x faster than VMEC2000 on this deck,
while ten-thread VMEC++ remains 2.24x faster and uses about one quarter the
RSS. The remaining storage gap is substantive even though the VMEX state
matches VMEC2000 to near floating-point accuracy.

Before the separable-transform change, explicit CPU and GPU runs of the same
deck on the office host converged in
the same 2737 final-stage iterations.  With the GPU persistent cache already
populated, CPU took 426.94 s and 6.52 GiB host RSS; the RTX A4000 took
1468.07 s and 5.40 GiB.  CPU was therefore 3.44x faster, while cached GPU
placement used 17.1% less host RSS.  The CPU/GPU WOUT relative L2 differences
were ``5.20e-11`` for ``rmnc``, ``3.27e-10`` for ``zmns``, ``9.76e-11`` for
``bmnc``, and ``6.54e-11`` for core iota.  An empty-cache GPU process took
1596.01 s and 6.98 GiB.  These measurements show both that explicit GPU
placement is correct and that it is not automatically faster for a
high-mode CLI run. This is a pre-change baseline; the updated CPU/GPU result
must be recorded before changing the measured automatic device cutoff.

The new synthesis repacks the signed helical coefficients into separable
theta/zeta blocks, evaluates zeta with ``jax.numpy.fft.irfft``, and
performs a short real poloidal contraction. Undersampled toroidal grids fall
back to the established dense DFT. The implicit callback also retains the
dense-real path: a direct FFT tangent expanded complex Jacobian probe batches
past 10 GiB RSS, and compiling fast primal plus dense tangent representations
in one process exceeded 7 GiB. Fixed-boundary
:func:`~vmex.core.solver.solve` and
:func:`~vmex.core.multigrid.solve_multigrid` select separate FFT lanes only
above 512 modes on accelerators and ARM CPUs. Smaller problems retain the
dense-real lane: on the M4, FFT was 38--88% slower warm on three 5--8-mode
routine decks and its 8% warm win at 128 modes came with a 13% first-solve
loss. At 162 modes, both lanes reached the supplied 10,000-iteration cap with
near-zero residuals, but dense was 4.3% faster (279.55 s versus 291.69 s).
x86 CPUs also remain dense. This is a measured default: on the office Xeon
the exact FFT run took 570.47 s,
while dense synthesis with stage-cache release took 413.24 s / 4.04 GiB
(3.2% faster and 38.0% lower peak RSS than the prior cache-retaining dense
baseline of 426.94 s / 6.52 GiB).
Explicit ``use_fft=True`` or ``use_fft=False`` always wins. Implicit AD
retains the dense lanes and their existing checksum/storage gate. The shared
runtime pytree is unchanged.

The one-shot fixed-boundary CLI additionally calls JAX's public
``clear_caches`` between distinct radial grids. This clears in-memory
compilation/staging entries while VMEX's persistent on-disk compilation cache
remains available. On the exact deck it changed 205.97 s / 2.970 GiB to
206.56 s / 1.634 GiB (0.3% wall-time cost, 45.0% lower peak RSS) with
identical geometry, iteration count, residuals, and energy. Library
:func:`~vmex.core.multigrid.solve_multigrid` retains warm stage executables by
default for scans and repeated solves; ``release_stage_cache=True`` opts into
the one-shot policy.

Column chunking bounds simultaneous design-variable probes, not the dominant
dense ``O(ns * m_block**2)`` block bands and factors. On an exact
``ns=201``, ``max_mode=5``, one-Jacobian HSX workload:

- the PR74 automatic block path took 355.40 s and 4.106 GiB RSS;
- an 11-column chunk took 304.24 s and 4.256 GiB;
- a proposed automatic chunk took 283.05 s but 5.612 GiB.

All three Jacobians were bit-identical. The candidate was rejected because
RSS increased 36.68%. Matrix-free GMRES sampled 3.122 GiB (about 24% below
the baseline) but did not finish one Jacobian in 30m08s, over five times the
block wall. A lower-storage factorization or a measured resolution-aware
solver policy is still needed.

``benchmarks/profile_high_resolution.py`` makes that gate reusable on any
input without hardware-selection environment variables:

.. code-block:: bash

   python benchmarks/profile_high_resolution.py implicit \
       --input /path/to/input.HSX_QHS_vacuum_ns201 \
       --max-mode 5 --device cpu --out implicit.json

It records the input resolution, devices, wall time, peak RSS, solve count,
and the complete Jacobian's finiteness, norm, and SHA-256.  On the case above,
the unmodified float64 path repeated in 355.25 s with the established
``74.70287727265259`` norm and checksum.  Four small lower-storage candidates
were rejected: float32 bands/factors and row scaling made this demanding HSX
Jacobian non-finite, while a regularized scaled factor took over twice the
baseline wall time.
Low precision is therefore not a safe drop-in replacement; future work must
change the representation or measured solver policy while preserving this
gate.

A fifth experiment streamed the three radial probe colors instead of retaining
their complete response tensor.  It preserved the norm and checksum and reduced
wall time to 284.69 s, but increased peak RSS by 8.6 % to 4.833 GiB as the
allocator retained loop intermediates into factorization.  It was also rejected:
lower storage must be demonstrated end to end, not inferred from one live array.

The same profiler isolates one free-boundary mirror resolution per process:

.. code-block:: bash

   python benchmarks/profile_high_resolution.py mirror \
       --ns 5 --nxi 7 --elements 4 --exterior-ntheta 8 \
       --betas 0,0.1,0.5 --device cpu --out mirror-coarse.json
   python benchmarks/profile_high_resolution.py mirror \
       --ns 9 --nxi 17 --elements 9 --exterior-ntheta 16 \
       --betas 0,0.1,0.5 --device cpu --out mirror-fine.json

Fresh office CPU processes gave:

.. list-table::
   :header-rows: 1

   * - ``(ns, nxi, elements, ntheta)``
     - iterations by beta
     - wall (s)
     - peak RSS (GiB)
   * - ``(5, 7, 4, 8)``
     - 5 / 8 / 10
     - 16.67
     - 2.359
   * - ``(7, 13, 7, 12)``
     - 42 / 43 / 44
     - 1457.30
     - 4.506
   * - ``(9, 17, 9, 16)``
     - 42 / 44 / 45
     - 3351.74
     - 8.085

All beta points converged with variational maxima below ``2e-13``.  Isolating
the fine process lowers the previous combined 11.04 GiB peak by 26.8 %, but
its 8.085 GiB peak is still 3.43 times the coarse result.  Allocator history
therefore explains part, not all, of the scaling gap.  This 56-minute fine
case remains manual/nightly coverage rather than a required PR check.

The existing optimization controls remain useful within that limitation:

- Scalar objectives default to one matrix-free reverse adjoint
  (``jac_solver="auto"``), avoiding dense block assembly entirely.
- The optimization Jacobian is column-chunked (``jac_chunk_size="auto"``, the
  device-aware choice conservatively capped at a square-root width), limiting
  the simultaneous probe batch for vector objectives.
- Factoring the residual and field pipelines into reusable compiled
  sub-computations cut the implicit-gradient compile ~20% in memory and ~21% in
  wall time, bit-identically (R16).
- The converged-state memo (R25.1) removed a redundant equilibrium
  solve per accepted optimizer iterate and cut the profiled ``opt_step``
  peak RSS from 6.0 to 3.5 GB.

GPU guidance
------------

Measured behavior (``benchmarks/gpu_baseline.json`` plus the supplied
high-mode HSX case):

- **Per-iteration throughput favours the GPU across the tested low- and
  moderate-mode cases** (0.83 ms vs 1.90 ms per iteration at
  ``ns=35, mpol=2, ntor=2``; up to ~3x on NuhrenbergZille-class decks:
  90 s vs 277 s wall).
- **The GPU pays fixed per-solve overheads** (~0.2-0.4 s dispatch/transfer
  floor plus compile or cache-load in cold processes), so small decks that
  finish in well under a second of CPU work stay faster on the CPU
  (``solovev``: 0.043 s CPU vs 0.29 s CUDA warm).
- **Fast desktop CPUs change the calculus**: the GPU wins above were
  measured against the office box's slower server cores. Against an idle
  Apple-Silicon CPU, the CPU wins every production workflow even at
  ``ns = 201`` (the table above) — on a modern desktop, treat the GPU as
  an option for very large or heavily batched solves, not a default.
- **High Fourier mode count is a separate limit**: on the same office host,
  the 858-mode HSX deck was 3.44x faster on CPU than on a cache-warm A4000,
  despite its large aggregate work proxy.

Device policy
~~~~~~~~~~~~~

:mod:`vmex.core.device` encodes this as a default placement rule using
the per-iteration work proxy ``ns * mnmax * nznt`` (the cost driver of the
batched-matmul transforms).  The solve stays on CPU below
``GPU_MIN_ITERATION_WORK = 100_000`` and above
``GPU_MAX_SPECTRAL_MODES = 512``; the middle region uses GPU.  The measured
GPU winners have at most 162 modes and the measured CPU winner has 858; the
intermediate range is not calibrated.  The round cutoff preserves prior AUTO
behavior for common stages through 288 modes while catching the HSX
regression, not a claimed universal hardware crossover.  The policy is a
*default* only:

- an explicit ``device=`` argument to ``solve``/``solve_multigrid`` always
  wins;
- ``device=None`` leaves placement to JAX;
- an active ``jax.default_device`` context or a user-pinned JAX platform makes
  the default ``device="auto"`` policy stand down entirely.

.. code-block:: python

   solve(inp, device="cpu")
   solve(inp, device="gpu")
   with jax.default_device(jax.devices("gpu")[0]):
       solve(inp)  # AUTO respects this context

The mirror solver uses its own measured default because host SciPy repeatedly
drives JAX callbacks. ``vmex.mirror.solve_fixed_boundary``,
``solve_free_boundary``, and ``solve_beta_scan`` choose CPU under ``"auto"``
(35.2 s CPU versus 44.2 s RTX A4000 on the office ``15x15`` case), but expose
the same explicit/``None``/active-context precedence.

Persistent compilation cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vmex`` enables JAX's persistent XLA compilation cache on accelerators,
so the multi-second compile cost is paid once per machine, not once per
process.

.. warning::

   **cwd-shadowing pitfall.** Running ``python`` with a working directory
   that contains a ``vmex`` source checkout can shadow the installed
   package as a namespace package: ``vmex/__init__.py`` never runs, the
   persistent compilation cache is never enabled, and every solve pays the
   full XLA recompile (measured ~7 s vs ~1.7 s warm on CUDA for solovev).
   If GPU runs are mysteriously slow, check that
   ``python -c "import vmex; print(vmex.__file__)"`` points where
   you expect.

Float64 is required (enforced at solver import). On GPUs this means fp64
arithmetic, but the solve is latency- rather than FLOP-bound at benchmark
sizes: the tridiagonal preconditioner solve, for instance, measures identical
fp32/fp64 GPU times (~15 us per radial row, independent of the number of
spectral columns).

Reproducing the numbers
-----------------------

.. code-block:: bash

   python benchmarks/run_baseline.py         # CPU suite -> benchmarks/baseline.json
   python benchmarks/run_freeboundary_multigrid.py  # free-bdy ladder + VMEC2000 parity
   python benchmarks/run_gpu_matrix.py       # GPU matrix -> benchmarks/gpu_baseline.json
   python benchmarks/profile_production.py --device cpu
   python benchmarks/profile_production.py --device gpu
   pytest tests/test_parity_breadth.py     # end-to-end parity suite

For a compact hardware-parity audit, ``device_parity.py`` runs the same small
nonzero-shear equilibrium on explicitly selected CPU/GPU devices and records
the forward state plus boundary derivatives of MHD energy, magnetic well, quasisymmetry,
quasi-isodynamicity, and the mean traceable ``DMerc``, ``jdotb``, and
Glasser ``D_R`` interior profiles in JSON. It does not set or require JAX
platform environment variables::

   python benchmarks/device_parity.py --quick --metrics mhd_energy --output /tmp/vmex-smoke.json
   python benchmarks/device_parity.py --devices cpu,gpu --output /tmp/vmex-parity.json

On a CPU-only host the default runs the CPU lane and marks the cross-device
comparison as skipped; ``--devices cpu`` requests that lane explicitly.
The first command is the short smoke lane; omit ``--metrics`` to audit all
seven objectives.

The parity suite needs the golden VMEC2000 fixtures (fetched release assets);
it is skipped automatically when they are unavailable.
