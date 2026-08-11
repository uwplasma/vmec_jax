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

.. begin generated-baseline-table (tools/render_performance_docs.py)

.. list-table::
   :header-rows: 1
   :widths: 34 14 14 14 14

   * - case
     - VMEC2000
     - vmex cold
     - vmex warm
     - VMEC++
   * - li383_low_res
     - 0.86
     - 3.36
     - **0.434**
     - 0.341
   * - solovev
     - 1.07
     - 3.23
     - **0.319**
     - 0.845
   * - circular_tokamak
     - 1.35
     - 4.12
     - **0.522**
     - 1.26
   * - nfp4_QH_warm_start
     - 1.42
     - 3.51
     - **0.641**
     - 0.782
   * - nfp4_QH_warm_start (multigrid)
     - 1.48
     - 11.9
     - **0.787**
     - 1.05
   * - DSHAPE
     - 1.83
     - 6.27
     - **0.812**
     - 1.87
   * - cth_like_fixed_bdy
     - 6.04
     - 6.65
     - **3.5**
     - failed
   * - cth_like_fixed_bdy (multigrid)
     - 7.08
     - 17.3
     - **4.57**
     - failed
   * - cth_like_free_bdy
     - 20.2
     - 28.9
     - **13**
     - 6.36
   * - LandremanPaul2021_QA_lowres
     - 34.7
     - 28.6
     - **22.2**
     - 12.2
   * - LandremanPaul2021_QA_lowres (multigrid)
     - 55.8
     - 45.2
     - **35.9**
     - 16.5
   * - LandremanPaul2021_QH_reactorScale_lowres
     - 61.7
     - 45.7
     - **38.4**
     - failed
   * - NuhrenbergZille_1988_QHS
     - 106
     - 98.6
     - **74.6**
     - 45.7
   * - cth_like_free_bdy_lasym_small
     - 154
     - 133*
     - **105**
     - failed

Bold marks vmex warm beating VMEC2000 (14 of 14 rows).
``*`` marks an equal-iteration-budget run whose CLI exit was nonzero
(the deliberately NITER-bounded LASYM stress row: both codes exhaust
the same budget, so the wall times compare equal work); ``failed``
marks an aborted run and ``n/a`` an unsupported configuration.

.. end generated-baseline-table

These are wall-clock seconds measured on an otherwise idle Apple-Silicon
host — one fresh process per code, run sequentially (never interleaved), so
each row is one controlled baseline rather than a statistical benchmark;
repeated runs move the small rows by tens of milliseconds and the ratios by
a few percent.  The comparable quantity across hosts is the warm/Fortran
*ratio*, not the absolute numbers.

Reading the table:

- **Warm** solves reuse the compiled executable — the number that matters
  inside optimization loops.  The generated caption above counts the rows
  where the warm solve beats VMEC2000; the converged symmetric
  **free-boundary** row is among them (the NESTOR path reaches VMEC2000
  parity *and* edges out the Fortran wall clock).
- **Cold** runs pay a one-time XLA compile, so a single fire-and-forget run
  is usually slower than Fortran — except on the biggest decks, where even
  the cold run, compile included, wins.  The persistent compilation cache
  removes most of the compile cost on subsequent processes.
- **VMEC++** (10-thread default; invoked once per
  deck through its Python API in a fresh process, same host, same sequential
  protocol) is faster on some converged large decks; its ``failed`` rows
  aborted during the first iterations.  ``vmex`` completes every supported
  convergent row and the deliberately NITER-bounded LASYM stress row
  (zero-crash policy); ``n/a`` marks a configuration the reference does not
  support (``lasym`` free boundary).

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
workflow, even at ns = 201.** Forward solves are close. Free-boundary GPU
runs use a hybrid decomposition: plasma iterations stay on the accelerator,
while the small dense NESTOR block runs on CPU with a reused LU factor. The
gradient pipeline is launch-bound on an accelerator,
which is why high-level optimization uses
:func:`vmex.core.device.resolve_implicit_device` to pin implicit-gradient work
to the CPU by default. Low-level :func:`vmex.core.implicit.run` follows JAX
placement when ``device`` is omitted; ``device="auto"`` opts into the
measured CPU policy. The
GPU's wins come against slower server cores and larger-than-production
problem sizes (see the GPU guidance below).

Optimization wall time
~~~~~~~~~~~~~~~~~~~~~~

Whole-campaign numbers, from a near-circular torus to a precise
configuration on a 36-core office CPU (details and scripts in
:doc:`/explanation/adjoint-gradients`): QA to QS 7.2e-6 in **14.5 min** with a single
ESS-scaled ``least_squares`` call (the staged ``max_mode`` 1–5 ladder
reaches 3.7e-7 in 25.5 min), and QI to a 25x omnigenity-residual reduction
in **17.3 min**. Two measured gradient-stack optimizations make that
possible — the block-tridiagonal implicit Jacobian (33x on the Jacobian
phase) and the perturbation warm start (3.7x fewer trial-solve iterations)
— both on by default and documented in :doc:`/explanation/adjoint-gradients`.

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
iterations), and VMEC++ follows a
near-identical path (501 iterations).
The vmex trace comes from ``SolveResult.fsq_history``, the VMEC2000
trace from its stdout iteration table run with ``NSTEP = 1``, and the
VMEC++ trace from the ``fsqt`` array of its wout payload.

.. figure:: /_static/figures/readme_convergence.png
   :alt: force residual vs iteration for VMEX, VMEC2000, and VMEC++
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

.. figure:: /_static/figures/readme_precond.png
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

One such stall is reproducible on the aspect-100 case at ``ns=51`` and
``FTOL=1e-11``.  With ``PRECON_TYPE='GMRES'`` and
``PREC2D_THRESHOLD=1e-6``, VMEX converges in 18 iterations, while VMEC2000's
finite-difference block GMRES remains at a maximum residual of ``2.05e-9``
after 1,600 explicit ``PRE_NITER`` steps.  A separate VMEC2000 1-D solve
converges and agrees with the VMEX result in ``wb`` to ``1.3e-11`` relative
and in the primary geometry to better than ``1e-5``.  The opt-in live test
``test_live_vmec2000_exact_jvp_gmres_robustness`` reproduces all three paths.
This is a robustness result, not a CPU speed or memory claim: the small
VMEC2000 1-D solve is still much cheaper than a cold JAX process.

Memory
------

Peak resident memory is 0.6–1.5 GB on most bundled rows and about 3.3 GB on
the largest bundled multigrid deck, but those figures are not a
high-resolution upper bound. The spectral state is small; compiled transform
graphs and implicit block factors are not. On high-mode decks the separable
toroidal FFT synthesis substantially reduces both wall time and peak memory
relative to the full mode-stacked contraction, and the stage-cache release
keeps peak RSS at the largest single rung. A residual memory gap to
single-purpose compiled implementations remains, dominated by XLA compiled
executables and the runtime floor rather than by the physics working set.
Current-head numbers for the reference high-mode deck are produced by the
reproducible harness (``benchmarks/profile_high_resolution.py`` and
``benchmarks/run_baseline.py``) rather than recorded here, so the
documentation cannot go stale against the code.

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
x86 CPUs also remain dense: on the x86 hosts measured so far the dense
contraction beat the FFT repacking, while ARM CPUs and accelerators prefer
the FFT path above the mode threshold.  Re-run
``benchmarks/profile_high_resolution.py`` on the target host to re-derive the
choice rather than trusting stale numbers.
Explicit ``use_fft=True`` or ``use_fft=False`` always wins. Implicit AD
retains the dense lanes and their existing checksum/storage gate. The shared
runtime pytree is unchanged.

Stage-cache release
~~~~~~~~~~~~~~~~~~~

The one-shot CLI calls JAX's public ``clear_caches`` between distinct radial
grids, so peak memory tracks the largest single rung instead of accumulating
every rung's executables; the persistent on-disk compilation cache is
unaffected.  Library :func:`~vmex.core.multigrid.solve_multigrid` and
:func:`~vmex.core.multigrid.solve_free_boundary_multigrid` retain warm stage
executables by default (the right policy for scans and repeated solves) and
accept ``release_stage_cache=True`` to opt into the one-shot behaviour.
The CLI and library compile solver lanes sequentially by default.
``--prefetch-compile`` (or ``prefetch_compile=True`` in the library) overlaps
the next rung's compilation.  This can reduce cold-start latency on a
core-rich host, but increases peak memory and can contend with the active
solve when the available CPU set is small.  ``--no-prefetch-compile`` remains
an explicit spelling of the default.

Reproducible resource profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``benchmarks/profile_resources.py`` is the common fixed-boundary,
free-boundary, implicit-AD, and mirror resource harness. Each case runs in a
fresh process and reports cold and warm wall time, OS peak RSS, device peak
memory when the backend exposes it, residuals, iterations, native thread
count, and output or gradient SHA-256. Prefetched fixed- and free-boundary
rows also report XLA executable memory. Other rows state why no executable
estimate is available.

The harness selects hardware only through the public ``device=`` API. It
records inherited JAX platform environment settings instead of creating
them. ``--device gpu --device-index 1`` selects a second visible GPU by
passing its JAX device object, without a platform environment pin. Fetch the
released mgrid assets before the default free-boundary row::

   python tools/fetch_assets.py --bundle reference-nc
   python benchmarks/profile_resources.py --device cpu --out /tmp/vmex-resources.json

An external high-resolution deck can replace the fixed and implicit inputs
without copying it into the repository::

   python benchmarks/profile_resources.py \
     --cases fixed,implicit \
     --fixed-input /path/to/input.hsx \
     --implicit-input /path/to/input.hsx \
     --vmec2000-executable /path/to/xvmec2000 \
     --vmec2000-source /path/to/STELLOPT \
     --vmecpp-python /path/to/vmecpp-python \
     --vmecpp-source /path/to/vmecpp \
     --vmecpp-threads 10 \
     --out /tmp/hsx-resources.json

The default retains compiled stages for a repeated library solve.
``--release-stage-cache --no-prefetch-compile`` instead measures the
lower-peak, one-shot policy used by the CLI. Its second timing can reload
released stages and is therefore not an in-memory warm-run measurement.

The report stores input and executable hashes, VMEX/JAX versions, VMEC++
version, hardware, and git revision without storing private paths. Mirror
scaling defaults to the ``5:7:4,7:13:7,9:17:9`` coarse/medium/fine ladder;
``--mirror-ladder`` changes it explicitly.

Implicit-storage experiments (recorded so they are not repeated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Column chunking bounds simultaneous design-variable probes, not the dominant
dense ``O(ns * m_block**2)`` block bands and factors.  Candidate reductions
were measured on a fixed high-mode implicit workload with
``benchmarks/profile_high_resolution.py`` (which records resolution, devices,
wall time, peak RSS, and the Jacobian's finiteness, norm, and SHA-256) and
rejected, each for a concrete reason:

- an automatic chunk schedule was faster but raised peak RSS by more than a
  third;
- float32 bands/factors and row scaling made the demanding Jacobian
  non-finite -- low precision is not a safe drop-in replacement;
- a regularised scaled factorisation more than doubled the wall time;
- matrix-free GMRES sampled ~24% less memory but did not finish one Jacobian
  in over five times the block-path wall;
- streaming the three radial probe colours preserved the checksum but the
  allocator retained loop intermediates into factorisation, *raising* RSS;
- differentiating a genuinely local three-surface kernel (which matches the
  global residual to ``2e-12``, LASYM included) still failed the end-to-end
  gate: compilation/allocator retention plus the unchanged dense factors
  erased the local-temporary saving.  The kernel remains as a tested
  foundation for a future lower-storage factor representation.

The conclusion stands until the factor *representation* changes: scalar
objectives use the matrix-free reverse adjoint, vector objectives keep the
exact block path, and any new storage candidate must reproduce the recorded
norm/checksum and beat both wall and RSS end to end.

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
batched-matmul transforms): the solve stays on CPU below
``GPU_MIN_ITERATION_WORK = 100_000`` and above
``GPU_MAX_SPECTRAL_MODES = 512``, and uses GPU in the middle region.  The
calibration evidence and the full precedence rules — an explicit ``device=``
argument always wins, ``device=None`` leaves placement to JAX, and an active
``jax.default_device`` context or user-pinned platform makes ``"auto"``
stand down — are in :ref:`explanation/architecture:Device policy (CPU/GPU)`.

.. code-block:: python

   solve(inp, device="cpu")
   solve(inp, device="gpu")
   with jax.default_device(jax.devices("gpu")[0]):
       solve(inp)  # AUTO respects this context

The mirror solver uses its own measured default because host SciPy repeatedly
drives JAX callbacks: ``vmex.mirror`` solves choose CPU under ``"auto"``
with the same explicit/``None``/active-context precedence (measurement in
:ref:`explanation/architecture:Device policy (CPU/GPU)`).

Persistent compilation cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vmex`` enables JAX's persistent XLA compilation cache on CPUs and
accelerators, so the multi-second compile cost is paid once per machine, not
once per process.

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

GPU decision sweep (office rig)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One command produces every CPU-vs-GPU crossover curve on a dual-GPU host —
warm per-iteration marginals on the ``ns x mnmax`` grid (``ns`` 51/101/201,
``mnmax`` 8/128/288) for the production CLI lane and the multigrid ladder,
the free-boundary NS sweep (steady vacuum lane, NESTOR included), the
537-mode probe where the GPU default switches to FFT synthesis, and the
fixed/free/gradient workflow profiles (cold+warm+memory)::

   python benchmarks/run_gpu_matrix.py --office --out benchmarks/gpu_office.json

Each cell is a fresh subprocess selecting hardware through the public
``device=`` API.  For the CUDA-graph A/B, repeat with command buffers and a
second output file, then compare the two ``stepscan`` sections::

   python benchmarks/run_gpu_matrix.py --office \
       --xla-flags "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL" \
       --out benchmarks/gpu_office_cmdbuf.json

The applied ``XLA_FLAGS`` are recorded in the artifact's ``meta`` block.

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
