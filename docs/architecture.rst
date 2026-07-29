Architecture
============

The production implementation lives in :mod:`vmex.core`: one concern per
module, each with a header docstring naming its VMEC2000 counterpart file(s)
and the equations it implements. Everything is pure-JAX and shared between
the CLI, the differentiable API, plotting, and the wout writer — there is a
single set of physics kernels.

Module map
----------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Module
     - Role
     - VMEC2000 counterpart
   * - :mod:`~vmex.core.input`
     - ``VmecInput``: INDATA + structured JSON parsing, round-trip writers
     - ``readin.f``, ``vmec_input.f``
   * - :mod:`~vmex.core.profiles`
     - pressure / iota / current parameterizations (power series, splines,
       two-power, pedestal, ...)
     - ``profile_functions.f``, ``profil1d.f``
   * - :mod:`~vmex.core.fourier`
     - ``(m,n)`` bookkeeping, parity tables, ``mscale/nscale``, trig tables,
       m=1 constraint maps
     - ``fixaray.f``
   * - :mod:`~vmex.core.transforms`
     - spectral <-> real-space transforms as batched matmuls
     - ``totzsps.f``/``tomnsps.f`` (``totzsp.f90``/``tomnsp.f90``)
   * - :mod:`~vmex.core.geometry`
     - real-space :math:`R, Z, \lambda`, half-mesh Jacobian ``tau``/``sqrt(g)``,
       metrics ``guu, guv, gvv``
     - ``jacobian.f``
   * - :mod:`~vmex.core.fields`
     - :math:`B^u, B^v, |B|`, covariant B, pressure, energies ``wb/wp``,
       ``tcon``
     - ``bcovar.f``, ``add_fluxes.f90``
   * - :mod:`~vmex.core.forces`
     - MHD force kernels + spectral-condensation constraint force
     - ``forces.f``, ``alias.f``
   * - :mod:`~vmex.core.residuals`
     - ``fsqr/fsqz/fsql``, m=1 constraint, ``fedge``
     - ``residue.f90``, ``getfsq.f``
   * - :mod:`~vmex.core.preconditioner`
     - 1D radial preconditioner, vectorized tridiagonal (Thomas) solve
     - ``precondn.f``, ``scalfor.f``, ``lamcal.f90``, ``tridslv``
   * - :mod:`~vmex.core.preconditioner_2d`
     - 2D block preconditioner: matrix-free Newton step (``jax.jvp``
       Hessian-vector products + SOLVAX GMRES)
     - ``Hessian/precon2d.f`` (jog-free)
   * - :mod:`~vmex.core.step`
     - damped 2nd-order Richardson step, ``dtau`` damping (``ndamp=10``),
       ``irst`` back-off
     - ``evolve.f``, ``restart.f``
   * - :mod:`~vmex.core.setup`
     - radial grids, 1D profile arrays, boundary processing, initial guess
     - ``profil1d.f``, ``profil3d.f``, ``readin.f``
   * - :mod:`~vmex.core.solver`
     - single-grid solve loop: ``lax.while_loop`` core + host-blocked CLI lane
     - ``funct3d.f``, ``eqsolve.f``
   * - :mod:`~vmex.core.multigrid`
     - fixed- and free-boundary ``NS_ARRAY`` ladders, coarse-to-fine
       interpolation, hot restart, vacuum continuation/rebuild
     - ``runvmec.f``, ``interp.f``
   * - :mod:`~vmex.core.vacuum`
     - NESTOR: Green's function, ``analyt``/``scalpot``, ``potvac`` solve
     - ``NESTOR_vacuum/`` (``precal``, ``surface``, ``bextern``, ``analyt``,
       ``greenf``, ``fourp``, ``scalpot``, ``solver``, ``bsqvac``)
   * - :mod:`~vmex.core.freeboundary`
     - free-boundary iteration, ``ivac``/``nvacskip`` cadence, external-field
       protocol
     - ``funct3d.f`` (free-boundary block)
   * - :mod:`~vmex.core.freeboundary_diff`
     - differentiable free-boundary residual via virtual casing
     - (no VMEC2000 equivalent)
   * - :mod:`~vmex.core.mgrid`
     - mgrid netCDF read/write, differentiable interpolated field, and
       ESSOS/SIMSOPT/``xyz -> B`` host-side Biot--Savart tabulation
     - MAKEGRID file format, ``mgrid_mod.f90``
   * - :mod:`~vmex.core.implicit`
     - implicit differentiation of the equilibrium (``custom_vjp`` + adjoint
       GMRES)
     - (no VMEC2000 equivalent)
   * - :mod:`~vmex.core.optimize`
     - objectives (quasisymmetry ratio residual, QI residual, aspect, iota,
       mirror, well, DMerc, ...) + least-squares driver
     - (no VMEC2000 equivalent)
   * - :mod:`~vmex.core.omnigenity`
     - traceable Boozer ``|B|`` spectrum + Goodman constructed-QI residual
     - (no VMEC2000 equivalent)
   * - :mod:`~vmex.core.bootstrap`
     - differentiable Redl bootstrap ``<J.B>``, mismatch objective,
       self-consistency Picard loop
     - (no VMEC2000 equivalent; BOOTSJ-adjacent scope)
   * - :mod:`~vmex.core.stability`
     - traceable Mercier profile and infinite-n ideal-ballooning eigenvalue
       objective (COBRA-style)
     - ``mercier.f`` / ``jxbforce.f``; COBRA companion code
   * - :mod:`~vmex.core.turbulence`
     - GK flux-tube geometry adapter + GKX turbulence proxies
     - (no VMEC2000 equivalent)
   * - :mod:`~vmex.core.nyquist`
     - Nyquist-resolution Fourier tables, ``bsubs``, jxbforce, Mercier
     - ``wrout.f``, ``bss.f``, ``jxbforce.f``, ``mercier.f``
   * - :mod:`~vmex.core.postprocess`
     - derived wout quantities (beta, currents, ``specw``, ``equif``, ...)
     - ``eqfor.f``, ``bcovar.f`` outputs
   * - :mod:`~vmex.core.wout`
     - VMEC-compatible ``wout_*.nc`` schema, writer and reader, including
       symmetric and LASYM NESTOR potential/surface tables
     - ``wrout.f``
   * - :mod:`~vmex.core.printing`
     - VMEC2000-format iteration lines, stage banners, termination summary
     - ``printout.f``, ``initialize_radial.f``, ``runvmec.f``
   * - :mod:`~vmex.core.plotting`
     - ``vmex --plot`` figures for wout and boozmn files
     - (no VMEC2000 equivalent)
   * - :mod:`~vmex.core.boozer`
     - Boozer transform driver (thin wrapper over ``booz_xform_jax``)
     - booz_xform
   * - :mod:`~vmex.core.device`
     - measured CPU/GPU placement policy for the solve lanes
     - (no VMEC2000 equivalent)
   * - :mod:`~vmex.core.errors`
     - typed zero-crash exceptions + the VMEC2000 ``werror`` message table
     - ``runvmec.f`` error flags
   * - :mod:`~vmex.core.cli`
     - the ``vmec`` entry point
     - ``vmec.f``/``runvmec.f`` driver

State and purity
----------------

The solver state is a frozen pytree
(:class:`~vmex.core.solver.SpectralState`): spectral coefficients of
:math:`R, Z, \lambda` (plus the asymmetric partners when ``lasym``), the
Richardson velocity, time step, damping history, iteration counters, and the
restart flag. All solver functions are pure ``state -> state`` maps, which is
what makes the same kernels usable from ``jit``, ``grad``, and ``vmap``.

Static configuration (resolutions, flags) is hashable and kept out of traced
signatures; mode and radial arrays are padded to the maximum multigrid
resolution so all ``NS_ARRAY`` stages share one compiled executable.

Two lanes, one physics
----------------------

- ``solver.solve(...)`` — a ``lax.while_loop`` over the jitted iteration,
  fully traceable: the forward solver of the differentiable API.
- the CLI lane — a Python ``while`` around the same jitted N-iteration block
  kernel, with host-side residual checks between blocks: exact-``ftol`` early
  exit, live VMEC2000-format printing, buffer donation, no AD bookkeeping.

Both lanes call identical physics kernels; a regression test asserts
per-block state agreement to machine precision.

Device policy (CPU/GPU)
-----------------------

:mod:`vmex.core.device` implements a *measured* CPU/GPU placement policy
for the solve lanes (calibrated against ``benchmarks/gpu_baseline.json``).
The cost driver of one iteration is the ``totzsps/tomnsps`` batched-matmul
work, proxied by

.. math::

   w = \mathrm{ns} \times \mathrm{mnmax} \times \mathrm{nznt}

(:func:`~vmex.core.device.iteration_work`). Per-iteration throughput
favours the GPU across the tested low- and moderate-mode cases, but the GPU
pays fixed per-solve overheads (dispatch/transfer floor plus compile or cache
load), so small decks finish faster on the CPU.  High-mode transforms are a
second measured exception: an 858-mode HSX deck was 3.44x faster on CPU than
on a cache-warm RTX A4000 on the same host.  Therefore
:func:`~vmex.core.device.recommended_device` returns ``"cpu"`` below
:data:`~vmex.core.device.GPU_MIN_ITERATION_WORK` (``100_000``), ``"cpu"``
above :data:`~vmex.core.device.GPU_MAX_SPECTRAL_MODES` (``512``), and
``"gpu"`` in the middle region, per multigrid stage.  The measured GPU
winners have at most 162 modes and the measured CPU winner has 858; the
intermediate range is not calibrated.  The round 512 cutoff preserves prior
AUTO behavior for common stages through 288 modes while catching that
high-mode regression; it is not a universal crossover.

:func:`~vmex.core.device.resolve_device` turns this into a concrete
placement with strict precedence rules: an explicit device always wins;
``device=None`` follows JAX placement; and the default ``device="auto"``
policy stands down for an active ``jax.default_device`` context or a
user-pinned JAX platform.  The recommendation is applied only when its
platform is available. :func:`~vmex.core.device.device_context` wraps a stage
in the corresponding ``jax.default_device``.

The optimization path is different:
:func:`~vmex.core.device.resolve_implicit_device` **pins high-level
optimization's implicit-gradient work to the CPU by default when VMEX owns
placement**. Low-level :func:`~vmex.core.implicit.run` follows JAX placement
when ``device`` is omitted; ``device="auto"`` opts into the same measured
policy. The
``jac="implicit"`` Jacobian builds a per-dof vmapped
forward-implicit-differentiation graph —
dozens of preconditioned GMRES solves with inner control flow — whose XLA
compile time grows with the dof count and whose execution is
kernel-launch-bound; measured on GPU it is slower than the CPU at every
optimization size tested. The forward equilibrium callback uses the solver's
independent automatic per-stage placement policy; the implicit-device choice
controls the residual and Jacobian graphs. Explicit ``device=`` arguments and
JAX placement contexts are still honored for those graphs.

Mirror equilibria have a third measured policy:
:func:`~vmex.core.device.resolve_mirror_device` selects CPU by default for
their SciPy-controlled JAX callback loop (35.2 s CPU versus 44.2 s RTX A4000
on the office ``15x15`` case). Fixed/free-boundary mirror solves and beta
scans still honor explicit devices, ``device=None``, and active JAX placement
without environment variables.

Naming conventions
------------------

Community-expected VMEC names are kept (``ns, mpol, ntor, nfp, lasym, iotaf,
presf, rmnc, zmns, lmns, bmnc, ...``). Internal Fortran temporaries get
descriptive names (``sqrt_g`` rather than ``gsqrt``); every module docstring
cross-references the VMEC2000 source it ports.
