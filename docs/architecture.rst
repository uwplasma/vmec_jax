Architecture
============

The production implementation lives in :mod:`vmec_jax.core`: one concern per
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
   * - :mod:`~vmec_jax.core.input`
     - ``VmecInput``: INDATA + VMEC++-style JSON parsing, round-trip writers
     - ``readin.f``, ``vmec_input.f``
   * - :mod:`~vmec_jax.core.profiles`
     - pressure / iota / current parameterizations (power series, splines,
       two-power, pedestal, ...)
     - ``profile_functions.f``, ``profil1d.f``
   * - :mod:`~vmec_jax.core.fourier`
     - ``(m,n)`` bookkeeping, parity tables, ``mscale/nscale``, trig tables,
       m=1 constraint maps
     - ``fixaray.f``
   * - :mod:`~vmec_jax.core.transforms`
     - spectral <-> real-space transforms as batched matmuls
     - ``totzsps.f``/``tomnsps.f`` (``totzsp.f90``/``tomnsp.f90``)
   * - :mod:`~vmec_jax.core.geometry`
     - real-space :math:`R, Z, \lambda`, half-mesh Jacobian ``tau``/``sqrt(g)``,
       metrics ``guu, guv, gvv``
     - ``jacobian.f``
   * - :mod:`~vmec_jax.core.fields`
     - :math:`B^u, B^v, |B|`, covariant B, pressure, energies ``wb/wp``,
       ``tcon``
     - ``bcovar.f``, ``add_fluxes.f90``
   * - :mod:`~vmec_jax.core.forces`
     - MHD force kernels + spectral-condensation constraint force
     - ``forces.f``, ``alias.f``
   * - :mod:`~vmec_jax.core.residuals`
     - ``fsqr/fsqz/fsql``, m=1 constraint, ``fedge``
     - ``residue.f90``, ``getfsq.f``
   * - :mod:`~vmec_jax.core.preconditioner`
     - 1D radial preconditioner, vectorized tridiagonal (Thomas) solve
     - ``precondn.f``, ``scalfor.f``, ``lamcal.f90``, ``tridslv``
   * - :mod:`~vmec_jax.core.preconditioner_2d`
     - 2D block preconditioner: matrix-free Newton step (``jax.jvp``
       Hessian-vector products + SOLVAX GMRES)
     - ``Hessian/precon2d.f`` (jog-free)
   * - :mod:`~vmec_jax.core.step`
     - damped 2nd-order Richardson step, ``dtau`` damping (``ndamp=10``),
       ``irst`` back-off
     - ``evolve.f``, ``restart.f``
   * - :mod:`~vmec_jax.core.setup`
     - radial grids, 1D profile arrays, boundary processing, initial guess
     - ``profil1d.f``, ``profil3d.f``, ``readin.f``
   * - :mod:`~vmec_jax.core.solver`
     - single-grid solve loop: ``lax.while_loop`` core + host-blocked CLI lane
     - ``funct3d.f``, ``eqsolve.f``
   * - :mod:`~vmec_jax.core.multigrid`
     - ``NS_ARRAY`` ladder, coarse-to-fine interpolation, hot restart
     - ``runvmec.f``, ``interp.f``
   * - :mod:`~vmec_jax.core.vacuum`
     - NESTOR: Green's function, ``analyt``/``scalpot``, ``potvac`` solve
     - ``NESTOR_vacuum/`` (``precal``, ``surface``, ``bextern``, ``analyt``,
       ``greenf``, ``fourp``, ``scalpot``, ``solver``, ``bsqvac``)
   * - :mod:`~vmec_jax.core.freeboundary`
     - free-boundary iteration, ``ivac``/``nvacskip`` cadence, external-field
       protocol
     - ``funct3d.f`` (free-boundary block)
   * - :mod:`~vmec_jax.core.freeboundary_diff`
     - differentiable free-boundary residual via virtual casing
     - (no VMEC2000 equivalent)
   * - :mod:`~vmec_jax.core.mgrid`
     - mgrid netCDF read/write, differentiable interpolated field; external
       coils live in ESSOS (``essos.coils.Coils``), consumed as an mgrid or
       ``xyz -> B`` callable
     - MAKEGRID file format, ``mgrid_mod.f90``
   * - :mod:`~vmec_jax.core.implicit`
     - implicit differentiation of the equilibrium (``custom_vjp`` + adjoint
       GMRES)
     - (no VMEC2000 equivalent)
   * - :mod:`~vmec_jax.core.optimize`
     - objectives (quasisymmetry ratio residual, QI residual, aspect, iota,
       mirror, well, DMerc, ...) + least-squares driver
     - (no VMEC2000 equivalent)
   * - :mod:`~vmec_jax.core.omnigenity`
     - traceable Boozer ``|B|`` spectrum + Goodman constructed-QI residual
     - (no VMEC2000 equivalent)
   * - :mod:`~vmec_jax.core.bootstrap`
     - differentiable Redl bootstrap ``<J.B>``, mismatch objective,
       self-consistency Picard loop
     - (no VMEC2000 equivalent; BOOTSJ-adjacent scope)
   * - :mod:`~vmec_jax.core.stability`
     - infinite-n ideal-ballooning eigenvalue objective (COBRA-style)
     - (no VMEC2000 equivalent; COBRA companion code)
   * - :mod:`~vmec_jax.core.turbulence`
     - GK flux-tube geometry adapter + SPECTRAX-GK turbulence proxies
     - (no VMEC2000 equivalent)
   * - :mod:`~vmec_jax.core.nyquist`
     - Nyquist-resolution Fourier tables, ``bsubs``, jxbforce, Mercier
     - ``wrout.f``, ``bss.f``, ``jxbforce.f``, ``mercier.f``
   * - :mod:`~vmec_jax.core.postprocess`
     - derived wout quantities (beta, currents, ``specw``, ``equif``, ...)
     - ``eqfor.f``, ``bcovar.f`` outputs
   * - :mod:`~vmec_jax.core.wout`
     - complete ``wout_*.nc`` schema, writer and reader
     - ``wrout.f``
   * - :mod:`~vmec_jax.core.printing`
     - VMEC2000-format iteration lines, stage banners, termination summary
     - ``printout.f``, ``initialize_radial.f``, ``runvmec.f``
   * - :mod:`~vmec_jax.core.plotting`
     - ``vmec --plot`` figures for wout and boozmn files
     - (no VMEC2000 equivalent)
   * - :mod:`~vmec_jax.core.boozer`
     - Boozer transform driver (thin wrapper over ``booz_xform_jax``)
     - booz_xform
   * - :mod:`~vmec_jax.core.device`
     - measured CPU/GPU placement policy for the solve lanes
     - (no VMEC2000 equivalent)
   * - :mod:`~vmec_jax.core.errors`
     - typed zero-crash exceptions + the VMEC2000 ``werror`` message table
     - ``runvmec.f`` error flags
   * - :mod:`~vmec_jax.core.cli`
     - the ``vmec`` entry point
     - ``vmec.f``/``runvmec.f`` driver

State and purity
----------------

The solver state is a frozen pytree
(:class:`~vmec_jax.core.solver.SpectralState`): spectral coefficients of
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

:mod:`vmec_jax.core.device` implements a *measured* CPU/GPU placement policy
for the solve lanes (calibrated against ``benchmarks/gpu_baseline.json``).
The cost driver of one iteration is the ``totzsps/tomnsps`` batched-matmul
work, proxied by

.. math::

   w = \mathrm{ns} \times \mathrm{mnmax} \times \mathrm{nznt}

(:func:`~vmec_jax.core.device.iteration_work`). Per-iteration throughput
favours the GPU at every tested size, but the GPU pays fixed per-solve
overheads (dispatch/transfer floor plus compile or cache load), so small
decks finish faster on the CPU. The measured crossover is
:data:`~vmec_jax.core.device.GPU_MIN_ITERATION_WORK` (``100_000``):
:func:`~vmec_jax.core.device.recommended_device` returns ``"cpu"`` below it
and ``"gpu"`` at or above it, per multigrid stage.

:func:`~vmec_jax.core.device.resolve_device` turns this into a concrete
placement with strict precedence rules: an explicit ``device=`` argument to
``solve``/``solve_multigrid`` always wins; a user pin via ``JAX_PLATFORMS``
or ``JAX_PLATFORM_NAME`` makes the automatic policy stand down entirely; and
the recommendation is applied only when the recommended platform is actually
available. :func:`~vmec_jax.core.device.device_context` wraps a stage in the
corresponding ``jax.default_device``.

The optimization path is different:
:func:`~vmec_jax.core.device.resolve_implicit_device` **always pins the
implicit-gradient work to the CPU** by default. The ``jac="implicit"``
Jacobian builds a per-dof vmapped forward-implicit-differentiation graph —
dozens of preconditioned GMRES solves with inner control flow — whose XLA
compile time grows with the dof count and whose execution is
kernel-launch-bound; measured on GPU it is slower than the CPU at every
optimization size tested, while the forward equilibrium solve inside it is a
host callback that never touches the accelerator anyway. Explicit
``device=`` arguments and user platform pins are still honored.

Naming conventions
------------------

Community-expected VMEC names are kept (``ns, mpol, ntor, nfp, lasym, iotaf,
presf, rmnc, zmns, lmns, bmnc, ...``). Internal Fortran temporaries get
descriptive names (``sqrt_g`` rather than ``gsqrt``); every module docstring
cross-references the VMEC2000 source it ports.
