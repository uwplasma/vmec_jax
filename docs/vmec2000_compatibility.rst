VMEC2000 compatibility and research scope
=========================================

Purpose
-------

This page is the normative disclosure of what VMEX does with VMEC2000 input,
solver, output, and differentiation features.  It is deliberately more
conservative than a feature list: a control is called *supported* only when it
reaches the production path, and validation evidence is stated separately.

The source of truth used for this audit is the STELLOPT VMEC2000 tree:

* ``LIBSTELL/Sources/Modules/vmec_input.f`` for the complete ``&INDATA``
  namelist and defaults;
* ``VMEC2000/Sources/Input_Output/readin.f`` for post-read normalization and
  preconditioner selection;
* ``VMEC2000/Sources/TimeStep/runvmec.f``, ``evolve.f`` and ``vmec.f`` for
  multigrid, convergence, continuation, and WOUT policy;
* ``VMEC2000/Sources/General`` for force, restart, RFP, and axis behavior;
* ``VMEC2000/Sources/NESTOR_vacuum`` for the vacuum solve; and
* ``VMEC2000/Sources/Input_Output/wrout.f`` and ``jxbforce.f`` for WOUT
  variables and derived diagnostics.

Status vocabulary
-----------------

``implemented``
   The production VMEX path consumes the control or implements the method.
   This does not by itself claim numerical parity for every equilibrium.

``parity-regressed``
   Tests compare the relevant trajectory, state, or WOUT quantity with
   VMEC2000 for at least one representative case.

``deliberate divergence``
   VMEX implements the same mathematical purpose with a disclosed different
   algorithm or extends VMEC2000 behavior.

``partial``
   A documented subset is implemented.  The omitted subset fails explicitly
   where silently dropping it could change a result.

``accepted no-op``
   The value cannot change equilibrium physics and is retained only so legacy
   decks parse.  VMEX does not produce the requested legacy artifact.

``accepted with warning``
   The equilibrium equations are unaffected, so the deck can run, but VMEX
   warns that a requested auxiliary artifact is not produced.

``rejected when active``
   The parser recognizes the VMEC2000 control but raises
   :class:`~vmex.core.input.UnsupportedInputModeError` before setup.  It never
   substitutes an ordinary equilibrium for the requested model.

``not implemented``
   No production implementation or parity claim exists.

No-silent-physics policy
------------------------

Input passes through four distinct gates:

``tokenize -> classify -> construct/normalize -> setup/solve -> output``.

Parsing a name is not evidence that the solver uses it.  VMEX therefore
applies these rules:

1. Unknown INDATA variables and unknown VMEC++ JSON keys are input errors.
2. Active controls which change the mathematical problem, iteration contract,
   or requested WOUT convention are either implemented or rejected before
   iteration 1.
3. Neutral spellings such as ``AH=0``, ``AT=[1,0,...]``,
   ``TRIP3D_FILE='NONE'``, and an inactive reconstruction block remain
   accepted.
4. Active legacy output requests which do not change the equilibrium are
   accepted with a warning when the artifact is unavailable; truly obsolete
   controls with no VMEC2000 production behavior are listed as no-ops.
5. Symmetry-limited derived methods raise on ``LASYM=T`` instead of omitting
   Fourier partners.

The privacy-preserving ``tools/diagnose_input.py`` reports the same
classification as the production parser.  Its stable ``D00*`` code contains
no filename, coefficient, input value, or equilibrium result.  It also reports
whether explicit angular grids meet VMEC2000's automatic-resolution floor;
``W01_ANGULAR_GRID_BELOW_VMEC_DEFAULT`` is a convergence-risk warning, not a
parser or physics-mode rejection.

Equilibrium capability matrix
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 19 57

   * - Capability
     - Status
     - Scope and evidence
   * - Fixed boundary, stellarator symmetric
     - parity-regressed
     - VMEC force iteration, multigrid, restart logic, profiles, and WOUT
       variables have representative VMEC2000 golden tests.
   * - Fixed boundary, ``LASYM=T``
     - parity-regressed
     - Full asymmetric solve and WOUT partner channels are implemented.
       Symmetry-limited *derived objectives* are a separate row below.
   * - Free boundary, NESTOR, symmetric
     - implemented; parity-regressed on representative cases
     - Mgrid external field, plasma-current filament, full/incremental vacuum
       cadence, pressure coupling, and WOUT equilibrium channels.
   * - Free boundary, NESTOR, ``LASYM=T``
     - implemented; limited regression envelope
     - A CTH-like asymmetric case exercises the solve.  Symmetric and
       asymmetric vacuum-potential WOUT completeness are tracked separately.
   * - Fixed-boundary ``NS_ARRAY``
     - parity-regressed
     - Increasing stages interpolate the final state, equal stages rerun, and
       ``readin.f`` ends the active ladder at the first decreasing or
       nonpositive entry.
   * - Free-boundary ``NS_ARRAY``
     - implemented by PR #70
     - Plasma state, ``ivac``, adaptive ``nvacskip``, boundary pressure, and
       vacuum continuation are carried.  Resolution-specific NESTOR basis,
       Green-function, filament, matrix, and cache structures are rebuilt or
       selected at every executed resolution.
   * - Hot restart
     - implemented
     - Fixed and free boundary accept ``initial_state``.  A user free-boundary
       restart repeats activation (reset-file semantics); continuation between
       radial stages carries the active vacuum state.
   * - Mgrid
     - implemented
     - MAKEGRID netCDF field and coil-group currents are interpolated in
       :class:`~vmex.core.mgrid.MgridField`.
   * - ESSOS/SIMSOPT field callable
     - deliberate VMEX extension
     - A Cartesian ``xyz -> B`` callable can be tabulated once into an
       ``MgridField``.  The table/current scale remains differentiable; coil
       geometry derivatives are not retained by tabulation.
   * - CLI ``--coils`` / ``DIRECT_COILS``
     - deliberate VMEX extension
     - ESSOS exports an mgrid representation which then follows the same
       NESTOR path.  This is not an interpolation-free coil solve.
   * - VMEC++ ``free_boundary_method='only_coils'``
     - rejected when active
     - This is a different boundary model, not an alias for choosing a coil
       input source.
   * - BIEST vacuum method
     - not implemented
     - VMEC++'s ``biest`` selector is rejected rather than mapped to NESTOR.
   * - TRIP3D coupling
     - rejected when active
     - A non-``NONE`` ``TRIP3D_FILE`` receives
       ``D00E_TRIP3D_MODE_UNSUPPORTED``.
   * - Reconstruction
     - rejected when active
     - Effective ``LRECON`` with ``IMSE>0`` or ``ITSE>0`` receives
       ``D00A_RECONSTRUCTION_MODE_UNSUPPORTED``.  An inert ``LRECON=T`` with no
       reconstruction signals remains inert, matching VMEC2000.
   * - Reversed-field pinch
     - rejected when active
     - ``LRFP=T`` receives ``D00B_RFP_MODE_UNSUPPORTED``.  VMEX does not claim
       the reciprocal-q profile, force, residue, or vacuum sign semantics.
   * - ANIMEC anisotropy/flow
     - rejected when active
     - Nonzero ``AH`` or non-default ``AT`` receives
       ``D00F_ANIMEC_MODE_UNSUPPORTED``.
   * - Boundary target-volume rescaling
     - rejected when active
     - ``TVOLUME>0`` / ``LVOLUME_RFIX`` geometry rescaling from
       ``vmec_input.f:RESCALE_BOUNDARY`` is not yet ported.

Complete INDATA disposition
---------------------------

Core grid, profiles, and geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - VMEC2000 variables
     - Status
     - Effective VMEX behavior
   * - ``LASYM, NFP, MPOL, NTOR, NTHETA, NZETA``
     - implemented
     - Select symmetry, Fourier resolution, and angular quadrature.
   * - ``NS_ARRAY``
     - implemented
     - Radial resolution-continuation ladder.  The explicit old-style
       ``NS_ARRAY(1)=0`` form expands through ``NSIN`` to ``[NSIN,31]``.
       ``readin.f`` accepts a positive nondecreasing prefix (equal grids
       rerun) and excludes the first decreasing/nonpositive value and its tail.
   * - ``FTOL_ARRAY``
     - implemented
     - Per-stage physical-force stopping tolerance.  Repeated dense/indexed
       namelist assignments overlay in source order.  An explicit zero first
       entry generates VMEC2000's geometric ``1e-8 -> FTOL`` ladder.
   * - ``FTOL``
     - implemented
     - Used directly for a single grid when ``FTOL_ARRAY(1)=0`` and as the
       final target of the generated multigrid tolerance ladder.
   * - ``NITER_ARRAY, NITER``
     - implemented
     - VMEC2000's ``ALL(NITER_ARRAY==-1) -> NITER`` fallback is reproduced;
       any explicit array write preserves unassigned ``-1`` elements.  Such a
       stage executes one force/evolution pass before the ``iter2 >= niter``
       limit is observed.
   * - ``DELT, TCON0, NSTEP``
     - implemented
     - Initial time step, spectral-condensation multiplier, and print cadence.
       VMEX's driver retries a fatal 75-reset stage from its best finite
       checkpoint with reduced ``DELT`` (two attempts by default); set
       ``jacobian_retries=0``/``--jacobian-retries 0`` for exact VMEC2000
       termination behavior.
   * - ``APHI, PHIEDGE``
     - implemented
     - Toroidal-flux map and edge flux.  Indexed and section assignments use
       Fortran lower bounds and column-major order.
   * - ``GAMMA, BLOAT, SPRES_PED, PRES_SCALE``
     - implemented
     - Profile/equation-of-state controls used by setup.
   * - ``PMASS_TYPE, AM, AM_AUX_S, AM_AUX_F``
     - implemented
     - Pressure profile types listed in :mod:`vmex.core.profiles`.
   * - ``PIOTA_TYPE, AI, AI_AUX_S, AI_AUX_F``
     - implemented, excluding RFP interpretation
     - Prescribed-iota profile for ``NCURR=0``.
   * - ``PCURR_TYPE, AC, AC_AUX_S, AC_AUX_F, CURTOR, NCURR``
     - implemented
     - Prescribed-current lane for ``NCURR=1`` and ordinary current data.
   * - ``RAXIS_CC, ZAXIS_CS, RAXIS_CS, ZAXIS_CC``
     - implemented
     - Initial axis Fourier coefficients.
   * - obsolete ``RAXIS, ZAXIS``
     - implemented compatibility
     - Nonzero legacy entries override their modern partners as in
       ``read_indata_namelist``.
   * - ``RBC, ZBS, RBS, ZBC``
     - implemented
     - Scalar, starting-element, and multidimensional Fortran namelist
       sections are supported with declared bounds, inclusive section limits,
       first-subscript-fastest order, and source-ordered overlay.
   * - ``TVOLUME, LVOLUME_RFIX``
     - rejected when active
     - Positive target-volume rescaling is not silently omitted.

Force, axis, and iteration controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 31 18 51

   * - VMEC2000 variables
     - Status
     - Effective VMEX behavior
   * - ``LFORBAL``
     - implemented by PR #70
     - Selects VMEC2000's non-variational average-force replacement for the
       ``m=1,n=0`` R/Z channels.  ``calc_fbal`` consumes full-mesh ``chipf``
       reconstructed from the effective half-mesh ``chips`` by the
       ``add_fluxes.f90`` formulas for both ``NCURR`` modes; WOUT uses the
       same reconstruction.
   * - ``LMOVE_AXIS``
     - implemented by PR #70
     - Enables the first-pass ``irst=4`` axis re-guess when the finite force sum
       exceeds the VMEC2000 threshold.  Missing/all-zero axis coefficients are
       not pre-inferred in production: VMEX now follows VMEC2000's supplied
       zero-axis first pass and exactly one ``eqsolve.f`` recovery transfer.
       The value is preserved in WOUT.
   * - ``LFULL3D1OUT``
     - implemented
     - On fixed or free NITER exhaustion, ``T`` writes the unconverged WOUT and
       ``F`` does not.  Numerical/Jacobian failures never write one.
   * - ``PRE_NITER``
     - rejected when active with 2-D GMRES
     - VMEC2000 changes the total post-activation iteration cap.  VMEX does not
       yet implement that budget mutation.
   * - ``MAX_MAIN_ITERATIONS``
     - rejected above 1
     - VMEC2000 can request additional ``NITER`` blocks after
       ``more_iter_flag``.  VMEX instead exposes explicit hot restart.
   * - ``LGIVEUP, FGIVEUP``
     - rejected when ``LGIVEUP=T``
     - VMEC2000's early stop between poorly converged radial stages is not yet
       implemented.
   * - ``TIME_SLICE``
     - implemented
     - Preserved in the VMEC-style run header.  It does not change the
       equilibrium equations.
   * - ``OMP_NUM_THREADS``
     - accepted no-op
     - JAX/XLA owns CPU threading; see :doc:`parallelization`.

Free-boundary controls
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - VMEC2000 variables
     - Status
     - Effective VMEX behavior
   * - ``LFREEB, MGRID_FILE, EXTCUR``
     - implemented
     - Select NESTOR and the external coil-group field.  ``MGRID_FILE='NONE'``
       selects fixed boundary.  An unreadable requested mgrid follows
       VMEC2000's warning/fixed-boundary fallback, including fixed WOUT
       metadata.
   * - ``NVACSKIP``
     - implemented
     - Full-vacuum cadence and adaptive lower bound; nonpositive input falls
       back to ``NFP``.
   * - ``MFILTER_FBDY, NFILTER_FBDY``
     - implemented
     - Suppress selected high boundary modes in setup and implicit boundary
       degrees of freedom.
   * - ``TRIP3D_FILE``
     - rejected when non-``NONE``
     - No external-field substitution occurs.

Preconditioner controls
~~~~~~~~~~~~~~~~~~~~~~~

VMEC2000's strings do not denote one interchangeable implementation.
``readin.f`` selects a 2-D block operator with four distinct evolution/Krylov
algorithms: ``CG`` (type 1), ``GMRES`` (type 2), ``GMRESR`` (type 3), and
``TFQMR`` (type 4).  ``NONE``, ``DEFAULT``, and unrecognized strings leave the
ordinary 1-D radial preconditioner in the audited source.

VMEX uses this explicit contract:

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - ``PRECON_TYPE``
     - VMEX status
     - Meaning
   * - ``NONE`` or ``DEFAULT``
     - implemented
     - VMEC-parity 1-D radial tridiagonal plus lambda preconditioner.  These
       spellings disable only the optional 2-D block preconditioner.
   * - ``GMRES``
     - deliberate divergence
     - Exact JAX JVP of the preconditioned force, solved matrix-free by
       restarted SOLVAX GMRES.  VMEC2000 instead finite-difference assembles a
       block-tridiagonal operator.
   * - ``CG``, ``GMRESR``, ``TFQMR``
     - rejected when active
     - These are not aliases for VMEX GMRES.
   * - any other string
     - rejected
     - Prevents typographical selection of unintended solver behavior.

``PREC2D_THRESHOLD`` is consumed by VMEX GMRES on the finest radial stage
after the minimum-iteration gate.  An explicit Python
:class:`~vmex.core.preconditioner_2d.Prec2DConfig` remains the VMEX-native
advanced interface.

The production radial solve uses SOLVAX's checked tridiagonal interface.  It
replays VMEC2000 ``serial_tridslv``'s unregularized modified-pivot test at the
same ``1e-8`` relative threshold and additionally verifies a normwise backward
residual.  VMEC2000 executes a process-wide ``STOP`` on a rejected pivot;
VMEX instead applies the identity action only to rejected coefficient columns,
keeps the update finite, and reports
``D04E_RADIAL_PRECONDITIONER_REJECTED``.  Well-conditioned columns retain the
ordinary platform-selected Thomas/fused solve and its existing parity tests.

Reconstruction, anisotropy, and legacy output controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete VMEC2000 reconstruction family is classified together because
its arrays have no ordinary-equilibrium meaning:

``LRECON, IMSE, ITSE, PSA, PFA, ISA, IFA, IMATCH_PHIEDGE, IOPT_RAXIS,
TENSI, TENSP, TENSI2, FPOLYI, MSEANGLE_OFFSET, MSEANGLE_OFFSETM, ISNODES,
IPNODES, RSTARK, DATASTARK, SIGMA_STARK, RTHOM, DATATHOM, SIGMA_THOM,
PRESFAC, PRES_OFFSET, PHIDIAM, SIGMA_DELPHID, NFLXS, INDXFLX, DSIOBT,
SIGMA_FLUX, NBFLD, INDXBFLD, BBC, SIGMA_B, SIGMA_CURRENT, LPOFR``.

They are accepted only while the effective reconstruction mode is inactive;
active reconstruction is rejected as a unit.

The ANIMEC family ``AH, AT, BCRIT, PH_TYPE, PT_TYPE, AH_AUX_S, AH_AUX_F,
AT_AUX_S, AT_AUX_F`` is similarly accepted at its isotropic defaults and
rejected when ``AH`` or ``AT`` activates anisotropic physics.

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - Output/legacy variables
     - Status
     - Disclosure
   * - ``LBSUBS=T``
     - rejected when active
     - Requests a different ``B_s`` diagnostic in ``jxbforce.f``.
   * - ``LNYQUIST=F``
     - rejected when active
     - VMEX currently writes its Nyquist WOUT contract.
   * - ``LMAC, LEDGE_DUMP, LOLDOUT, LWOUTTXT, LDIAGNO``
     - accepted with warning
     - These request auxiliary VMEC2000 monitor, edge, legacy, text-WOUT, or
       DIAGNO artifacts which VMEX does not produce.  The netCDF WOUT and
       equilibrium solve remain available.
   * - ``LMOVIE, LSPECTRUM_DUMP, LOPTIM``
     - accepted no-op
     - These are obsolete and have no production behavior in the audited
       VMEC2000 source.  They do not change the equilibrium force solve.
   * - ``LBOOZ, MBOOZ, NBOOZ, BOOZ_SURFACES``
     - VMEX extension; ``LBOOZ=T`` rejected
     - Use ``vmex --booz --mbooz ... --nbooz ... --booz-surfaces ...``.

WOUT contract and limitations
-----------------------------

VMEX writes a VMEC2000-shaped netCDF WOUT; :doc:`wout_reference` lists every
variable.  The following distinctions are important:

* ``lrecon`` and ``lrfp`` are false because active modes are rejected.
* ``lmove_axis`` records the actual input value.
* ``lfreeb`` records the *effective* solve.  A missing-mgrid fixed-boundary
  fallback is not labeled free-boundary.
* ``ier_flag=2`` WOUT output requires ``LFULL3D1OUT=T`` for either boundary
  mode.
* On the base of PR #70, NESTOR potential/surface variables are present as
  netCDF fill because the solver result does not expose them.  PR #73 adds the
  symmetric potential/surface export; its asymmetric extension remains a
  separate requirement and must not be lost during merge.
* PR #72 adds ``curlabel`` schema preservation.  It is complementary to, not a
  replacement for, the free-boundary solve changes.

Differentiation and derived-method matrix
-----------------------------------------

``AD`` below means algorithmic/automatic differentiation of the stated VMEX
map.  ``FD-validated`` means a test compares that derivative with a finite
difference; it does not mean the method itself uses FD.

.. list-table::
   :header-rows: 1
   :widths: 27 20 53

   * - Method
     - Status
     - Exact scope
   * - Fixed-boundary implicit equilibrium derivative
     - implemented; FD-validated
     - Boundary/profile/current/flux parameters at a converged fixed point.
       Multigrid and adaptive iteration history are initializers, not
       differentiated.
   * - NESTOR forward solve derivative
     - not implemented
     - The host-driven full/incremental NESTOR fixed point has no adjoint or
       custom derivative.
   * - Virtual-casing external-field residual
     - implemented; FD-validated
     - Coil/current derivatives on a specified plasma boundary.  This is not
       the derivative of a fully reconverged NESTOR equilibrium.
   * - Simultaneous boundary + coil surface-field construction
     - partial
     - Traceable state-to-surface data exists for ``LASYM=F`` with the optional
       ``virtual_casing_jax`` dependency.  ``LASYM=T`` is rejected as
       unvalidated.
   * - Mgrid tabulation
     - partial derivative contract
     - Differentiable in table values/current scale, not in the coil geometry
       used to generate a frozen table.
   * - Mercier WOUT diagnostic
     - implemented; host/FD-only
     - VMEC2000-style WOUT engine.  The traceable Mercier objective is
       stellarator-symmetric.
   * - Ballooning, traceable Boozer/QI/omnigenity, turbulence geometry
     - partial
     - Current traceable implementations require ``LASYM=F`` and raise
       otherwise.
   * - ``L_grad_B`` WOUT and state objectives
     - partial
     - ``LASYM=F`` only.  Both public lanes now raise on asymmetric input;
       asymmetric partners are never discarded.
   * - Quasilinear/nonlinear-window turbulence proxies
     - value-level
     - Eigenvector-weighted objectives use finite-difference optimization;
       the documented growth-rate lane is AD-capable.

Terms and design decisions
--------------------------

Fixed boundary
   The last radial surface is prescribed by ``RBC/ZBS/RBS/ZBC`` and is not
   evolved by the plasma force iteration.

Free boundary
   The plasma edge evolves and is coupled to exterior magnetic pressure.
   ``LFREEB`` is a physics selection, not merely an output flag.

NESTOR
   VMEC2000's boundary-integral vacuum solver.  It solves for a harmonic
   scalar potential so the total exterior field is tangent to the plasma
   boundary.

Mgrid
   A cylindrical grid of external magnetic-field components, partitioned by
   coil group and weighted by ``EXTCUR``.

Multigrid
   In VMEC terminology, a radial *resolution-continuation ladder*
   (``NS_ARRAY``), not a V-cycle correction method.  Each stage performs a
   nonlinear equilibrium solve and transfers the state to the next stage.
   Fixed and free boundary use the same normalized nondecreasing prefix.

Hot restart
   Seeding a solve with an existing spectral state.  Fixed-boundary hot restart
   adapts the edge smoothly to a changed boundary; free-boundary restart
   distinguishes user reset semantics from within-ladder vacuum continuation.

Raw/physical force residual
   ``FSQR``, ``FSQZ``, and ``FSQL`` are normalized physical force channels and
   determine convergence.

Preconditioned update residual
   Lower-case/internal force norms measure the update after the radial or
   optional 2-D preconditioner.  Changing a preconditioner must not change the
   equilibrium root.

``LFORBAL``
   Replaces selected R/Z force coefficients with a non-variational
   flux-averaged force-balance form.  Its radial force uses full-mesh
   ``phipf/chipf`` and half-mesh current/pressure differences exactly as
   ``fbal.f``/``add_fluxes.f90`` specify.  It changes the force operator and
   must be propagated through setup, solve, diagnostics, and differentiation.

``LMOVE_AXIS``
   Enables the VMEC2000 first-force axis-recovery control transfer.  It is not
   equivalent to supplying an axis and is independent of the preconditioner.

``LFULL3D1OUT``
   Controls output after iteration-budget exhaustion; it does not declare the
   unconverged state converged.

CLI lane / JIT lane
   Two controllers around the same force and update kernels.  The CLI lane
   prints/checks between compiled blocks; the JIT lane is a traced loop.
   ``VMEX_FAST_COMPILE`` and device policy may change compilation/execution
   strategy, never the intended physics mode.

AD / FD
   Automatic differentiation computes derivatives of a coded map.  Finite
   differences perturb inputs and rerun that map.  An AD result is
   research-grade only for the exact map disclosed and independently checked
   over a stated parameter/regime envelope.

Open-PR ownership and merge preservation
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 13 33 54

   * - PR
     - Primary scope
     - Relationship to this audit
   * - #61--#64
     - Device contract and accelerator selection/CI
     - Performance/placement only; they do not repair ignored VMEC2000 physics
       controls.
   * - #65--#66
     - Traceable Mercier method/objective
     - Adds an explicitly symmetry-limited derived method; does not change the
       equilibrium parser.
   * - #67--#68
     - Accelerator documentation, doctor, and benchmarks
     - Complement the device design disclosure.
   * - #69
     - Nonfinite/axis diagnostics and parser hardening
     - Originally detected active reconstruction/RFP only in the diagnostic.
       Production enforcement and the complete no-silent-mode policy belong to
       the revised #70 stack and must survive merge.
   * - #70
     - Free-boundary multigrid and VMEC2000 parity hardening
     - Owns multidimensional namelist sections, APHI/profile parsing, finite
       first-force diagnosis, exact single-transfer axis recovery, LFORBAL,
       full-mesh force-balance profiles, bounded JAC75 best-checkpoint recovery,
       angular-grid risk diagnostics, free-boundary vacuum
       continuation/rebuild, guarded radial solves, effective fallback
       metadata, preconditioner semantics, and this compatibility ledger.
   * - #71
     - Hot-restart example/documentation
     - Depends on #70's free-boundary multigrid behavior; keep after #70.
   * - #72
     - ``curlabel`` WOUT schema
     - Complementary output parity; retain when resolving WOUT conflicts.
   * - #73
     - Symmetric NESTOR potential/surface WOUT export
     - Closes part of the documented fill-value gap; asymmetric export remains
       explicit follow-up work.
   * - #74
     - Bounded nightly validation
     - CI scheduling/limits only; no solver semantics.

Public integrated stress gate
-----------------------------

``tests/test_vmec2000_feature_stress.py`` constructs a reproducible difficult
case solely from the tracked public
``input.serial2500170_surface_points_mpol12_ntor12`` boundary.  The stress
header combines ``MPOL=13``, ``NTOR=9`` (238 modes), no supplied axis,
``LFORBAL=T``, ``PRECON_TYPE='NONE'``, compact multidimensional boundary
sections, repeated dense array overlays, an ``APHI`` starting-element write,
and a four-stage ``21,34,55,89`` ladder.  The mandatory gate parses the exact
combination and reaches a finite first force with automatic axis recovery and
an accepted radial solve.  The full gate runs the 21-surface fixed-boundary
problem to the VMEC2000 equilibrium and pins its residual channels, energy,
axis, iteration count, and reset count.

The same PR's public free-boundary matrix uses the bundled CTH-like mgrid
fixtures to cover one-time vacuum activation, increasing/equal radial grids,
NESTOR structure rebuild/reuse, carried ``ivac/nvacskip/rbsq`` and invariant
residuals, first-fine-grid edge-force norms, evolved-edge hot restart,
``LFORBAL``, ``PRECON_TYPE='NONE'``, active VMEX GMRES, JAC75 checkpoint
recovery, and WOUT comparison.  Its ``7 -> 15`` converged trajectory pins the
same first fine-grid ``FSQR/FSQZ/FSQL`` screen row as local VMEC2000 before
comparing the final WOUT.  The ESSOS/SIMSOPT adapters and free-boundary
AD-versus-FD tests remain separate so an optional external package cannot
weaken the mandatory mgrid/NESTOR gates.

Research-grade completion criteria
----------------------------------

A remaining row is complete only when all applicable evidence exists:

1. a production implementation, not a diagnostic-only branch;
2. unit tests that prove the control reaches the intended kernel/controller;
3. VMEC2000 comparison of trajectory and converged state where parity is the
   goal;
4. fixed/free and symmetric/asymmetric coverage where the method claims those
   modes;
5. multigrid and hot-restart coverage where state is transferred;
6. WOUT schema/value comparison where outputs are claimed;
7. AD-versus-FD checks for every differentiability claim, with the exact
   differentiated map identified; and
8. benchmarks that report hardware, precision, cold/warm compilation, solver
   tolerance, iteration count, and failure policy.

The highest-priority unimplemented parity work exposed by this audit is:
VMEC2000 continuation controls (``PRE_NITER``, ``MAX_MAIN_ITERATIONS``,
``LGIVEUP``), target-volume rescaling, the non-GMRES 2-D preconditioner modes,
TRIP3D/reconstruction/RFP/ANIMEC physics where required by research programs, a
true NESTOR equilibrium derivative, and complete asymmetric NESTOR WOUT
structures.
