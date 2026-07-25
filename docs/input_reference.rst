Input file reference
====================

``vmex`` accepts two input formats, auto-detected by
:meth:`vmex.core.input.VmecInput.from_file`:

- the classic VMEC2000 ``&INDATA`` Fortran namelist (``input.<case>``), and
- VMEC++-style JSON (``.json`` suffix or leading ``{``) with identical key
  names.

Both round-trip: ``VmecInput.to_json`` writes a VMEC++-schema JSON deck that
parses back to the same input. Implemented VMEC2000 defaults and ``readin.f``
normalizations are applied on construction (see the
:mod:`vmex.core.input` docstring for the exact rules). Parsing does not imply
solver support: the complete supported/rejected/no-op classification is in
:doc:`vmec2000_compatibility`. Unknown names and active unsupported physics
fail before setup.

Fortran namelist assignment semantics
--------------------------------------

INDATA assignments are replayed in source order, as a Fortran namelist reader
writes into the already initialized ``vmec_input.f`` arrays.  A later short
dense assignment replaces only the elements it supplies; it does not clear an
earlier tail.  Indexed writes and dense writes override one another according
to their textual order.  For example,

.. code-block:: fortran

   FTOL_ARRAY = 1e-6, 1e-11, 1e-30, 1e-30
   FTOL_ARRAY = 1e-7, 1e-7

leaves entries 3 and 4 at ``1e-30``.

An indexed designator followed by several values is a *starting-element*
assignment, so ``APHI(1)=1,0`` writes elements 1 and 2.  Array sections use
inclusive Fortran bounds, and the first subscript varies fastest.  Thus compact
boundary syntax such as ``RBC(-6:6,0)=...`` is equivalent to the corresponding
13 scalar assignments.  Omitted section limits (``RBC(:,0)`` or
``RBC(-6:,0)``) are resolved from the declared VMEC2000 array bounds.  Bounds,
rank, and value count are checked before setup.

INDATA variables
----------------

Symmetry and resolution
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Variable
     - Default
     - Meaning
   * - ``LASYM``
     - ``F``
     - non-stellarator-symmetric mode (enables the ``rbs/zbc`` and
       ``*mns/*mnc`` partners)
   * - ``NFP``
     - 1
     - number of field periods
   * - ``MPOL``
     - 6
     - poloidal modes ``m = 0 .. mpol-1``
   * - ``NTOR``
     - 0
     - toroidal modes ``n = -ntor .. ntor``
   * - ``NTHETA`` / ``NZETA``
     - 0
     - angular grid points (0 selects the VMEC default).  Explicit values
       below ``2*MPOL+6`` / ``2*NTOR+4`` remain legal for VMEC2000 parity but
       can alias nonlinear force products; the privacy-safe diagnostic reports
       ``W01_ANGULAR_GRID_BELOW_VMEC_DEFAULT`` without printing the values.

Multigrid ladder and stepping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Variable
     - Default
     - Meaning
   * - ``NS_ARRAY``
     - ``[31]``
     - radial surfaces per multigrid stage; explicit ``NS_ARRAY(1)=0`` uses
       the VMEC2000 ``NSIN -> 31`` legacy expansion.  The active ladder ends
       at the first nonpositive or decreasing entry; equal entries rerun.
   * - ``FTOL_ARRAY``
     - ``[1e-10]``
     - force tolerance per stage (converged when ``fsqr, fsqz, fsql`` are all
       below it).  If its first element is explicitly zero, ``readin.f``'s
       geometric ladder from ``1e-8`` to scalar ``FTOL`` is generated.
   * - ``NITER_ARRAY`` (or ``NITER``)
     - ``[100]``
     - iteration cap per stage.  Scalar ``NITER`` fills the complete array
       only when every initialized ``NITER_ARRAY`` entry remains ``-1``.
       A preserved ``-1`` in a partially assigned array still executes the
       first ``eqsolve`` pass before the iteration-limit check, as in VMEC2000.
   * - ``DELT``
     - 1.0
     - initial time step
   * - ``TCON0``
     - 1.0
     - constraint-force multiplier (spectral condensation)
   * - ``APHI``
     - ``[1, 0, ...]``
     - radial-flux remap polynomial
   * - ``PHIEDGE``
     - 1.0
     - total enclosed toroidal flux [Wb]
   * - ``NSTEP``
     - 10
     - iterations between progress prints
   * - ``TIME_SLICE``
     - 0
     - informational value preserved in the VMEC-style run header
   * - ``LFORBAL``
     - ``F``
     - use VMEC2000's non-variational average-force replacement for the
       ``m=1,n=0`` R/Z channels
   * - ``LMOVE_AXIS``
     - ``T``
     - permit automatic first-pass axis recovery for a large finite force
   * - ``LFULL3D1OUT``
     - ``F``
     - write an unconverged WOUT after NITER exhaustion (fixed or free
       boundary); fatal numerical/Jacobian failures never write one

Pressure profile
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Variable
     - Default
     - Meaning
   * - ``PMASS_TYPE``
     - ``power_series``
     - one of ``power_series``, ``two_power``, ``gauss_trunc``,
       ``cubic_spline``, ``akima_spline``, ``line_segment``, ``pedestal``,
       ... (see :mod:`vmex.core.profiles`)
   * - ``AM``
     - zeros
     - profile coefficients (dense, indices 0..20)
   * - ``AM_AUX_S`` / ``AM_AUX_F``
     - —
     - spline knots in ``s`` / values for tabulated profile types
   * - ``PRES_SCALE``
     - 1.0
     - pressure scale factor [Pa]
   * - ``GAMMA``
     - 0.0
     - adiabatic index (JSON alias: ``adiabatic_index``)
   * - ``SPRES_PED``
     - 1.0
     - pressure pedestal location in ``s``

Current and iota profiles
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Variable
     - Default
     - Meaning
   * - ``NCURR``
     - 0
     - 0: prescribed iota (``AI``), 1: prescribed toroidal current (``AC``)
   * - ``PCURR_TYPE``
     - ``power_series``
     - current profile type; ``*_i`` forms prescribe :math:`I(s)`, ``*_ip``
       forms prescribe :math:`I'(s)`
   * - ``AC`` / ``AC_AUX_S`` / ``AC_AUX_F``
     - zeros / —
     - current profile coefficients / spline knots and values
   * - ``CURTOR``
     - 0.0
     - total toroidal current [A]
   * - ``PIOTA_TYPE``
     - ``power_series``
     - iota profile type
   * - ``AI`` / ``AI_AUX_S`` / ``AI_AUX_F``
     - zeros / —
     - iota profile coefficients / spline knots and values
   * - ``BLOAT``
     - 1.0
     - profile-argument expansion factor

Axis and boundary shape
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Variable
     - Default
     - Meaning
   * - ``RAXIS_CC`` / ``ZAXIS_CS``
     - zeros
     - axis initial guess, cos/sin coefficients for ``n = 0 .. ntor``
   * - ``RAXIS_CS`` / ``ZAXIS_CC``
     - zeros
     - asymmetric axis partners (``LASYM = T``)
   * - ``RBC(n,m)`` / ``ZBS(n,m)``
     - zeros
     - boundary Fourier coefficients of :math:`R\cos / Z\sin(m\theta - n\,\mathrm{NFP}\,\zeta)`
   * - ``RBS(n,m)`` / ``ZBC(n,m)``
     - zeros
     - asymmetric boundary partners (``LASYM = T``)

Free boundary
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Variable
     - Default
     - Meaning
   * - ``LFREEB``
     - ``T``
     - free-boundary mode; forced ``F`` when ``MGRID_FILE = 'NONE'``
   * - ``MGRID_FILE``
     - ``'NONE'``
     - MAKEGRID vacuum-field file, or ``'DIRECT_COILS'`` to ask ESSOS to
       tabulate a temporary mgrid (with ``vmex --coils``)
   * - ``EXTCUR``
     - —
     - external coil-group currents [A]
   * - ``NVACSKIP``
     - 1
     - vacuum-solve cadence (``<= 0`` falls back to ``NFP``)
   * - ``MFILTER_FBDY`` / ``NFILTER_FBDY``
     - -1
     - boundary spectral filtering
   * - ``PRECON_TYPE`` / ``PREC2D_THRESHOLD``
     - ``NONE`` / 1e-30
     - ``NONE``/``DEFAULT`` use the 1-D VMEC path; ``GMRES`` selects VMEX's
       exact-JVP matrix-free Newton--GMRES path. VMEC2000 ``CG``, ``GMRESR``,
       and ``TFQMR`` are distinct and are rejected, not aliased.
       ``NONE`` does **not** disable ``scalfor`` or lambda scaling; it disables
       only the optional 2-D block preconditioner, as in VMEC2000.

VMEC++-style JSON
-----------------

The JSON schema follows the implemented ``vmecpp.VmecInput`` subset: the keys
are the
lower-case INDATA names (``lasym, nfp, mpol, ntor, ntheta, nzeta, ns_array,
ftol_array, niter_array, delt, tcon0, aphi, phiedge, nstep, pmass_type, am,
am_aux_s, am_aux_f, pres_scale, adiabatic_index, spres_ped, ncurr,
pcurr_type, ac, ac_aux_s, ac_aux_f, curtor, piota_type, ai, ai_aux_s,
ai_aux_f, bloat, raxis_c, zaxis_s, raxis_s, zaxis_c, rbc, zbs, rbs, zbc,
lfreeb, mgrid_file, extcur, nvacskip, ...``).

Boundary coefficients are **sparse** lists of
``{"n": int, "m": int, "value": float}`` objects; axis arrays are dense of
length ``ntor + 1``:

.. code-block:: json

   {
     "lasym": false,
     "nfp": 5,
     "mpol": 5,
     "ntor": 4,
     "ns_array": [31],
     "ftol_array": [1e-12],
     "niter_array": [2000],
     "phiedge": 0.5,
     "raxis_c": [1.0, 0.1, 0.0, 0.0, 0.0],
     "zaxis_s": [0.0, 0.1, 0.0, 0.0, 0.0],
     "rbc": [
       {"n": 0, "m": 0, "value": 1.0},
       {"n": 0, "m": 1, "value": 0.3}
     ],
     "zbs": [
       {"n": 0, "m": 1, "value": 0.3}
     ]
   }

Extensions beyond VMEC++ (ignored by VMEC++ but accepted here): the spline
and pedestal profile types listed above, with the same key names as INDATA.
``free_boundary_method="nestor"`` is accepted explicitly. ``only_coils`` and
``biest`` select different models and are rejected. Unknown JSON keys are
errors rather than silent no-ops.

Unsupported and output-only INDATA controls
-------------------------------------------

Active reconstruction, RFP, TRIP3D, ANIMEC, target-volume rescaling, and
unsupported continuation/preconditioner modes fail with a typed error.
Neutral legacy defaults remain accepted. Legacy output-only switches are
listed individually in :doc:`vmec2000_compatibility`; they do not alter the
equilibrium and VMEX does not imply that their auxiliary files are produced.
