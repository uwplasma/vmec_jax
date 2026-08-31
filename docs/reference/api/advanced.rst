Advanced API
============

Everything below :doc:`basic`: the solver internals of :mod:`vmex.core` and
the open-field-line lane :mod:`vmex.mirror`, grouped as in
:doc:`/explanation/architecture`. Every docstring names the VMEC2000
counterpart it ports.

Profiles
--------

.. automodule:: vmex.core.profiles
   :members:

Radial basis and axis regularity
--------------------------------

.. automodule:: vmex.core.radial_basis
   :members:

High-order reconstruction and force certificate
------------------------------------------------

.. automodule:: vmex.core.strong_force
   :members:

High-order correction transfer and preconditioner
-------------------------------------------------

.. automodule:: vmex.core.polish
   :members:

.. automodule:: vmex.core.polish_driver
   :members:

.. automodule:: vmex.core.polish_implicit
   :members:

Both :func:`vmex.solve` and :func:`vmex.solve_multigrid` accept
``polish_force_balance=False`` (unchanged behavior), ``True`` (required
correction), or ``"auto"`` (skip an already-certified state). The shorter
``polish`` keyword remains an alias on the single-grid call. A standard VMEC
deck enables the same path with comment directives that VMEC2000 ignores::

   !@VMEX POLISH = AUTO
   !@VMEX POLISH_TOL = 1.0E-8
   !@VMEX POLISH_FAIL = ERROR
   !@VMEX POLISH_DEGREE = 5

(the original single-flag spelling ``! VMEX: POLISH_FORCE_BALANCE = .TRUE.``
still parses).  Directives are execution metadata, owned by
:mod:`vmex.core.run_options` — they never become :class:`vmex.VmecInput`
fields, and ``VmecInput.from_file`` ignores them while preserving all physics.
Structured JSON carries the same keys in a reserved ``_vmex`` section that is
removed before schema validation.  :func:`vmex.solve_file` runs a deck the way
the CLI does — directives honored, ``wout_<case>.nc`` written — and explicit
Python keywords override the file::

   result = vmex.solve_file("input.case", polish="auto")

Precedence is exactly ``CLI option > Python keyword > file directive >
package default``; the CLI prints which layer a polish request came from.
``polish_fail`` selects the failure behavior: ``"error"`` raises,
``"fallback"`` returns the unpolished state, ``"warn"`` does the same with a
:class:`RuntimeWarning` — never a failure silently presented as polished. A successful result retains the ordinary ``state`` and exposes
the VMEC-grid projection as ``polished_state``; an
:class:`vmex.core.optimize.Equilibrium` requested with polishing uses that
projected state, while the continuous solution remains in
``native_equilibrium``.  CLI WOUT files instead sample
``native_equilibrium`` on a denser radial mesh
(:func:`vmex.core.polish_driver.polished_wout_ns`): on the solve mesh the
stable wout reconstruction cannot resolve the between-node correction, so a
solve-resolution export would silently discard most of the certified gain.

The public primal path solves the overdetermined physical collocation residual
with SOLVAX Gauss--Newton and accepts it only after independent force,
radial-refinement, and nestedness checks.  A successful result carries a
``polish_context`` for :func:`vmex.collocation_polish_tangent`,
:func:`vmex.collocation_polish_adjoint`, and
:func:`vmex.implicit_collocation_polished_state`.  These differentiate the
exact least-squares stationarity equation, including its nonzero-residual
Hessian term, without replaying nonlinear iterations.  The earlier square-root
diagnostics remain internal to :mod:`vmex.core.polish_implicit`.

A scalar objective differentiates through that converged state directly.  This
example minimizes relative field-strength variation on one interior surface::

   import jax
   import jax.numpy as jnp
   import vmex as vj

   native = result.polish_context.runtime.native

   def objective(value):
       polished = vj.implicit_collocation_polished_state(
           value, result.polish_context)
       theta = jnp.linspace(0, 2 * jnp.pi, 12, endpoint=False)
       zeta = jnp.linspace(0, 2 * jnp.pi, 6, endpoint=False)
       tt, zz = jnp.meshgrid(theta, zeta, indexing="ij")
       B = vj.evaluate_high_order_fields(polished, 0.7, tt, zz).B
       magnitude = jnp.linalg.norm(B, axis=-1)
       return jnp.var(magnitude / jnp.mean(magnitude))

   gradient = jax.grad(objective)(native)

For boundary objectives, :func:`vmex.evaluate_high_order_surface` returns a
one-field-period array view accepted by ESSOS, and
:func:`vmex.surface_field_data_from_high_order` converts the same analytic
geometry and edge field for ``virtual_casing_jax``.  Neither path writes a
``wout`` file or finite-differences a surface tangent.  Field-aligned
objectives use :func:`vmex.boozer_bmnc_high_order`, which sends continuous
geometry and field tables to BOOZ_XFORM_JAX without reconstructing a sampled
radial mesh.

Spectral representation and physics kernels
-------------------------------------------

.. automodule:: vmex.core.fourier
   :members:

.. automodule:: vmex.core.transforms
   :members:

.. automodule:: vmex.core.geometry
   :members:

.. automodule:: vmex.core.fields
   :members:

.. automodule:: vmex.core.forces
   :members:

.. automodule:: vmex.core.residuals
   :members:

Solver
------

.. automodule:: vmex.core.setup
   :members:

.. automodule:: vmex.core.preconditioner
   :members:

.. automodule:: vmex.core.preconditioner_2d
   :members:

.. automodule:: vmex.core.step
   :members:

.. automodule:: vmex.core.solver
   :members:

.. automodule:: vmex.core.multigrid
   :members:

.. automodule:: vmex.core.restart
   :members:

.. automodule:: vmex.core.device
   :members:

Free boundary
-------------

.. automodule:: vmex.core.vacuum
   :members:

.. automodule:: vmex.core.freeboundary
   :members:

.. automodule:: vmex.core.freeboundary_implicit
   :members:

.. automodule:: vmex.core.freeboundary_linear
   :members:

.. automodule:: vmex.core.virtual_casing
   :members:

``vmex.core.freeboundary_diff`` remains as a compatibility name for this
prescribed-interface API. It does not differentiate through a moving-boundary
NESTOR equilibrium solve.

.. automodule:: vmex.core.mgrid
   :members:

.. automodule:: vmex.core.extender
   :members:

Physics objectives
------------------

The objective catalog with usage snippets is :doc:`/reference/objectives`.

.. automodule:: vmex.core.omnigenity
   :members:

.. automodule:: vmex.core.bounce
   :members:

.. automodule:: vmex.core.qi
   :members:

.. automodule:: vmex.core.maxj
   :members:

.. automodule:: vmex.core.gammac
   :members:

.. automodule:: vmex.core.bootstrap
   :members:

.. automodule:: vmex.core.stability
   :members:

.. automodule:: vmex.core.turbulence
   :members:

Outputs
-------

.. automodule:: vmex.core.scaling
   :members:

.. automodule:: vmex.core.nyquist
   :members:

.. automodule:: vmex.core.postprocess
   :members:

.. automodule:: vmex.core.printing
   :members:

.. automodule:: vmex.core.plotting
   :members:

.. automodule:: vmex.core.boozer
   :members:

Straight-axis mirrors
---------------------

.. automodule:: vmex.mirror.analytic
   :members:

.. automodule:: vmex.mirror.splines
   :members:

.. automodule:: vmex.mirror.model
   :members:

.. automodule:: vmex.mirror.solver
   :members:

.. automodule:: vmex.mirror.free_boundary
   :members:

.. automodule:: vmex.mirror.implicit
   :members:

.. automodule:: vmex.mirror.output
   :members:

Errors and CLI
--------------

.. automodule:: vmex.core.errors
   :members:

.. automodule:: vmex.core.cli
   :members:
