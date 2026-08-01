Capability contract
===================

This table is the public support contract, generated from
``benchmarks/capabilities.json``. ``validated`` means that committed
evidence exercises the path; ``limited`` means that only the scope
stated in the row is validated; ``—`` means no public path.

.. list-table::
   :header-rows: 1
   :widths: 12 15 8 9 8 7 7 7 7 7 7 12 24

   * - topology
     - configuration
     - boundary
     - symmetry
     - pressure
     - CPU
     - GPU
     - forward
     - JVP
     - VJP
     - optimize
     - status
     - scope and evidence
   * - toroidal
     - stellarator / tokamak
     - fixed
     - symmetric
     - scalar
     - validated
     - validated
     - validated
     - validated
     - validated
     - validated
     - supported
     - Converged implicit derivatives. Evidence: `test_solver_end_to_end.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_solver_end_to_end.py>`__, `test_implicit_grad.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_implicit_grad.py>`__, `test_gpu_ci.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_gpu_ci.py>`__.
   * - toroidal
     - stellarator / tokamak
     - fixed
     - LASYM
     - scalar
     - validated
     - validated
     - validated
     - validated
     - validated
     - validated
     - supported
     - Converged implicit derivatives; some diagnostics retain independent LASYM guards. Evidence: `test_parity_breadth.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_parity_breadth.py>`__, `test_implicit_grad.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_implicit_grad.py>`__, `test_gpu_ci.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_gpu_ci.py>`__.
   * - toroidal
     - stellarator / tokamak
     - free
     - symmetric
     - scalar
     - validated
     - validated
     - validated
     - limited
     - limited
     - limited
     - supported
     - Public B2-B4 APIs differentiate a supplied converged symmetric VMEX-NESTOR root for scalar external-field parameters. Public B5 adds a CPU-validated, projected-root-and-chart-gated common-anchor scalar host solve with a custom VJP; B5 GPU qualification, vector design, and a common optimizer are not yet public. Evidence: `test_freeboundary.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_freeboundary.py>`__, `test_freeboundary_implicit_fast.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_freeboundary_implicit_fast.py>`__, `test_freeboundary_implicit_integration.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_freeboundary_implicit_integration.py>`__, `test_gpu_ci.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_gpu_ci.py>`__.
   * - toroidal
     - stellarator / tokamak
     - free
     - LASYM
     - scalar
     - validated
     - validated
     - validated
     - —
     - —
     - —
     - supported
     - Forward solve and NESTOR WOUT fields only. Evidence: `test_lasym_free_convergence.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_lasym_free_convergence.py>`__, `test_gpu_ci.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_gpu_ci.py>`__.
   * - open mirror
     - axisymmetric
     - fixed
     - axisymmetric
     - scalar
     - validated
     - validated
     - validated
     - validated
     - validated
     - limited
     - supported
     - Implicit boundary derivatives are validated; no common objective driver yet. Evidence: `mirror_fixed_boundary.json <https://github.com/uwplasma/VMEX/blob/main/benchmarks/mirror_fixed_boundary.json>`__, `test_implicit.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_implicit.py>`__, `test_gpu_ci.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_gpu_ci.py>`__.
   * - open mirror
     - rotating ellipse
     - fixed
     - nonaxisymmetric
     - scalar
     - validated
     - limited
     - validated
     - validated
     - validated
     - limited
     - release-candidate
     - Corrected-cut rotating-ellipse lane; broader straight-field-line validation is incomplete. Evidence: `mirror_fixed_boundary.json <https://github.com/uwplasma/VMEX/blob/main/benchmarks/mirror_fixed_boundary.json>`__, `test_implicit.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_implicit.py>`__, `test_splines.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_splines.py>`__.
   * - open mirror
     - axisymmetric
     - free
     - axisymmetric
     - scalar
     - validated
     - validated
     - validated
     - limited
     - limited
     - limited
     - supported
     - Supported from β=0 through β=10%; field adjoint validated against reconverged finite differences. Evidence: `mirror_free_boundary_axisymmetric.json <https://github.com/uwplasma/VMEX/blob/main/benchmarks/mirror_free_boundary_axisymmetric.json>`__, `test_free_boundary.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_free_boundary.py>`__, `test_implicit.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_implicit.py>`__.
   * - open mirror
     - axisymmetric
     - free
     - axisymmetric
     - scalar
     - validated
     - validated
     - limited
     - —
     - —
     - —
     - extended-validation
     - 10% < β ≤ 50%; converges variationally but the independent-force promotion gate fails. Evidence: `mirror_free_boundary_axisymmetric.json <https://github.com/uwplasma/VMEX/blob/main/benchmarks/mirror_free_boundary_axisymmetric.json>`__, `test_free_boundary.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_free_boundary.py>`__.
   * - open mirror
     - nonaxisymmetric
     - free
     - nonaxisymmetric
     - scalar
     - limited
     - limited
     - limited
     - —
     - —
     - —
     - deferred
     - Local observables do not yet converge under refinement. Evidence: `mirror_free_boundary_nonaxisymmetric.json <https://github.com/uwplasma/VMEX/blob/main/benchmarks/mirror_free_boundary_nonaxisymmetric.json>`__.
   * - closed hybrid
     - circular section
     - fixed
     - nonaxisymmetric
     - scalar
     - validated
     - validated
     - validated
     - validated
     - validated
     - limited
     - supported
     - Closed periodic spline axis and boundary derivatives; no common objective driver yet. Evidence: `mirror_hybrid_fixed_boundary.json <https://github.com/uwplasma/VMEX/blob/main/benchmarks/mirror_hybrid_fixed_boundary.json>`__, `test_implicit.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_implicit.py>`__, `test_splines.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_splines.py>`__.
   * - closed hybrid
     - rotating ellipse
     - fixed
     - nonaxisymmetric
     - scalar
     - validated
     - validated
     - limited
     - limited
     - limited
     - —
     - extended-validation
     - The independent strong-force promotion gate remains open. Evidence: `mirror_hybrid_fixed_boundary.json <https://github.com/uwplasma/VMEX/blob/main/benchmarks/mirror_hybrid_fixed_boundary.json>`__, `test_splines.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_splines.py>`__.
   * - all
     - all
     - fixed / free
     - all
     - anisotropic
     - —
     - —
     - —
     - —
     - —
     - —
     - not-implemented
     - ANIMEC-derived and open-mirror anisotropic equilibria are roadmap work. Evidence: `mirror_geometry.rst <https://github.com/uwplasma/VMEX/blob/main/docs/mirror_geometry.rst>`__.

Free-boundary differentiation
-----------------------------

A supported forward free-boundary solve does not imply differentiation
through its adaptive reconvergence path. For stellarator-symmetric
toroidal equilibria, the public lower-level B2--B4 APIs provide
projected tangents and state adjoints at a supplied converged
VMEX--NESTOR root. Public B5 adds a deterministic common-anchor scalar
host solve with a custom VJP and gates every accepted endpoint against
that projected root and the anchor's affine constraint slice. Adaptive
host decisions stay outside AD; vector
design and a common optimizer remain outside the public path. B5 is
currently CPU-qualified; its GPU wrapper gate remains open. The
specified-boundary virtual-casing residual is a separate differentiable
interface.

Mirror beta labels
------------------

The axisymmetric open-mirror free-boundary lane is supported through
10% requested beta. The 25% and 50% cases remain extended validation:
they converge variationally, but do not pass the independent strong-force
promotion gate recorded in the benchmark artifact.
