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
     - Forward NESTOR equilibrium. AD covers the virtual-casing residual on a specified boundary, not a reconverged plasma-vacuum root. Evidence: `test_freeboundary.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_freeboundary.py>`__, `test_freeboundary_diff.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_freeboundary_diff.py>`__, `test_gpu_ci.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_gpu_ci.py>`__.
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
     - ANIMEC-derived and open-mirror anisotropic equilibria are roadmap work. Evidence: `mirror-geometry.rst <https://github.com/uwplasma/VMEX/blob/main/docs/explanation/mirror-geometry.rst>`__.

Free-boundary differentiation
-----------------------------

A supported forward free-boundary solve does not imply differentiation
through the reconverged plasma-vacuum equilibrium. The toroidal
specified-boundary virtual-casing residual is differentiable, but the
fully coupled NESTOR root is not yet a public AD path.

Mirror beta labels
------------------

The axisymmetric open-mirror free-boundary lane is supported through
10% requested beta. The 25% and 50% cases remain extended validation:
they converge variationally, but do not pass the independent strong-force
promotion gate recorded in the benchmark artifact.
