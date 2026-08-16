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
     - —
     - limited
     - limited
     - supported
     - Forward NESTOR and exterior fields are supported. An experimental CPU reverse-mode path differentiates the reconverged VMEC--NESTOR root with direct ESSOS coil parameters; cold compile time, GPU memory, and failed-trial recovery are not yet production-ready. Evidence: `test_freeboundary.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_freeboundary.py>`__, `test_freeboundary_implicit.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_freeboundary_implicit.py>`__, `test_virtual_casing_physics.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_virtual_casing_physics.py>`__, `test_examples.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_examples.py>`__.
   * - toroidal
     - stellarator / tokamak
     - free
     - LASYM
     - scalar
     - validated
     - validated
     - validated
     - —
     - limited
     - limited
     - supported
     - Forward solve and NESTOR WOUT fields are supported; the experimental CPU current/field VJP is reconverged-FD certified, with the same performance limitations as the symmetric path. Evidence: `test_lasym_free_convergence.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_lasym_free_convergence.py>`__, `test_freeboundary_implicit.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_freeboundary_implicit.py>`__, `test_gpu_ci.py <https://github.com/uwplasma/VMEX/blob/main/tests/test_gpu_ci.py>`__.
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
     - 10% < β ≤ 80%; converges variationally and the example force gate passes, but refined-grid promotion above 50% is incomplete. Evidence: `mirror_free_boundary_axisymmetric.json <https://github.com/uwplasma/VMEX/blob/main/benchmarks/mirror_free_boundary_axisymmetric.json>`__, `test_free_boundary.py <https://github.com/uwplasma/VMEX/blob/main/tests/mirror/test_free_boundary.py>`__.
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

A supported forward free-boundary solve does not imply that every derivative
mode is production-ready. VMEX exposes an experimental reverse derivative
of the reconverged plasma-vacuum root on CPU, certified against independent
free-boundary re-solves. Forward JVPs, low-memory GPU compilation, and robust
failed-trial walls remain open promotion gates. The prescribed-boundary
virtual-casing derivative is the mature path for fixed-LCFS coil objectives.

Mirror beta labels
------------------

The axisymmetric open-mirror free-boundary lane is supported through
10% requested beta. The 25%, 50%, and 80% cases remain extended validation:
the 80% example passes its force gate, but refined-grid promotion above
the current 50% campaign is incomplete.
