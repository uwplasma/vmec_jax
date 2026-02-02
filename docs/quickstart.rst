Quickstart
==========

Run the validated example chain
-------------------------------

All examples can be run directly from the repo root without installing::

  python examples/00_parse_and_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out boundary_step0.npz --verbose
  python examples/02_init_guess_and_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out coords_step1.npz --verbose
  python examples/04_geom_metrics.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out geom_step2.npz --verbose
  python examples/05_profiles_and_volume.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out profiles_step3.npz --verbose
  python examples/06_field_and_energy.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
  python examples/07_solve_lambda.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
  python examples/08_solve_fixed_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose
  python examples/09_solve_fixed_boundary_lbfgs.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose

Run the test suite::

  pytest -q

ParaView export (VTK)
---------------------

Export a surface ``B`` field (including ``Bx``, ``By``, ``Bz``, and ``|B|``) and a
field-line trace for ParaView (requires ``netCDF4``)::

  python examples/3_Advanced/02_vtk_field_and_fieldlines.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --hi-res --outdir vtk_out

Implicit differentiation (Step-9)
---------------------------------

Differentiate through the lambda-only equilibrium sub-solve (no backprop through iterations) and write a publication-ready figure::

  python examples/2_Intermediate/02_implicit_lambda_gradients.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --outdir figures_implicit_lambda

VMEC2000 parity diagnostics (Step-10)
-------------------------------------

Compare covariant B components (``bsubu``, ``bsubv``) reconstructed from the metric and ``wout`` contravariant fields against the ``wout`` covariant fields (writes figures; requires ``netCDF4`` + ``matplotlib``)::

  python examples/2_Intermediate/03_bsub_parity_figures.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --outdir figures_bsub_parity

Force-like residual report (Step-10 target; advanced)::

  python examples/3_Advanced/05_force_residual_report.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --hi-res

Advanced: implicit differentiation through fixed-boundary equilibrium::

  python examples/3_Advanced/03_implicit_fixed_boundary_sensitivity.py examples/input.circular_tokamak --outdir figures_implicit_fixed_boundary

A minimal API sketch
--------------------

The primary “dataflow” objects are:

- ``InData``: parsed ``&INDATA`` namelist
- ``VMECConfig``: discretization and run parameters
- ``VMECStatic``: precomputed grids/basis tables
- ``VMECState``: Fourier coefficients for (R, Z, lambda)

Typical usage::

  from vmec_jax.config import load_config
  from vmec_jax.static import build_static
  from vmec_jax.init_guess import init_state_from_boundary
  from vmec_jax.geom import eval_geom

  cfg, indata = load_config(\"examples/input.LandremanSenguptaPlunk_section5p3_low_res\")
  static = build_static(cfg)
  state0 = init_state_from_boundary(indata, static)
  geom = eval_geom(state0, static)
