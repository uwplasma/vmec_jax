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

