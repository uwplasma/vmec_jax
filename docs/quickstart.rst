Quickstart
==========

Run the validated example chain
-------------------------------

All examples can be run directly from the repo root without installing.
The canonical scripts live under the categorized folders (``1_Simple/``,
``2_Intermediate/``, ``3_Advanced/``); thin compatibility wrappers also exist at
the top level of ``examples/``::

  python examples/1_Simple/00_parse_and_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out boundary_step0.npz --verbose
  python examples/1_Simple/02_init_guess_and_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out coords_step1.npz --verbose
  python examples/2_Intermediate/04_geom_metrics.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out geom_step2.npz --verbose
  python examples/2_Intermediate/05_profiles_and_volume.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out profiles_step3.npz --verbose
  python examples/2_Intermediate/06_field_and_energy.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
  python examples/3_Advanced/07_solve_lambda.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
  python examples/3_Advanced/08_solve_fixed_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose
  python examples/3_Advanced/09_solve_fixed_boundary_lbfgs.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose

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

Compare contravariant B components (``bsupu``, ``bsupv``) reconstructed by vmec_jax against the ``wout`` contravariant fields (writes figures; requires ``netCDF4`` + ``matplotlib``)::

  python examples/2_Intermediate/05_bsup_parity_figures.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --outdir figures_bsup_parity

Compare covariant B components (``bsubu``, ``bsubv``) reconstructed from the metric and ``wout`` contravariant fields against the ``wout`` covariant fields (writes figures; requires ``netCDF4`` + ``matplotlib``)::

  python examples/2_Intermediate/03_bsub_parity_figures.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --outdir figures_bsub_parity

Compare ``|B|`` parity (bmnc/bmns) against reconstructed ``|B|`` (writes figures; requires ``netCDF4`` + ``matplotlib``)::

  python examples/2_Intermediate/04_bmag_parity_figures.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --outdir figures_bmag_parity

Roundtrip a reference ``wout_*.nc`` file through vmec_jax's minimal writer (requires ``netCDF4``)::

  python examples/3_Advanced/06_wout_roundtrip.py --wout examples/wout_circular_tokamak_reference.nc --out wout_roundtrip.nc

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
