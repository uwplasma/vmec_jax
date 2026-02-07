Quickstart
==========

Run the minimal showcase (recommended)
--------------------------------------------

The simplest way to get started is the axisymmetric showcase. It runs a small
suite of bundled tokamak-like inputs, writes a ``wout_*.nc`` for each, produces
VMEC-style plots, and prints a small parity summary against the bundled VMEC2000
reference ``wout`` files::

  python examples/showcase_axisym_input_to_wout.py

This is designed to feel like running VMEC2000: inputs in, ``wout`` out, plus
standard plots.

Run the test suite::

  pytest -q

ParaView export (VTK)
---------------------

Export a surface ``B`` field (including ``Bx``, ``By``, ``Bz``, and ``|B|``) and a
field-line trace for ParaView (requires ``netCDF4``)::

  python examples/visualization/vtk_field_and_fieldlines.py examples/data/input.li383_low_res --hi-res --outdir vtk_out

Stepwise kernel demo (optional)
-------------------------------

If you want to see the lowest-level pieces in isolation, start with the boundary
evaluation script::

  python examples/tutorial/00_parse_and_boundary.py examples/data/input.circular_tokamak --out boundary_step0.npz --verbose

VMEC2000 parity diagnostics (Step-10)
-------------------------------------

Compare contravariant B components (``bsupu``, ``bsupv``) reconstructed by vmec_jax against the ``wout`` contravariant fields (writes figures; requires ``netCDF4`` + ``matplotlib``)::

  python examples/validation/bsup_parity_figures.py examples/data/input.li383_low_res --wout examples/data/wout_li383_low_res_reference.nc --outdir figures_bsup_parity

Compare covariant B components (``bsubu``, ``bsubv``) reconstructed from the metric and ``wout`` contravariant fields against the ``wout`` covariant fields (writes figures; requires ``netCDF4`` + ``matplotlib``)::

  python examples/validation/bsub_parity_figures.py examples/data/input.li383_low_res --wout examples/data/wout_li383_low_res_reference.nc --outdir figures_bsub_parity

Compare ``|B|`` parity (bmnc/bmns) against reconstructed ``|B|`` (writes figures; requires ``netCDF4`` + ``matplotlib``)::

  python examples/validation/bmag_parity_figures.py examples/data/input.li383_low_res --wout examples/data/wout_li383_low_res_reference.nc --outdir figures_bmag_parity

Roundtrip a reference ``wout_*.nc`` file through vmec_jax's minimal writer (requires ``netCDF4``)::

  python examples/validation/wout_roundtrip.py --wout examples/data/wout_circular_tokamak_reference.nc --out wout_roundtrip.nc

Developer-only diagnostics (advanced)
--------------------------------------------

Additional parity breakdown scripts and research diagnostics live under
``tools/diagnostics/``. These are not part of the minimal examples set and may
change frequently.

A minimal API sketch (recommended)
----------------------------------

Most users should start from the small public API in ``vmec_jax.api``::

  import vmec_jax.api as vj

  run = vj.run_fixed_boundary(\"examples/data/input.shaped_tokamak_pressure\", solver=\"vmecpp_iter\", max_iter=30, verbose=True)
  wout = vj.write_wout_from_fixed_boundary_run(\"wout_shaped_tokamak_pressure_vmec_jax.nc\", run, include_fsq=True)

  # Compare to a VMEC2000 reference wout (bundled in examples/data/)
  wref = vj.read_wout(\"examples/data/wout_shaped_tokamak_pressure_reference.nc\")
  print(\"fsq_total(ref)=\", float(wref.fsqr + wref.fsqz + wref.fsql))
  print(\"fsq_total(new)=\", float(wout.fsqr + wout.fsqz + wout.fsql))

High-level driver helpers
-------------------------

Advanced users can still drop down to the lower-level building blocks
(``load_config``, ``build_static``, ``eval_geom``, etc.) as needed.
