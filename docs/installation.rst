Installation
============

Requirements
------------

- Python 3.10+
- NumPy (core requirement)
- JAX + jaxlib (for performance + autodiff)

Recommended:

- ``netCDF4`` to read VMEC2000 ``wout_*.nc`` files for validation

Install from source (editable)
------------------------------

From the repo root::

  pip install -e .

Enable JAX::

  pip install -e .[jax]

VMEC relies heavily on float64. JAX defaults to float32 unless x64 is enabled.
We recommend setting::

  export JAX_ENABLE_X64=1

Enable netCDF support::

  pip install -e .[netcdf]

Build docs locally
------------------

Install doc dependencies::

  pip install -e .[docs]

Then build docs::

  python -m sphinx -b html docs docs/_build/html

