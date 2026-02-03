Installation
============

Requirements
------------

- Python 3.10+
- NumPy (core requirement)
- JAX + jaxlib (for performance + autodiff)

Recommended:

- ``netCDF4`` to read VMEC2000 ``wout_*.nc`` files for validation

Install from source
-------------------

From the repo root (non-editable install)::

  python -m pip install -U pip
  python -m pip install .

Enable JAX::

  python -m pip install ".[jax]"

VMEC relies heavily on float64. JAX defaults to float32 unless x64 is enabled.
We recommend setting::

  export JAX_ENABLE_X64=1

Enable netCDF support::

  python -m pip install ".[netcdf]"

Editable install (recommended for development)::

  python -m pip install -e .

Build docs locally
------------------

Install doc dependencies::

  python -m pip install ".[docs]"

Then build docs::

  python -m sphinx -b html docs docs/_build/html
