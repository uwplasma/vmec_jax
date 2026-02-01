# Installation

## Requirements

- Python 3.10+
- NumPy (core requirement)
- JAX + jaxlib (for performance + autodiff)

Recommended:

- `netCDF4` to read VMEC2000 `wout_*.nc` files for validation

## Install from source (editable)

From the repo root:

```bash
pip install -e .
```

### Enable JAX

```bash
pip install -e .[jax]
```

VMEC relies heavily on float64. JAX defaults to float32 unless x64 is enabled.
We recommend setting:

```bash
export JAX_ENABLE_X64=1
```

`vmec-jax` also attempts to enable x64 by default when importing JAX, but explicit is better.

### Enable netCDF support

```bash
pip install -e .[netcdf]
```

## Install doc dependencies

```bash
pip install -e .[docs]
```

Then build docs:

```bash
python -m sphinx -b html docs docs/_build/html
```

