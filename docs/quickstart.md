# Quickstart

## Run the validated example chain

All examples can be run directly from the repo root without installing:

```bash
python examples/00_parse_and_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out boundary_step0.npz --verbose
python examples/02_init_guess_and_coords.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out coords_step1.npz --verbose
python examples/04_geom_metrics.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out geom_step2.npz --verbose
python examples/05_profiles_and_volume.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --out profiles_step3.npz --verbose
python examples/06_field_and_energy.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
python examples/07_solve_lambda.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --wout examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc --verbose
python examples/08_solve_fixed_boundary.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose
python examples/09_solve_fixed_boundary_lbfgs.py examples/input.LandremanSenguptaPlunk_section5p3_low_res --verbose
```

## Run the test suite

```bash
pytest -q
```

## A minimal API sketch

The primary “dataflow” objects are:

- `InData`: parsed `&INDATA` namelist
- `VMECConfig`: discretization and run parameters
- `VMECStatic`: precomputed grids/basis tables
- `VMECState`: Fourier coefficients for `(R,Z,lambda)`

Typical usage:

```python
from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.geom import eval_geom

cfg, indata = load_config("input.vmec")
static = build_static(cfg)
bdy = boundary_from_indata(indata, static.modes)
state0 = initial_guess_from_boundary(static, bdy, indata)
geom = eval_geom(state0, static)  # sqrtg, metric, derivatives
```

