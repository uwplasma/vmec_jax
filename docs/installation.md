# Installation

`pip install vmex` installs everything needed for solving, plotting, and the
Boozer transform — no user-facing extras to remember. Verify with
`vmex --doctor` and `vmex --test`.

## Requirements

- Python 3.10+ (Python 3.12+ recommended for current accelerator-enabled JAX)
- `numpy`, `jax` + `jaxlib`, `netCDF4`, `matplotlib`, `booz_xform_jax`
  (all installed automatically)

## From PyPI

```console
pip install vmex
vmex --doctor
vmex --test
```

`vmex --doctor` diagnoses mixed-Python environments: it prints the active
interpreter, pip location, package versions, JAX backend and devices, the
active JAX default device, and VMEX's forward/implicit placement policies. If
an install misbehaves, first check that `pip --version` and
`python -m pip --version` point at the same Python.

`vmex --test` runs the bundled fixed-boundary QH case end to end: it copies
the packaged `input.nfp4_QH_warm_start` deck into `./vmex_test/`, solves it
(with `FTOL_ARRAY = 1e-12` for a fast first check), writes
`wout_nfp4_QH_warm_start.nc`, and renders diagnostic figures into
`vmex_test/figures/`. It also prints the equivalent manual commands so you
can reproduce each step yourself.

JAXopt and Optax are optional because SciPy and the public problem callables
are part of the core install. Install the external-optimizer examples with:

```console
pip install "vmex[optimizers]"
```

## From conda-forge

```console
conda install --channel conda-forge vmex
```

or, with [Pixi](https://pixi.prefix.dev/), `pixi add vmex`. The
[feedstock](https://github.com/conda-forge/vmex-feedstock) may lag PyPI.

## From source

```console
git clone https://github.com/uwplasma/vmex
cd vmex
pip install -e .          # editable install, recommended for development
```

## Float64 (required)

VMEC's numerics require double precision. VMEX enables JAX x64 mode itself
when you use the CLI or the core solver entry points; if you drive JAX
directly in your own scripts, set:

```console
export JAX_ENABLE_X64=1
```

or `jax.config.update("jax_enable_x64", True)` before solving.

## GPU support

GPU-enabled JAX is intentionally not forced by VMEX because the right wheel
depends on your platform and CUDA/ROCm version. Install the CPU package
first, then install JAX for your accelerator following the
[official JAX installation matrix](https://docs.jax.dev/en/latest/installation.html),
e.g.:

```console
pip install -U "jax[cuda13]"
```

CUDA 13 wheels currently require an NVIDIA driver version of at least 580
and a Python version supported by the current JAX release. On older Python
versions, package resolution can select an older JAX release whose
accelerator extras differ; always confirm the result with `vmex --doctor`.
CUDA 12, ROCm, TPU, and platform-specific alternatives remain documented in
JAX's installation matrix.

VMEX then picks CPU or GPU per forward solve using a measured device policy —
when the GPU actually pays off, and how to pin a device explicitly, is
{doc}`howto/run-on-gpu`.

## Build the documentation locally

```console
pip install ".[docs]"
python -m sphinx -W -j auto -b html docs docs/_build/html
```
