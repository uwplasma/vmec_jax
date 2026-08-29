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
active JAX default device, a real float64 JIT device probe, and VMEX's
forward/implicit placement policies. Under WSL2 it also reports the
Windows-provided NVIDIA driver seen by `nvidia-smi`. If an install misbehaves,
first check that `pip --version` and
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

### Windows with WSL2 and NVIDIA GPUs

Use the NVIDIA driver installed on Windows. Do not install a Linux NVIDIA
driver inside WSL2; NVIDIA exposes the Windows driver there through a stub
`libcuda.so`. If `nvidia-smi` is not on `PATH`, VMEX also checks its standard
WSL location, `/usr/lib/wsl/lib/nvidia-smi`.

JAX/jaxlib 0.9.2 has two upstream logging defects that are especially visible
in this environment:

```text
Assume version compatibility. PjRt-IFRT does not track XLA executable versions.
Could not get kernel mode driver version: Version does not match the format X.Y.Z
```

The first is a spurious message on persistent-compilation-cache hits
([JAX issue 36294](https://github.com/jax-ml/jax/issues/36294), fixed by
[OpenXLA PR 40018](https://github.com/openxla/xla/pull/40018)). The second
rejects a valid two-component Windows driver version such as `566.36`; it does
not by itself mean CUDA failed ([OpenXLA PR
41380](https://github.com/openxla/xla/pull/41380)). Both upstream fixes are
present in JAX/jaxlib 0.10.1 and newer. Upgrade the matching accelerator
installation, then rerun the doctor:

```console
python -m pip install --upgrade "jax[cuda13]>=0.10.1"
vmex --doctor
```

Use the CUDA extra selected by the current official JAX installation matrix
if CUDA 13 is not appropriate. A healthy report must show the `gpu` backend,
at least one `cuda:` device, and a passed JIT device probe. VMEX retains
error-level XLA logging: setting `TF_CPP_MIN_LOG_LEVEL=3` would hide genuine
CUDA failures and is not recommended. Disabling VMEX's persistent cache also
removes the PJRT cache-hit message on affected JAX releases, but makes cold
processes slower and is not the fix.

VMEX then picks CPU or GPU per forward solve using a measured device policy —
when the GPU actually pays off, and how to pin a device explicitly, is
{doc}`howto/run-on-gpu`.

## Build the documentation locally

```console
pip install ".[docs]"
python -m sphinx -W -j auto -b html docs docs/_build/html
```
