# vmec-jax documentation

`vmec-jax` is an incremental, JAX-based rewrite of **VMEC2000**, targeting:

- **fixed-boundary equilibria first** (free-boundary + MPI later),
- **end-to-end differentiability** (JAX autodiff),
- **laptop-friendly performance** (careful JIT boundaries, minimal allocations),
- **stepwise validation** against VMEC2000 output (`wout_*.nc`).

```{toctree}
:maxdepth: 2
:caption: User guide

overview
installation
quickstart
theory
algorithms
validation
performance
code_structure
contributing
references
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

