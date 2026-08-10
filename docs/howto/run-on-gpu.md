# Run on GPU

Pass `--device gpu` (CLI) or `device="gpu"` (Python) to place a solve on an
accelerator; the default `auto` applies a measured policy that picks the GPU
only where it wins — per-iteration work `ns*mnmax*nznt >= 100_000` and at
most 512 active Fourier modes, where the measured advantage is 2-3x wall
(`benchmarks/gpu_baseline.json`).

## Select the device

```console
vmex input.case --device gpu     # explicit: always wins
vmex input.case --device cpu
vmex input.case --device auto    # default: measured policy
vmex input.case --device none    # leave placement to JAX
```

```python
import vmex as vj

result = vj.solve_multigrid(inp, device="gpu")
```

Explicit `device=` always wins. `auto` stands down when you pinned a JAX
default device or platform yourself (`jax.config.update("jax_default_device", ...)`,
`JAX_PLATFORMS`, `jax.default_device(...)`), so VMEX never fights your
placement. Install notes for GPU wheels: {doc}`/installation`.

## When the GPU pays off

The policy in {mod}`vmex.core.device` is measured, not guessed
(`benchmarks/gpu_baseline.json`; regenerate with
`benchmarks/device_parity.py` and the benchmark scripts):

- Per-iteration throughput favors the GPU — up to 3x wall on
  NuhrenbergZille-class decks — but the GPU pays fixed per-solve overheads
  (~0.2-0.4 s dispatch/transfer floor plus compile/cache-load on cold
  processes), so small decks that converge in under a second of CPU work
  finish faster on the CPU.
- The work proxy is `ns * mnmax * nznt` (radial surfaces x spectral modes x
  angular grid — the cost driver of the batched `totzsps/tomnsps` matmuls).
  Measured decks split into two clusters: proxies up to ~24e3 where the CPU
  wins (and misclassification costs < 0.5 s either way), and >= ~490e3 where
  the GPU wins 2-3x. `GPU_MIN_ITERATION_WORK = 100_000` sits between them
  (geometric mean ~109e3). The range between the clusters is not calibrated.
- Mode count is an independent guard: the measured GPU winners have at most
  162 active modes, while a high-resolution HSX deck (`mnmax=858`) ran ~3.4x
  *slower* on the GPU even warm despite a large work proxy.
  `GPU_MAX_SPECTRAL_MODES = 512` sits between the largest measured GPU
  winner (288 modes) and that high-mode CPU winner; the cutoff is not
  claimed as a hardware-independent crossover.

Ask the policy directly:

```python
from vmex.core.device import GPU_MIN_ITERATION_WORK, iteration_work, recommended_device

print(iteration_work(runtime.resolution), GPU_MIN_ITERATION_WORK)
print(recommended_device(runtime.resolution))    # "cpu" or "gpu"
```

## What stays on CPU regardless

- **Ensembles.** Multi-solve ensembles are CPU-threaded
  ({doc}`parallel-ensembles`): the host solver's `pure_callback` cannot run
  on a GPU.
- **Implicit Jacobians in optimization.** High-level optimization defaults
  its implicit-gradient path to CPU because it is launch-bound on the tested
  GPUs; low-level {func}`vmex.core.implicit.run` follows JAX placement when
  `device` is omitted, and accepts `device="gpu"` plus
  {func}`~vmex.core.device.device_scope` for explicit accelerator gradients.
- **The dense NESTOR factor.** On a GPU free-boundary run the plasma
  iteration stays on the accelerator while the dense vacuum
  assembly/factor/solve is explicitly placed on CPU
  ({doc}`/explanation/nestor-vacuum`).

## Verify what you got

```console
vmex --doctor
```

prints the JAX backend, visible devices, the active default device, and
VMEX's forward/implicit placement policies. Per-deck CPU-vs-GPU timings and
the decision sweep for a new machine are in {doc}`/reference/performance`.
