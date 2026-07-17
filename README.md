# vmec-jax

[![PyPI version](https://img.shields.io/pypi/v/vmec-jax.svg)](https://pypi.org/project/vmec-jax/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/vmec-jax.svg)](https://github.com/conda-forge/vmec-jax-feedstock)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/uwplasma/vmec_jax/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/vmec_jax)](https://github.com/uwplasma/vmec_jax/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/vmec_jax/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/vmec_jax/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/vmec-jax/latest?label=docs)](https://vmec-jax.readthedocs.io/en/latest/)

**vmec-jax** is a clean-room, JAX-native reimplementation of the
[VMEC2000](https://princetonuniversity.github.io/STELLOPT/VMEC) ideal-MHD
equilibrium code for stellarators and tokamaks. It reproduces VMEC2000
iteration-for-iteration on benchmark decks — and, unlike the Fortran
original, it is differentiable and runs on GPUs.

- **VMEC2000 parity.** The solver ports VMEC2000's algorithms
  constant-for-constant (steepest-descent moment method, radial
  preconditioner, spectral condensation, NESTOR vacuum solve). Benchmark
  decks converge in the *same* number of iterations and reproduce the
  plasma energy at machine precision. An optional **2D block
  preconditioner** cuts iterations 2.5–11x on stiff cases while leaving the
  default path byte-identical.
- **Differentiable.** Gradients of *fixed-boundary* equilibrium outputs with
  respect to boundary shape and profile parameters by implicit
  differentiation of the converged fixed point — no finite differences, no
  unrolling — validated against central finite differences to ~1e-6 relative
  (see the gradient table in the docs), with an O(1)-memory adjoint. **Free
  boundary** is differentiable end-to-end through the virtual-casing vacuum
  field (coil / `extcur` derivatives), finite-difference-validated.
- **Drop-in.** Reads VMEC2000 `input.*` namelists and VMEC++-style JSON,
  prints VMEC2000-format iteration output, and writes `wout_*.nc` files
  that load unchanged in simsopt and booz_xform.
- **Batteries included.** Plotting (`vmec --plot`), Boozer transform
  (`vmec --booz`), spline profiles, multigrid, hot restart, free boundary
  from mgrid files *or* directly from coils,
  typed zero-crash errors — with the shared linear/adjoint solver layer
  factored out into [SOLVAX](https://pypi.org/project/solvax/).

![Flux surfaces, 3-D geometry, and Boozer |B| of the bundled quick-start QH case](docs/_static/figures/readme_equilibrium_showcase.png)

*The bundled quick-start case (`vmec --test`): flux-surface cross sections,
the 3-D plasma boundary coloured by `|B|`, and `|B|` in **Boozer coordinates**
on the last closed flux surface (the near-straight diagonal contours are the
signature of quasi-helical symmetry) for a four-field-period stellarator —
all from the built-in `vmec_jax.core.plotting` / `core.boozer` helpers.*

## Install

Install with **one** of the following — PyPI is recommended:

```bash
pip install vmec-jax                        # PyPI (recommended)
# ...or, if you prefer conda:
conda install -c conda-forge vmec-jax       # conda-forge
```

Development install from source:

```bash
git clone https://github.com/uwplasma/vmec_jax
cd vmec_jax && pip install -e .
```

## Quickstart

```bash
vmec --doctor     # check the installation and JAX backend
vmec --test       # solve the bundled QH case, write wout + plots
vmec input.X      # run any VMEC2000 input deck (or VMEC++-style JSON)
```

`vmec input.X` writes `wout_X.nc` next to the input (`--outdir` to
redirect). To try it on a real deck:

```bash
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/input.nfp4_QH_warm_start
vmec input.nfp4_QH_warm_start
```

Post-process any wout file, including ones written by VMEC2000:

```bash
vmec --plot wout_nfp4_QH_warm_start.nc     # surfaces, |B|, profiles, 3D
vmec --booz wout_nfp4_QH_warm_start.nc     # Boozer transform -> boozmn_*.nc
vmec --plot boozmn_nfp4_QH_warm_start.nc   # Boozer |B| contours + spectrum
```

## Parity with VMEC2000

vmec-jax is validated end-to-end against golden VMEC2000 (PARVMEC 9.0) runs:
benchmark decks converge in **exactly** the golden iteration count — including
DSHAPE's mid-run jacobian reset — and reproduce the plasma energy `wb` to
1 part in 10¹⁵. Across the full benchmark suite (14 rows, all at `ns ≥ 201`),
the iteration count matches VMEC2000 exactly on 12 rows; on the free-boundary
CTH-like row it converges in a ~9% iteration tail, and on Nuhrenberg–Zille QHS
it converges in *fewer* iterations (1681 vs 2829). Per-variable wout agreement
and the full test gates live in the
[documentation](https://vmec-jax.readthedocs.io/en/latest/).

![Force residual vs iteration for vmec_jax, VMEC2000, and VMEC++](docs/_static/figures/readme_convergence.png)

*Parity is per-iteration, not just end-to-end: the total force residual
(`fsqr + fsqz + fsql`) of the quick-start QH case at ns=51, per iteration.
The vmec_jax trajectory lies exactly on top of VMEC2000's (both converge in
502 iterations); VMEC++ follows a near-identical path (501 iterations).
Traces: vmec_jax `SolveResult.fsq_history`, VMEC2000 `NSTEP=1` stdout,
VMEC++ wout `fsqt`.*

### Optional 2D preconditioner: fewer iterations on stiff cases

The default radial (1D) preconditioner reproduces VMEC2000 iteration-for-iteration.
An opt-in **2D block preconditioner** (matrix-free Newton: a Jacobian-vector-product
Hessian on SOLVAX's GMRES) cuts the iteration count **2.5–11×** at *identical*
accuracy — the converged `wb` matches the 1D result to ~1e-10 (it changes the path,
not the fixed point).

![2D vs 1D preconditioner iteration counts on stiff cases](docs/_static/figures/readme_precond.png)

**Why it is opt-in, not the default.** Fewer iterations is not the same as less
wall-clock: each 2D Newton step (a GMRES solve of Hessian-vector products) costs far
more than a 1D radial sweep. Measured across easy and stiff decks the wall-clock
ranges 0.55–1.16× — a wash to *slower* (e.g. ~2× slower on a plain circular tokamak,
a tie even on an aspect-ratio-100 stiff case) — and peak memory is ~30% higher (the
extra GMRES/HVP compile graph). So the 1D path stays the byte-identical default, and
the 2D preconditioner is there for cases where the 1D iteration count is the
bottleneck or stalls.

## Performance

![Wall-clock comparison against VMEC2000 and VMEC++](docs/_static/figures/readme_runtime_compare.png)

Full-solve wall-clock times on the bundled benchmark suite (Apple Silicon
CPU, single thread; `benchmarks/baseline.json`; reproduce with
`python benchmarks/run_baseline.py`):

- **Warm** — kernels already compiled; the number that matters inside an
  optimization loop or scan. Faster than VMEC2000 on most decks (typically
  1.3–2.2×, up to ~4× on small ones); the only exceptions are the
  free-boundary rows, which now converge to parity but whose NESTOR vacuum
  solve is not yet speed-tuned. Ratios measured on a shared CPU are
  conservative lower bounds.
- **Cold** — a fresh CLI process pays a one-time 5–25 s JAX/XLA compile, so a
  single run is slower than Fortran. Executables cache per solver structure, so
  scans, ladders, and optimizations recompile nothing — which is why *warm* is
  the workflow number.
- **GPU** — at these sizes a fixed per-solve dispatch cost dominates and the CPU
  wins outright; per-iteration throughput favours the GPU ~3× on the largest
  decks. The device policy picks CPU or GPU per stage.
- **Memory** — peak (0.6–3.3 GB) is the transient XLA *compile* working set, not
  the data: the equilibrium state is a few MB. The optimization Jacobian is
  bounded by column chunking (`jac_chunk_size="auto"`), so it does not grow with
  the number of design variables.

## Features

| | vmec-jax | VMEC2000 | VMEC++ |
|---|:---:|:---:|:---:|
| Fixed-boundary equilibria | ✅ | ✅ | ✅ |
| Free boundary from an mgrid file | ✅ | ✅ | ✅ |
| Free boundary directly from coils (no mgrid) | ✅ | ❌ | ❌ |
| Free-boundary tokamaks (`ntor = 0`) | ✅ | ✅ | ❌ |
| Non-stellarator-symmetric (`LASYM = T`) | ✅ | ✅ | ❌ |
| Fixed-boundary fallback on missing mgrid | ✅ | ✅ | ❌ |
| Spline profiles (cubic / Akima) | ✅ | ✅ | ❌ |
| VMEC++-schema JSON input | ✅ | ❌ | ✅ |
| Hot restart from a previous state | ✅ | ❌ | ✅ |
| Typed zero-crash errors | ✅ | ❌ | ✅ |
| Boozer transform built in (`--booz`) | ✅ | ❌ | ❌ |
| Plotting built in (`--plot`) | ✅ | ❌ | ❌ |
| GPU execution | ✅ | ❌ | ❌ |
| Differentiable fixed boundary (implicit diff, O(1) memory) | ✅ | ❌ | ❌ |
| Differentiable free boundary (virtual casing) | ✅ | ❌ | ❌ |
| 2D block preconditioner (stiff-case speedup) | ✅ | ❌ | ❌ |

### Free boundary straight from coils

The "free boundary directly from coils" row is a workflow, not a checkbox:
tabulate an [ESSOS](https://github.com/uwplasma/ESSOS) coil set onto the solver
grid in memory (`essos.coils.Coils.to_mgrid`) and pass it as `external_field=` —
no MAKEGRID file to manage, no on-disk round-trip. For gradients the
differentiable free boundary evaluates a JAX Biot-Savart at exactly the boundary
points it needs, every iteration (a plain `xyz→B` callable), so the coil degrees
of freedom stay differentiable end-to-end. vmec_jax keeps no coil code of its
own; coils live in ESSOS.

![Free-boundary Landreman-Paul QA pressure scan directly from ESSOS coils](docs/_static/figures/readme_essos_beta_scan.png)

*Free-boundary equilibria of the Landreman–Paul precise-QA configuration held
by its 16 modular coils as optimized in
[ESSOS](https://github.com/uwplasma/ESSOS) (3 KB coil JSON bundled in
`examples/data/`). Pressure is ramped at fixed coil currents with each point
warm-started from the previous boundary, and `PRES_SCALE` is calibrated per
point so the **actual** volume-average beta of the converged wout
(`betatotal`) — not a nominal input value — lands on 0, 1, 2, 3 % (all within
0.08 %, force residual ~2e-10 at ns = 51). The plasma dilates and the magnetic
axis Shafranov-shifts 14 cm outboard at the φ = 0 section (right panel) while
the coils never move. Reproduce with
`python examples/free_boundary_essos_coils.py`.*

### Single-stage plasma + coil optimization

Starting **cold** — a circular torus and four circular coils, no warm start —
the plasma boundary Fourier modes, the coil curve degrees of freedom, *and* the
coil currents are co-optimized by **one exact gradient**: a single
`jax.value_and_grad` threads the implicit-adjoint derivative of the
fixed-boundary equilibrium, the differentiable virtual casing, and Biot–Savart
off the ESSOS coil filaments through one backward pass. It is benchmarked
against the classical **two-stage** baseline from the *same* seeds: stage 1
optimizes the boundary alone for quasi-axisymmetry, stage 2 then fits the coils
to that frozen boundary. Both approaches get identical coil budgets (same
length and curvature limits, same number of coils), and both are scored on the
**coil-realized equilibrium** — a re-solve of each final boundary, with `B·n`
evaluated from each approach's actual final coils. The finite-β column runs the
same joint optimization with a pressure profile — a capability with essentially
no published general-purpose counterpart.

The headline use is the literature's canonical one (arXiv:2302.10622 runs
single-stage as a "stage 3"): **polish the two-stage result** — warm-start the
joint objective from the stage-1 boundary + stage-2 coils and let both
co-adapt. In ~10–30 minutes of polish the normal-field error drops **33 %
(vacuum) / 17 % (finite β)** below the two-stage result at held quasisymmetry
and on-target iota — the coil↔plasma inconsistency that frozen-boundary
stage 2 cannot fix. The cold-start column shows what the same joint descent
does from the crude seeds alone in 50 iterations: it drives ⟨|B·n|⟩ hard
(coils well inside every budget) but cannot match a dedicated stage-1 on
quasisymmetry — which is exactly why polish is the recommended pattern.

![Cold-start single-stage vs two-stage plasma+coil optimization, vacuum and finite beta](docs/_static/figures/readme_single_stage.png)

*Top: seed (grey, dashed) vs two-stage (orange) vs cold-start single-stage
(blue) boundaries at φ = 0 and a half field period — the polish boundary is
visually indistinguishable from two-stage (same aspect and iota), so it is not
drawn. Middle/bottom: each approach's final LCFS coloured by |B| inside its
own final coils.*

Vacuum (measured; identical seeds and coil budgets across columns):

| metric (vacuum) | two-stage | + single-stage polish | single-stage (cold) |
|---|---|---|---|
| QS ratio residual | 9.3e-05 | 1.6e-04 | 2.4e-02 |
| mean iota (target 0.42) | 0.420 | 0.420 | 0.396 |
| ⟨\|B·n\|⟩/⟨B⟩ | 2.38e-03 | **1.60e-03** | 3.05e-03 |
| max\|B·n\|/⟨B⟩ | 1.30e-02 | **7.84e-03** | 1.18e-02 |
| coil lengths [m] (≤ 4.40) | 4.12–4.39 | 4.11–4.40 | 3.60–3.87 |

Finite β (⟨β⟩ ≈ 1.5 %, same pressure profile in all columns):

| metric (finite β) | two-stage | + single-stage polish | single-stage (cold) |
|---|---|---|---|
| QS ratio residual | 4.4e-05 | 2.4e-04 | 2.3e-02 |
| mean iota (target 0.42) | 0.420 | 0.422 | 0.100 |
| ⟨\|B·n\|⟩/⟨B⟩ | 2.80e-03 | **2.34e-03** | 6.20e-03 |
| max\|B·n\|/⟨B⟩ | 1.37e-02 | **1.27e-02** | 1.75e-02 |
| coil lengths [m] (≤ 4.40) | 3.91–4.18 | 3.91–4.19 | 3.25–3.28 |

Reproduce with `python examples/single_stage_vs_two_stage.py --case vacuum
--phase all` (and `--case beta`). Measured on a 36-core CPU: stage 1 ≈ 7–9 min,
stage 2 ≈ 6 min, polish ≈ 10–30 min; the optional cold-start single column is
the long pole (≈ 1.5 h vacuum, several hours at finite β). The phases are
resumable, so long runs can be split across sessions.

## Code size

vmec-jax delivers that superset of capabilities in little more than **half the
code**, and is the most densely documented of the three. Solver source only (tests,
language bindings, and vendored third-party excluded), counted with
[`pygount`](https://pypi.org/project/pygount/) 3.2:

| code base | language | files | code (SLOC) | comments / docstrings | doc-to-code |
|---|---|---:|---:|---:|---:|
| **vmec-jax** | Python | 41 | **13,326** | 6,744 | **0.51** |
| VMEC2000 (PARVMEC) | Fortran | 115 | 24,190 | 8,425 | 0.35 |
| VMEC++ | C++ / Python | 117 | 22,824 | 7,646 | 0.34 |

vmec-jax is little more than half the SLOC of VMEC2000 and VMEC++, while
*adding* differentiability, GPU execution, direct-coil free boundary, and a
built-in Boozer transform — and it carries the highest comment/docstring
density of the three (reproduce with
`pygount --format=summary vmec_jax`).

## Python API

```python
from vmec_jax.core.input import VmecInput
from vmec_jax.core.multigrid import solve_multigrid
from vmec_jax.core.wout import wout_from_state, write_wout
from vmec_jax.core.plotting import plot_wout

inp = VmecInput.from_file("input.nfp4_QH_warm_start")
result = solve_multigrid(inp)          # full NS_ARRAY ladder, VMEC2000 numerics
print(result.converged, result.iterations, result.wmhd)

wout = wout_from_state(inp=inp, state=result.state, niter=result.iterations,
                       fsqr=result.fsqr, fsqz=result.fsqz, fsql=result.fsql)
write_wout("wout_nfp4_QH_warm_start.nc", wout)
plot_wout(wout, "figures/")
```

Optimization building blocks live in `vmec_jax.core.optimize`
(quasisymmetry and omnigenity residuals; aspect ratio, iota, mirror ratio,
magnetic well, ballooning-stability targets; a least-squares driver over
boundary Fourier coefficients) with implicit-differentiation gradients from
`vmec_jax.core.implicit` (`jac="implicit"`). The recommended pattern is **one
`least_squares` call** — no `max_mode` continuation loop — with **Exponential
Spectral Scaling** ordering the harmonics through the trust region:

```python
from vmec_jax import optimize as opt

qs = opt.QuasisymmetryRatioResidual(surfaces, helicity_m=1, helicity_n=0)
result = opt.least_squares(
    [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0), (opt.mean_iota, 0.42, 1.0)],
    inp, max_mode=5, jac="implicit",
    use_ess=True,        # exp(-alpha*max(|m|,|n|)) trust radius per dof:
)                        # high harmonics on short leashes — no ladder needed
```

Measured on a 36-core CPU from a near-circular torus (single call, all
harmonics released at once; `examples/optimization/*_ess.py`; the staged
`max_mode`-ladder variants live alongside for comparison):

| class | nfp | residual | seed | achieved | max_mode | wall | status |
|-------|-----|----------|------|----------|----------|------|--------|
| QA | 2 | QS (1, 0)  | 2.04e-01 | **7.2e-06** | 5 | **14.5 min** | precise; aspect 6.00, iota 0.42 (ladder: 3.7e-07 in 25.5 min) |
| QH | 4 | QS (1, −1) | 6.91e-01 | **5.83e-05** | 5 | 25.5 min (ladder) | precise; aspect 8.00, iota −1.22 |
| QP | 2 | QS (0, 1)  | 4.46e-01 | 3.3e-02 | 5 | ~3.4 h (ladder + refinement) | hardest QS class — see caption |
| QI | 1 | omnigenity | 4.52e-01 | **1.81e-02** | 6 | **17.3 min** | 25× via the traceable Goodman constructed-QI residual |

![QA/QH/QP optimization: seed vs optimized boundary, 3-D |B| geometry, and Boozer |B| on the LCFS](docs/_static/figures/readme_optimization.png)

*Each quasisymmetry class starts from a near-circular torus (grey, dashed) and
is shaped into a quasi-symmetric stellarator (blue) by the least-squares driver
(top row); the middle row is the optimized last-closed flux surface in 3-D
coloured by `|B|`, and the bottom row is `|B|` in Boozer coordinates on the LCFS
(jet line contours), whose contour geometry reads off the symmetry family —
horizontal for QA, diagonal for QH, vertical for QP. `QS` is the quasisymmetry
residual measured on the plotted equilibrium: QA **1.1e-6**, QH **5.8e-5**
(note QH's near-straight diagonal contours), QP **3.3e-2**. Quasi-poloidal QP
is the hardest class: the ladder plateaus near 5e-2, and an extended
warm-start refinement of the shipped deck reaches 3.3e-2. Reproduce with
`python benchmarks/make_readme_figures.py --only optimization` from the decks
in `benchmarks/opt_decks/`.*

Quasi-isodynamic (QI) shaping is intrinsically harder than quasisymmetry, so it
gets its own row across field periods:

![QI equilibria at nfp 1-4: boundary, 3-D |B| geometry, and Boozer |B| on the LCFS](docs/_static/figures/readme_qi.png)

*Quasi-isodynamic (QI) equilibria at nfp 1, 2, 3, 4 (bundled decks in
`examples/data/`): boundary cross-sections (top), 3-D `|B|` geometry (middle),
and `|B|` in Boozer coordinates on the LCFS (jet, bottom). The label is the QI
(omnigenity) residual — **not** QS; QI is hard, so ~1e-3–1e-2 is expected here,
not the ~1e-5 reachable for quasisymmetry. Reproduce with
`python benchmarks/make_readme_figures.py --only qi`.*

Implicit gradients are *essential*, not merely faster: for the helical (QH)
target the exact-axisymmetric seed is a saddle where finite differences stall,
and for QP the implicit path reaches a far better basin than FD. Three
measured accelerations make the minutes-scale campaigns above possible: the
residual Jacobian is solved through a **block-tridiagonal factorization** of
the force linearization (33× over per-dof GMRES), each trial equilibrium is
seeded with a **first-order perturbation prediction** from that same
factorization (3.7× fewer solver iterations), and a converged-state memo means
the Jacobian never re-solves the point the residual just converged. The
implicit path is **CPU-pinned by default** (it is kernel-launch-bound; GPUs
lose at every production size measured), while forward solves at high radial
resolution are GPU-competitive — the device policy picks per stage.

### Self-consistent bootstrap current

vmec-jax implements the **Redl** analytic bootstrap-current formula
([Redl et al. 2021](https://doi.org/10.1063/5.0012664)) as a differentiable
objective, and a fixed-boundary self-consistency loop that regenerates the
toroidal current from the plasma geometry and kinetic profiles. Below,
reproducing [Landreman, Buller & Drevlak 2022](https://arxiv.org/abs/2205.02914):
the published precise QA and QH optima are loaded, their current profile is
**erased**, and `self_consistent_bootstrap` recovers it from the Redl formula
plus the paper's density/temperature profiles.

![Self-consistent bootstrap current vs the published equilibria and SFINCS](docs/_static/figures/readme_bootstrap.png)

*Recovered current density &#10216;J·B&#10217; (VMEC, blue) matches the analytic
Redl profile (green), the published self-consistent equilibrium (grey), and —
for QA — the paper's SFINCS drift-kinetic benchmark (circles). Converged in 7
(QA) / 4 (QH) Picard iterations to bootstrap mismatch `f_boot` = 2.0e-6 / 7.5e-6;
the recovered plasma current lands within **1.9 % (QA)** and **0.3 % (QH)** of
the published `CURTOR`. Reproduce with
`python examples/optimization/{QA,QH}_bootstrap_selfconsistent.py` (needs the
paper's Zenodo dataset).*

## vmec-jax vs DESC

[DESC](https://desc-docs.readthedocs.io/) is the other JAX-native,
differentiable, GPU-capable stellarator-equilibrium code. The key difference:
DESC minimises the MHD force in a global Zernike–Fourier basis — *its own*
equilibrium — while vmec-jax reproduces VMEC exactly. They are complementary;
the honest trade-off, side by side:

| Where **vmec-jax** wins | Where **DESC** wins |
|---|---|
| **Is VMEC**: iteration-for-iteration VMEC2000 parity, standard `wout_*.nc`, VMEC-format prints | **Low-resolution accuracy**: global Zernike basis converges in fewer radial points |
| **Drop-in**: reads VMEC2000 `input.*` and VMEC++ JSON unchanged | **Objective library**: large, mature set of built-in optimization targets |
| **Full namelist**: non-symmetric surfaces (`LASYM = T`), NESTOR *and* virtual-casing free boundary | **Optimizers**: more built-in stochastic / constrained optimizers |
| **O(1)-memory adjoint**: peak memory flat in the number of design variables | Adjoint gradients (both codes are differentiable) |

Reach for **vmec-jax** to drop a differentiable code that *is* VMEC into an
existing VMEC workflow (simsopt, `booz_xform`, near-axis tooling). Reach for
**DESC** for its spectral accuracy at low radial resolution or its mature
objective library.

## CLI reference

```text
vmec input.X             solve (INDATA or VMEC++ JSON), write wout_X.nc
vmec --plot wout_*.nc    diagnostic plots from a WOUT file
vmec --booz wout_*.nc    run booz_xform_jax, write boozmn_*.nc
vmec --plot boozmn_*.nc  Boozer contour/spectrum plots
vmec --test              run and plot the bundled quick-start case
vmec --doctor            installation and JAX backend diagnostics

options:
  --outdir PATH          directory for wout/boozmn/figure output
  --mode {cli,jit}       jitted blocks with live printing (cli, default)
                         or a single lax.while_loop (jit)
  --ftol F               override the final-stage FTOL_ARRAY tolerance
  --max-iter N           override the final-stage NITER_ARRAY cap
  --coils PATH           ESSOS-style coils file: drive an LFREEB = T deck
                         by direct Biot-Savart instead of an mgrid file
  --mbooz/--nbooz N      Boozer spectral resolution (default 32/32)
  --booz-surfaces S      Boozer surfaces ('all' or a list of s values)
  --quiet                silence the VMEC-style stdout
```

`vmec` follows the selected JAX backend: with CPU-only JAX it runs on the
CPU; with CUDA-enabled JAX it uses the GPU for the solver stages where that
is faster (`JAX_PLATFORMS=cpu|cuda` pins it explicitly).

## Documentation

Full documentation — installation, quickstart, theory and numerics with
equation-to-source cross-references, API reference, and
performance/validation notes — at
[vmec-jax.readthedocs.io](https://vmec-jax.readthedocs.io/en/latest/).

## License

MIT. If you use vmec-jax in published work, please cite this repository and
the original VMEC papers (Hirshman & Whitson, *Phys. Fluids* 1983;
Hirshman, van Rij & Merkel, *Comput. Phys. Commun.* 1986).
