# vmec_jax Overhaul Plan — from research prototype to a fast, differentiable, research-grade VMEC

**Audience:** an autonomous agent (Cowork) with full local access, executing this plan end-to-end.
**Owner / git identity:** all commits and pushes authored as `rogeriojorge` (GitHub user `rogeriojorge`;
`gh` is already authenticated as `rogeriojorge` with `repo`+`workflow` scopes on this machine).
**Status:** this file is the single active plan. It replaces every file in `vmec_jax_plan/`
(`plan.md`, `plan_freeb.md`, `plan_differentiability.md`, `discrete_adjoint_2506_plan.md`, and the
1.5 MB / 32,088-line `plan_research_grade_performance_differentiability.md`). Those files are Phase-0
inputs (read once, extract anything still relevant into scratch `NOTES.md`) and are then deleted from
the working tree **and from git history** in Phase 1. Keep this plan.md under 200 KB; it is a working
document, updated with a short status line per phase as work proceeds, and moved to `docs/dev/` (or
deleted) at the v0.1.0 release.

---

## 0. Mission statement

Turn `vmec_jax` into the reference JAX implementation of the VMEC ideal-MHD equilibrium solver:

1. **End-to-end differentiable** library API (fixed and free boundary), fast on CPU and GPU, using
   implicit differentiation of the converged equilibrium — not unrolled iteration tapes and not the
   current "fingerprint-gated branch-local" machinery.
2. **A non-differentiable CLI fast path** that may use Python-side control flow, host callbacks,
   early exits, and donated buffers to beat the differentiable path in wall time.
3. **VMEC2000 parity**: iteration prints, `wout_*.nc` contents, threed1-style summaries, and
   converged physics quantities match VMEC2000 within per-quantity validation tolerances.
4. **Performance parity or better** than VMEC2000 single-thread CPU on the benchmark suite,
   including multigrid (`NS_ARRAY` ladders), which is currently slower than VMEC2000 — a named bug.
5. **A small, readable codebase**: 30–40 Python files in `vmec_jax/`, ~25–30k library lines
   (revised 2026-07-09 from the original ≤15k after the fixed-boundary core alone measured ~10k
   well-documented lines; still a >4x reduction from **229 files / ~123k lines**), physically
   meaningful names, docstrings everywhere, ≥95% coverage without repo bloat (tests currently
   ~140k lines with coverage-padding files; target ≤ ~10k test lines).
6. **A ~10 MB repository** after a `git filter-repo` history rewrite (currently 57.4 MiB packed);
   large assets move to GitHub Releases; no Claude in the contributors panel.
7. **User-friendly docs** with full derivations (energy functional → forces → spectral condensation
   → preconditioner → time stepping → free boundary → adjoint), every equation linked to the
   implementing source.
8. **simsopt-style optimization examples** for QA / QH / QP / QI that start from a circular torus
   and converge to precise configurations in a single, short, readable script each.
9. **Free-boundary showcase**: β = 0→5% scans driven by ESSOS coils (stellarator + tokamak), run
   both through generated mgrid files and through direct Biot–Savart evaluation (no mgrid),
   demonstrating agreement — and that the direct path is the interpolation-free reference.
10. **Feature superset vs VMEC++** where VMEC++ has gaps: `lasym` (non-stellarator-symmetric),
    free boundary for tokamaks (`ntor=0`) and stellarators, fixed-boundary fallback on missing
    mgrid, spline/pedestal profile types, and a 2D preconditioner option — while borrowing VMEC++'s
    hot restart, JSON input, zero-crash policy, and validation methodology.

Every decision below optimizes for: *simpler to use, fewer files, faster, more manageable*.

---

## 0.5 Completion roadmap (updated 2026-07-10) — honest status + ordered remaining work

This is the actionable index of what is DONE and what REMAINS. Sections 1-16 below are the original
phase specs (still authoritative for detail); this roadmap supersedes the scattered STATUS notes and
folds in every requirement from the user prompts and the two independent reviews.

### Done and verified on current `main` (d26bed44)
- Legacy tree deleted; package is `vmec_jax/core/` only — **34 files / 19.2k lines** (revised target
  25-30k), tests **34 files / 6.7k lines**. Fresh clone ≤ ~12 MB; no Claude contributor; all commits
  rogeriojorge.
- Fixed-boundary equilibrium at **VMEC2000 machine-precision parity** across the 9 golden fixtures
  (exact iteration counts incl. lasym after the fixaray dnorm fix; wb ~1e-16, geometry ~1e-12).
- Structural executable reuse (SolverRuntime pytree; 0 recompiles on same-Resolution re-solve),
  NS_ARRAY multigrid ladder, hot restart, wout (full wrout.f set), plotting, Boozer, CLI on the core,
  VMEC++-JSON input, zero-crash typed errors, mgrid fallback.
- Free-boundary NESTOR solve (vacuum activates at the golden iteration exactly) + direct-coil field;
  CLI routes LFREEB decks + `--coils`.
- Implicit differentiation (custom_vjp + preconditioned-GMRES adjoint), FD-validated 1e-6..1e-12 for
  fixed-boundary boundary/profile/phiedge params; `jac="implicit"` wired into `least_squares`.
- GPU fixes (cuSPARSE tridiagonal, cache hardening, device policy) measured on office 2x A4000.
- README ns≥51 benchmark + vmecpp-style convergence figure; docs restructured (17 pages) with honest
  fixed-vs-free differentiability scoping; **CI green: 7 shards, wall ~9 min, 95% coverage gate**.
- Examples: 5 clean simsopt-style files (541 lines) on the core API; `jac="implicit"` used.

### Remaining work — ordered by priority (each item has an acceptance gate)

**R1. Optimization examples actually reach precise QA/QH/QP/QI (biggest open lane; user's #1 ask).**
Current: scripts run and objectives decrease at CI budget; genuine convergence from a circular torus
is UNVALIDATED. Gate: from a near-circular seed, staged `max_mode` 1→5 with ESS and `jac="implicit"`,
running **thousands of iterations/nfev as needed**, reach *precise* QS/QI — QA (nfp=2) & QH (nfp=4)
QS residual ≤ ~1e-3 of seed and aspect on target; QP (nfp=2, note the max_mode≥4 bad-basin caveat)
and QI (nfp=1, QP-first-then-QI) to documented precision. Record achieved values in each docstring;
add a `full`-marked convergence test per class asserting the achieved bound. Run the heavy ones on
`ssh office` GPU; compare CPU vs GPU wall + autodiff-vs-FD accuracy. Verify the commented DMerc/
LgradB/magnetic-well terms work uncommented (already CI-tested) and read well pedagogically.

**Seed policy (user 2026-07-10; ties to the R1 saddle finding).** The seed does NOT have to be an
exact circular torus — an exact-axisymmetric/circular boundary is a *saddle* of the QS residual (the
symmetry-breaking harmonics are even → gradient vanishes there), which is exactly why FD stalls and
implicit gradients are needed. So seed from a **near-circular torus with the would-be-zero boundary
harmonics initialized to ~1e-4** (a small symmetry-/shape-breaking "kick"), with one shaping mode
seeded a bit larger to give the optimizer a defined descent direction, instead of exact zeros. This
makes even the first step well-posed, is physically honest, and matches how the QA precise result was
obtained ("kicked circular seed"). Make the kick amplitude an example parameter-at-top (default
~1e-4) and document it. The alternative, richer seed is the **near-axis (pyQSC_JAX/pyQIC) seed of
R19** — offer both: the tiny-kick circular seed (simplest, in-repo) and the near-axis seed (best
starting point) so users learn both routes.

_R1 status (2026-07-10, office 2x A4000 / 36-core CPU, on f45a6491):_
- **QA (nfp2) — PRECISE, validated.** `jac="implicit"` + ESS from the kicked circular seed:
  QS total 2.043e-01 → 9.82e-03 (max_mode=1) → **1.701e-04 (max_mode=2)**, aspect 6.000 &
  mean iota 0.420 on target. >3 orders, ≤1e-3-of-seed gate met. Docstring updated.
- **QH (nfp4) — descends via implicit; precise not re-validated this session.** QS 6.908e-01 →
  1.401e-01 (max_mode=1, iota −0.917). Key finding: the exact-axisymmetric seed is a *saddle*
  (QS residual even in the symmetry-breaking harmonic) — finite differences STALL (njev=1, QS
  unchanged); implicit escapes it, so the example is correct to use implicit (no kick). Higher
  max_mode continuation is compile/eval-bound (see below); stage 2 did not finish in budget.
- **QP (nfp2) — basin-limited, not precise (as documented).** Implicit reaches QS 4.458e-01 →
  9.42e-02 (max_mode=1, ~the docstring's 7e-2 basin); FD stalls much worse at 2.32e-01. Confirms
  the QP bad-basin caveat AND that the gradient method selects the basin (implicit ≫ FD here).
  Schedule capped at 3.
- **QI (nfp1) — partial (hardest).** QP-basin (FD) 2.43 → ~1.15 (QI total), QP 6.7e-2 → 3.85e-2;
  the Boozer QI-stage refinement barely moves from a crude circular nfp1 seed — precise QI needs
  more than the current 4-term FD path (documented, not overclaimed).
- **Autodiff accuracy:** implicit gradients drive the same descent as FD (QA reaches precise;
  `tests/core_new/test_implicit_grad.py` already validates implicit vs central FD, rtol ≤1e-6
  solovev / ~1e-5 li383-3D). Implicit is *essential* for the helical/basin cases (FD stalls/worse
  basin), not just faster.
- **CPU vs GPU / per-step:** warm forward solve ~0.9 s; the implicit forward solve is a host (CPU)
  callback so the GPU only accelerates the adjoint GMRES — and does NOT help this small problem
  (cold solve CPU 13 s vs GPU 27 s; GPU per-stage dominated by a one-time XLA compile that grows
  with dof count: QA stage1 761 s / stage2 1261 s, amortized 20–37 s/step; a 24-dof QH GPU stage
  hung >37 min in a single kernel-launch-bound GMRES eval). **Actionable for R3/R6:** the
  per-dof-vmap implicit Jacobian compile/eval is the scaling bottleneck for max_mode ≥ 2.
- Added `tests/core_new/test_optimization_convergence.py` (`full`-marked, per class).
- REMAINING: get QH to precise (run implicit continuation on CPU where GMRES eval is fast, or
  reduce the adjoint/vmap cost); improve QP basin & QI omnigenity residual.

**R2. Free boundary to production.** Current: CTH free-bdy stops at NITER (fsq~9e-2), not converged;
warm 14.4 s ≫ Fortran 1.95 s; coil derivatives unsupported by the implicit residual. Gate: a
*converged* free-boundary golden fixture (raise NITER; validate wout vs VMEC2000), NESTOR/vacuum
performance profiled and tuned (target within ~3x Fortran warm), then free-boundary implicit
derivatives (coil dofs → boundary/QS) FD-validated. β-scan showcase example (ESSOS Landreman-Paul QA
coils + a tokamak coil set; β=0..5% hot-restarted; mgrid vs direct Biot-Savart agreement) → one
compressed README panel. Do NOT promote coil-derivative claims before the converged fixture exists.

**R3. Memory + cold-start workstream.** Current: solves 0.7-1.5 GB (Fortran 27-43 MB), implicit grad
3.4 GB; cold CLI pays 5-25 s XLA setup. Gate: profile XLA graph construction + peak buffers; donate
CLI-lane buffers; eliminate redundant structural jit variants; reuse fixed-shape residual/adjoint
kernels; persistent-cache guidance documented. Targets: ≥2x solve-memory reduction; cold small-deck
CLI < ~2.5 s; record in benchmarks. This is the lane the reviews scored lowest (cold/memory 40%).

**R4. GPU production evidence.** Current: microbenchmarks + fixes done; production runs across all
solver types/modes/geometries not yet benchmarked. Gate: `benchmarks/gpu_baseline.json` extended on
office to fixed+free, sym+lasym, single+multigrid, small→large ns, with the crossover documented and
the auto device policy validated end-to-end; README/docs GPU section updated.

**R5. Finite-beta + diagnostics parity.** Validate finite-beta wout channels (currents, `DMerc`,
`D_R`/Glasser, jdotb) against VMEC2000 and their AD-vs-central-FD gradients; add golden fixtures for a
finite-beta case. Gate: per-variable tolerances met; gradient tests permanent.

**R6. Refactor + docstring hygiene.** Split the 4 files >1000 lines when next touched (optimize 1482,
solver 1308, nyquist 1046, setup 1005); close the **71/355 public-def docstring gap**. Gate: no core
file >~1000 lines; 0 public defs without docstrings; ruff+mypy clean without blanket ignores.

**R7. Docs completion.** Per-example tutorial pages with rendered figures; theory-with-equations pass
(every equation linked to its implementing function); `docs/glossary.rst` (VMEC2000↔vmec_jax names).
Gate: docs `-W` green; each example has a tutorial; glossary present.

**R8. Mirror geometry (dev).** Rebase/port PR #22 (codex/mirror-geometry) onto the core, or implement
the `docs/mirrors.rst` design + xfail scaffolds per §7 Phase 5.5. Gate: design doc + a fixed-boundary
axisymmetric mirror smoke.

**R9. Release v0.1.0.** After R1-R5: regenerate benchmarks/README, refresh the release asset bundle,
tag, publish PyPI + conda-forge, verify `pip install vmec-jax && vmec --test` on a clean machine.

### Standing constraints (apply to all remaining work)
- CI wall ≤10 min, coverage ≥95%, no brittle absolute wall-clock asserts (use ratios / compile counts).
- Optimization runs use thousands of iterations for real convergence; CI uses reduced budgets.
- Docs/README claims stay honest: separate validated fixed-boundary from in-progress free-boundary.
- Use `ssh office` (2x RTX A4000) for GPU/heavy runs; keep the local machine responsive (watchdog).
- Every commit rogeriojorge, no AI trailer; push small, let CI verify; avoid rapid successive pushes
  that cancel in-flight runs via concurrency.

### R10-R16 — detailed resumable tasks (added 2026-07-10 from user review; specific steps)

**R10. Prove functionality completeness vs VMEC2000 + VMEC++ (the "is it all there?" question).**
*(R10.2 DONE 2026-07-10, 2980d812: 2D block preconditioner — matrix-free Newton via
jax.jvp HVP on solvax.gmres; 2.5-11x iteration reduction on stiff cases (aspect-100 97->18,
163->15; nfp4_QH finite-beta 1885->204); default 1D path byte-identical; CI green incl. 95%
gate. Wall neutral on CPU cold — GPU/warm-cache/gcrot-recycling win pending. Showcase = R20.)*
The core is small (34 files / 19.2k lines) because JAX/Python is far denser than Fortran/C++ and we
dropped VMEC2000's MPI, v3fit reconstruction, and ANIMEC boilerplate — NOT because physics is missing.
Verified present: fixed + free boundary (NESTOR), lasym, ntor=0 free-bdy, multigrid + hot restart,
**18 profile parameterizations** (power_series/two_power/gauss_trunc/pedestal/cubic+akima splines/
line_segment + _i/_ip integrated variants), ncurr=0/1, full wrout.f wout set, Mercier, jxbforce, bss,
Boozer, JSON+INDATA, zero-crash, implicit diff (fixed-bdy), direct-coil free boundary, GPU.
Steps:
  1. Write `docs/functionality_matrix.rst` (+ a README summary): a feature-by-feature table
     vmec_jax vs VMEC2000 vs VMEC++ with a "where implemented (module:function)" column, so
     completeness is provable and auditable. Include the LOC/file-count comparison row (see R11.3).
  2. Close the **genuine gaps**, in priority: (a) **2D preconditioner** — currently only accepted as
     an input key (`precon_type`/`prec2d_threshold` in input.py) but NOT implemented; only the 1D
     radial preconditioner exists. Implement the optional 2D block preconditioner via matrix-free
     `jax.jvp` Hessian-vector products + preconditioned GMRES (plan §7.5), activated below a
     threshold on the finest grid, to cut iteration counts on stiff cases. (b) document ANIMEC
     (anisotropic pressure) and RFP (`lrfp`) as explicitly out-of-scope/niche, or add if a user
     needs them. Gate: functionality_matrix has no unexplained ❌ for a mainstream VMEC2000 feature.

**R11. README overhaul (all ns≥51; optimization + Boozer; code-size comparison; better showcase).**
  **(R11 DONE 2026-07-11, commit caf6166c.)** All four subitems shipped: free-bdy row runs at ns≥51
  (cth_like_free_bdy + cth_like_free_bdy_lasym_small, both converged in baseline.json); README
  `readme_optimization.png` (QA/QH/QP/QI initial-vs-optimized + Boozer |B|); code-size table via
  pygount 3.2 (vmec-jax **36 files / 11,789 SLOC / 5,532 comment / 0.47 doc-ratio** vs VMEC2000 115f
  /24,190/8,425/0.35 and VMEC++ 117f/22,824/7,646/0.34 — the shipped numbers, slightly refined from
  the 2026-07-10 estimate below as the core grew during R15/R18); `readme_equilibrium_showcase.png`
  now has 3D |B|-on-surface + Boozer-|B| with jet cmap (plotting.py `cmap` arg + `boozer_modB_on_surface`).
  Performance section reconciled to baseline.json (warm faster than VMEC2000 on 9/13 rows) with an
  explicit shared-CPU caveat: the ratios are conservative lower bounds measured under load; **R9
  re-runs the runtime figure on a clean machine.** docs -W green, ruff clean.
  1. **All benchmark rows ns≥51.** `benchmarks/run_baseline.py` already ramps fixed-bdy to ns≥51;
     the **free-boundary row (cth_like_free_bdy) must also run at ns≥51** (currently its deck ns may
     be <51). Bump the free-bdy deck's final NS_ARRAY stage to ≥51 (regenerate its mgrid if the grid
     resolution requires), re-run the whole suite, regenerate `readme_runtime_compare.png`. Verify no
     row in the figure/table uses ns<51.
  2. **Optimization panels.** Add README figures showing QA/QH/QI/QP **initial vs optimized** boundary
     cross-sections + **Boozer |B| on the LCFS** for each, from the R1 converged results. One compact
     multi-panel `readme_optimization.png` (compressed <150 KB). Generate via a tracked
     `benchmarks/make_readme_figures.py` addition using `core.plotting` + `core.boozer`.
  3. **Code-size comparison.** Add a README table: source files + lines of code for vmec_jax vs
     VMEC2000 vs VMEC++ (count with `cloc`/`tokei` over `/Users/rogerio/local/STELLOPT/VMEC2000/Sources`
     and `/Users/rogerio/local/vmecpp/src`; state what's counted). Message: comparable/greater
     capability in a fraction of the code. **Measured 2026-07-10 (solver source only; tests/
     bindings/third-party excluded): vmec_jax 34 files / 19,237 Python lines; VMEC2000 115
     files / 36,693 Fortran lines; VMEC++ 117 files / ~39,677 (34,255 C++ + 5,422 Python) —
     vmec_jax is ~half the code of both, with a superset of capabilities.** ALSO report the
     **comment/docstring vs actual-code split** (user 2026-07-10) to show vmec_jax is better
     documented and more user-friendly. Measured (tokenize for Python; comment-line count for
     Fortran/C++): vmec_jax **11,274 actual code (SLOC)** + 5,112 comments/docstrings (27% of
     total) → **doc-to-code ratio 0.45**; VMEC2000 24,164 code + 8,451 comment → 0.35; VMEC++
     ~23,149 code + ~5,841 comment → ~0.25. Headline: **vmec_jax has <half the actual code AND
     the highest documentation density of the three.** Use `pygount`/`cloc` for the README
     table (install if needed) so the comment/code split is reproducible.
  4. **Showcase figure.** `readme_equilibrium_showcase.png`: show the **3D geometry with |B| color on
     the surface**; and change the current flat |B| plot to **|B| in Boozer coordinates with the `jet`
     colormap** (the STELLOPT/Boozer convention). Update `core.plotting`/`core.boozer` plot helpers as
     needed (add a `cmap` arg + a boozer-|B|-on-LCFS plot).

**R12. Rename `tests/core_new/` → `tests/`.**
  **(R12 DONE 2026-07-11.)** `git mv` of all 38 test modules + `data/` up one level; merged the golden-fixture
  + `_module_jit_enabled` machinery from `core_new/conftest.py` into the root `tests/conftest.py`; decremented
  the 28 `Path(__file__).resolve().parents[N]` depth anchors (27×[2]→[1], 1×[1]→[0]) while leaving
  `DATA_DIR.parents[1]` and the unmoved conftest untouched; updated every `tests/core_new` path in ci.yml
  (shard lists, ignores, the golden-prefetch import), benchmarks, docs, examples, and core docstrings.
  Verified: 513 tests collect, golden prefetch resolves via `tests/conftest.py`, golden parity test passes.
  Done standalone (decoupled from R21 per the 2026-07-11 reordering). Original note follows:
  "new" is meaningless to users and the legacy `tests/`
is gone. Steps: `git mv tests/core_new/* tests/` (handle conftest.py merge — root `tests/conftest.py`
already exists with the RUN_FULL/jit gates; merge the core_new conftest fixtures into it), update
every CI path in `.github/workflows/ci.yml` (parity shard file lists, ignores, prefetch), update the
golden-fetch import path, `pyproject.toml` pytest config, and any `tests/core_new` string in docs.
Gate: CI green with the flat `tests/` layout; no `core_new` anywhere.

**R13. Many more pedagogic examples (study STELLOPT / VMEC2000 / hiddenSymmetries simsopt / DESC /
VMEC++ example layouts).**
  **(R13 IN PROGRESS 2026-07-11.)** Shipped so far (each simsopt-style, params-at-top, CI-smoke-tested,
  indexed in examples/README.md): `plot_and_boozer.py` (every plot_wout figure + Boozer on the LCFS),
  `profiles_power_and_spline.py` (power-series vs cubic-spline profiles → identical equilibrium; NCURR
  0 vs 1), `take_gradients.py` (implicit-adjoint d(aspect)/d(RBC) and d(wb)/d(phiedge) vs central FD,
  rel ~1e-9), `run_from_json.py` (VMEC++ JSON ↔ &INDATA round-trip → one equilibrium),
  `hot_restart_scan.py` (seed each scan point from the previous state → warm converges in ~1 iter vs
  ~309 cold, no recompile). Light ones run in the PR examples shard; take_gradients is nightly (`full`).
  `finite_beta_scan.py` (pressure ramp → beta, Shafranov axis shift, Mercier DMerc; hot-restarted).
  `free_boundary_mgrid.py` (NESTOR free boundary from coil EXTCUR + mgrid; LCFS solved for, nightly),
  `free_boundary_beta_scan.py` (free-bdy pressure ramp → beta 0→2.6%, LCFS re-solved each point, nightly).
  8 examples shipped this session. DEFERRED (need real coil data / advanced, follow-up):
  free_boundary_essos_coils (direct-coil free bdy — no bundled CTH coils that reproduce mgrid_cth_like
  and converge; needs a purpose-built coil set), single_stage_free_boundary_opt (gated on R15 free-bdy
  IFT wrap). These two are the remaining R13 items; the rest of R13 is done. Each is one simsopt-style file (params at top, no `main()`, prints
initial→progress→final, teaches one feature) and is CI-smoke-tested (reduced budget) + doubles as the
docs tutorial (R14.3). Target set:
  - `run_fixed_boundary.py` (exists), `run_from_json.py` (VMEC++ JSON in/out + convert),
  - `free_boundary_mgrid.py` (mgrid path), `free_boundary_essos_coils.py` (ESSOS coils tabulated to
    an in-memory mgrid; needs ESSOS), `free_boundary_beta_scan.py` (β=0..5% hot-restarted; the README β-scan),
  - `profiles_power_and_spline.py` (power_series vs cubic/akima; pressure/iota/current; ncurr=0 vs 1),
  - `finite_beta_scan.py` (pressure ramp; beta, Mercier, Shafranov shift),
  - `take_gradients.py` (implicit d(aspect|iota|QS)/d(boundary|profile) vs FD; jacrev usage),
  - `plot_and_boozer.py` (all plot types + `--booz` on LCFS),
  - `hot_restart_scan.py` (reuse a converged state across a parameter scan; warm speedups),
  - `single_stage_free_boundary_opt.py` (ESSOS coils → free-bdy equilibrium → QS/aspect targets;
    advanced) — gated on R15 free-bdy differentiation.
  Keep QA/QH/QP/QI optimization examples. `examples/README.md` indexes them by feature. Gate: every
  example smoke-passes in CI; each maps to a docs tutorial.

**R14. Complete the documentation (full theory + algorithms + tutorials, not an overview).**
  **(R14 DONE 2026-07-12, commit 8681e25d.)** R14.1 theory complete: +536 lines across
  theory/equations/algorithms/architecture — energy functional + Hirshman-Whitson moment method,
  parities/lasym, metric→|B| pipeline, force kernels + spectral condensation, 1D preconditioner
  derivation + NEW 2D block-preconditioner section, NESTOR Green's-function formulation,
  virtual-casing free bdy, full IFT/adjoint math with the O(1)-memory argument, device-policy section.
  Every :mod:/:func: ref verified; sphinx -W green. R14.2 reference already substantial; R14.3 below.
  **(R14.3 DONE 2026-07-11.)** `docs/tutorials.rst` rewritten from a "coming soon" stub into a real
  gallery: every R13 example (`literalinclude` so the page stays in sync with the tested code) grouped
  by theme — getting started (fixed run, plot+Boozer, JSON), profiles & finite-beta, hot restart,
  differentiation (implicit + free-bdy), free boundary (mgrid + beta scan), optimization (QA + the
  QH/QP/QI note). docs `-W` green. R14.1 theory (theory/equations/algorithms/architecture, ~1100 lines)
  and R14.2 reference (api/input/wout/cli) already substantial from prior lanes; remaining R14 polish is
  incremental, not a gap.
  1. **Theory & numerics, exhaustive** (`docs/theory/` split into pages, each equation linked to its
     implementing `core` function): ideal-MHD energy functional + Hirshman-Whitson moment method;
     flux coordinates + λ; Fourier representation + parities + lasym; **how |B| is computed** (metric
     → B^u/B^v → covariant B → |B|, from `core.fields`); the MHD **forces** (`core.forces`) and
     spectral condensation; **preconditioners** (1D radial derivation + tridiagonal solve; the 2D
     extension from R10.2); Richardson time-stepping + restart; multigrid + hot restart; **NESTOR**
     free-boundary vacuum (Green's function); the **implicit differentiation** adjoint (custom_vjp +
     preconditioned GMRES, with the basin/saddle finding from R1); the CLI-vs-jit lanes; device policy.
  2. **Reference**: API autodoc (all core modules), input reference (INDATA + JSON), wout reference
     (Appendix A rendered), glossary (VMEC2000↔vmec_jax names), CLI reference.
  3. **Tutorials = the examples** (R13): one docs page per example with rendered figures + expected
     output; the examples ARE the tutorials (docs currently reference tutorials that don't exist).
  Gate: docs `-W` green; a reader can follow B-field→forces→preconditioner→solve→differentiate→optimize
  entirely from the docs; every example has a tutorial page.

**R15. Free boundary to production parity + performance + differentiability (make it excellent AND
*(R15.1 DONE 2026-07-11, c83bb2a1: free boundary now CONVERGES to VMEC2000 parity — fixed a
double-nfp vacuum phase bug (boundary synthesis used xn=n·nfp against per-period zeta; the
geometric angle is phi=zeta·onp). input.cth_like_free_bdy: was stalling at NITER; now 574
iters to fsqr=9.9e-11, wb parity 2.1e-7 vs VMEC2000's 476-iter golden. Remaining: iters 574
vs 476 (~20% tail); R15.2 DONE (f197e144): vacuum fused into jitted JAX, 27 host↔device
round-trips/iter → ~0, warm 9.43→3.48 s (2.7×, now 4.5× VMEC2000), convergence bit-identical.
R15.3 DONE (4dcbbb54): differentiable free boundary via virtual_casing_jax — coil/extcur
gradients WITHOUT NESTOR-adjoint (B_out.n=0 + pressure balance as a differentiable objective);
FD-validated 2.2e-13..1.2e-10 vs central FD. **The whole free-boundary lane (converge/fast/
differentiable) is DONE.** Remaining: full single-stage needs an IFT wrap d(boundary)/d(coils);
R15.3
differentiation via virtual_casing_jax (R19) still to do; add cth golden+mgrid to the CI
bundle.)*
show it).** Current: CTH free-bdy stops at NITER (fsq~9e-2, unconverged), warm 17 s ≫ Fortran 6.6 s,
coil derivatives unsupported. Steps:
  1. **Converge as well as VMEC2000.** Diagnose why the free-bdy solve stalls (nvacskip cadence, ivac
     activation threshold, edge-force/preconditioner interaction at js=ns, delt policy) vs VMEC2000 on
     the same deck; raise NITER and match VMEC2000's converged fsq. Produce a **converged
     free-boundary golden fixture** (validate wout per-variable vs VMEC2000). This is the acceptance
     authority — no coil-derivative claims before it.
  2. **Fast.** Profile the NESTOR/vacuum solve (dominant cost); the dense scalar-potential solve and
     Green's-function assembly are the suspects. Target free-bdy warm within ~3× VMEC2000 (from 17 s).
  3. **Differentiable.** Extend the implicit residual to include the free-boundary/NESTOR contribution
     so coil-dof and pressure gradients flow through the converged free-boundary fixed point;
     FD-validate d(boundary|QS)/d(coil-dof) and d/d(extcur).
  4. **Show it.** README free-boundary parity + performance row (ns≥51); a
     `single_stage_free_boundary_opt.py` example (R13); the β-scan showcase (R13) with mgrid vs direct
     agreement. Gate: converged free-bdy wout parity vs VMEC2000; free-bdy warm within ~3× Fortran;
     free-bdy gradients FD-validated; examples + README updated.

**R16. Memory reduction (reason + act; the biggest quantitative gap).**
*(R16 FINDING 2026-07-10: the DFT-transform-tensor premise is REFUTED by profiling — those
are 0.017-2.1 MB, negligible. The peak (0.6 GB floor; 3.8 GB implicit gradient) is XLA COMPILE
working set, not data. remat/jax.checkpoint tested + REJECTED (3885 vs 3809 MB — nothing to
save). What worked: jit-factoring the implicit residual F + _field_chain → implicit gradient
3809→3045 MB (−20%) AND 40→31.6 s (−21%), bit-identical; jac_chunk_size='auto' default
(bounds GPU/large-dof runtime memory); donate CLI carry (neutral CPU). The ≥2× CPU gate is NOT
met because the bottleneck is the compiler; <1.5 GB needs a custom_vjp split of the monolithic
jacrev program (correctness risk) or a smaller XLA footprint. REFRAME R16: 'reduce the XLA
compile working set' — the real levers are jit-factoring + GPU chunking + persistent cache.)* Measured: solves use
0.6-1.5 GB (NuhrenbergZille 3.3 GB, free-bdy 2.6 GB) vs VMEC2000's 28-102 MB — **20-30×**; implicit
gradient 3.4 GB. This IS improvable — the causes are architectural, not fundamental:
  1. **Profile** peak device/host buffers with `jax.profiler.device_memory_profile()` +
     `memory_stats()` on a mid + large deck; attribute MB to: the batched-DFT transform matrices
     (dense `(nznt × mnmax)` per parity/derivative — the prime suspect; VMEC2000 uses O(N) DFT loops),
     the trajectory/history buffers, un-donated carry copies, and jit residual variants.
  2. **Act**, in impact order: donate solver-carry buffers in the CLI lane
     (`jax.jit(donate_argnums=...)`); free/rematerialize the large transform tensors instead of
     holding all parities/derivatives simultaneously (or use the FFT path where it wins); shrink the
     trajectory buffer (store only what prints needs); collapse redundant structural jit variants
     (padded shapes); for the implicit gradient, chunk the per-dof Jacobian (DESC's `jac_chunk_size`
     idea — see R17) so peak memory doesn't scale with dof count.
  3. Gate: **≥2× peak-memory reduction** on the benchmark median (target <~700 MB for mid decks,
     <1.5 GB for the largest), implicit-gradient peak <~1.5 GB, recorded in `benchmarks/baseline.json`
     and the README performance notes. Correctness (parity + gradient tests) unchanged.

**R17. Apply DESC ideas (deep-dive done 2026-07-10; https://github.com/PlasmaControl/DESC).** DESC is
a JAX Fourier-Zernike force-residual code — numerics don't transfer 1:1, but these architecture/UX
patterns do. Ordered by value, each cross-referenced into the lane it strengthens:

  *Memory (feeds R16):*
  1. **`jac_chunk_size` column-chunking of the optimization Jacobian** — DESC's headline memory knob:
     build the residual Jacobian in column blocks so peak memory = m0 + m1·chunk (time ≈ t0 +
     t1/chunk), `"auto"` picks the largest that fits. We chunk only coil eval today, NOT the objective
     Jacobian. Add a `jac_chunk_size` kwarg to `least_squares` (both FD and `jac="implicit"`), chunk
     the per-dof loop with `jax.lax.map(..., batch_size=chunk)`. THE fix for optimization memory.
  2. **`jax.checkpoint`/remat on the adjoint + field chain** — core has NO remat anywhere; wrapping
     `implicit._field_chain`/force-eval in `jax.checkpoint` recomputes in backward instead of storing,
     the direct lever on the 3.4 GB implicit-gradient backward.
  3. **Expose GPU knobs**: `XLA_PYTHON_CLIENT_MEM_FRACTION` (0.75→0.9), `XLA_PYTHON_CLIENT_ALLOCATOR=
     platform` for OOM debugging; surface in `doctor.py`. Verify our persistent cache is as aggressive
     as DESC (`jax_persistent_cache_min_compile_time_secs=0`) to help the cold small-deck target (R16).

  *Optimization depth (feeds R1):*
  4. **Block-solve all dof columns against ONE shared linearization** (block-GMRES / recycled Krylov
     subspace) instead of one preconditioned GMRES per dof — DESC's "factorize once, reuse" lesson;
     the most direct per-dof Jacobian-cost win, complements the CPU-pin fix (a37d0ec3).
  5. **Perturbation (analytic Newton) warm-start for trial solves** — seed each trial boundary with a
     first-order step `dx = −(∂F/∂x)^{-1}(∂F/∂c)dc` (we already have `∂F/∂x`, `∂F/∂c` VJPs in
     `implicit.py`) before iterating; cuts per-trial iterations → deeper QA/QH/QP/QI at fixed budget.
  6. **`bounds=(lo,hi)` inequality targets + generic `loss_function` (min/max/mean) on every term** —
     DESC's objective contract; removes weight-tuning guesswork (aspect∈(7,9), mirror∈(0.18,0.22))
     and unifies the `l_grad_b`/`mirror_ratio` bespoke reductions. Extend the `(fun,target,weight)`
     term to `(fun, target|bounds, weight, loss_function)`.
  7. **Richer objective library** into `core.optimize`, high-value first: `QuasisymmetryTripleProduct`
     (local `f_T`, no FSA — cheap complement to our ratio residual); **`EffectiveRipple` ε_eff**
     (1/ν neoclassical — we have the bounce primitives in `quasi_isodynamic`); `GammaC` (fast-ion);
     `Omnigenity`+`OmnigenousField` target (a cleaner QI formulation than our 4-term residual — study
     for QI depth); `BootstrapRedlConsistency`, `BallooningStability` (fuller stability vs Mercier);
     medium: elongation/curvature/BScaleLength/rotational-transform+shear profile targets.

  *Free-boundary differentiation (feeds R15):*
  8. **Virtual-casing `BoundaryError` as an ADDITIONAL differentiable free-boundary formulation** —
     DESC gets free-boundary gradients WITHOUT a NESTOR subsolve by making `B·n=0` and the pressure
     balance a differentiable objective via the virtual-casing principle. This sidesteps
     differentiating our NESTOR fixed point and is the cleanest route to coil→boundary→QS gradients.
     Keep NESTOR for the forward VMEC2000-parity solve; add virtual casing for the differentiable path.

  *UX / capability (feeds R13/R14/R1):*
  9. **Near-axis (pyQSC/pyQIC) seeding** — `from_near_axis` builds a physically-good `VmecInput`
     boundary from a QSC/QIC solution instead of a circular seed → better QA/QH/QI starts (our input
     decks even note "B0 not yet implemented"). Direct optimization-depth lever.
  10. **`Equilibrium` save/load (HDF5) + `EquilibriaFamily`** — return the staged `max_mode`/multigrid
      sequence as an inspectable family; make campaigns resumable (feeds the save/load UX).
  11. **Plot helpers to mirror**: `plot_qs_error` (QS `f_B/f_C/f_T` vs flux), `plot_comparison`
      (overlay before/after optimization surfaces — the README optimization panels of R11.2),
      `plot_boozer_surface`/`plot_boozer_modes` (LCFS Boozer, feeds R11.4), `plot_coefficients`
      (spectral-convergence diagnostic we lack). Add to `core.plotting`.
  12. **Notebook tutorials + output-analysis notebook** — DESC ships 7 rendered notebooks; convert the
      flagship examples to narrated notebooks with inline plots (feeds R13/R14.3); add a "how to read
      a wout / compute QS error / plot Boozer" analysis tutorial.
  13. **CI: split fast-unit vs slow-regression workflows** (DESC pattern) — we already shard; formalize
      the golden/regression split and add the I/O-format reference doc (every INDATA + wout var, §827).

  Gate: items 1-2 (chunk+remat) land first (biggest measurable win, feed R16's ≥2× gate); each other
  adopted idea lands as a tested change in its cross-referenced lane. Note: our matrix-free O(1)-memory
  adjoint is already BETTER than DESC's for a single scalar gradient — keep it; borrow the chunking,
  remat, warm-start, objectives, virtual-casing, and UX. **Route the solver-generic ones (chunk,
  remat, block/recycled Krylov, warm-start) through SOLVAX (R18); the physics ones (virtual casing,
  near-axis) through the uwplasma packages (R19).**

**R18. SOLVAX integration — slim vmec_jax, share solver infra with the uwplasma ecosystem.**
*(STATUS 2026-07-10: R18a + R18b DONE.* SOLVAX PR #1 merged + released v0.2.0 to PyPI
(backend-aware tridiagonal_solve + chunked-autodiff, example-per-capability + full docs).
vmec_jax imports them (d6b4c938): preconditioner tridiagonal, adjoint GMRES, jac_chunk_size
all via solvax; preconditioner + gradient tests bit-identical; CI green incl. 95% gate; core
−56 net lines now. Remaining: the big reduction with the 2D preconditioner on
solvax.block_thomas_truncated (R10.2).)*
`uwplasma/SOLVAX` (local `/Users/rogerio/local/SOLVAX`, v0.1.0, "differentiable structured linear
solvers, preconditioners and matrix-free methods in JAX", built on lineax) ALREADY ships: `banded`
LU (+periodic), `krylov` (`gmres`, `gcrot`=recycled Krylov), `implicit` (`linear_solve`,
`root_solve` = implicit-function-theorem custom_vjp), `direct` (`block_thomas`,
`block_thomas_truncated` = block-tridiagonal Schur elimination), `precond` (jacobi/block_jacobi/
coarse_operator/line_smoother/p_multigrid/mixed_precision/kronecker), `refine` (iterative refinement),
`operators` (MatrixFree/Sum/Kronecker). This overlaps heavily with vmec_jax's solver needs, so the
work is **bidirectional** and the net effect is a SLIMMER, better-integrated vmec_jax.

  *18a. Assess + migrate FROM vmec_jax TO SOLVAX (make the SOLVAX PR).* Read SOLVAX `src/solvax/*`
  and vmec_jax `core/preconditioner.py` + `core/implicit.py`; migrate ONLY genuinely-generic pieces
  SOLVAX lacks, as a PR to `uwplasma/SOLVAX` (branch, as rogeriojorge):
    - **Backend-aware batched tridiagonal solve**: our `tridiagonal_solve` (CPU vectorized Thomas +
      GPU `jax.lax.linalg.tridiagonal_solve`/cuSPARSE, `platform_dependent` selection, batched over
      RHS/columns). SOLVAX `banded` is general banded LU; add this specialized fast tridiagonal path
      if absent. (Not the VMEC-specific precondn/lamcal/scalfor — those stay in vmec_jax.)
    - **`jac_chunk_size` chunked-Jacobian utility** (R17.1): a generic `jax.lax.map`-based column-
      chunked forward/reverse Jacobian builder with `"auto"` sizing — a solver/AD utility that belongs
      in SOLVAX, reused by vmec_jax and sfincs_jax.
    - **Perturbation/Newton-predictor warm-start** (R17.5) if it generalizes cleanly on top of
      SOLVAX's recycled-Krylov continuation.
    - The PR MUST also add (user requirement): **one example per SOLVAX capability** (including the
      new ones and the pre-existing banded/krylov/gcrot/root_solve/block_thomas/precond/refine), and
      **comprehensive up-to-date docs** explaining every method, its equations, the source, the
      architecture, inputs/outputs, and use cases (mirror the vmec_jax docs bar). Gate: SOLVAX CI
      green, coverage kept, PR opened with examples + docs.
  *18b. Import FROM SOLVAX INTO vmec_jax (slim the core).* After (or alongside) 18a, refactor:
    - `core/implicit.py` adjoint → use SOLVAX `root_solve`/`linear_solve` + `krylov.gmres`/`gcrot`
      (gcrot gives the recycled/block Krylov of R17.4 for free) instead of the hand-rolled custom_vjp
      + `jax.scipy.sparse.linalg`. Keep our preconditioner as the Krylov `M`.
    - The 1D radial preconditioner tridiagonal solve in `core/preconditioner.py` → SOLVAX's
      tridiagonal/banded solve (the migrated one), deleting the duplicated Thomas kernel.
    - The **2D block preconditioner** (R10.2) → build on SOLVAX `direct.block_thomas_truncated`
      (truncated block-tridiagonal storage — exactly VMEC's BCYCLIC analogue) + `krylov` + `precond`,
      NOT from scratch.
    - Add `solvax` to vmec_jax runtime deps (unpinned). Gate: parity + gradient tests unchanged;
      vmec_jax core LOC drops (target −1 to −2k lines); one place to maintain the solver math.
  Net: vmec_jax gets slimmer and faster, SOLVAX gets battle-tested methods + examples + docs, and the
  uwplasma ecosystem (sfincs_jax, vmec_jax) shares one solver layer.

**R19. Physics-package reuse (uwplasma) — don't re-implement.**
  - **Virtual casing (R17.8 differentiable free boundary)**: reuse `uwplasma/virtual_casing_jax`
    (local; JAX virtual-casing with examples incl. `simsopt_stage_two_optimization_finite_beta.py`,
    `w7x_gradB.py`, `vmec_extender_python_api.py`) instead of re-implementing the virtual-casing
    `B·n` differentiable free-boundary formulation. Wire it as the differentiable free-boundary path
    (keep NESTOR for the VMEC2000-parity forward solve).
  - **Near-axis seeding (R17.9)**: use `uwplasma/pyQSC_JAX` (QA/QH near-axis) and
    `github.com/rogeriojorge/pyQIC` (local `pyQIC`, import `qic`, for QI) to build physically-good
    `VmecInput` boundary seeds via `from_near_axis`, replacing the circular-torus seed for deeper
    QA/QH/QI optimization. Both differentiable/JAX where possible so seeds flow into single-stage.
  Gate: free-boundary gradients via virtual_casing_jax FD-validated; near-axis-seeded QA/QH/QI reach
  deeper precision than the circular seed (feeds R1); examples added (R13).

**R20. Showcase everything new (README + docs + examples) — the differentiators.**
  **(R20 DONE 2026-07-11, commit caf6166c, with R11.)** README now carries `readme_precond.png`
  (2D-vs-1D iteration counts, 2.5–11x on stiff decks), a **DESC comparison section** (VMEC2000
  iteration-parity + standard wout, INDATA/JSON drop-in, NESTOR *and* virtual-casing free boundary,
  lasym, VMEC2000-format prints; honest about DESC's Zernike-at-low-res and objective-library edge),
  and a capability matrix enumerating every beyond-VMEC2000 feature (implicit diff, direct coils, 2D
  precond, chunked memory, virtual-casing free-bdy, near-axis seeding, SOLVAX-shared solvers). Each
  user-facing capability has an example (R13) + tutorial (R14). Residual follow-up: a couple of
  examples/tutorials still to broaden under R13/R14, tracked there — the showcase evidence itself is
  shipped.
  - **2D preconditioner advantages** (once R10.2/R18b land): README + docs figure — iteration-count
    and wall-time reduction vs the 1D preconditioner on a stiff case; explain the method (docs R14).
  - **DESC comparison where vmec_jax WINS** (README table + notes), beyond the O(1)-memory adjoint:
    exact VMEC2000 iteration-for-iteration parity + standard `wout` (DESC is a different equilibrium,
    not VMEC-parity); direct INDATA/JSON drop-in; free boundary with NESTOR *and* virtual casing;
    lasym; VMEC2000-format prints; and any measured speed/memory wins from R16. Be honest where DESC
    wins (Zernike accuracy at low resolution, mature objective library — which R17.7 narrows).
  - **Mention every new capability** (DESC-derived AND our own beyond-VMEC2000: implicit diff, direct
    coils, 2D preconditioner, chunked/remat memory, virtual-casing free-bdy, near-axis seeding, the
    SOLVAX-shared solvers) in BOTH README and docs, and for each user-facing one ship an example
    (R13) + a tutorial page (R14). Gate: README/docs enumerate the differentiators with evidence;
    each important new capability has an example.

**R21. Rename everything `vmec_jax` → VMEX (user 2026-07-10).**
  **ORDERING (user 2026-07-11): R21 is now the ABSOLUTE LAST step — do everything else in the plan first
  (R12 tests rename, R14 docs, R9 v0.1.0 release), THEN the VMEX rename as a standalone cutover.** R21 is
  decoupled from R12 (do R12 now, standalone) and no longer "right before R9" — v0.1.0 releases as
  `vmec-jax`, and VMEX becomes a later renamed version with a `vmec_jax` compatibility shim. Names: GitHub repo
`uwplasma/vmec_jax` → `uwplasma/VMEX`; Python import package `vmec_jax` → **`vmex`** (lowercase,
PEP 8; `import vmex`); PyPI distribution `vmec-jax` → **`vmex`** (verified AVAILABLE on PyPI
2026-07-10, HTTP 404); CLI command → **`vmex`** (keep `vmec` as an alias — do NOT rename the output
`wout_*.nc`/`boozmn_*.nc` files, those are the VMEC community conventions, not our package name).
Scope measured: 96 files / 444 Python occurrences. Do it as ONE ATOMIC sweep (a partial rename breaks
everything), paired with R12 (`tests/core_new/` → `tests/`):
  1. `git mv vmec_jax vmex`; global identifier replace `vmec_jax` → `vmex` across .py/.rst/.md/.toml/
     .yml (mind word boundaries: `vmec_jax` the package vs `vmec2000`/`vmec_input`/wout var names that
     must NOT change; and the display string "vmec-jax"/"vmec_jax" in prose → "VMEX"). Update
     `pyproject.toml` name=`vmex`, `[project.scripts] vmex = "vmex.core.cli:main"` (+ `vmec` alias),
     all `[project.urls]` to the VMEX repo, package-data paths. Update `.github/workflows/*.yml`
     (test paths, the golden fetch, size check), `docs/conf.py` + readthedocs slug, README badges
     (PyPI/docs/CI URLs → vmex / VMEX), `CITATION.cff`, `.readthedocs.yaml`.
  2. Ship a thin **`vmec_jax` compatibility shim** for one release: a stub package that
     `from vmex import *` and emits a `DeprecationWarning` (pre-1.0 courtesy so existing imports don't
     hard-break); document the deprecation. (Optional — a clean break is acceptable at v0.0.x, but the
     shim is user-friendly.)
  3. GitHub repo rename (auto-creates redirects); update the local `git remote`; re-point the
     readthedocs project and the conda-forge feedstock (if it exists) to `vmex`; keep the old
     `vmec-jax` PyPI project with a final `0.0.x` release whose long-description points to `vmex`.
  4. NO-BUGS GATE (the user's explicit requirement): after the sweep — `pip install -e .` resolves;
     `python -c "import vmex; print(vmex.__version__)"`; `vmex --test` + `vmex input.solovev` +
     `--plot` + `--booz` all work; FULL test suite green (rename any `tests/` import of the package);
     docs `-W` green; CI green incl. the 95% coverage gate; ruff clean; and a grep confirms **zero
     stray `vmec_jax` identifiers** remain except the intentional compat shim. Verify on a fresh clone.
  Gate: fresh clone installs as `vmex`, CLI/docs/CI all green, no stray identifiers, PyPI `vmex`
  published; then proceed to R9 release under the VMEX name.

**R24. Full production profiling (CPU + GPU) + runtime/memory/simplicity pass (user 2026-07-12).**
  Harness: `benchmarks/profile_production.py` — five production workflows (fixed ns=201, multigrid
  51/101/201, NESTOR free boundary, implicit value_and_grad, least_squares opt step), each reporting
  cold (compile) vs warm wall, iterations, ms/iter, peak RSS, and GPU device memory. Run with
  `JAX_PLATFORMS=cpu` for the CPU profile; plain on a GPU box for the GPU profile.
  **Finding #1 (2026-07-12, HIGH):** on a GPU box the QA/QH/QP optimization examples ran the hot loop
  on the GPU by default and took HOURS per stage vs MINUTES pinned to `JAX_PLATFORMS=cpu` (office 2x
  A4000: capped QH still in stage 2 after 100+ min on GPU; QA stage 1 in ~minutes on CPU). The
  forward-solve policy (`device.recommended_device`, 100k iteration-work threshold) and the implicit
  pin (`resolve_implicit_device` → CPU) exist, but something in the `least_squares(jac="implicit")`
  hot path still lands on the accelerator — diagnose with the GPU-vs-CPU profile pair and either fix
  the placement or make the optimizer pin its whole session to the recommended backend. Also: JAX had
  preallocated 13.3 GB VRAM (default 75%) — consider XLA_PYTHON_CLIENT_PREALLOCATE=false guidance.
  **Finding #2:** `opt_step` (2-nfev max_mode-1 least_squares, minimal_seed_nfp2) warm ~63 s / peak
  RSS ~6 GB on a contended local CPU — the heaviest per-call production path; profile where the time
  goes (per-dof implicit JVP solves vs forward solves vs trf overhead) and cut it.
  **CPU profile (local M-series, idle, post-R25.1, 2026-07-12):** fixed ns=201 warm 5.5 s
  (4.3 ms/iter), multigrid 7.8 s, implicit_grad warm 17.3 s (cold 59.9 — compile is 42 s, an R26c
  target), opt_step warm 88.8 s (the R25 baseline number on idle hardware). **CPU beats the A4000 GPU
  on every case even at ns=201** (GPU: 6.9/9.3/27.8/151 s) — on fast desktop CPUs the accelerator adds
  nothing at production sizes; the device policy's GPU wins were vs the office box's slower cores.
  **Measured (office 2x A4000, CPU contended by QA/QP):** before-fix GPU warm — fixed ns=201 6.9 s
  (5.4 ms/iter), multigrid 9.3 s, free-bdy 8.0 s (13.9 ms/iter) → forward solves ARE GPU-competitive at
  ns=201; implicit_grad 57 s, opt_step 151 s → gradients pathological on GPU. **After the params-pin fix
  (9f239620): implicit_grad 57→27.8 s (2×+, on contended cores; more when idle). opt_step unchanged
  (152 s) — its dof vector was already pinned; the remaining cost is ALGORITHMIC:** scipy trf's fun(x0)
  + jac(x0) each trigger a separate solve_implicit at the same x, and least_squares re-solves the final
  equilibrium via solve_equilibrium for the .equilibrium attribute (no hot restart). Next levers: share
  the frozen solve between fun/jac at the same x (hot-restart makes the 2nd cheap but not free), reuse
  the last trial state for result.equilibrium, and re-measure. CPU profile still pending (local bench
  occupies the box).

**R26. FINAL PRE-VMEX SWEEP (user 2026-07-12; the last content pass — after R24/R25 conclude, before
R9 release and the VMEX rename R21).** Ten items (+k added 2026-07-12):
  k. **Single-stage ESS examples (user, DO NEXT):** two example scripts — one QA, one QI — that do
     NOT use the max_mode continuation for-loop: a SINGLE least_squares call directly at large
     max_mode (>=5) with Exponential Spectral Scaling as the trust-region x_scale (use_ess=True /
     _ess_scale — higher-|m|,|n| dofs get exponentially smaller trust radii, which is what makes the
     ladder unnecessary). Simsopt-style, CI-smoke-tested, indexed in examples/README.md; QI version
     uses the new traceable omnigenity residual (h2) once landed. Compare achieved QS/QI and wall vs
     the laddered examples in the docstrings.
  a. **Trim + simplify the code** — one more dead-code/duplication/altitude sweep over vmec_jax/core.
  b. **Port more functionality to SOLVAX** — **(MOSTLY DONE 2026-07-12.)** Audit: the generic
     linear-algebra IS already in SOLVAX and imported by vmec_jax — block_thomas_factor/solve,
     chunk_map, auto_chunk_size, chunked_jacfwd/rev, gmres, gcrot (no local copies remain; the R25.2
     block Jacobian and R25.3 recycle both use solvax). The only vmec-specific glue kept local is
     implicit._adjoint_solve/_recycled_solve (they wrap solvax.gmres/gcrot with VMEC's residual
     operator — correctly not generic). REMAINING: the harmonic-Ritz GCRO-DR upgrade INSIDE SOLVAX
     (solvax.gcrot's FIFO recycle space poisons reuse — the R25.3 negative result); a focused SOLVAX
     numerics enhancement that would unlock recycle=True in vmec_jax. Optional (block Jacobian already
     gives 33x).
  c. **Performance:** (i) COLD STARTS — **(DONE 2026-07-12.)** The persistent XLA compile cache was
     enabled but had NO default directory on CPU (accelerator-only gate), so CPU CLI/API runs
     recompiled every process. Made it default-on for every backend (_compat._default_compilation_cache_dir);
     the machine fingerprint already hashes CPU model+flags (AVX2/AVX512; +macOS sysctl brand added) so
     heterogeneous shared-FS machines never collide. Measured: li383 CLI 4.48 s -> 1.34 s on the 2nd run
     (3.3x), zero user config; opt out with VMEC_JAX_COMPILATION_CACHE=disabled. (ii) FREE BOUNDARY **(DONE)** —
     measured: the mgrid and external_field (direct-coil) lanes are identical (both 1.32 s warm / 574
     iters on cth ns=15; mgrid file-read overhead ~0), and NESTOR was already fused into jitted JAX in
     R15.2 (9.43->3.48 s). No further tune warranted.
  d. **Faster optimizations/gradients** — continue past the R25 gate (block-tridiag amortization,
     recycling, perturbation warm starts all landed and measured together).
  e. **Memory reduction with DEFAULT controls** — good defaults, no advanced user knobs required.
     **(R26e ADDRESSED 2026-07-12.)** Defaults already do the work: jac_chunk_size="auto" caps the
     Jacobian column memory, the R25.1 converged-state memo cut opt_step peak RSS 6.0->3.5 GB, and
     peak memory is dominated by the transient XLA COMPILE working set (R16 finding), not data. The
     block-tridiagonal Jacobian (now the jac_solver default) trades ~40% more compile working set
     (opt_step 3.5->4.9 GB) for its 33x speed — the right default; a memory-constrained user sets
     jac_solver="gmres" for the lower-memory path (documented). No further default change warranted.
  f. **README example: free boundary from ESSOS coils** — Landreman-Paul QA with increasing pressure,
     vol-avg beta = 0%, 1%, 2%, 3% (needs a coil set reproducing the LP QA boundary).
  g. **README: QA (nfp 2) + QH (nfp 4) optimization with SELF-CONSISTENT BOOTSTRAP CURRENT**,
     reproducing arXiv:2205.02914 (Landreman-Buller-Drevlak) against the Zenodo data in
     /Users/rogerio/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current
     (calculations/ + configurations/). Requires the **Redl (2021) bootstrap formula in vmec_jax,
     DIFFERENTIABLE**, + a loop iterating the current profile to self-consistency with the equilibrium.
  h. **Literature/code deep dive — DONE + USER-APPROVED SCOPE (2026-07-12).** Proposal in
     notes_r26h_research_proposal.md. User approved for implementation (in this order, respecting
     file-conflict windows):
       h1. **Stability objectives**: infinite-n ballooning (JAX COBRA port, batched 1D eigh; blueprint
           arXiv:2302.07673) + differentiable Mercier/magnetic-well optimization terms. NEW module
           core/stability.py + tests; no conflicts — START FIRST.
       h2. **General omnigenity residual** (Dudt arXiv:2305.08026) + constructed-QI targets (Goodman
           arXiv:2211.09829), reusing the in-tree Boozer transform — fixes the documented "QI not
           precise" weakness. Touches optimize.py — AFTER R25.3 merges.
       h3. **Single-stage plasma–coil optimization with exact gradients**, using ESSOS from
           github.com/uwplasma/ESSOS for the coil side (differentiable Biot-Savart from ESSOS coils,
           consumed as an mgrid or xyz->B callable) + virtual casing + implicit diff composed end-to-end.
       h4. **Turbulence proxies from spectrax-gk (github.com/uwplasma/spectrax-gk)** — its available
           LINEAR, NONLINEAR and QUASILINEAR proxies wired as optimization objectives on vmec_jax
           geometry.
     NOT selected: bounce-averaged eps_eff/Gamma_c module (deferred).
  i. **Docs upgrade** — deeper algorithms/performance/differentiability/equations explanations, more
     engaging with plots; better tutorials/examples/use cases; user-friendly.
  j. **Release hygiene:** repo <= 10 MB; coverage >= 95% with SIMPLE, CONCISE tests in a SMALL number
     of files covering every functionality; future-proof; literature-anchored; real physics + numerical
     testing; a VMEC2000 parity/accuracy check that does NOT require storing large wout files (scalar
     digests of golden quantities); then release v0.1.0 (R9) with all of it.

**R25. Optimization wall-time: multi-hour → under one hour (user 2026-07-12; DO BEFORE the VMEX
rename R21).** Same modes and resolution — the win must come from a more efficient/performant
GRADIENT, not from shrinking the problem. Ground it in the literature (papers, preprints, reports,
docs and source of DESC / SIMSOPT / VMEC++ / adjoint-stellarator work) and pick the best fast,
accurate gradient strategy. Candidate directions to evaluate against the measured R24 profile
(opt_step cost is algorithmic — one solve_implicit per fun(x) AND per jac(x), one GMRES per dof per
Jacobian, redundant final solve_equilibrium):
  1. Reuse/share solves: one frozen solve per trust-region iterate serving both fun and jac; reuse
     the last trial state for result.equilibrium; deeper hot restarts across trials.
  2. Krylov recycling across Jacobian evals (SOLVAX gcrot — nearby systems share spectrum) and/or
     block-GMRES over all dof right-hand-sides at once instead of one GMRES per dof.
  3. Quasi-Newton Jacobian recycling: Broyden secant updates between exact implicit Jacobians
     (recompute exactly only every k-th iterate or on trust-region step rejection).
  4. trf internals: LSMR with a Jacobian LinearOperator built from JVP/VJP closures (no dense J
     assembly), scipy tr_solver options.
  5. Literature scan: DESC chunked-jacfwd + Levenberg-Marquardt practice, SIMSOPT MPI-FD baselines,
     Landreman-Paul analytic shape-gradient adjoints, one-shot (SAND) simultaneous optimization,
     VMEC++ perturbation/hot-restart papers.
  **RESEARCHED (2026-07-12, notes_r25_gradient_research.md — DESC source-verified, papers cited).
  Ranked implementation order:**
  1. **Memoize the converged state + perturbation warm start** (DESC `_update_equilibrium`/`f_where_x`
     pattern): jac(x) never re-solves at the x fun(x) just converged (~20 lines, zero accuracy risk);
     seed trial solves with the first-order perturbation z-(dF/dz)^-1(dF/dp)dp (arXiv:2203.15927).
     ~1.5-2x on the solve phase. DO FIRST.
  2. **Amortized block-tridiagonal factorization of dF/dz**: assemble the radial blocks with 3-colored
     jax.jvp probes (cost independent of dof count), solvax.block_thomas factor once, backsolve all
     24-120 dof RHS, keep a 2-3-iter GMRES corrector for exactness. 2-5x on the jac phase; also reuses
     as forward Newton preconditioner + perturbation seed.
  3. **GCROT recycle-space carry — LANDED AS OPT-IN, NEGATIVE RESULT (f1fdd509).** Plumbing exact
     (lax.scan carry, vmap-shared within chunks) but solvax v0.1's FIFO cycle-correction recycle space
     POISONS later solves (1.7-3.4x more iterations, Jacobian drift 1.6e-2): needs harmonic-Ritz
     GCRO-DR in SOLVAX (→ R26b item). Default recycle=False keeps the exact path; flip when SOLVAX
     upgrades.
  Runner-up Broyden secant (only option that weakens the per-iterate exactness); long-term: one-shot/
  SAND (~4x one solve for a whole optimization, needs replacing the scipy driver).
  **GATE PASSED (2026-07-12): the full max_mode 1->5 QA campaign on the office CPU box completes in
  1532 s (25.5 min, target <3600 s) with memo + block-tridiagonal Jacobian + CPU pin, reaching QS
  3.73e-7 (440x deeper than the prior README deck; iota 0.42, aspect 6.000). QP (the stall-prone
  class) still running at its own pace. Gradient FD validation and CI gradient shard green throughout.**

**R22. README/showcase refinement round 2 (user 2026-07-11; DO these before R21/R9; VMEX rename deferred
as a longer refactor).**
  **(R22 DONE 2026-07-11 except the ns=201 figure regen, which is folded into R9.)** Commits caf…→3e8296b2:
  (1) install now reads PyPI-recommended-OR-conda; (2) all `|B|` plots are line contours (plot_modB,
  Boozer, showcase, optimization); (3) QI reworked into its own `readme_qi.png` at nfp 1-4 with the
  OMNIGENITY residual (1.3e-2→3.2e-3) + 3D `|B|`, and `readme_optimization.png` split to QA/QH/QP with
  3D `|B|` and the PRECISE R1 QH deck (QS 5.83e-5, straight diagonal Boozer contours; QA 1.6e-4; QP
  0.094 — honestly the hardest class); (4) DESC comparison rewritten as a concise where-each-wins table;
  (5) warm/cold/GPU/memory bullets cut to 1-2 sentences; (6) benchmark harness RAMP_NS=201 (figure regen
  at R9 — needs idle box + vmecpp). NOTE learned: the QH/QP re-optimization is genuinely multi-hour even
  capped (implicit-gradient per-dof GMRES); reused the saved R1 campaign decks from office instead.
  1. **Install: PyPI (recommended) OR conda-forge.** Make the README install section state clearly that
     it is *either* `pip install vmex/vmec-jax` (recommended) *or* `conda install -c conda-forge`, not
     both — one path, recommended = PyPI.
  2. **`|B|` as line contours, not filled.** All `|B|` plots (`plot_modB`, `plot_boozmn_modB`, the
     showcase Boozer `|B|`) should be line contours (`ax.contour`), not filled `contourf`. Regenerate the
     affected README/docs figures.
  3. **QI optimization row reworked.** In `readme_optimization.png`: QI gets its **own row** with
     **nfp = 1, 2, 3, 4** examples, displays the achieved **QI residual** (omnigenity), *not* the QS
     residual, at **higher `max_mode`**; add **3D geometry plots with `|B|` on the surface** for the
     optimized configs. QA/QH/QP keep the QS residual. Regenerate the figure + the example/deck set.
  4. **DESC comparison rewritten for clarity.** Replace the prose with a concise
     advantages/disadvantages presentation — the same clear style as the vmec-jax vs VMEC2000 vs VMEC++
     feature matrix: what vmec-jax does better, what DESC does better, in short bullets/table cells.
  5. **Concise performance descriptions.** The warm / cold / GPU / memory bullets are too long — tighten
     each to one or two clear sentences.
  6. **Full-solve benchmarks at ns=201, not ns=51.** The full-equilibrium wall-clock table/figure should
     run at **ns = 201** (not 51) so the JIT compile time is small relative to run time (a fairer warm
     comparison). Regenerate `readme_runtime_compare.png` + reconcile the numbers. (Watch memory/wall on
     the heaviest decks; ns=201 is much larger — keep the suite honest and machine-load-caveated per R11.)
     **(R22.6 CONFIG DONE 2026-07-11: `run_baseline.py` `RAMP_NS=201`, multigrid ladders extended to
     201, docstring updated. Probe confirms the point — cth_like ns=201 warm=3.8 s vs cold=10.5 s, so the
     solve dominates. Figure REGEN DEFERRED TO R9: a clean, complete ns=201 figure needs (a) an idle
     machine — the box is currently contended by the user's sfincs at ~300% CPU, and the heaviest deck
     (NuhrenbergZille ns=201) is minutes-long — and (b) `vmecpp`, which is not importable in the current
     venv (the figure would drop its column). R9 finalizes benchmarks on a clean machine per R11; ns=201
     is now the harness default it will use.)**

**R23. Decide whether the 2D block preconditioner should be the DEFAULT (user 2026-07-11: "if it is so
good, why isn't it the default?").** Measure 2D-vs-1D on representative decks: (a) **accuracy** — the
converged `wout` must match the 1D/VMEC2000 golden to the same tolerance (the 2D precond changes the
*path*, not the fixed point; verify it doesn't shift the answer); (b) **wall-clock** warm speed;
(c) **peak memory**; (d) **iteration counts** (already known 2.5–11× fewer on stiff decks). Decision
rule: if 2D is *at least as accurate* AND *faster in wall-clock* (not just iterations) AND *not
materially heavier in memory* across the board (not only stiff decks), make it the **default**;
otherwise keep it opt-in with a **documented rationale** (e.g. per-iteration cost or memory on easy
decks). Update the README precond section + the `prec2d`/`precon_type` default in `solver.py` accordingly.
  **(R23 DONE 2026-07-11 — DECISION: keep opt-in.)** Measured 1D vs 2D (ns=51, warm, ftol=1e-11):
  accuracy IDENTICAL (wb parity 1e-9…1e-11 — 2D changes the path, not the fixed point); iterations
  2.6–5.4× fewer everywhere; but **wall-clock 0.55–1.16×** — a wash-to-slower (solovev 1.16×, cth_like
  1.14×, circular_tokamak **0.55× = ~2× slower**, aspect-100 stiff **0.97× tie**): the per-iteration
  GMRES+HVP cost offsets the iteration savings even on the stiffest deck. Peak RSS ~30% higher
  (cth_like 761→987 MB, the GMRES/HVP compile graph). So default 1D unchanged; 2D stays opt-in for
  iteration-count-bound / stalling cases. README precond section rewritten with this honest rationale;
  `solver.py` default (`prec2d=None`) left as-is (correct).

---

## 1. Ground truth — current state (audited 2026-07-08)

Facts established by direct audit; the executor should trust these and not re-derive them.

### 1.1 Repository

- 57.38 MiB packed git history; 89 MB working tree; 955 tracked files; single `main` plus 4 remote
  branches. Top history bloat: ~30 historical revisions of a ~1.7 MB `plan_differentiability.md`
  (now a 39-line stub — the blobs live only in history), multi-MB PNGs
  (`readme_best_optimization_qh.png` 2.0 MB, `minimal_seed_showcase_state_panel.png` 1.7 MB), and
  the 1.5 MB `vmec_jax_plan/plan_research_grade_performance_differentiability.md` at tip.
- **Claude in contributors:** authorship is clean (`git shortlog -sne --all` shows only Rogerio
  Jorge ×2 emails + Matthew Feickert ×1). Claude appears **only via 70 commit-message trailers**
  `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`. Fix = history rewrite of commit
  *messages* (strip the trailer), not a mailmap.
- **Branches:** `origin/codex/differentiability-refactor-plan`,
  `origin/feature/freeb-essos-coil-single-stage`, `origin/phase2/freeb-adjoint-validation` are all
  **0 commits ahead** of main → delete after the rewrite. `origin/codex/mirror-geometry` is **511
  commits ahead** with genuinely unmerged experiments: a *native state block preconditioner*,
  *native spline matrix-free loop*, and *square hybrid solver method lanes*. Phase 0 skims it for
  ideas (esp. for the 2D preconditioner and mirror/spline work), records conclusions in NOTES.md,
  then it is deleted too — the rewrite invalidates its ancestry anyway.

### 1.2 Library (`vmec_jax/`, 229 files, ~123k lines, 49 root modules)

- **Core physics exists and is comprehensive**: Fourier transforms (`kernels/tomnsp.py` 1513,
  `fourier.py`), geometry/metrics (`kernels/bcovar.py` 1434, `kernels/jacobian.py`), forces
  (`kernels/forces.py` 2010, `kernels/residue.py`), 1D preconditioner (`preconditioner_1d_jax.py`
  2247), fixed-boundary loop (`solvers/fixed_boundary/residual/` — iteration.py 2957, update.py
  2350, runtime.py 1957, policy.py 1708 — plus a parallel `scan/` tree), multigrid (`multigrid.py`
  331, a port of VMEC2000 `interp.f`), JAX NESTOR (`solvers/free_boundary/jax_nestor_operator.py`
  1671), mgrid IO + JAX interpolation, direct-coil Biot–Savart (`external_fields/coils_jax.py`,
  `essos_adapter.py`, routed via `MGRID_FILE='DIRECT_COILS'`), wout writer (`io/wout_files/` tree
  incl. jxbforce, mercier, bsubs, nyquist), CLI with `--plot/--booz/--doctor/--test`, and
  booz_xform_jax as a hard dependency.
- **The bloat** is: (a) duplicated NumPy/JAX kernel pairs (`kernels/numpy_forces.py` 1102 vs
  `kernels/forces.py`; `preconditioner_1d.py` vs `preconditioner_1d_jax.py`), (b) facade/compat
  shims (`solve.py`, `_compat.py`, `_solve_runtime.py`), (c) the entire "branch-local adjoint"
  apparatus (`discrete_adjoint.py` 2002, `implicit.py` 1865, `solvers/*/adjoint/` ≈ 30 files with
  fingerprints, replay tapes, gate reports), (d) overlapping driver layers (`driver.py`,
  `drivers/{staging,policy,lifecycle,...}.py`, `solvers/fixed_boundary/{residual,scan}/`), and
  (e) an oversized optimization/QI workflow layer (`optimization.py` 1993,
  `optimization_workflow.py` 1881, `quasi_isodynamic/` ≈ 6k lines, `solvers/free_boundary/
  coil_optimization.py` 2646).
- ~15 files violate the project's own 1500-line rule. mypy is configured with most error codes
  disabled and per-module `ignore_errors` on the core physics; ruff ignores E402/F821/F841/E501.
  The refactor must make these crutches unnecessary.

### 1.3 Tests, examples, docs

- `tests/`: 296 files, ~138k lines, codecov project gate 95%. Includes obvious coverage-padding
  ("`test_solve_wave3..10_coverage`" files) and 3–4k-line monsters
  (`test_direct_coil_finite_pressure_sensitivity.py` 4478). Only one tiny committed fixture
  (1.1 KB); large assets already come via `tools/fetch_assets.py` — keep that pattern.
- `examples/`: QA/QH/QP scripts are ~220–250 lines (close to acceptable); **QI is the mess**:
  `QI_optimization.py` (499 lines, ~49 helper/stage references) + per-NFP variants (256 lines each,
  3-phase circular→QP-basin→QI pipelines) + helper modules `qi_optimization_cases.py` (1066),
  `qi_staged_runner.py` (805). Free-boundary `free_boundary_QS_coil_optimization.py` is 1284 lines.
  `examples/data/` and `examples/data/single_grid/` duplicate ~40 input decks.
- `docs/`: Sphinx+furo, ~35 rst pages; theory pages exist (`theory.rst`, `equations.rst`,
  `algorithms.rst`, `discrete_adjoint.rst`) but are interleaved with internal plan/lane pages
  (`aggressive_performance_plan.rst`, `accelerated_merge_readiness.rst`,
  `optimization_sweep_results.rst`, `piecewise_omnigenous_plan.rst`) that must go.

### 1.4 Performance (README 2026-07-06 snapshot)

37 normalized single-grid rows (`NS_ARRAY=151`, `FTOL=1e-14`): **warm** vmec_jax beats VMEC2000 on
33/37; **cold** on only 14/37 (Python/JAX/XLA setup dominates small cases). VMEC++ converges cleanly
on only 17/37 rows. Multigrid ladders are slower than VMEC2000 multigrid (primary suspects in §5).

### 1.5 Differentiability

AD-vs-central-FD agreement is demonstrated for fixed-boundary geometry/profile scalars, QS/QI
residuals, `DMerc`, `D_R`, and *branch-local* direct-coil free-boundary scalars. Arbitrary
differentiation through adaptive solver-control branches is explicitly unclaimed. The whole
"promoted lane / fingerprint" claim policy disappears in this overhaul: a feature is either
supported and tested, or it does not exist in the public API. Implicit differentiation (Phase 4)
makes the adaptive-branch problem moot: only the converged fixed point defines the derivative.

### 1.6 External pieces

- **ESSOS** is cloned at `/Users/rogerio/local/ESSOS`. The mgrid PR is
  **uwplasma/ESSOS #33 "Add VMEC mgrid export from ESSOS coils"**, branch `feature/mgrid-from-coils`
  (open, +325/−21 over 7 files): adds `essos.mgrid.MGrid` (SIMSOPT-compatible cylindrical grid
  layout), `coils_to_mgrid(...)`, and `Coils.to_mgrid(...)`, validated by round-trip and SIMSOPT
  parity tests. Use this branch; if it merges, use main; push fixes to the PR (as rogeriojorge) if
  needed and pin the commit hash in example docstrings.
- **booz_xform_jax** exists at `uwplasma/booz_xform_jax` (pure JAX, MIT, pip-installable from
  GitHub; also a local clone at `/Users/rogerio/local/booz_xform_jax`) and is already a runtime
  dependency wired to `vmec --booz`.
- **jaxopt is deprecated.** Modern implicit-diff stack: **Optax** (first-order optimizers) +
  **Optimistix** (root/fixed-point solves with implicit differentiation) + **Lineax** (linear
  solves). We hand-roll a thin `custom_vjp` (§6) so none of these becomes a hard dependency;
  `lineax` is an acceptable optional backend for the adjoint linear solve.

---

## 2. Environment, locations, ground rules

### 2.1 Local paths

| Code | Path | Notes |
|---|---|---|
| vmec_jax | `/Users/rogerio/local/vmec_jax` | the repo being rewritten |
| VMEC2000 (STELLOPT) | `/Users/rogerio/local/STELLOPT/VMEC2000` | executable `Release/xvmec2000`; INDATA module in `LIBSTELL/Sources/Modules/vmec_input.f` |
| VMEC++ | `/Users/rogerio/local/vmecpp` | C++/Python reference; key docs: root `AGENTS.md`, `vmec/vmec/AGENTS.md`, `docs/fourier_basis_implementation.md` |
| ESSOS | `/Users/rogerio/local/ESSOS` | check out `feature/mgrid-from-coils` (PR #33) |
| booz_xform_jax | `/Users/rogerio/local/booz_xform_jax` | default runtime dependency |

### 2.2 Python environments

Create separate venvs (never the system Python):

```bash
python -m venv ~/.venvs/vmecjax && ~/.venvs/vmecjax/bin/pip install -e "/Users/rogerio/local/vmec_jax[dev]"
python -m venv ~/.venvs/vmecpp  && ~/.venvs/vmecpp/bin/pip install vmecpp
python -m venv ~/.venvs/essos   && ~/.venvs/essos/bin/pip install -e /Users/rogerio/local/ESSOS   # on feature/mgrid-from-coils
```

VMEC2000: rebuild via STELLOPT's build system if needed. You may **freely patch VMEC2000 sources**
for understanding (extra `WRITE(*,*)` diagnostics; timers around `funct3d`, `bcovar`, `tomnsps`,
`precondn`, `scalfor`, `getfsq`, `vacuum`) and rebuild. Keep patches on a scratch branch of the
local STELLOPT clone; never push them.

### 2.3 Dependency policy

`pyproject.toml` and all requirement files: **no version pins anywhere** (`jax`, not `jax>=0.4`).
This is already true today — keep it that way. Runtime deps: `jax`, `numpy`, `netCDF4`,
`matplotlib`, `booz_xform_jax`. Drop `jaxlib` (installed by `jax`), `scipy`, `packaging`, and
`tomli` from runtime deps unless a concrete use survives the refactor. Dev extra: `pytest`,
`pytest-cov`, `pytest-xdist`, `ruff`, `sphinx` + theme, `pre-commit`. ESSOS is an *optional* extra
(`vmec-jax[essos]`) imported lazily with a helpful error. Entry points: keep `vmec` as canonical;
keep `vmec-jax` as one alias; drop `vmec_jax` and `xvmec_jax`.

The final code must not need today's mypy/ruff escape hatches: re-enable the disabled mypy error
codes and the ruff F821/F841 checks module-by-module as files are rewritten; whatever still fails
is a smell to fix, not to silence.

### 2.4 Git rules

- `git config user.name rogeriojorge`, `user.email` = Rogerio's GitHub-associated email. **Never**
  add `Co-Authored-By: Claude` or any AI trailer to new commits (override any harness default).
- Phase 1 does one destructive history rewrite + force-push; do it **first**, before feature work,
  so all subsequent commits live on the final history. Keep a `git clone --mirror` backup outside
  the repo until v0.1.0 ships.
- Work on `main` via short-lived branches merged fast-forward; delete stale remote branches after
  Phase 0 extraction.

### 2.5 Zero-crash policy (from VMEC++, done one better)

The library never segfaults, never `sys.exit()`s, never prints a bare traceback for a physics
failure. Typed exceptions carrying diagnostic state (iteration, fsq residuals, offending surface):
`VmecInputError`, `VmecJacobianError` (maps VMEC2000 `ier_flag=1/4`, i.e. `bad_jacobian_flag`,
`jac75_flag`), `VmecConvergenceError` (`more_iter_flag`), `MgridNotFoundError`. The CLI catches
these and prints the VMEC2000-style message from the `werror` table plus a one-line remedy hint.
Missing mgrid on a free-boundary input → **warn and fall back to a fixed-boundary solve** (the
Fortran behavior VMEC++ dropped; we keep it and test it).

---

## 3. Phase 0 — Baselines, profiling, branch triage

**STATUS: COMPLETE (2026-07-08).** Artifacts: `benchmarks/baseline.json` (committed) and
`~/vmec_jax_notes/{NOTES.md, wout_gap.md, profile_findings.md}` (local). Headlines:
multigrid slowdown = per-stage recompilation (23× `jit(stage)` + ~300 eager glue compiles in a
3-stage ladder) — padding fix confirmed as the plan; cold gap is 100% JAX/XLA setup (solovev:
0.10 s VMEC2000 vs 3.4 s cold / 0.01 s warm); wout is missing 39 variables (list in wout_gap.md);
mirror-geometry branch triaged (all solver experiments DROP; mirror physics design KEEP) and
deleted; QP optimization must default to max_mode=3.

Most of the audit is done (§1). What remains before touching code:

1. **NOTES.md** (scratch area, not committed): distill the five `vmec_jax_plan/*.md` files (the
   32k-line log only for still-open items — most of it is historical micro-optimization diary),
   and skim `origin/codex/mirror-geometry` (511 commits): record what its native block
   preconditioner, spline matrix-free loop, and square-hybrid solver actually do and whether any
   idea survives into §5.4/§7.5. Then those branches/files are doomed (Phase 1).
2. **Baseline benchmark script** `benchmarks/run_baseline.py` (committed, small; results JSON
   committed as `benchmarks/baseline.json`): fixed suite — `solovev`, `DSHAPE`, `HELIOTRON`,
   `cth_like_fixed_bdy` (+ lasym variant), `cth_like_free_bdy`, a DIII-D-like tokamak free-boundary
   case, `nfp4_QH_warm_start`, `w7x`, precise QA/QH (Landreman–Paul) — recording wall time
   (cold+warm), peak RSS, iterations to each `ftol` stage, for: VMEC2000 (xvmec2000,
   single-thread), VMEC++ (where it converges), vmec_jax CLI (CPU; GPU if available), each
   single-grid and multigrid. This script regenerates the README plot in Phase 10.
3. **Profile vmec_jax now** (`jax.profiler` + `py-spy`): per-case split of trace/compile vs run;
   count XLA compilations across a multigrid ladder (each `ns` stage recompiling is the #1 suspect
   for "multigrid slower than VMEC2000"); host↔device syncs per iteration in the hot loop (target
   0); transform vs preconditioner vs residual cost shares.
4. **Profile VMEC2000** (timer patches per §2.2, or gprof) on the same cases: per-part budget of
   funct3d / bcovar / transforms / precondn / tridslv / vacuum, and its multigrid stage timings —
   so we know exactly what budget each part of vmec_jax must beat.
5. **wout gap list**: `ncdump -h` a VMEC2000 wout vs a vmec_jax wout for the same case; diff
   against the authoritative variable list in Appendix A; record missing/mismatched variables.

**Exit criteria:** NOTES.md (branch verdicts, open TODOs worth keeping), `benchmarks/baseline.json`,
a ranked list of multigrid slowdown causes with profile evidence, and the wout gap checklist.

---

## 4. Phase 1 — Repository consolidation and history rewrite

**STATUS: COMPLETE (2026-07-08).** History rewritten with git-filter-repo (57.4 → 11.8 MiB
packed; 0 Claude trailers; contributors = rogeriojorge + matthewfeickert), force-pushed; 4 stale
branches deleted; vmec_jax_plan/, validation/, tools/diagnostics, examples/data/single_grid
removed; figures pruned/compressed (4.8 → 1.8 MB); interim CI (fast tests + smoke + build + size
check) green locally; pre-commit 200 KB guard added. Mirror backup at ~/vmec_jax_backup.git
(keep until v0.1.0).

Goal: one branch, ≤10 MB fresh clone, no Claude in contributors, drastically fewer files.

1. **Working-tree consolidation first** (so the rewrite also shrinks the tip):
   - Delete `vmec_jax_plan/` entirely (this plan.md replaces it), `validation/`, obsolete docs
     pages (plan/lane rst files, §11), `examples/data/single_grid/` (fold unique decks into
     `examples/data/`), and the archived optimization showcase PNGs.
   - Recompress every kept image (`oxipng -o4 --strip all` / `pngquant`); target <150 KB each; keep
     only figures referenced by README/docs.
   - `tools/` shrinks to `fetch_assets.py`, `make_release_assets.py`, `compress_figures.py`.
   - Large fixtures (reference wouts, mgrids, golden stdout captures, benchmark provenance) → a
     versioned GitHub Release bundle (`vmec-jax-test-assets-vX.tar.gz`, sha256-checked, cached
     under `~/.cache/vmec_jax/`). Keep in-repo only text input decks and one tiny mgrid (<1 MB) so
     `pytest -m quick` works offline.
2. **History rewrite** with `git filter-repo` on a fresh clone (destructive; mirror-backup first):
   - `--strip-blobs-bigger-than 300K`, plus explicit `--path <old plan files, old figures, old
     fixtures> --invert-paths`.
   - **Strip the Claude trailers** with a message callback:
     `git filter-repo --message-callback 'return re.sub(rb"\n?Co-Authored-By: Claude[^\n]*", b"", message)'`.
     Verify with `git log --all --format=%b | grep -ci co-authored-by: claude` → 0. (A mailmap pass
     can also normalize `rogerio.jorge@ist.utl.pt` vs `@wisc.edu` if desired — optional.)
   - Verify: `git count-objects -vH` ≤ ~10 MB packed; `git shortlog -sne` clean.
   - Force-push `main` + tags; delete the 4 stale remote branches; confirm the GitHub contributors
     page no longer lists Claude (may take a cache cycle).
3. **Hygiene going forward:** `.gitignore` covers `wout_*.nc`, `boozmn_*.nc`, `mgrid_*.nc` (except
   the tiny fixture), `figures/`, `results/`; pre-commit `check-added-large-files` (200 KB); a CI
   job fails if packed size exceeds 15 MB.

**Exit criteria:** fresh clone ≤ 10 MB; single `main`; contributors clean; CI green on the pruned
tree (tests may be temporarily reduced — full restructure lands in Phase 9).

---

## 5. Phase 2 — Core library refactor (architecture, naming, fixed-boundary parity)

**STATUS (2026-07-10): PHASE COMPLETE — legacy deleted (3ce3402c).** vmec_jax/ = 33 files /
19k lines all-core; tests/ = 24 files / 5k lines; 323 tests vs golden; ruff clean; docs -W green.
Remaining project work tracked in §10 (examples), §12 (tutorials), §13 (release), plus follow-ups:
free-boundary vacuum tuning + potvac export + freeb ladder, full radial padding, coverage gate,
mirror design doc.

**Independent review (2026-07-10, ~65% overall) — binding follow-ups:**
1. **CI must run in <=10 minutes** (currently ~22-26) at equal-or-better coverage: shard
   (fast / parity / gradient / optional full-physics), re-enable JIT inside gradient tests (a
   global fixture disables JIT, making implicit tests 105-160 s each — the main CI cost), cache
   goldens, and restore the **95% coverage gate** (currently 90%; weak: profiles 31%, step 72%,
   printing 77% — add targeted tests, don't pad).
   *(2026-07-10 progress, 0f9aca65):* sharded into fast/parity/gradient/examples/coverage-gate/
   cli-smoke/build; module-scoped JIT fixture landed (solver tests 5-40x faster); the parity long
   pole was test_examples subprocess smokes (QA 145s/QH 101s/QP 81s/QI 67s) — isolated into their
   own shard with QH/QP/QI gated to nightly RUN_FULL; device.py (was 71%, untested) + mgrid/
   optimize error branches given targeted tests. **CI GREEN (run 29110749965)**: 7 parallel
   shards each under a 9-min timeout (parity-a 8.0m, parity-b 4.3m, gradient 8.8m, others <2m;
   wall ~9m, meets <=10m) and the **95% coverage gate PASSES** (8521 stmts, 414 missing = 95%).
   parity split into two balanced shards (a=8 heavy solver modules/104 tests, b=rest/370); one
   brittle warm-solve wall bound relaxed (0.1s->1.0s; the zero-recompile check remains the gate).
   PR #23 closed (already on main, pre-deletion base); orphan branch/worktrees cleaned.
2. **Docs/README honesty**: distinguish validated fixed-boundary implicit differentiation from
   NOT-yet-supported free-boundary/coil derivatives and the optimizer wiring status; fix
   optimization.rst "no special handling" claim; README examples claim must become true when the
   examples land.
3. **Free boundary to production**: a CONVERGED free-boundary fixture (raise NITER; the CTH deck
   stops at 1000 with fsq~9e-2), vacuum-solve performance tuning (warm 14.4 s vs Fortran 1.95 s),
   then freeb implicit derivatives — do not promote coil-derivative claims before this.
4. **Memory workstream**: solves use 0.7-1.5 GB vs VMEC2000's 27-43 MB; implicit gradient 3.4 GB.
   Profile buffers, donate in the CLI lane, audit temporaries; targets in §7.7.
5. **Optimization convergence budgets**: examples run as many iterations as needed (thousands)
   for genuine convergence; CI smoke uses reduced budgets via VMEC_JAX_EXAMPLES_CI.
6. Line/docstring hygiene: 4 files >1000 lines to split when touched; ~142 functions lack
   docstrings — close during Phase-9 doc pass.

*(superseded status of 2026-07-09:)* core landed, integration/perf hardening next. `vmec_jax/core/` has 20
modules (~10k lines), each A/B-proven vs the legacy kernels (420+ tests) — including the solve
loop (solovev 215/215 iterations vs VMEC2000, cth 434, machine-precision wout parity), the
complete wout writer (all 39 missing variables; found legacy lasym output bugs: buco/jcur*/ctor
x1/2, jdotb x1/16, fast-bcovar bsubsmns corruption), mgrid IO/field (ESSOS PR#33 cross-verified),
and multigrid interpolation. Golden fixtures are a GitHub release (golden-v1) with sha256 fetch;
core suite is in CI with coverage reporting.
**(1) DONE + (2) DONE (2026-07-09, per-stage variant):** `SolverRuntime` is now a registered
pytree passed to module-level jitted lanes (`_while_lane`/`_block_lane`): RunSetup arrays +
`rcon0/zcon0` as data, `Resolution` + scalar config as meta, numpy trig/mode/gather tables
derived from meta via lru-cached `_static_tables`. Proven (tests/core_new/
test_multigrid_ladder.py, JAX_LOG_COMPILES subprocess counts): second solve at same Resolution
with different boundary = 0 compiles, 0.02 s on solovev. `solve(initial_state=...)` hot restart
added (boundary delta spread with the profil3d radial profile; bare edge-row swap measured
fsqr~0.5, spread ~4e-6; cth 1% RBC(0,1) restart 298 vs 434 iters — <25% is stepper-rate-limited,
needs Phase-4 Newton/2D precond). `multigrid.solve_multigrid` (runvmec.f ladder: skip-decreasing,
per-stage banners/ftol/niter, interp.f handoff) matches xvmec2000 ladders to machine precision
(cth 5/9/15 rel 9e-15; nfp4_QH 9/17/35 identical printed wb; NOTE ladder-vs-single-grid wb
scatter 1.36e-8 on nfp4_QH is inherent to VMEC — reproduced by xvmec2000 itself — m=1 freeze).
Compile behavior: one block-lane compile per distinct stage structure per session (3 for a
3-stage ladder, ~3 s each — cold ladder 9.3 s vs cold direct 5.1 s on cth; warm ladder 0.3 s,
0 compiles; direct-after-ladder reuses the final-stage executable, 0 compiles). **Follow-up
(padding, not attempted — >2 h):** ONE executable for all stages = pad radial arrays to
max(ns_array) as pytree *data* (s grids/hs/profiles already data, so per-stage values reuse one
executable) + a static `ns_active` mask threaded through the radial reductions (energies/force
norms in fields.py `energies_and_force_norms`, getfsq sums in residuals.py, precondn/lamcal
integrals and the tridiagonal jmax in preconditioner.py, jacobian half-mesh differences at the
padded rows) with masked rows pinned to identity updates in the loop body; validate vs the
per-stage ladder to 1e-15 per stage, then flip solve_multigrid to a single lane. (3)-(4)
unchanged below.
(3) parity breadth: 3D/lasym/finite-beta/ncurr=1/high-mode across all nine golden fixtures
(known gap: legacy lasym solver drifts ~5% on asym harmonics — validate the new core against
golden directly); (4) switch one public vertical slice (CLI fixed-boundary path) to the core,
then delete the corresponding legacy modules and migrate tests — repeat until the legacy tree is
gone. Implicit diff (Phase 3) starts once the residual API is frozen by (1)-(2).

### 5.1 Target layout (~30 files; one concern per file, none over ~1000 lines)

```
vmec_jax/
  __init__.py       # public API: run, Equilibrium, VmecInput, wout io, plotting entries
  input.py          # INDATA parser + VMEC++-compatible JSON input + VmecInput pytree; convert CLI
  profiles.py       # power_series, gauss_trunc, two_power, pedestal, cubic/akima splines, line_segment
  fourier.py        # (m,n) bookkeeping, parity tables, mscale/nscale, angle grids, m=1 constraint maps
  transforms.py     # totzsps/totzspa + tomnsps/tomnspa equivalents: batched DFT matmuls + FFT path
  geometry.py       # R,Z,λ real-space fields, jacobian tau/sqrt(g), metrics guu,guv,gvv   (jacobian.f)
  fields.py         # B^u,B^v, |B|, covariant B, pressure, energies wb/wp, tcon           (bcovar.f)
  forces.py         # MHD force kernels + spectral-condensation constraint force          (forces.f, alias.f)
  residuals.py      # fsqr/fsqz/fsql via getfsq, m=1 constraint, fedge                    (residue.f90)
  preconditioner.py # precondn/lamcal 1D radial precond, vectorized tridiagonal solve, 2D option (scalfor.f, tridslv, precon2d ideas)
  step.py           # damped 2nd-order Richardson step, dtau damping (ndamp=10), irst back-off (evolve.f, restart.f)
  solver.py         # single-grid loop: lax.while_loop core + host-blocked CLI variant    (eqsolve.f)
  multigrid.py      # NS_ARRAY ladder, coarse→fine interpolation, hot restart             (runvmec.f, interp.f)
  vacuum.py         # NESTOR: Green's function, analyt/scalpot, potvac solve              (NESTOR_vacuum/)
  freeboundary.py   # free-boundary iteration, ivac/nvacskip cadence, MagneticField protocol
  mgrid.py          # mgrid netCDF read/write, interpolated MagneticField
  # (no in-tree coils module — coils live in ESSOS (essos.coils.Coils); the
  #  external field is consumed as an mgrid or a generic xyz->B callable)
  implicit.py       # custom_vjp implicit differentiation of the equilibrium (Phase 4)
  wout.py           # wout writer/reader — full Appendix-A variable set incl. jxbforce, mercier, bss
  printing.py       # VMEC2000-format iteration lines, stage banners, threed1 summary     (printout.f)
  plotting.py       # vmec --plot for wout and boozmn files
  boozer.py         # thin wrapper over booz_xform_jax (--booz)
  optimize.py       # objectives: QS ratio residual, QI (Goodman-style), aspect, iota, mirror; least-squares driver
  errors.py         # typed exceptions + werror message table
  cli.py            # `vmec` entry point
```

Names follow physics with a **VMEC-canonical glossary**: community-expected names stay (`ns, mpol,
ntor, nfp, lasym, iotaf, presf, rmnc, zmns, lmns, bmnc, ...`); internal Fortran temporaries get
descriptive names (`force_R_cos` not `armn`, `dpressure_ds` not `pres1`, `sqrt_g` not `gsqrt` — with
the glossary mapping both ways). Ship `docs/glossary.rst`: VMEC2000 name ↔ vmec_jax name ↔ meaning ↔
defining equation ↔ source location. Every module header docstring names its VMEC2000 counterpart
file(s) and the equations it implements.

Deletions (absorbed or dropped): `kernels/numpy_forces.py` (single JAX implementation, used by both
lanes), `preconditioner_1d.py` (keep only the JAX one), `solve.py`/`_compat.py` facades,
`discrete_adjoint.py` + both `adjoint/` trees + `optimizers/fixed_boundary/exact_replay*` (replaced
by `implicit.py`), `drivers/` + `solvers/fixed_boundary/{residual,scan}/` (merged into
`solver.py`/`step.py`), `quasi_isodynamic/` (distilled into `optimize.py`), `robust_coils.py`,
`optimization_workflow.py`, `finite_beta.py`/`bootstrap_current.py`/`redl_bootstrap.py` (move to
`optimize.py`-adjacent helpers only if an example/test uses them; otherwise drop — record in NOTES).

### 5.2 State and purity

- `EquilibriumState`: frozen pytree dataclass — spectral coefficients (`rmnc, zmns, lmns` +
  `rmns, zmnc, lmnc` when `lasym`), velocity `xcdot`, `time_step`, damping history `otau[10]`,
  iteration counters, residual history, `irst`-equivalent restart flag. All solver functions are
  pure `state -> state`.
- One set of spectral kernels shared by everything (solver, wout, plotting, objectives) — today's
  lane duplication is the main divergence source; eliminate it.
- Static configuration (resolutions, flags) in a hashable `VmecConfig`; **mode/radial arrays padded
  to the maximum multigrid resolution** so `ns` stages share one compiled executable (§7.1).

### 5.3 Two execution lanes, one physics

- `solver.solve(...)`: `lax.while_loop` over a jitted iteration, fully traceable — the
  differentiable API's forward solver.
- `solver.solve_cli(...)`: Python `while` around the same jitted *N-iteration block* kernel
  (e.g. `nstep=10`-aligned blocks via `lax.scan`), residuals checked on host between blocks —
  enabling exact-`ftol` early exit, live VMEC2000-format prints, buffer donation
  (`jax.jit(..., donate_argnums=...)`), and zero AD bookkeeping. Both lanes call identical physics
  kernels; a regression test asserts per-block state agreement to machine precision.

### 5.4 Algorithmic parity targets (the VMEC2000 details that matter — verified from source)

These constants/behaviors must be ported exactly; they are why VMEC2000 converges in few iterations:

- **Richardson step** (`evolve.f`): `dtau = min(|log(fsq1/fsq)|, 0.15)`, averaged over the last
  `ndamp=10` steps; `b1 = 1−dtau/2·Δt·⟨otau⟩`… concretely:
  `otav = mean(otau)`, `dtau = delt*otav/2`, `xcdot = (1−dtau)/(1+dtau)·xcdot + delt·gc`,
  `xc += delt·xcdot`.
- **Back-off** (`restart.f`): on `irst=2` (Jacobian sign change) restore saved state, zero
  velocity, `delt *= 0.90`, count `ijacob`; on `irst=3` (residual grew >1e4× best) restore,
  `delt /= 1.03`. Escalation in `eqsolve`: try `guess_axis` on first bad Jacobian; reset delt at
  `ijacob=25,50`; give up at 75 (`jac75_flag`).
- **Preconditioner cadence**: `precondn`+`lamcal`+force norms+`tcon` recomputed every
  `ns4=25` iterations, not every step.
- **tcon**: `tcon(js) = min(|ard/arnorm|,|azd/aznorm|)·tcon0-scaled·(32·hs)²` per surface,
  `tcon(ns)=½·tcon(ns−1)`; constraint force spectrally filtered to `m ∈ [1, mpol−2]` (`alias.f`)
  with `faccon(m)` weights.
- **m=1 constraint** (`residue.f90`): internally rotate `(gcr,gcz)_{m=1}` to `((gcr+gcz)/√2, 0)`;
  released when `fsqz<1e-6` etc. Boundary input applies `rbss=½(rbs+zbc)`-style conversion
  (`lconm1`, `readin.f`).
- **Radial start indices**: R,Z evolved from `jmin2`, λ from `jlam` (m-dependent; `vmec_params.f`).
- **1D preconditioner matrices**: `precondn` builds `axm/axd/bxm/bxd/cx` from
  `ptau = r12²·bsq·wint/gsqrt`-type integrals; `scalfor` forms tridiagonal
  `dx = axd + bxd·m² + cx·(n·nfp)²` with `edge_pedestal=0.05` and the ZC(0,0)(ns) `fac=0.25`
  stabilization; `tridslv` = Thomas algorithm vectorized over all (m,n) columns. λ uses the
  diagonal `faclam` from `lamcal` (`1/(blam·(n·nfp)² + clam·m² ± 2mn·nfp)`-shaped, √s-damped for
  m>16).
- **Free-boundary cadence** (`funct3d.f`): vacuum activates when `fsqr+fsqz ≤ 1e-3`; full NESTOR
  solve when `mod(iter2−iter1, nvacskip)==0`, incremental otherwise; adaptive
  `nvacskip = max(nvskip0, 1/max(0.1, 1e11·(fsqr+fsqz)))`; edge force `rbsq` from
  `bsqvac + presf(ns)` enters `forces` at js=ns; `rcon0,zcon0` ramp ×0.9/step in free-boundary.
- **Stopping**: converged when `fsqr, fsqz, fsql ≤ ftolv` simultaneously (physical, not
  preconditioned, residuals).

**Exit criteria for Phase 2:** all fixed-boundary benchmark cases (sym + lasym) converge with wout
parity vs VMEC2000 per Appendix-A tolerances; file/LoC budget met; ruff+mypy clean without today's
blanket ignores; every public function documented.

---

## 6. Phase 3 (interleave with Phase 2) — Differentiability done right

Adopt implicit differentiation of the equilibrium fixed point (DESC precedent; Skene & Burns
arXiv:2506.14792 for reuse-the-forward-machinery adjoints; jaxopt paper for the IFT formulation):

- Equilibrium = root of the preconditioned force residual `F(x, p) = 0`
  (`x` = spectral state, `p` = boundary coefficients / profile params / phiedge / coil currents &
  geometry / extcur). Wrap the solve in **`jax.custom_vjp`** (implemented in `implicit.py`):
  - forward: run the fast CLI-style solver (non-traced host loop is fine — it's opaque to AD),
    return converged `x*`;
  - backward: solve the adjoint linear system `(∂F/∂x)ᵀ λ = ḡ` matrix-free — `∂F/∂x`-vector
    products via `jax.vjp(residual_fn, x*)` — with **the 1D preconditioner as the preconditioner**
    for GMRES/BiCGStab (`jax.scipy.sparse.linalg.gmres`/`bicgstab`, or lineax); then return
    `−λᵀ ∂F/∂p` via one more VJP. Cost target: a handful of residual evaluations per gradient,
    O(1) memory in iteration count.
- This **replaces** `discrete_adjoint.py`, the replay tapes, fingerprints, and branch-local gates
  entirely. Multigrid/adaptive control lives inside the opaque forward solve; only the final fixed
  point defines the derivative (coarse stages are an initializer — stop-gradient by construction).
- Free boundary: identical scheme; NESTOR is inside `F` (traceable JAX code already exists), so
  coil parameters differentiate with no special handling. This deletes the "same-branch
  fingerprint-gated" hedging from the README.
- Provide `diff_mode="implicit"` (default) and `"unrolled"` (debug-only, small cases).
- **Permanent gradient tests** (rtol ≤ 1e-6 vs central FD): boundary coefficients → aspect, iota,
  QS residual, volume, `DMerc`; pressure profile / `pres_scale` → beta, wout scalars; coil currents
  and coil Fourier dofs (ESSOS) → free-boundary boundary shape and QS residual. Plus an adjoint
  linear-solve convergence test (preconditioned GMRES residual < 1e-10 in ≤ ~50 iterations).

**Exit criteria:** gradient tests pass; an L-BFGS boundary optimization with implicit gradients
matches/beats the FD-driven result at a fraction of cost; backward memory ≤ 2× forward.

---

## 7. Phase 4 — Performance (fast everywhere; multigrid faster than VMEC2000)

Ranked workstreams — confirm ranking against Phase-0 profiles before executing:

1. **Kill recompilation and host syncs.** One compiled solver for the whole `NS_ARRAY` ladder:
   pad radial arrays to `max(ns_array)` and mask; mode arrays sized once. No `.item()`/`float()`
   in the hot loop; prints via `jax.debug.callback` (jit lane) or between blocks (CLI lane).
   Measure: exactly one XLA compile per (mpol,ntor,lasym,lfreeb) tuple per session.
2. **Hot restart, VMEC++-style but stronger.** Public API `run(input, restart_from=output)`.
   VMEC++ restores only `rmnc/zmns/lmns` at a single matching `ns` and immediately activates the
   vacuum contribution; we do the same *plus* allow resolution changes by reusing our multigrid
   interpolation (radial interp in √s of scaled coefficients, odd-m axis extrapolation
   `2x₁−x₂`, spectrum pad/truncate — VMEC++ `_continuation.py` mechanics, already half-present in
   `multigrid.py`). Reset `delt` conservatively; carry λ. Hot restart powers the β-scan example and
   every optimization loop.
3. **Cold-start cost.** The 23/37 cold-slower rows are XLA setup. Mitigations: JAX persistent
   compilation cache enabled by default in the CLI (`JAX_COMPILATION_CACHE_DIR` under
   `~/.cache/vmec_jax/xla`), smaller/fused graphs (fewer distinct jitted entry points), lazy
   imports so `vmec --help` stays <100 ms.
4. **Transforms.** Keep batched-DFT matmuls with basis-baked weights (GPU-optimal, AD-friendly;
   VMEC++'s FFTX codelets only buy 10–20%). Add a `jnp.fft.rfft` path selected at trace time when
   `ntheta·nzeta` is large enough to win on CPU (benchmark the crossover once, hard-code the rule).
   Fuse totzsp→geometry→forces→tomnsp; verify with XLA cost analysis that intermediates stay
   fused.
5. **Preconditioner.** Exact 1D port (§5.4) with the Thomas solve vectorized over (m,n) — this is
   *the* convergence-rate feature; recompute on the ns4=25 cadence. Then an optional
   **`precond="2d"`**: VMEC2000's precon2d builds the Hessian by finite-difference "jogs" and
   block-tridiagonal LU (BCYCLIC); in JAX we get exact Hessian-vector products for free via
   `jax.jvp(residual_fn, ...)`, so implement 2D as matrix-free GMRES on the Newton step
   preconditioned by the 1D operator, activated below a `prec2d_threshold` on the finest grid
   (mirroring `ictrl_prec2d`/GMRES lanes in `evolve.f`/`gmres_mod.f`). Check NOTES for anything
   worth stealing from the `codex/mirror-geometry` native block preconditioner before deleting it.
6. **CPU threading / vectorization.** VMEC++ beats Fortran via OpenMP over radial partitions; our
   analog is XLA CPU multi-threading over the big batched matmuls — ensure kernels are large
   enough to parallelize, document `XLA_FLAGS=--xla_cpu_multi_thread_eigen=...`/thread pinning in
   docs/performance.
7. **Memory.** Donate state buffers in the CLI lane; float64 mandatory
   (`jax.config.update("jax_enable_x64", True)` at solver import); audit temporaries; peak-RSS in
   the benchmark output.

**Exit criteria:** CLI ≥ VMEC2000 speed on ≥80% of suite rows (cold, CPU, single-grid);
multigrid strictly faster than our single-grid *and* faster than VMEC2000 multigrid on the suite
median; GPU runs validated; README plot regenerated from `run_baseline.py`.

---

### 7.8 GPU profiling workstream (added 2026-07-09; hardware available)

`ssh office` (pop-os, 2x RTX A4000 16GB, repo at ~/vmec_jax) is available for GPU work. Reported
symptom: vmec_jax is sometimes SLOWER on GPU than CPU — cause unknown. Plan:

1. **Environment**: venv on office with CUDA jax (`pip install -U "jax[cuda12]"`), editable
   vmec_jax at current main, golden fixtures via the conftest downloader.
2. **Benchmark matrix** (extend `benchmarks/run_baseline.py` with a `--device {cpu,gpu}` axis and
   an office-runner mode): all baseline decks x {cpu, gpu} x {legacy solver, core solver
   cli/jit lanes} x {single-grid, multigrid} x {cold, warm}, recording wall, device memory,
   compile vs run time (jax.profiler), and per-iteration step time across problem sizes
   (ns=11 -> 151, low and high mpol/ntor) — locate the GPU crossover point.
3. **Hypotheses to test** for GPU-slower-than-CPU: (a) small kernels + dispatch overhead at low
   resolution (GPU should win only at high ns*mnmax); (b) host<->device syncs per iteration in
   the legacy driver; (c) the tridiagonal Thomas solve serializes over ns on GPU (lax.scan) —
   consider cyclic reduction or a batched parallel solve, or pin the tridiagonal solve to CPU;
   (d) float64 throughput on A4000 (GA104 fp64 = 1/32 fp32) — measure; experiment with fp32
   preconditioner + fp64 physics; (e) recompiles from per-solve closures (identity-cache landed;
   structural runtime caching pending).
4. **Deliverables**: `benchmarks/gpu_baseline.json`, a docs/performance section explaining the
   crossover + tuning guidance, and implementation changes ranked by measured impact feeding
   Phase 4.

## 8. Phase 5 — Free boundary, ESSOS, mirrors

1. **mgrid path** (VMEC2000-compatible, tokamaks *and* stellarators — VMEC++ can do neither
   `ntor=0` free-boundary nor lasym; we support both): validate on `cth_like_free_bdy`
   (sym + lasym) and a DIII-D-like tokamak mgrid case against VMEC2000.
2. **Direct-coil path**: `CoilField` (ESSOS Biot–Savart, differentiable in coil dofs) evaluated on
   the NESTOR grid each vacuum update; no interpolation. `mgrid.write_mgrid(field, ...)` generates
   VMEC2000-compatible mgrids from any field (use/align with ESSOS PR #33's
   `essos.mgrid.MGrid`/`coils_to_mgrid` so the two codes interchange files).
3. **Fallback**: `lfreeb=T` + missing mgrid → `MgridNotFoundError`-grade warning + fixed-boundary
   solve (§2.5).
4. **Single-stage optimization with ESSOS** (both directions the plan's development goals name):
   fixed-boundary single-stage (coil objectives + VMEC QS objectives on one gradient tape, boundary
   from coils via a quadratic-flux surface or direct constraint) and free-boundary single-stage
   (coils → direct field → free-boundary equilibrium → QS/aspect targets; gradients via §6).
   One example each, marked advanced.
5. **Mirror geometry (design + skeleton; implement only if time remains after Phase 10)**:
   fixed/free-boundary open-field-period mirror-like configurations; boundary as Fourier series
   *or* splines (spline boundary + spline geometry representation). Deliver `docs/mirrors.rst`
   design doc, input-schema extension, and xfail test scaffolds. Mine `codex/mirror-geometry`
   NOTES for its spline matrix-free loop before discarding.

---

## 9. Phase 6 — Outputs: prints, wout completeness, JSON, Boozer

1. **Prints**: replicate VMEC2000 layout byte-for-column. The authoritative formats (from
   `printout.f`, `initialize_radial.f`, `runvmec.f`) are in Appendix B — implement `printing.py`
   directly from them (screen lane: `iter, fsqr, fsqz, fsql, RAX(v=0)[, ZAX], DELT, WMHD[, DEL-BSQ]`;
   threed1 lane adds preconditioned `fsqr1,fsqz1,fsql1`, `BETA`, `<M>`, `FEDGE`). Golden stdout
   captures of VMEC2000 per benchmark case go in the release asset bundle; tests diff structure
   exactly and values within tolerance. Also: `BEGIN FORCE ITERATIONS` banner, per-stage
   `NS = … NO. FOURIER MODES = … FTOLV = … NITER = …` banners, `VACUUM PRESSURE TURNED ON AT n
   ITERATIONS`, final timing + `EXECUTION TERMINATED NORMALLY`-style `werror` messages, and the
   threed1 summary file.
2. **wout completeness**: implement the full Appendix-A variable set (unit conventions included:
   `presf/pres/mass/jcuru/jcurv/ctor` divided by μ0 on write; `phipf/chipf` × 2π·signgs; `qfact =
   1/iotaf`; `lmns` half-mesh; `bsubsmns` full-mesh). Add VMEC++'s useful extras where free
   (`fsqt`, `wdot` already exist in VMEC2000; consider `lmns_full` as an extension attribute).
   Parity test: per-variable `CompareWOut`-style relative+absolute tolerances (global default +
   looser `currumnc/currvmnc`), following vmecpp-validation methodology. wout must load in simsopt
   and booz_xform unchanged.
3. **JSON input**: accept VMEC++'s exact schema alongside INDATA (keys in Appendix C; sparse
   `{"n":…,"m":…,"value":…}` boundary coefficients; dense axis arrays). `vmec convert input.foo
   --to json|indata` round-trips. We additionally support the profile types VMEC++ lacks
   (splines, pedestal, two_power, gauss_trunc…) — same key names as INDATA.
4. **Boozer**: `booz_xform_jax` stays a default dependency; `vmec --booz [--plot]` writes
   `boozmn_*.nc` + |B| contours + spectrum plots; one integration test.

---

## 10. Phase 7 — Examples (the public face; simsopt-simple)

Rules: each example is **one file**, < ~120 lines, top docstring stating goal/physics/expected
runtime/achieved result, public API only, smoke-tested in CI at reduced resolution (`--ci` flag).

```
examples/
  data/                          # text input decks + one tiny mgrid
  fixed_boundary_run.py          # run + plot + wout + --booz equivalent, ~30 lines
  free_boundary_mgrid.py         # cth_like via mgrid
  free_boundary_direct_coils.py  # same physics via ESSOS CoilField; compares to the mgrid run
  free_boundary_beta_scan.py     # THE README example (below)
  QA_optimization.py             # circular torus -> precise QA   (nfp=2)
  QH_optimization.py             # circular torus -> precise QH   (nfp=4)
  QP_optimization.py             # circular torus -> precise QP   (nfp=2 or 3)
  QI_optimization.py             # circular torus -> precise QI   (nfp=1)
  single_stage_fixed_boundary.py # advanced, ESSOS
  single_stage_free_boundary.py  # advanced, ESSOS
```

**Example style requirements (user directive 2026-07-09; binding):** pedagogic and user-friendly —
no `main()` functions; input parameters at the top; the user writes their own objective function
in the script (importing gradient-ready building blocks from `vmec_jax.core.optimize`); scripts
teach: reading input files AND creating a `VmecInput` from scratch, writing outputs (wout),
plotting, and printing initial conditions, per-iteration progress, and final results. Not so
minimal that users cannot generalize; no auxiliary-function mazes (anything reusable moves into
the source). Optimization examples target **precise QA/QH/QP/QI at several nfp, max_mode=5, with
ESS, from a circular torus**; for QI, try **QP-first-then-QI** as the simple route before any
seed/preconditioner machinery (compare against the legacy seed approach and report). Runs must be
fast on CPU and GPU: reuse the warm solver (structural executable cache), use implicit autodiff
with measured accuracy (test gradient accuracy in CI), and include **commented-out but fully
tested** extra objective terms (DMerc, LgradB, magnetic well, ...) that work when uncommented
(tests exercise them uncommented).

**Optimization examples** mirror simsopt's `QH_fixed_resolution.py` (66 lines: build equilibrium →
`QuasisymmetryRatioResidual(surfaces, helicity_m, helicity_n)` + aspect target → one least-squares
call). Ours: `vmec_jax.optimize.QuasisymmetryResidual(m, n)` with (QA: m=1,n=0; QH: m=1,n=−nfp;
QP: m=0,n=1) + aspect (+ iota/mirror targets), staged `max_mode` 1→2→3 continuation inside one
visible loop, gradient-based least squares using Phase-6 implicit gradients (highlight: no finite
differences, no MPI). **QI** uses a Goodman-style penalty (B-contour alignment, mirror-ratio /
target-B shaping, elongation + iota + aspect practical targets) implemented and documented inside
`optimize.py` — the example stays one file; continuation detail (grow ntor ~2× faster than mpol) is
a documented option, not example-level machinery. Today's QP-basin-then-QI trick, if still needed,
lives inside `optimize.py` with a docstring, honestly stated. Achieved objective values go in each
docstring and are loosely asserted by the CI smoke test.

**`free_boundary_beta_scan.py`** (featured in README): Landreman–Paul precise-QA coils from ESSOS
+ a simple ESSOS tokamak coil set; for β = 0,1,2,3,4,5%: free-boundary solve (hot-restarting each β
from the previous), once via generated mgrid and once via direct Biot–Savart; plot boundary
cross-sections evolving with β for both machines, overlay mgrid vs direct, report the difference
(direct = interpolation-free reference). Output: one compressed panel figure for the README.

---

## 11. Phase 8 — Tests and coverage (≥95%, no bloat)

- `tests/` mirrors the module layout: one file per module + `test_parity/` (wout/print goldens
  from the asset bundle) + `test_gradients/` (FD checks) + `test_examples.py` (smoke).
- Markers: `quick` (offline, <2 min, every push), `parity` (asset bundle), `slow` (nightly).
  Coverage gate ≥95% on `vmec_jax/` from quick+parity (`--cov-fail-under=95`).
- Property tests: transform round-trip (tomnsp∘totzsp = identity on band-limited data), residual
  invariance under nfp rotation, lasym-off ≡ symmetric path, CLI-lane ≡ jit-lane per block,
  JSON↔INDATA round-trip.
- Delete the wave/coverage-padding files and the 3–4k-line lane tests. Budget: ≤ ~6k lines total.
- Keep the `VMEC2000_INTEGRATION=1` opt-in gate that runs xvmec2000 side-by-side locally/nightly.

---

## 12. Phase 9 — Documentation overhaul

Sphinx with `furo` (already) or `sphinx-book-theme`, MathJax, `sphinx-copybutton`, `myst-parser`.
Landing page: what/why, 3-command quickstart, gallery. Delete the internal plan/lane pages
(`aggressive_performance_plan`, `accelerated_merge_readiness`, `optimization_sweep_results`,
`piecewise_omnigenous_plan`, `free_boundary_plan`, `discrete_adjoint` in its current form).

Structure:
1. **Getting started** — install, `vmec --test`, first run, plotting, Boozer.
2. **Tutorials** — one page per example, rendered figures, expected output.
3. **Theory & numerics** (the differentiator; every equation links to the implementing function):
   ideal-MHD energy functional and the Hirshman–Whitson steepest-descent moment method; flux
   coordinates and λ; Fourier representations, parities, lasym; force residuals and the m=1
   constraint; spectral condensation (`alias`/tcon); half/full radial meshes and jmin/jlam
   conventions; the 1D preconditioner derivation + tridiagonal solve + the 2D matrix-free
   extension; Richardson time stepping, damping, and irst back-off; multigrid + hot restart;
   NESTOR (Merkel Green's-function method) and the free-boundary cadence (ivac/nvacskip); the
   implicit-differentiation adjoint with derivation and cost analysis (cite Skene & Burns 2026,
   jaxopt, DESC); CLI lane vs differentiable lane.
4. **Reference** — API autodoc; input reference (every INDATA variable + JSON schema); wout
   variable reference (Appendix A rendered); glossary; CLI reference.
5. **Performance & validation** — benchmark methodology, plots, parity tables, GPU notes,
   profiling guide (including how VMEC2000 was instrumented).
6. **Developer guide** — architecture map (module ↔ VMEC2000 subroutine), adding an objective,
   release checklist.

---

## 13. Phase 10 — Benchmarks, README, release

1. Re-run `benchmarks/run_baseline.py`; regenerate the README benchmark figure (compressed):
   vmec_jax CPU/GPU cold+warm vs VMEC2000 vs VMEC++, single-grid and multigrid.
2. Rewrite README: short pitch; install; quickstart; β-scan figure; one optimization figure;
   feature table vs VMEC2000/VMEC++ (differentiable ✓, lasym ✓, free-boundary tokamak+stellarator
   ✓, JSON ✓, hot restart ✓, zero-crash ✓, mgrid fallback ✓, GPU ✓, Boozer built-in ✓, spline
   profiles ✓, 2D preconditioner ✓); CLI reference; docs link. Remove all "research lane" language.
3. Upload the release asset bundle; tag `v0.1.0`; publish to PyPI; update the conda-forge
   feedstock; verify `pip install vmec-jax && vmec --test` in a clean venv.

---

## 14. Acceptance checklist (definition of done)

- [ ] Fresh clone ≤ 10 MB; single branch; zero `Co-Authored-By: Claude` trailers in history; Claude
      absent from the GitHub contributors panel; all new commits authored by rogeriojorge.
- [ ] `vmec_jax/` ≤ ~35 files / ~15k lines; no file >~1000 lines; docstrings + VMEC2000 cross-refs
      everywhere; ruff and mypy pass without today's blanket ignores.
- [ ] Fixed + free boundary (mgrid and direct-coil; tokamak and stellarator; sym and lasym)
      converge with wout + print parity vs VMEC2000 per Appendix-A tolerances; missing-mgrid
      fixed-boundary fallback works and is tested.
- [ ] CLI ≥ VMEC2000 speed on ≥80% of suite rows (cold CPU); multigrid faster than VMEC2000
      multigrid on the suite median and faster than our own single-grid; GPU benchmarked;
      hot restart works and is used by examples.
- [ ] Implicit-diff gradients validated vs central FD (boundary, profiles, coil dofs, extcur);
      backward memory ≤2× forward; no fingerprint/replay machinery remains.
- [ ] QA/QH/QP/QI examples: single-file, <~120 lines, from circular torus to precise
      configurations with achieved values in docstrings; β-scan free-boundary example with ESSOS
      coils (mgrid + direct, agreeing) featured in README.
- [ ] VMEC++-schema JSON inputs accepted and round-trip converted; `--booz` works out of the box;
      typed zero-crash exceptions throughout.
- [ ] Coverage ≥95% with tests ≤ ~6k lines; goldens in release assets; CI green including example
      smoke tests and a repo-size check.
- [ ] Docs rebuilt per §12 with equations linked to source; README benchmark plot regenerated;
      v0.1.0 on PyPI + conda-forge.

---

## 15. Risks and mitigations

- **History rewrite is destructive** → fresh clone + `git clone --mirror` backup kept until
  v0.1.0; verify trailer count and pack size before force-pushing.
- **Multigrid slowdown may not be recompilation** → Phase-0 profiles decide; alternates to check:
  preconditioner recompute cadence, interpolation quality on restart, `delt` reset policy across
  stages (compare `irst`/`delt` handling line-by-line with `runvmec.f`/`restart.f` and VMEC++
  `_iteration.py`).
- **Deleting 100k+ lines can drop silent capabilities** → NOTES.md records every deleted module's
  purpose; parity + gradient + example tests are the safety net; the mirror-geometry branch and
  pre-rewrite mirror keep everything recoverable.
- **Adjoint linear solve may converge slowly near marginal equilibria** → 1D preconditioner as
  GMRES preconditioner; fall back to more inner iterations; document conditioning diagnostics.
- **QI from a circular torus is genuinely hard** → staged continuation + Goodman-style residual
  live in `optimize.py`, documented; if NFP-specific seeding is truly unavoidable, the example
  docstring says so honestly.
- **ESSOS PR #33 churn** → pin the commit hash in example docstrings; push fixes to the PR as
  rogeriojorge if needed.
- **float32 GPUs** → require x64 at solver import; document the performance implication.
- **Parity tolerance fights** → per-quantity rel+abs tolerances (CompareWOut methodology) with a
  looser current-density tolerance; never invent ad-hoc tolerances per test.

---

## 16. Key references (cite in docs)

- Hirshman & Whitson, Phys. Fluids 26, 3553 (1983) — steepest-descent moment method.
- Hirshman, van Rij & Merkel, Comput. Phys. Commun. 43, 143 (1986) — NESTOR.
- Merkel, J. Comput. Phys. 66, 83 (1986) — vacuum Green's-function method.
- Schilling et al., *The Numerics of VMEC++*, arXiv:2502.04374 — hot restart, JSON, zero-crash,
  validation methodology.
- Skene & Burns, *Fast automated adjoints for spectral PDE solvers*, arXiv:2506.14792 — adjoints
  reusing forward spectral machinery; template for `implicit.py`.
- Blondel et al., *Efficient and Modular Implicit Differentiation*, NeurIPS 2022 (jaxopt) — the
  IFT/custom_vjp formulation (note: jaxopt itself is deprecated; Optax/Optimistix/Lineax are the
  living successors).
- Dudt & Kolemen (2020); Conlin et al. (2023) — DESC: JAX equilibrium solver + implicit-derivative
  optimization precedent.
- Goodman et al., *Constructing precisely quasi-isodynamic magnetic fields*, JPP (2023),
  arXiv:2211.09829 — QI objective; ntor-faster-than-mpol continuation.
- simsopt `examples/2_Intermediate/QH_fixed_resolution.py` — style target for optimization examples.
- STELLOPT VMEC wiki (princetonuniversity.github.io/STELLOPT/VMEC) — INDATA semantics.

---

## Appendix A — wout variable checklist (from VMEC2000 `wrout.f`; implement all)

**Scalars:** `version_`, `input_extension`, `mgrid_file`, `pcurr_type`, `pmass_type`, `piota_type`,
`wb`, `wp`, `gamma`, `rmax_surf`, `rmin_surf`, `zmax_surf`, `nfp`, `ns`, `mpol`, `ntor`, `mnmax`,
`mnmax_nyq`, `iter2→niter`, `itfsq`, `lasym`, `lrecon`, `lfreeb`, `lrfp`, `ier_flag`, `aspect`,
`betatotal`, `betapol`, `betator`, `betaxis`, `b0`, `rbtor0`, `rbtor`, `signgs`, `IonLarmor`,
`volavgB`, `ctor` (/μ0), `Aminor_p`, `Rmajor_p`, `volume_p`, `ftolv`, `fsql`, `fsqr`, `fsqz`,
`nextcur`, `extcur(:)`, `mgrid_mode`; if lfreeb: `mnpd`, `nobser`, `nobd`, `nbsets`, `nbfld(:)`,
`curlabel(:)`.
**Mode arrays:** `xm`, `xn`, `xm_nyq`, `xn_nyq`.
**Axis:** `raxis_cc`, `zaxis_cs` (+ `raxis_cs`, `zaxis_cc` if lasym).
**Profile inputs:** `am`, `ac`, `ai`, `am_aux_s/f`, `ac_aux_s/f`, `ai_aux_s/f`.
**Radial 1D:** `iotaf`, `q_factor` (=1/iotaf), `presf` (/μ0), `phi`, `phipf` (2π·signgs·),
`chi`, `chipf` (2π·signgs·), `jcuru` (/μ0), `jcurv` (/μ0), `iotas`, `mass` (/μ0), `pres` (/μ0),
`beta_vol`, `buco`, `bvco`, `vp`, `specw`, `phips`, `over_r`, `jdotb`, `bdotb`, `bdotgradv`,
`DMerc`, `DShear`, `DWell`, `DCurr`, `DGeod`, `equif`.
**History:** `fsqt(:)`, `wdot(:)`.
**Free-boundary potential (lfreeb):** `potvac` sin (+cos if lasym), `xmpot`, `xnpot` — note VMEC++
skips these; we implement them.
**2D (mode×radius):** `rmnc`, `zmns`, `lmns` (half mesh), `gmnc` (half), `bmnc` (half),
`bsubumnc` (half), `bsubvmnc` (half), `bsubsmns` (full), `currumnc`, `currvmnc`, `bsupumnc`,
`bsupvmnc`; if lfreeb the `*_sur` surface arrays; if lasym all `*mns/*mnc` partners
(`rmns, zmnc, lmnc, gmns, bmns, bsubumns, bsubvmns, bsubsmnc, currumns, currvmns, bsupumns,
bsupvmns` + `*_sur`).
**Tolerances:** CompareWOut-style combined rel+abs per variable; global default (start 1e-10 for
geometry Fourier coefficients, 1e-8 for derived profiles) with a documented looser bound for
`currumnc/currvmnc`; calibrate against actual VMEC2000-vs-VMEC2000 run-to-run scatter in Phase 0.

## Appendix B — VMEC2000 print formats (from `printout.f` / `initialize_radial.f` / `runvmec.f`)

```
Headers:
  iter_line  = "  ITER    FSQR      FSQZ      FSQL   "
  fsq_line   = "   fsqr      fsqz      fsql      DELT    "
  raxis_line = "RAX(v=0) "        zaxis_line = "  ZAX(v=0)      "
  threed1 hdr suffixes: "WMHD      BETA      <M>   DEL-BSQ   FEDGE" (freeb)
                        "WMHD      BETA      <M>"                  (fixed)
Data lines (Fortran FORMATs):
  screen fixed sym : (i5,1p,3e10.2, e11.3, e10.2, e12.4)             ! ITER FSQR FSQZ FSQL RAX DELT WMHD
  screen freeb sym : (i5,1p,3e10.2, e11.3, e10.2, e12.4, e11.3)      ! + DEL-BSQ
  screen lasym     : adds ZAX(v=0) column (2e11.3)
  threed1 sym      : (i6,1x,1p,7e10.2, e11.3, e12.4, e11.3, 0p,f7.3, 1p,2e9.2)
Banners:
  ' FSQR, FSQZ = Normalized Physical Force Residuals' /
  ' fsqr, fsqz = Preconditioned Force Residuals' / 23('-') /
  ' BEGIN FORCE ITERATIONS' / 23('-')
  '  NS = ',i4,' NO. FOURIER MODES = ',i4,' FTOLV = ',1p,e10.3,' NITER = ',i6   (per stage)
  '  VACUUM PRESSURE TURNED ON AT ',i4,' ITERATIONS'
```

## Appendix C — VMEC++ JSON input keys (adopt verbatim; extend for our extra features)

`lasym, nfp, mpol, ntor, ntheta, nzeta, ns_array, ftol_array, niter_array, delt, tcon0, aphi,
phiedge, nstep, pmass_type, am, am_aux_s, am_aux_f, pres_scale, adiabatic_index(=gamma), spres_ped,
ncurr, pcurr_type, ac, ac_aux_s, ac_aux_f, curtor, piota_type, ai, ai_aux_s, ai_aux_f, bloat,
raxis_c, zaxis_s, raxis_s, zaxis_c, rbc, zbs, rbs, zbc, lfreeb, mgrid_file, extcur, nvacskip,
free_boundary_method, lforbal` — boundary coefficients as sparse `{"n": int, "m": int,
"value": float}` lists; axis arrays dense length `ntor+1`. Our extensions (documented, ignored by
VMEC++): `precon_type`, `prec2d_threshold`, spline profile types, mirror-geometry keys (Phase 5.5).

## Appendix D — VMEC2000 algorithm constants (parity-critical; from source)

| Item | Value / rule | Source |
|---|---|---|
| Richardson damping window | `ndamp = 10` | vmec_params.f |
| dtau cap | `bprec·0.15` (bprec=6 with 2D precond on) | evolve.f |
| Velocity update | `xcdot ← (1−dtau)/(1+dtau)·xcdot + delt·gc; xc += delt·xcdot` | evolve.f |
| Precond recompute cadence | every `ns4 = 25` iterations | bcovar.f |
| Jacobian reset (`irst=2`) | restore state, zero velocity, `delt ×= 0.90` | restart.f |
| Residual-growth back-off (`irst=3`) | growth >1e4× best after >10 steps; `delt /= 1.03` | evolve.f/restart.f |
| Escalation | guess_axis on 1st bad jac; delt reset at ijacob=25,50; abort at 75 | eqsolve.f |
| Constraint scaling | `tcon(js)=min(|ard/arnorm|,|azd/aznorm|)·tcon0-scaled·(32hs)²; tcon(ns)=½tcon(ns−1)` | bcovar.f |
| Constraint spectrum | m ∈ [1, mpol−2], weights `faccon(m)` | alias.f |
| m=1 constraint | rotate (gcr,gcz)_{m=1} → ((gcr+gcz)/√2, 0); input `rbss=½(rbs+zbc)` when lconm1 | residue.f90, readin.f |
| Edge pedestal / ZC00 stabilization | 0.05 / fac=0.25 | scalfor.f |
| λ precond | `faclam ∝ 1/(blam(n·nfp)²+clam·m²±2mn·nfp)`, √s damping m>16 | lamcal.f90 |
| Vacuum turn-on | `fsqr+fsqz ≤ 1e-3` | funct3d.f |
| Vacuum cadence | full solve when `mod(iter2−iter1,nvacskip)=0`; adaptive `nvacskip=max(nvskip0, 1/max(0.1,1e11(fsqr+fsqz)))` | funct3d.f |
| Free-bdy constraint ramp | `rcon0,zcon0 ×= 0.9` per iteration | funct3d.f |
| Convergence | `fsqr,fsqz,fsql ≤ ftolv` simultaneously | evolve.f |
| 2D precond activation | finest grid, `fsqr+fsqz+fsql < prec2d_threshold`; GMRES/CG/TFQMR lanes via precon_type | evolve.f, precon2d.f, gmres_mod.f |
