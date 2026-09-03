# What is validated, and against what

This page is the map of VMEX's evidence. For every class of claim the code
makes, it names the gate that enforces it, the tolerance in that gate, and
the artifact behind the number. It also names what is *not* validated, which
is the part a reader checking a solver actually needs.

Two rules run through the whole page. Where a number appears here it is a
tolerance some test asserts or a value some committed artifact records — not
a remembered result. And where the honest answer is "this is a record, not a
gate", the page says so instead of implying coverage.

## The evidence hierarchy

Not all agreement is worth the same. VMEX's evidence falls into five tiers,
strongest first, and the rest of this page is organized by them.

1. **Analytic limits.** The answer is known in closed form, so agreement is
   evidence about the code and nothing else. The mirror module's paraxial
   and vacuum identities are here.
2. **An independent oracle.** A second implementation of the physics, written
   against a different discretization, scores the same equilibrium. VMEX's
   continuum force-balance oracle is here, and it is the only tier that can
   compare VMEX against other codes on a quantity none of them optimizes.
3. **Reference-code parity.** VMEC2000 is the reference implementation; VMEX
   reproduces its trajectory and its output file. This is the largest tier by
   test count, and its weakness is structural — it can only show that VMEX
   agrees with VMEC2000, never that either is right.
4. **Internal consistency.** Adjoint against finite difference, forward
   against transpose, CPU against accelerator, one lane against another.
   These catch implementation error, not modelling error.
5. **Committed records without a gate.** Measurements that are stored and
   hashed but that no test asserts on. They are provenance, not validation,
   and this page marks them as such.

`tests/manifest.json` classifies every test by the oracle it checks against,
so the balance of tiers is countable rather than asserted:

```bash
python -c "import json,collections; print(collections.Counter(
    r[6] for r in json.load(open('tests/manifest.json'))['records']))"
```

At the time of writing that reads `analytic` 46, `vmec2000` 18, `fd` 16,
`none` 15, `external` 10, `golden` 7. Run it rather than trusting the
snapshot; it moves with every test added.

## VMEC2000 parity, tier by tier

Parity is checked at four different resolutions of detail. Each tier is a
separate suite with its own gate.

### Trajectory: the same iterations, not just the same answer

`tests/test_parity_breadth.py` runs six decks — `DSHAPE`,
`circular_tokamak`, `li383_low_res`, `LandremanPaul2021_QA_lowres`,
`nfp4_QH_warm_start`, `up_down_asymmetric_tokamak` — against stored VMEC2000
goldens and asserts:

| quantity | gate |
|---|---|
| convergence | `fsqr`, `fsqz`, `fsql` all at or below the deck `ftol` |
| iteration count | within `[0.75 g, ceil(1.25 g)]` of the golden `g` |
| `wb` | relative difference below `1e-7` |
| mode tables `xm`, `xn` | bit-exact |
| `iotaf` | `rtol=1e-5` |
| `rmnc`, `zmns` (and the LASYM partners) | `rtol=1e-5` on three surfaces: axis-adjacent, mid, edge |

The iteration window is the one to read carefully. Iteration counts are the
parity quantity that legitimately moves with the floating-point path, so the
enforced gate is `+-25%`, not equality — even though the counts observed on
the recorded runs match exactly on most decks. Anywhere this documentation
prints an iteration table, it is reporting what a run did, not what CI
requires.

Two decks carry a looser harmonic `atol` because their goldens are
NITER-capped rather than converged: `5e-6` for `LandremanPaul2021_QA_lowres`
and `2e-5` for `up_down_asymmetric_tokamak`. For the latter, a fully
converged VMEC2000 rerun agrees with VMEX's converged core to `7.3e-7` on
every checked harmonic. Two decks are deliberately absent:
`NuhrenbergZille_1988_QHS` (over the 120 s budget, no golden in the bundle)
and `cth_like_free_bdy_lasym_small` (free boundary).

These tests need no VMEC2000 binary — they read a stored, sha256-verified
golden bundle, resolved from `VMEX_GOLDEN_DIR`, then `~/vmex_notes/golden`,
then a verified download. Tests that do need a live binary are marked
`vmec2000_live` and run only under `--run-vmec2000`.

```{figure} /_static/figures/readme_convergence.webp
:alt: force-residual traces for VMEX, VMEC2000 and VMEC++ on the NFP=4 QH case
:align: center
:width: 90%

Parity along the whole trajectory, not only at the endpoint: total force
residual per iteration for the bundled NFP=4 QH case at `ns=51` through all
three codes. Regenerate with
`python benchmarks/make_readme_figures.py --only convergence`.
```

### Scalars: a cheap gate that needs no golden bundle

`tests/golden_digests.json` stores ten scalars per case (`wb`, `wp`,
`aspect`, `volume_p`, `betatotal`, `b0`, `betapol`, `betator`, `rmax_surf`,
`rmin_surf`), the `iotaf` and `presf` endpoints, and two boundary geometry
checksums. `tests/test_golden_digests.py` compares them at `rtol` `2e-4` for
`wb` and `aspect`, `5e-4` for the volume, field, extent and geometry
checksums, `3e-3` for the `iotaf` endpoints, and `3e-3` by default, all with
`atol=1e-8` so quantities passing through zero do not produce false
failures. Two cases run on every pull request; five more run under
`RUN_FULL=1`. Regenerate with `python tools/make_golden_digests.py`.

### The output file: every variable, by class

`tests/test_wout_golden.py` compares a written `wout` against the golden
one variable at a time on `solovev`, `cth_like_fixed_bdy`, `li383_low_res`
and `up_down_asymmetric_tokamak`. It asserts structure first (same
variables, dimensions and dtypes, no unexpected additions) and then values
under a per-variable policy rather than one global tolerance:

- geometry, pressure, flux and axis arrays: `rtol=1e-6`, `atol=1e-7`;
- Jacobian, field and contravariant families: `rtol=5e-5`;
- `iotaf`/`iotas`: `rtol=1e-5`; scalars: `rtol=1e-6`;
- near-zero covariant channels and near-axis rows get their own atol and skip
  the first two surfaces, because a relative tolerance is meaningless there;
- the Mercier family gets `rtol=5e-2` with a scale-relative atol.

The loose Mercier tier is not a claim that Mercier terms agree to 5%; it is a
bound chosen to catch normalization regressions in quantities built from
nested radial derivatives. A separate drift tier applies to decks whose
`ftol` is above `1e-9`, where the two trajectories are not comparable
coefficient by coefficient.

### Fresh decks, never benchmarked before

`benchmarks/fresh_decks_vs_vmec2000_2026-09-02.json` records six decks that
had never been run on VMEX, solved against a locally built `xvmec2000`
(sha256 prefix `f7e9034f7d9d7ae5`): `ITERModel`, `estell_24_scaled`,
`n3are_R7.75B5.7`, `HSX_QHS_vacuum_ns201`, `W7-X_standard_configuration`
and `NuhrenbergZille_1988_QHS`, spanning tokamak, vacuum, finite beta and
net-current cases.

Quote it by its measured maxima, never as "machine precision":

- worst relative difference over all decks and all compared fields:
  **`2.5e-10`**, on `iotaf` for `HSX_QHS_vacuum_ns201`;
- worst `betatotal` difference `2.2e-13`; worst `volume_p` difference
  `1.2e-16`;
- worst boundary coefficient difference `1.4e-17` where it was measured;
- iteration counts identical on five of six decks, and different by exactly
  one on `ITERModel` (1469 against 1470); Jacobian resets identical wherever
  recorded.

`tests/test_performance_docs.py::test_fresh_deck_parity_artifact_is_provenanced_and_cited`
guards the record's provenance and requires the docs to cite it by path.

## The independent oracle, and the cross-code table

Reference-code parity cannot say whether VMEC2000 is right. For that VMEX
carries a separate continuum evaluation of the residual
$\mathbf{J} \times \mathbf{B} - \nabla p$ on a high-order B-spline lift of a
converged state (`vmex.core.strong_force`), overintegrated to produce a
certificate independent of the solver's own discretization. It is validated
in its own right before it is used as a judge:

- against an analytic constant-toroidal-field case, `rtol` `2e-12` on
  $\mathbf{B}$ and `2e-11` on $\mathbf{J}$ and the force;
- against DESC's pointwise current and force on a stored circular case,
  `rtol=3e-13` on the Jacobian and field and `rtol=3e-10` on the current and
  force components.

Because the oracle is applied to *each code's exported equilibrium*, it gives
the one genuinely comparable cross-code number. From
`benchmarks/strong_force_comparison_m4.json`, normalized force-balance $L_2$:

| source | shaped tokamak, finite pressure | NFP=2 QA, finite beta |
|---|---|---|
| VMEX (certified polished) | **0.00179** | 0.52590 |
| VMEC2000 | 0.01711 | 0.52518 |
| VMEC++ | 0.01711 | 0.52518 |
| DESC | 0.02962 | 0.87926 |

Read both rows. On the tokamak, VMEX's polished state reaches roughly a
tenth of the VMEC2000 and VMEC++ residual, which agree with each other to
nine digits. **On the stellarator VMEX is not better** — 0.5259 against
0.5252 is a tie, marginally on the wrong side, and only DESC is clearly
worse. The stellarator row is published as a tie rather than omitted; the
3-D production polish that would move it is not yet tractable. The polish
gain itself is recorded as a before/after pair on the tokamak, `0.0505` to
`0.0019`.

The figure and every source hash are guarded by
`tests/test_performance_docs.py::test_readme_strong_force_figure_matches_committed_sources`,
and the renderer refuses to draw any source measured from a dirty tree or a
failed external solve. That guard checks provenance and one ordering claim;
it does not re-run the physics.

## Derivative certificates

Gradients come from the implicit function theorem at the converged fixed
point, so the question is whether the linear solve is the true derivative.

**Fixed boundary** (`tests/test_implicit_grad.py`). Four adjoint gradients on
`solovev` — `d(wb)/d(RBC)`, `d(aspect)/d(RBC)`, `d(wb)/d(phiedge)`,
`d(wp)/d(pres_scale)` — must match central finite differences to
`rel <= 1e-6`. The 3-D `li383_low_res` boundary gradient is checked at
`rtol=2e-4` against a finite difference whose own noise floor is about
`3e-5`. The adjoint preconditioner has its own certificate: the
preconditioned residual falls below `1e-10` within 300 matrix-vector
products while the raw residual stays above `1e-6`, a ratio above `1e4`.

**Free boundary** (`tests/test_freeboundary_implicit.py`). The coupled
plasma-vacuum root's reverse derivative is certified factor by factor
(relative error below `1e-6`), against independent free-boundary re-solves,
and cross-checked between two adjoint formulations. The most informative
gate is the honesty one: the gradient's agreement is bounded *both* sides,
`1e-6 < gap < 1e-2`, so the test fails if the claimed agreement is tighter
than the solver's own root reproducibility.

**End to end** (`tests/test_examples.py::test_take_gradients`). The bundled
example prints its adjoint-versus-finite-difference agreement and the test
fails above `1e-4`. Documentation quotes that gate rather than a particular
run's number.

For solver-sensitive outputs — `iota`, `DMerc`, `jdotb`, `D_R` — a naive
re-solving finite difference is not a valid reference at all, and can even
sign-flip. The frozen-path check described in
{doc}`/explanation/adjoint-gradients` is the reference there.

## Free boundary: the ladder

Free-boundary evidence is built in rungs, each with its own gate.

1. **The vacuum operator.** `tests/test_freeboundary.py` pins the NESTOR
   solve against a reference at `rtol=1e-12`, asserts bit-exactness of the
   skip branch, and matches the first-call diagnostics against the golden
   VMEC2000 print block.
2. **The radial ladder with an mgrid.**
   `tests/test_freeboundary_multigrid.py` matches VMEC2000 through the
   pre-vacuum stages, pins `r00` to `2e-10` relative, and requires the
   converged multigrid state to agree within `1e-2` on geometry and `1.5e-2`
   on `iota`.
3. **A second 3-D geometry.** `tests/test_ncsx_free_boundary_parity.py`
   reproduces the published NCSX c09r00 mgrid currents at `rtol=1e-12` and
   the field components at `rtol=1e-8`, from a coil file recorded by sha256.
4. **High mode count.** `tests/test_high_mode_free_boundary_parity.py` takes
   the 238-mode CTH-like free-boundary ladder to `fsqr <= 1e-8` and pins
   `r00` and `wb` against VMEC2000 at `1e-4` and `2e-4`.
5. **Finite beta.** The beta-scan panels are recorded in
   `docs/_static/figures/freeb_diiid_mgrid_beta_ns101_panel_summary.csv`
   (mgrid, to 3.33% actual beta) and
   `freeb_lpqa_direct_coil_beta_ns101_panel_summary.csv` (direct coils, to
   1.93%), both holding `fsq_total` near `1e-12` across the scan.

Rung 5 is tier 5 evidence: those CSVs, and
`docs/_static/figures/pr20_wout_parity_summary.json`, are committed records
that **no test asserts on**. They are provenance for a campaign that was
run, not gates that would fail if the physics regressed. Read them as such.

## Mirror geometry: analytic limits

The mirror module is the one place where the answer is known in closed form,
so `tests/mirror/test_analytic.py` is tier 1 evidence. Vacuum and flux
identities hold to round-off: curl and divergence residuals at `atol` `2e-15`
to `2e-14`, flux preservation at `rtol=3e-15`, the Riccati residual of the
rotating-ellipse construction at `2e-15`, and the endpoint quarter-turn to
`2e-15`.

Two gates are genuinely physical rather than round-off. The straight-field-line
mirror's ellipticity and its flux-area-times-axis-field product are checked at
`rtol=5e-4` — that is the paraxial truncation error at the chosen aspect
ratio, not a numerical tolerance. And
`tests/mirror/test_boundary_conditions.py` checks long-thin pressure balance
with an *aspect-ratio-scaled* bound: the deviation of $p + B^2/2\mu_0$ across
the midplane must stay within $1.5\,\epsilon^2$ of the magnetic scale,
where $\epsilon = a/L$, and the magnetic rise must cancel the pressure drop
to $2\,\epsilon^2$. That is the diamagnetism certificate.

## Downstream parity

Equilibria are inputs to other codes, so several tests check what those codes
see rather than what VMEX stores.

| consumer | compared | gate |
|---|---|---|
| Boozer tables (`vmex.core.boozer_tables`) | `bmnc`, `R`/`Z` tables, `iota`/`G`/`I` against the wout rows | `rtol=1e-8` |
| ESSOS field handoff | `AbsB` and coordinates, wout path against in-memory equilibrium | `rtol=0.0` (exact) |
| `neo_jax` effective ripple | `epsilon_eff` on three surfaces | `rtol=5e-5` |
| DESC | pointwise current and force through the shared oracle | `rtol=3e-13` / `3e-10` |

The ESSOS and `neo_jax` tests are `importorskip`-gated, so they run in the
nightly optional-integrations lane and silently skip in the dependency-minimal
core lane. A green core run is not evidence that they passed.

## Device and lane consistency

Float64 is required and enforced at solver import. Across devices the
contract is numerical equivalence, not bit-identity — the batched tridiagonal
solver on an accelerator is a different algorithm from the CPU Thomas sweep.

`tests/test_gpu_ci.py` (marked `gpu`, skipped without a real device) pins
CPU-against-GPU agreement per quantity: `1e-10` for the confinement
diagnostics, `5e-10` for the forward solve, `5e-9` for a converged LASYM free
boundary, and `2e-7` for implicit gradients, which are the loosest because a
reverse pass accumulates over the whole trajectory. A separate test asserts
that no `JAX_PLATFORMS` pin is in the environment and that the default
backend really is the GPU, so a silently-CPU run cannot pass as a GPU lane.
`benchmarks/device_parity.py` is the standalone audit; its default tolerance
is `rtol=1e-7`, recorded in every output JSON it writes.

Lane consistency — jit against eager, and the traced against the stepped
lane — is checked where it matters rather than in one suite: the Boozer
tables, the setup and geometry chains, the mgrid path and the bootstrap
state-versus-wout lanes each carry their own comparison. Note that the test
session disables JIT globally and solver-heavy modules opt back in, so a
module without that fixture is exercising the interpreted path.

## What is NOT validated

This section is deliberately specific. Absence of a claim here does not mean
a claim exists elsewhere.

- **Free-boundary forward derivatives.** JVPs are `not-available`. The
  reverse derivative of the reconverged free-boundary root is `limited` and
  explicitly experimental, CPU only. Low-memory GPU compilation and failed-trial
  handling are open promotion gates. `tests/test_capability_docs.py`
  asserts these statuses, so the claim cannot quietly widen.
- **Mirror beta above 10%.** The axisymmetric open-mirror free-boundary lane
  is supported to 10% requested beta. The 25%, 50% and 80% cases converge
  variationally and the 80% case passes its force gate, but refined-grid
  promotion is incomplete and they remain extended validation.
- **Anisotropic and ANIMEC-derived equilibria.** Not implemented.
- **Untested `wout` modes.** Parity is claimed for the variables the golden
  comparison covers. Fill-valued or untested modes are not covered.
- **Parsed input names.** Recognizing a VMEC2000 namelist name is not
  evidence that the solver uses it. Unsupported physics fails with a typed
  error rather than being silently ignored.
- **The 2D block preconditioner as a speed feature.** Its evidence is an
  iteration-count reduction and a `wb` agreement, not a wall-clock or memory
  win; it is opt-in for that reason.
- **The oracle against an exact solution.** The independent oracle is
  validated against an analytic constant-field case and against DESC, but
  there is no analytic Solov'ev equilibrium in its validation set yet.
- **`virtual_casing.py` and `turbulence.py`** are excluded from the coverage
  gate, because their finite-difference gradient tests are optional-dependency
  gated and skip in the core lane.
- **Memory figures** in {doc}`/reference/performance` are measurements on
  particular decks, not high-resolution upper bounds.

## How to rerun any of this

The local gate before pushing:

```bash
python tools/preflight.py --static   # lint, types, docs prose, guard tests
python tools/preflight.py            # also the diff-affected suites
python tools/preflight.py --docs     # also a warning-free sphinx build
```

Test selection is manifest-driven, not path-driven. Every CI lane resolves
its files the same way:

```bash
python tools/test_manifest.py select pr-fast
RUN_FULL=1 pytest -q $(python tools/test_manifest.py select full-core-d-f0)
```

Four pytest markers gate the slower evidence: `full` (needs `RUN_FULL=1`),
`weekly` (high-resolution campaigns, excluded from nightly), `gpu` (needs a
real device) and `vmec2000_live` (needs `--run-vmec2000` and a local
`xvmec2000`). There is no `nightly`, `slow` or `live` marker. Passing
`--vmex-report PATH` writes a machine-readable record of what ran, what was
slowest, and — importantly for reading a green run — every test that skipped
and why.

Regenerating the artifacts this page cites:

```bash
python tools/fetch_assets.py --bundle golden-v1     # the VMEC2000 goldens
python tools/make_golden_digests.py                 # tests/golden_digests.json
python benchmarks/run_baseline.py                   # benchmarks/baseline.json
python benchmarks/run_freeboundary_multigrid.py     # the free-boundary ladder
python benchmarks/run_gpu_matrix.py                 # benchmarks/gpu_baseline.json
python benchmarks/preconditioner_2d_stiff.py        # the 2D preconditioner run
python benchmarks/make_strong_force_comparison.py   # the cross-code table
python benchmarks/device_parity.py --devices cpu,gpu --output parity.json
```

Figures carry their own provenance in
`docs/_static/figures/figures.json` — generator, inputs, date, hardware, and
whether one command reproduces the committed file. `python
tools/update_figure_manifest.py` refreshes it and
`tests/test_figure_provenance.py` fails when a figure and its row disagree.
