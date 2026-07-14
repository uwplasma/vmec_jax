"""Smoke tests for the plan.md §10 examples at reduced budgets.

Each example script reads ``VMEC_JAX_EXAMPLES_CI=1`` (parameters-at-top
hook) and shrinks its continuation schedule / trial budget; the tests run
the scripts as subprocesses in a temp cwd and assert that

- the script exits cleanly,
- the reported least-squares cost decreases from its first evaluation,
- the promised outputs (optimized input deck, wout file, figures) exist.

The "commented-out but CI-tested" extra objective terms of the optimization
examples (magnetic well, DMerc floor, L_grad_B floor) are exercised
*uncommented* in ``test_extra_terms_work_uncommented`` — exactly the
expressions the example comments show, one finite-difference least-squares
iteration each, plus the traceable magnetic-well term through
``jac="implicit"``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("netCDF4")

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
DATA_DIR = EXAMPLES / "data"

_COST_RE = re.compile(r"\[least_squares\] cost = ([0-9.eE+-]+)")


def _run_example(script: Path, cwd: Path, timeout: int = 2400) -> str:
    env = dict(os.environ, VMEC_JAX_EXAMPLES_CI="1")
    env.pop("JAX_DISABLE_JIT", None)
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"{script.name} failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-4000:]}")
    return proc.stdout


def _assert_cost_decreased(stdout: str, name: str) -> None:
    costs = [float(c) for c in _COST_RE.findall(stdout)]
    assert len(costs) >= 2, f"{name}: expected verbose cost lines, got {costs}"
    assert min(costs) < costs[0], (
        f"{name}: least-squares cost did not decrease: first {costs[0]:.6e}, "
        f"best {min(costs):.6e}")


def test_fixed_boundary_run(tmp_path):
    out = _run_example(EXAMPLES / "fixed_boundary_run.py", tmp_path, timeout=900)
    assert "converged = True" in out
    outdir = tmp_path / "output_fixed_boundary_run"
    assert (outdir / "wout_li383_low_res.nc").exists()
    assert (outdir / "li383_low_res_summary.png").exists()


def test_plot_and_boozer(tmp_path):
    out = _run_example(EXAMPLES / "plot_and_boozer.py", tmp_path, timeout=900)
    assert "converged = True" in out
    outdir = tmp_path / "output_plot_and_boozer"
    assert (outdir / "wout_li383_low_res.nc").exists()
    # every plot_wout figure kind is written unconditionally
    for suffix in ("summary", "surfaces", "modB", "profiles", "boundary3d"):
        assert (outdir / f"li383_low_res_{suffix}.png").exists()


def test_profiles_power_and_spline(tmp_path):
    out = _run_example(EXAMPLES / "profiles_power_and_spline.py", tmp_path, timeout=900)
    # both profile representations converge to the same equilibrium
    assert out.count("converged=True") == 2
    match = re.search(r"\|d aspect\| = ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(1)) < 1e-3


@pytest.mark.full  # nightly: ~1 min (2 adjoint grads + 4 FD solves, subprocess cold-start)
def test_take_gradients(tmp_path):
    out = _run_example(EXAMPLES / "take_gradients.py", tmp_path, timeout=900)
    # both implicit-adjoint gradients agree with central finite differences
    rels = [float(m) for m in re.findall(r"rel=([0-9.eE+-]+)", out)]
    assert len(rels) == 2, f"expected two AD-vs-FD checks, got {rels}"
    assert max(rels) < 1e-4, f"adjoint gradient disagrees with FD: rel={rels}"


def test_run_from_json(tmp_path):
    out = _run_example(EXAMPLES / "run_from_json.py", tmp_path, timeout=900)
    match = re.search(r"\|diff\|=([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(1)) < 1e-6
    assert (tmp_path / "output_run_from_json" / "circular_tokamak.json").exists()
    assert (tmp_path / "output_run_from_json" / "wout_circular_tokamak.nc").exists()


def test_hot_restart_scan(tmp_path):
    out = _run_example(EXAMPLES / "hot_restart_scan.py", tmp_path, timeout=900)
    base = re.search(r"cold base solve:\s*(\d+) iters", out)
    warm = [int(m) for m in re.findall(r"^\s*[0-9.]+\s+(\d+)\s+[0-9.]+\s+warm", out, re.M)]
    assert base is not None and int(base.group(1)) > 10, "base should need many iters"
    assert len(warm) == 5 and max(warm) <= 5, f"warm restarts should be cheap: {warm}"


@pytest.mark.full  # nightly: free-bdy NESTOR solve ~10s; parity already covered in shard-a
def test_free_boundary_mgrid(tmp_path):
    out = _run_example(EXAMPLES / "free_boundary_mgrid.py", tmp_path, timeout=900)
    assert "converged = True" in out
    assert (tmp_path / "output_free_boundary_mgrid" / "wout_cth_like_free_bdy.nc").exists()


def test_take_free_boundary_gradients(tmp_path):
    # skips where the optional virtual_casing_jax dep is absent (core CI);
    # validates the FD-checked coil/extcur gradients where it is installed.
    pytest.importorskip("virtual_casing_jax")
    out = _run_example(EXAMPLES / "take_free_boundary_gradients.py", tmp_path, timeout=900)
    # each gradient row ends with its AD-vs-FD relative error in scientific notation
    rels = [float(m) for m in re.findall(r"\s([0-9.]+e[+-]\d+)\s*$", out, re.M)]
    assert "FD-validate" in out and rels and max(rels) < 1e-3, f"gradient rel errors: {rels}"


@pytest.mark.full  # nightly: one NESTOR solve per pressure point (~40s)
def test_free_boundary_beta_scan(tmp_path):
    out = _run_example(EXAMPLES / "free_boundary_beta_scan.py", tmp_path, timeout=1200)
    betas = [float(b) for _, b in re.findall(
        r"^\s*([0-9.]+)\s+([0-9.eE+-]+)\s+[0-9.]+\s+\d+\s*$", out, re.M)]
    assert len(betas) == 3 and betas[-1] > 1e-2, f"beta should reach finite values: {betas}"


@pytest.mark.full  # nightly: free-bdy NESTOR solve with direct-coil Biot-Savart (~30s)
def test_free_boundary_essos_coils(tmp_path):
    pytest.importorskip("essos")
    out = _run_example(EXAMPLES / "free_boundary_essos_coils.py", tmp_path, timeout=900)
    # table rows: nominal%  PRES_SCALE  actual-beta%  iters  fsq  aspect  axis-R
    rows = re.findall(r"^\s*([0-9.]+)%\s+([0-9.]+)\s+([0-9.]+)%\s+\d+\s+([0-9.eE+-]+)",
                      out, re.M)
    assert len(rows) == 1, f"CI mode should solve exactly one beta point:\n{out}"
    nominal, _pres_scale, actual, fsq = (float(x) for x in rows[0])
    assert abs(actual - nominal) <= 0.15, (
        f"actual betatotal {actual}% not calibrated to nominal {nominal}%")
    assert fsq < 1e-7, f"free-boundary point should converge, fsq={fsq}"


def test_finite_beta_scan(tmp_path):
    out = _run_example(EXAMPLES / "finite_beta_scan.py", tmp_path, timeout=900)
    # rows: pres_scale  beta_tot  R_axis  Shafranov  minDMerc
    rows = re.findall(r"^\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+[0-9.]+\s+([+-][0-9.]+)\s",
                      out, re.M)
    betas = [float(b) for _, b, _ in rows]
    shafr = [float(s) for _, _, s in rows]
    assert len(betas) == 3, f"expected 3 pressure points, got {rows}"
    assert betas[-1] > betas[0] and betas[-1] > 5e-3, "beta should rise into finite-beta"
    assert shafr[-1] > shafr[0], "magnetic axis should shift outward (Shafranov)"


@pytest.mark.parametrize("case", [
    "QA",  # PR smoke: proves the QS optimization pipeline end-to-end
    pytest.param("QH", marks=pytest.mark.full),  # nightly (subprocess cold-start heavy)
    pytest.param("QP", marks=pytest.mark.full),
])
def test_qs_optimization_examples(case, tmp_path):
    script = EXAMPLES / "optimization" / f"{case}_optimization.py"
    out = _run_example(script, tmp_path)
    _assert_cost_decreased(out, case)
    outdir = tmp_path / f"output_{case}_optimization"
    assert (outdir / f"input.{case}_optimized").exists()
    assert (outdir / f"wout_{case}_optimized.nc").exists()
    assert (outdir / f"{case}_optimized_summary.png").exists()
    # the final printout carries the achieved QS total (docstring claim hook)
    match = re.search(r"QS total: seed ([0-9.eE+-]+) -> final ([0-9.eE+-]+)", out)
    assert match is not None and np.isfinite(float(match.group(2)))


@pytest.mark.full  # nightly: single-stage ESS variants (one least_squares call, no ladder)
@pytest.mark.parametrize("case", ["QA", "QI"])
def test_ess_optimization_examples(case, tmp_path):
    script = EXAMPLES / "optimization" / f"{case}_optimization_ess.py"
    out = _run_example(script, tmp_path)
    _assert_cost_decreased(out, f"{case}-ESS")
    assert "one call, no max_mode ladder" in out


@pytest.mark.full  # nightly: QP-basin + QI stages + Boozer, subprocess cold-start heavy
def test_qi_optimization_example(tmp_path):
    pytest.importorskip("booz_xform_jax")
    script = EXAMPLES / "optimization" / "QI_optimization.py"
    out = _run_example(script, tmp_path)
    _assert_cost_decreased(out, "QI")
    match = re.search(r"QI total: seed ([0-9.eE+-]+) -> final ([0-9.eE+-]+)", out)
    assert match is not None
    seed, final = float(match.group(1)), float(match.group(2))
    assert np.isfinite(final) and final <= seed * 1.05
    outdir = tmp_path / "output_QI_optimization"
    assert (outdir / "wout_QI_optimized.nc").exists()


def test_extra_terms_work_uncommented():
    """The commented-out example terms run as objectives when uncommented."""
    import jax

    from vmec_jax.core.input import VmecInput
    from vmec_jax.core import optimize as opt

    was_disabled = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)  # conftest disables jit globally
    try:
        inp = VmecInput.from_file(DATA_DIR / "input.minimal_seed_nfp2")
        # exactly the expressions the example comments show:
        extra_terms = [
            (opt.magnetic_well, 0.05, 1.0),
            (lambda eq: np.minimum(opt.d_merc(eq)[2:-1], 0.0), 0.0, 100.0),
            (lambda eq: max(1.0 / opt.l_grad_b(eq) - 1.0 / 0.35, 0.0), 0.0, 1.0),
        ]
        terms = [(opt.aspect_ratio, 5.0, 1.0)] + extra_terms
        result = opt.least_squares(terms, inp, max_mode=1, jac=None,
                                   use_ess=True, max_nfev=2)
        assert np.all(np.isfinite(result.fun))
        # the traceable extra term also differentiates through jac="implicit"
        result2 = opt.least_squares(
            [(opt.aspect_ratio, 5.0, 1.0), (opt.magnetic_well, 0.05, 1.0)],
            inp, max_mode=1, jac="implicit", use_ess=True, max_nfev=2)
        assert np.all(np.isfinite(result2.fun))
        # host wout-engine terms are rejected with a clear error in implicit mode
        with pytest.raises(ValueError, match="implicit-differentiable"):
            opt.least_squares([(opt.d_merc, 0.0, 1.0)], inp, max_mode=1,
                              jac="implicit", max_nfev=1)
    finally:
        jax.config.update("jax_disable_jit", was_disabled)


# The self-consistent-bootstrap examples reproduce arXiv:2205.02914 against the
# Zenodo dataset, which is a large local-only archive (not in CI) — skip when
# absent.  Nightly-gated: each runs a multi-iteration Picard loop of solves.
_ZENODO_2205 = Path(os.environ.get(
    "VMEC_JAX_ZENODO_2205_02914",
    str(Path.home() / "local" /
        "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current")))


@pytest.mark.full
@pytest.mark.skipif(not _ZENODO_2205.is_dir(),
                    reason="arXiv:2205.02914 Zenodo dataset not present")
@pytest.mark.parametrize("case", ["QA", "QH"])
def test_bootstrap_selfconsistent_examples(case, tmp_path):
    script = EXAMPLES / "optimization" / f"{case}_bootstrap_selfconsistent.py"
    out = _run_example(script, tmp_path, timeout=1200)
    m = re.search(r"final f_boot = ([0-9.eE+-]+)", out)
    assert m is not None and float(m.group(1)) < 5e-2, f"{case} f_boot: {out[-400:]}"
    assert (tmp_path / f"output_{case}_bootstrap_selfconsistent"
            / f"wout_{case}_bootstrap_selfconsistent.nc").exists()


@pytest.mark.full
def test_single_stage_free_boundary_opt(tmp_path):
    pytest.importorskip("virtual_casing_jax")
    wout = EXAMPLES / "data" / "single_grid" / "wout_cth_like_free_bdy.nc"
    if not wout.exists() or not (EXAMPLES / "data" / "mgrid_cth_like.nc").exists():
        pytest.skip("fetched free-boundary assets absent (tools/fetch_assets.py)")
    out = _run_example(EXAMPLES / "single_stage_free_boundary_opt.py", tmp_path, timeout=900)
    m = re.search(r"recovered to ([0-9.]+)% of the confining", out)
    assert m is not None and float(m.group(1)) < 5.0, f"coil recovery: {out[-400:]}"


@pytest.mark.full  # nightly: true single-stage boundary+coil co-optimization, 2 cases
def test_single_stage_essos_coils_opt(tmp_path):
    # coil-agnostic single-stage: needs both the ESSOS coils and virtual_casing_jax
    pytest.importorskip("essos")
    pytest.importorskip("virtual_casing_jax")
    out = _run_example(EXAMPLES / "single_stage_essos_coils_opt.py", tmp_path, timeout=1800)
    # one machine-parseable "J <initial> -> <final>" line per case; each must drop
    rows = re.findall(r"\[single_stage\] (\S+): J ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert len(rows) == 2, f"expected two single-stage cases, got {rows}\n{out[-800:]}"
    for name, j0, jf in rows:
        assert float(jf) < float(j0), f"{name}: single-stage J did not decrease ({j0} -> {jf})"
    outdir = tmp_path / "output_single_stage_essos_coils_opt"
    for name, _j0, _jf in rows:
        assert (outdir / f"wout_{name}_final.nc").exists()
