"""Smoke tests for the plan.md §10 examples at reduced budgets: each script
reads ``VMEX_EXAMPLES_CI=1`` and shrinks its budget; the tests run them as
subprocesses and assert clean exit, decreasing least-squares cost, and the
promised outputs.  The examples' "commented-out but CI-tested" extra
objective terms are exercised uncommented in
``test_extra_terms_work_uncommented``.
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

_COST_RE = re.compile(r"^\s*\d+\s+\d+\s+([0-9.eE+-]+)", re.MULTILINE)


def _run_example(script: Path, cwd: Path, timeout: int = 2400,
                 args: tuple[str, ...] = ()) -> str:
    env = dict(os.environ, VMEX_EXAMPLES_CI="1")
    env.pop("JAX_DISABLE_JIT", None)
    proc = subprocess.run(
        [sys.executable, str(script), *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"{script.name} failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-4000:]}")
    return proc.stdout


def _assert_cost_decreased(stdout: str, name: str) -> None:
    costs = [float(c) for c in _COST_RE.findall(stdout)]
    assert len(costs) >= 2, f"{name}: expected scipy iteration rows, got {costs}"
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
    for suffix in ("summary", "surfaces", "modB", "profiles", "stability", "boundary3d"):
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


def test_parallel_ensemble_scan(tmp_path):
    out = _run_example(EXAMPLES / "parallel_ensemble_scan.py", tmp_path, timeout=900)
    # the correctness contract: threaded ensemble is bit-identical to serial
    assert "max|state diff| vs serial = 0.0e+00" in out
    assert "iterations identical: True" in out
    # the strong-scaling table printed at least one worker row
    assert re.search(r"^\s*\d+\s+[0-9.]+\s+[0-9.]+x", out, re.M) is not None


@pytest.mark.full  # nightly: free-bdy NESTOR solve ~10s; parity already covered in shard-a
def test_free_boundary_mgrid(tmp_path):
    out = _run_example(EXAMPLES / "free_boundary_mgrid.py", tmp_path, timeout=900)
    assert "converged = True" in out
    assert (tmp_path / "output_free_boundary_mgrid" / "wout_cth_like_free_bdy.nc").exists()


def test_take_free_boundary_gradients(tmp_path):
    # skips where the optional virtual_casing_jax dep is absent (core CI);
    # validates the FD-checked coil/extcur gradients where it is installed.
    pytest.importorskip("virtual_casing_jax")
    wout = EXAMPLES / "data" / "single_grid" / "wout_cth_like_free_bdy.nc"
    if not wout.exists():
        pytest.skip("fetched free-boundary asset absent (tools/fetch_assets.py)")
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


@pytest.mark.full
def test_mirror_fixed_boundary_nonaxisymmetric_example(tmp_path):
    import json
    _run_example(
        EXAMPLES / "mirror_fixed_boundary_nonaxisymmetric.py",
        tmp_path,
        timeout=1200,
    )
    outdir = tmp_path / "results" / "mirror_fixed_boundary_nonaxisymmetric"
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["rotating_ellipse"]["status"] == "supported"
    assert summary["rotating_ellipse"]["variational_max"] < 1.0e-12
    assert summary["rotating_ellipse"]["strong_force_normalized_rms"] < 5.0e-2
    assert summary["rotating_ellipse"]["boundary_gradient_relative_error"] < 1.0e-4
    assert summary["rotating_ellipse"]["adjoint_relative_residual"] < 1.0e-8
    assert summary["straight_field_line"]["status"].startswith("paraxial")
    assert summary["straight_field_line"]["variational_max"] < 1.0e-12
    assert summary["straight_field_line"]["final_linear_residual"] < 1.0e-8
    assert summary["straight_field_line"]["linear_iterations"] < 1000
    # Paraxial benchmark: the unconstrained bulk force is clean and gated,
    # while the expected cut boundary layer dominates the all-volume and
    # end-collar norms (device-normalized all-volume above 0.1).
    assert summary["straight_field_line"]["strong_force_bulk_rms"] < 5.0e-2
    assert (
        summary["straight_field_line"]["strong_force_end_collar_rms"]
        > summary["straight_field_line"]["strong_force_bulk_rms"]
    )
    assert summary["straight_field_line"]["strong_force_device_normalized_rms"] > 0.1
    assert summary["straight_field_line"]["axial_flux_derivative_min"] > 4.49e-4
    for case in summary:
        for suffix in ("3d", "cross_sections", "modB", "summary"):
            assert (outdir / f"{case}_{suffix}.png").stat().st_size > 10_000


@pytest.mark.full
def test_mirror_free_boundary_beta_scan_example(tmp_path):
    import json
    pytest.importorskip("essos")
    _run_example(EXAMPLES / "mirror_free_boundary_beta_scan.py", tmp_path, timeout=2400)
    outdir = tmp_path / "results" / "mirror_free_boundary_beta_scan"
    summary = json.loads((outdir / "beta_scan_summary.json").read_text())
    assert [row["requested_beta"] for row in summary] == [0.0, 0.10, 0.25, 0.50]
    assert [row["supported_lane"] for row in summary] == [True, True, False, False]
    assert summary[-1]["center_radius"] > summary[0]["center_radius"]
    assert summary[-1]["center_axis_field"] < summary[0]["center_axis_field"]
    for beta in ("000p0", "010p0", "050p0"):
        for suffix in ("3d", "cross_sections", "modB", "summary"):
            assert (outdir / f"mirror_beta_{beta}pct_{suffix}.png").stat().st_size > 10_000


@pytest.mark.full  # nightly: free-bdy NESTOR solve with direct-coil Biot-Savart (~90s)
def test_free_boundary_essos_coils(tmp_path):
    # The example needs only ``essos.coils`` (loading) + ``essos.fields.BiotSavart``
    # (tabulation) — present in every released ESSOS >= 0.16, with an in-example
    # fallback to the legacy ``Coils_from_json`` loader.  Under VMEX_EXAMPLES_CI=1
    # the script solves a single coarse beta point (ns=16), keeping this bounded.
    pytest.importorskip("essos.coils")
    pytest.importorskip("essos.fields")
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
    assert (tmp_path / f"input.{case}_optimized").exists()
    assert (tmp_path / f"wout_{case}_optimized.nc").exists()
    assert (tmp_path / f"{case}_optimized_summary.png").exists()
    match = re.search(r"\[final\] QS total = ([0-9.eE+-]+)", out)
    assert match is not None and np.isfinite(float(match.group(1)))


@pytest.mark.full  # nightly: shared Boozer + bounce-action Jacobian is cold-compile heavy
def test_qi_maxj_continuation_example(tmp_path):
    """Reduced-budget QI+maximum-J continuation smoke test."""
    script = EXAMPLES / "optimization" / "QI_maxJ_continuation.py"
    out = _run_example(script, tmp_path, timeout=900)
    _assert_cost_decreased(out, "QI-maxJ")
    seed = re.search(r"\[seed\] QI = ([0-9.eE+-]+)", out)
    final = re.search(r"\[final\] QI = ([0-9.eE+-]+)", out)
    assert seed is not None and final is not None
    assert np.isfinite(float(final.group(1))) and float(final.group(1)) <= 1.05 * float(seed.group(1))
    assert re.search(r"J-invariance = ([0-9.eE+-]+), maximum-J = ([0-9.eE+-]+)", out)
    assert (tmp_path / "input.QI_maxJ_optimized").exists()
    assert (tmp_path / "wout_QI_maxJ_optimized.nc").exists()
    assert (tmp_path / "QI_maxJ_optimized_summary.png").stat().st_size > 10_000


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
    assert (tmp_path / "wout_QI_optimized.nc").exists()


def test_extra_terms_work_uncommented():
    """The commented-out example terms run as objectives when uncommented."""
    import jax

    from vmex.core.input import VmecInput
    from vmex.core import optimize as opt

    was_disabled = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)  # conftest disables jit globally
    try:
        inp = VmecInput.from_file(DATA_DIR / "input.minimal_seed_nfp2")
        # exactly the expressions the example comments show:
        extra_terms = [
            (opt.magnetic_well, 0.05, 1.0),
            (opt.mercier_stability_residual, 0.0, 100.0),
            (lambda eq: max(1.0 / opt.l_grad_b(eq) - 1.0 / 0.35, 0.0), 0.0, 1.0),
        ]
        terms = [(opt.aspect_ratio, 5.0, 1.0)] + extra_terms
        result = opt.least_squares(terms, inp, max_mode=1, jac=None,
                                   use_ess=True, max_nfev=2)
        assert np.all(np.isfinite(result.fun))
        # the traceable extra term also differentiates through jac="implicit"
        result2 = opt.least_squares(
            [(opt.aspect_ratio, 5.0, 1.0),
             (opt.magnetic_well, 0.05, 1.0),
             (opt.mercier_stability_residual, 0.0, 100.0)],
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
    "VMEX_ZENODO_2205_02914",
    str(Path.home() / "local" /
        "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current")))


@pytest.mark.full
@pytest.mark.skipif(not _ZENODO_2205.is_dir(),
                    reason="arXiv:2205.02914 Zenodo dataset not present")
@pytest.mark.parametrize("case", ["QA", "QH"])
def test_bootstrap_selfconsistent_examples(case, tmp_path):
    script = REPO / "benchmarks" / f"{case}_bootstrap_selfconsistent.py"
    out = _run_example(script, tmp_path, timeout=1200)
    m = re.search(r"final f_boot = ([0-9.eE+-]+)", out)
    assert m is not None and float(m.group(1)) < 5e-2, f"{case} f_boot: {out[-400:]}"
    assert (tmp_path / f"output_{case}_bootstrap_selfconsistent"
            / f"wout_{case}_bootstrap_selfconsistent.nc").exists()


@pytest.mark.full  # nightly: Picard seed + one exact finite-beta optimization stage
@pytest.mark.parametrize("case", ["QA", "QH"])
def test_bootstrap_optimization_examples(case, tmp_path):
    script = EXAMPLES / "optimization" / f"{case}_optimization_bootstrap.py"
    out = _run_example(script, tmp_path, timeout=1800)
    _assert_cost_decreased(out, f"{case}-bootstrap")
    assert "self-consistent seed" in out and "[final] QS" in out
    match = re.search(r"\[final\].*beta = ([0-9.]+)%", out)
    assert match is not None and 1.0 < float(match.group(1)) < 4.0
    assert (tmp_path / f"input.{case}_bootstrap_optimized").exists()
    assert (tmp_path / f"wout_{case}_bootstrap_optimized.nc").exists()
    assert (tmp_path / f"{case}_bootstrap_current.png").exists()


@pytest.mark.full  # nightly: optional optimizer interoperability, cold JAX compilation
@pytest.mark.parametrize(("script_name", "dependency", "output"), [
    ("QA_optimization_scipy.py", None, "wout_QA_scipy_BFGS.nc"),
    ("QI_optimization_scipy.py", None, "wout_QI_scipy_BFGS.nc"),
    ("QI_optimization_jaxopt.py", "jaxopt", "wout_QI_jaxopt_LBFGS.nc"),
    ("QI_optimization_optax.py", "optax", "wout_QI_optax_adam.nc"),
])
def test_scalar_optimizer_examples(script_name, dependency, output, tmp_path):
    if dependency is not None:
        pytest.importorskip(dependency)
    out = _run_example(EXAMPLES / "optimization" / script_name, tmp_path, timeout=1800)
    assert "final cost" in out
    assert (tmp_path / output).exists()


@pytest.mark.full  # nightly: exact VMEX+ESSOS reverse-mode graph and ParaView output
def test_fixed_boundary_single_stage_optimization(tmp_path):
    pytest.importorskip("essos")
    out = _run_example(
        EXAMPLES / "optimization" / "single_stage_optimization.py", tmp_path, timeout=1800)
    match = re.search(r"Objective: ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(2)) < float(match.group(1))
    for diagnostic in ("B.n/B: area-weighted RMS", "Minimum coil-surface distance",
                       "Minimum coil-coil distance", "Maximum curvature", "Coil lengths"):
        assert diagnostic in out
    normal = re.search(r"B\.n/B: area-weighted RMS = ([0-9.]+)%, max = ([0-9.]+)%", out)
    assert normal is not None and all(np.isfinite(float(value)) for value in normal.groups())
    for name in ("wout_single_stage_optimized.nc", "single_stage_objectives.png",
                 "surface_single_stage_initial.vts", "coils_single_stage_initial.vtu",
                 "surface_single_stage_optimized.vts", "coils_single_stage_optimized.vtu"):
        assert (tmp_path / name).exists()
    surface_vtk = (tmp_path / "surface_single_stage_optimized.vts").read_bytes()
    assert b'Name="B_BiotSavart"' in surface_vtk
    assert b'Name="B_dot_n_over_B"' in surface_vtk


@pytest.mark.full  # nightly: one finite-beta VMEX + VCJ + ESSOS graph (~1 min)
def test_finite_beta_single_stage_optimization(tmp_path, monkeypatch):
    pytest.importorskip("essos")
    pytest.importorskip("virtual_casing_jax")
    # Keep this independent compile from filling a shared developer cache.
    monkeypatch.setenv("VMEX_COMPILATION_CACHE", "disabled")
    out = _run_example(
        EXAMPLES / "optimization" / "single_stage_optimization_finite_beta.py",
        tmp_path, timeout=1800)
    for diagnostic in ("[final] QA", "B.n/B RMS", "Normalized total-pressure jump RMS",
                       "Coil lengths", "Maximum curvature"):
        assert diagnostic in out
    for name in ("wout_single_stage_finite_beta_optimized.nc",
                 "single_stage_finite_beta_objectives.png",
                 "single_stage_finite_beta_bootstrap_current.png",
                 "surface_single_stage_finite_beta_initial.vts",
                 "coils_single_stage_finite_beta_initial.vtu",
                 "surface_single_stage_finite_beta_optimized.vts",
                 "coils_single_stage_finite_beta_optimized.vtu"):
        assert (tmp_path / name).exists()


def test_field_query_examples_cover_inside_outside_and_vjps() -> None:
    """Keep the two runnable API examples explicit without another slow solve."""
    interior = (EXAMPLES / "vmex_get_B_gradB.py").read_text()
    exterior = (EXAMPLES / "vmex_get_B_outside_plasma.py").read_text()
    for source in (interior, exterior):
        for call in ("set_points_xyz", "set_points_flux", ".B()", ".absB()", ".gradB()", ".B_vjp(",
                     ".gradB_vjp(", ".gradgradB_vjp(", ".gradgradgradB_vjp("):
            assert call in source
        assert "VmecProblem.from_input" in source and "SimpleNamespace" not in source
    assert "get_points_flux" in interior
    assert "uses_virtual_casing" in exterior and "exterior_field" in exterior
    assert "ESSOS_biot_savart_LandremanPaulQA_finite_beta.json" in exterior


@pytest.mark.full  # nightly: two bounded high-order field/VJP compilations (~3 min)
@pytest.mark.parametrize(("script", "message"), [
    ("vmex_get_B_gradB.py", "gradgradgradB VJP shapes"),
    ("vmex_get_B_outside_plasma.py", "uses virtual casing = True"),
])
def test_field_query_examples_run(script, message, tmp_path):
    if "outside" in script:
        pytest.importorskip("essos")
        pytest.importorskip("virtual_casing_jax")
    out = _run_example(EXAMPLES / script, tmp_path, timeout=360)
    assert message in out and "dof_names =" in out


def test_fieldline_example_uses_vmex_virtual_casing_and_actual_essos_coils() -> None:
    """Keep the integration path explicit without adding a slow solve to PR CI."""
    vacuum = (EXAMPLES / "vmex_fieldline_tracing_vacuum.py").read_text()
    finite = (EXAMPLES / "vmex_fieldline_tracing_finite_beta.py").read_text()
    for source in (vacuum, finite):
        for contract in ("BiotSavart", "exterior_field", "field_in_flux_coordinates",
                         "Tracing", "poincare_plot"):
            assert contract in source
    assert 'plasma="vacuum"' in vacuum
    assert "ESSOS_biot_savart_LandremanPaulQA_finite_beta.json" in finite


def test_single_stage_examples_use_general_surface_output_and_movie_colors() -> None:
    vacuum = (EXAMPLES / "optimization" / "single_stage_optimization.py").read_text()
    finite = (EXAMPLES / "optimization" / "single_stage_optimization_finite_beta.py").read_text()
    assert "from pyevtk" not in finite
    assert "surface_initial.to_vtk" in finite and "extra_data=" in finite
    for source in (vacuum, finite):
        assert "MOVIE_SURFACE_COLOR" in source and "color_factory=" in source


@pytest.mark.full  # nightly: two bounded ESSOS tracing integrations (~40 s total)
@pytest.mark.parametrize(("script", "message", "output"), [
    ("vmex_fieldline_tracing_vacuum.py", "VMEX exterior API outside", "vmex_fieldline_tracing_vacuum.png"),
    ("vmex_fieldline_tracing_finite_beta.py", "VMEX coil + virtual-casing field outside",
     "vmex_fieldline_tracing_finite_beta.png"),
])
def test_vmex_fieldline_tracing_examples(script, message, output, tmp_path):
    pytest.importorskip("essos")
    pytest.importorskip("virtual_casing_jax")
    out = _run_example(EXAMPLES / script, tmp_path, timeout=300)
    assert message in out
    assert (tmp_path / output).stat().st_size > 10_000


@pytest.mark.full  # nightly: fixed equilibrium + one 136-DOF exact coil step (~20 s)
def test_finite_beta_coil_optimization_example(tmp_path):
    pytest.importorskip("essos")
    pytest.importorskip("virtual_casing_jax")
    out = _run_example(
        EXAMPLES / "optimization" / "finite_beta_coil_optimization.py", tmp_path, timeout=300)
    assert "B.n/B RMS" in out and "Normalized total-pressure jump RMS" in out
    assert (tmp_path / "coils_LandremanPaulQA_finite_beta_optimized.json").exists()
    assert (tmp_path / "finite_beta_coil_objectives.png").stat().st_size > 10_000
