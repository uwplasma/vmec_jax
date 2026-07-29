"""New-core end-to-end parity breadth: six benchmark decks vs VMEC2000
golden (DSHAPE, circular_tokamak, li383_low_res,
LandremanPaul2021_QA_lowres, nfp4_QH_warm_start,
up_down_asymmetric_tokamak).  Multi-stage decks run the full ``NS_ARRAY``
ladder via ``solve_multigrid`` (golden iteration counts are per-stage,
warm-started); single-stage decks call ``solver.solve`` directly.

Golden caveats (details in the ``CASES`` table): QA_lowres and up_down are
NITER-capped goldens, so their final-stage ftol is relaxed to the matched
residual and the harmonic atol covers golden's OWN non-convergence, not
core drift (a fully converged VMEC2000 rerun of up_down moves golden by up
to 7e-5; the converged new core matches that rerun to <= 7.3e-7).  The
historic ~3% lasym drift was the fixaray.f dnorm defect fixed in
``vmex/core/fourier.py``.

Per case: convergence at the (possibly relaxed) ftol; iterations within
+-25% of the golden stdout; ``wb`` within 1e-7 relative; ``rmnc/zmns`` on
first-interior/mid/boundary surfaces (rtol 1e-5, atol 1e-9 unless noted);
``iotaf`` rtol 1e-5.  Results are pickle-cached under ``/tmp`` keyed by
case + git describe.  Deliberately skipped: ``NuhrenbergZille`` (no golden
fixture, > 120 s) and ``cth_like_free_bdy_lasym_small`` (free boundary).
"""

from __future__ import annotations

import pickle
import re
import subprocess
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from vmex.core.input import VmecInput
from vmex.core import solver
from vmex.core.multigrid import solve_multigrid

from tests.conftest import resolve_golden_dir

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
GOLDEN_DIR = resolve_golden_dir()
pytestmark = [
    pytest.mark.skipif(
        GOLDEN_DIR is None, reason="golden VMEC2000 fixtures unavailable (offline?)"
    ),
    pytest.mark.usefixtures("_module_jit_enabled"),  # full solves: run jitted
]

netCDF4 = pytest.importorskip("netCDF4")

#: bump when the solve recipe below changes (invalidates /tmp caches).
#: v2: lasym dnorm fix in core/fourier.py (fixaray.f parity) changed the
#: up_down_asymmetric_tokamak trajectory/fixed point.
_SPEC_VERSION = "v2"

#: case -> solve recipe + golden facts.
#:   multigrid:   run the full NS_ARRAY ladder (False -> single solve() call)
#:   ftol/niter:  final-stage overrides (None -> deck values)
#:   golden_iters: final(-stage) iteration count printed in golden stdout.txt
CASES: dict[str, dict] = {
    "DSHAPE": dict(multigrid=True, ftol=None, niter=None, golden_iters=908),
    "circular_tokamak": dict(multigrid=True, ftol=None, niter=None, golden_iters=368),
    "li383_low_res": dict(multigrid=False, ftol=None, niter=None, golden_iters=123),
    # golden final stage NITER-capped at 1000 with fsq (2.63e-13, 8.07e-14,
    # 1.20e-13) > deck FTOL 1e-13: stop at the matched residual instead.
    "LandremanPaul2021_QA_lowres": dict(
        multigrid=True, ftol=3e-13, niter=2000, golden_iters=1000,
        harmonic_atol=5e-6,
    ),
    "nfp4_QH_warm_start": dict(multigrid=False, ftol=None, niter=None, golden_iters=450),
    # golden NITER-capped at 2000 with fsq (5.25e-14, 1.11e-13, 5.80e-16)
    # > deck FTOL 1e-14: converge at the matched residual 1.5e-13 instead
    # (measured: 1951 iterations, fsq {7.2e-14, 1.5e-13, 7.3e-16}).  The
    # harmonic atol 2e-5 absorbs golden's own non-convergence at that
    # residual (largest measured coefficient diff 9.2e-6, mid-surface rmnc
    # m=0; a fully converged VMEC2000 rerun moves golden by up to 7e-5).
    "up_down_asymmetric_tokamak": dict(
        multigrid=False, ftol=1.5e-13, niter=3000, golden_iters=2000,
        harmonic_atol=2e-5,
    ),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git_describe() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return re.sub(r"[^A-Za-z0-9._-]", "_", out) or "nogit"
    except Exception:
        return "nogit"


_CACHE_DIR = Path("/tmp") / "vmex_parity_breadth"
_GIT_TAG = _git_describe()

_RESULT_FIELDS = (
    "converged", "iterations", "jacobian_resets", "fsqr", "fsqz", "fsql",
    "wb", "iotaf", "rmnc", "zmns", "rmns", "zmnc", "xm", "xn",
)


def golden_final_iteration(stdout_path: Path) -> int:
    """Final-stage final iteration from a golden VMEC2000 ``stdout.txt``.

    Iteration rows look like ``  908  9.96E-13  8.27E-14 ...`` (lasym rows
    carry one extra column); counters restart at 1 per NS stage, so the last
    matching row is the final stage's final iteration.
    """
    final = None
    for line in stdout_path.read_text().splitlines():
        tok = line.split()
        if len(tok) >= 6 and tok[0].isdigit() and "E" in tok[1]:
            final = int(tok[0])
    if final is None:
        raise ValueError(f"no iteration rows found in {stdout_path}")
    return final


def _solve_case(name: str) -> dict:
    """Run (or load from cache) one case; returns plain numpy fields."""
    spec = CASES[name]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _CACHE_DIR / f"{name}-{_GIT_TAG}-{_SPEC_VERSION}.pkl"
    if cache.is_file():
        try:
            with cache.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            cache.unlink(missing_ok=True)

    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    if spec["multigrid"]:
        n_stages = int(np.atleast_1d(np.asarray(inp.ns_array)).size)
        ftol_arr = niter_arr = None
        if spec["ftol"] is not None:
            ftol_arr = np.resize(np.asarray(inp.ftol_array, float), n_stages)
            ftol_arr[-1] = spec["ftol"]
        if spec["niter"] is not None:
            niter_arr = np.resize(np.asarray(inp.niter_array, int), n_stages)
            niter_arr[-1] = spec["niter"]
        result = solve_multigrid(
            inp, ftol_array=ftol_arr, niter_array=niter_arr, mode="cli"
        )
    else:
        result = solver.solve(
            inp, ftol=spec["ftol"], max_iterations=spec["niter"], mode="cli"
        )

    out = {}
    for f in _RESULT_FIELDS:
        v = getattr(result, f)
        out[f] = np.asarray(v) if isinstance(v, np.ndarray) else v
    with cache.open("wb") as fh:
        pickle.dump(out, fh)
    return out


def _target_ftol(name: str) -> float:
    spec = CASES[name]
    if spec["ftol"] is not None:
        return float(spec["ftol"])
    inp_ftol = np.atleast_1d(np.asarray(
        VmecInput.from_file(str(DATA_DIR / f"input.{name}")).ftol_array, float))
    ns_stages = int(np.atleast_1d(np.asarray(
        VmecInput.from_file(str(DATA_DIR / f"input.{name}")).ns_array)).size)
    idx = min(ns_stages, inp_ftol.size) - 1
    return float(inp_ftol[idx])


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request):
    name = request.param
    return name, _solve_case(name)


# ---------------------------------------------------------------------------
# convergence + iteration counts
# ---------------------------------------------------------------------------


def test_converges_at_deck_ftol(case):
    name, res = case
    ftol = _target_ftol(name)
    assert res["converged"], f"{name}: did not converge"
    assert res["fsqr"] <= ftol and res["fsqz"] <= ftol and res["fsql"] <= ftol, (
        f"{name}: final fsq ({res['fsqr']:.2e}, {res['fsqz']:.2e}, "
        f"{res['fsql']:.2e}) above ftol {ftol:.1e}"
    )


def test_iteration_count_within_25pct_of_golden(case):
    name, res = case
    golden_iters = golden_final_iteration(GOLDEN_DIR / name / "stdout.txt")
    assert golden_iters == CASES[name]["golden_iters"], (
        f"{name}: stdout parser returned {golden_iters}, "
        f"expected {CASES[name]['golden_iters']}"
    )
    low, high = int(0.75 * golden_iters), int(np.ceil(1.25 * golden_iters))
    assert low <= res["iterations"] <= high, (
        f"{name}: {res['iterations']} iterations vs golden {golden_iters} "
        f"(band [{low}, {high}])"
    )


# ---------------------------------------------------------------------------
# wout parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_wout(case):
    name, _ = case
    ds = netCDF4.Dataset(str(GOLDEN_DIR / name / f"wout_{name}.nc"))
    yield ds
    ds.close()


def test_wb_parity(case, golden_wout):
    name, res = case
    wb_gold = float(golden_wout["wb"][()])
    rel = abs(res["wb"] / wb_gold - 1.0)
    assert rel < 1e-7, f"{name}: wb {res['wb']} vs golden {wb_gold} (rel {rel:.2e})"


def test_mode_tables_match(case, golden_wout):
    name, res = case
    np.testing.assert_array_equal(res["xm"], np.asarray(golden_wout["xm"][()]))
    np.testing.assert_array_equal(res["xn"], np.asarray(golden_wout["xn"][()]))


def test_iotaf_parity(case, golden_wout):
    name, res = case
    np.testing.assert_allclose(
        res["iotaf"], np.asarray(golden_wout["iotaf"][()]), rtol=1e-5,
        err_msg=f"{name}: iotaf",
    )


def _check_surfaces(name: str, ours: np.ndarray, gold: np.ndarray, field: str,
                    atol: float) -> None:
    ns = gold.shape[0]
    for js in (1, ns // 2, ns - 1):
        np.testing.assert_allclose(
            ours[js], gold[js], rtol=1e-5, atol=atol,
            err_msg=f"{name}: {field} surface js={js} (of {ns})",
        )


def test_boundary_and_mid_surface_harmonics(case, golden_wout):
    name, res = case
    atol = CASES[name].get("harmonic_atol", 1e-9)
    _check_surfaces(
        name, res["rmnc"], np.asarray(golden_wout["rmnc"][()]), "rmnc", atol)
    _check_surfaces(
        name, res["zmns"], np.asarray(golden_wout["zmns"][()]), "zmns", atol)


def test_asymmetric_harmonics_lasym(case, golden_wout):
    name, res = case
    if res["rmns"] is None:
        pytest.skip(f"{name}: stellarator-symmetric deck (lasym=F)")
    # Same golden-non-convergence allowance as the symmetric blocks.
    atol = CASES[name].get("harmonic_atol", 1e-9)
    _check_surfaces(
        name, res["rmns"], np.asarray(golden_wout["rmns"][()]), "rmns", atol)
    _check_surfaces(
        name, res["zmnc"], np.asarray(golden_wout["zmnc"][()]), "zmnc", atol)


# ---------------------------------------------------------------------------
# deliberately skipped fixtures
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="NuhrenbergZille_1988_QHS exceeds the 120 s budget and ships no "
           "golden fixture in the golden-v1 bundle"
)
def test_nuhrenberg_zille():  # pragma: no cover - documented skip
    raise NotImplementedError
