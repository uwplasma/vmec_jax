"""VMEC2000 scalar-parity gate that needs NO large golden ``wout`` files.

``tests/golden_digests.json`` holds a few-KB set of reference scalars extracted
from the VMEC2000 golden runs (``tools/make_golden_digests.py``).  Here we solve
each case with vmex and check the wout scalars against those digests — the
physics accuracy (energies, aspect, beta, iota/pressure, converged boundary
shape) matched to VMEC2000, self-contained in the repo.  The full
variable-by-variable comparison against the stored ``wout`` bundle lives in
``tests/test_wout_golden.py`` (nightly, needs the download).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")

import vmex as vj  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
DIGESTS = json.loads((REPO / "tests" / "golden_digests.json").read_text())

# Per-scalar relative tolerance vs VMEC2000.  wb/aspect/volume are the
# tightly-converged invariants (matched to << 0.1%); beta/geometry a touch
# looser to absorb ns-interpolation and the pressure-scale round-off.
RTOL = {
    "wb": 2e-4, "aspect": 2e-4, "volume_p": 5e-4, "b0": 5e-4,
    "rmax_surf": 5e-4, "rmin_surf": 5e-4, "rmnc_bdy_rms": 5e-4,
    "zmns_bdy_rms": 5e-4, "iotaf_axis": 3e-3, "iotaf_edge": 3e-3,
}
DEFAULT_RTOL = 3e-3          # wp, betatotal/pol/tor, presf endpoints
ATOL = 1e-8                  # for values that pass through ~0

# Light cases run on every PR; the heavier finite-beta / high-ns cases nightly.
LIGHT = ["solovev", "circular_tokamak"]
HEAVY = ["li383_low_res", "cth_like_fixed_bdy", "DSHAPE", "nfp2_QA_finite_beta", "nfp4_QH_finite_beta"]


def _solve_scalars(case: str) -> dict:
    inp = vj.VmecInput.from_file(DATA / f"input.{case}")
    res = vj.solve_multigrid(inp, verbose=False)
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged))
    out = {k: float(getattr(wout, k)) for k in
           ("wb", "wp", "aspect", "volume_p", "betatotal", "b0",
            "betapol", "betator", "rmax_surf", "rmin_surf")
           if hasattr(wout, k)}
    iota = np.asarray(wout.iotaf, dtype=float)
    pres = np.asarray(wout.presf, dtype=float)
    out["iotaf_axis"], out["iotaf_edge"] = float(iota[0]), float(iota[-1])
    out["presf_axis"], out["presf_edge"] = float(pres[0]), float(pres[-1])
    rmnc = np.asarray(wout.rmnc, dtype=float)
    zmns = np.asarray(wout.zmns, dtype=float)
    out["rmnc_bdy_rms"] = float(np.sqrt(np.mean(rmnc[-1] ** 2)))
    out["zmns_bdy_rms"] = float(np.sqrt(np.mean(zmns[-1] ** 2)))
    return out


def _check(case: str) -> None:
    ref = DIGESTS[case]
    got = _solve_scalars(case)
    problems = []
    for key, refval in ref.items():
        if key == "ns" or key not in got:
            continue
        rtol = RTOL.get(key, DEFAULT_RTOL)
        if not np.isclose(got[key], refval, rtol=rtol, atol=ATOL):
            rel = abs(got[key] / refval - 1.0) if refval else abs(got[key])
            problems.append(f"{key}: vmex {got[key]:.6e} vs VMEC2000 "
                            f"{refval:.6e} (rel {rel:.2e} > {rtol:.0e})")
    assert not problems, f"{case} scalar parity:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("case", LIGHT)
def test_scalar_parity_light(case):
    _check(case)


@pytest.mark.full
@pytest.mark.parametrize("case", HEAVY)
def test_scalar_parity_heavy(case):
    _check(case)
