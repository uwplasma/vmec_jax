#!/usr/bin/env python
"""Extract scalar VMEC2000-parity digests from the golden ``wout_*.nc`` files.

Writes ``tests/golden_digests.json`` — a tiny (few-KB) file of reference
scalars (energies, aspect, beta, iota/pressure endpoints, and low-order
geometry checksums) taken from the stored VMEC2000 golden runs.  This lets
``tests/test_golden_digests.py`` check vmex against VMEC2000 to physical
accuracy **without** the multi-MB golden ``wout`` bundle on disk — the scalars
are the accuracy that matters and they live in the repo.

Run once whenever the golden set changes::

    python tools/make_golden_digests.py            # needs the golden dir present
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from conftest import resolve_golden_dir  # noqa: E402

# case -> the examples/data input deck vmex solves to reproduce it.
CASES = {
    "solovev": "input.solovev",
    "DSHAPE": "input.DSHAPE",
    "circular_tokamak": "input.circular_tokamak",
    "li383_low_res": "input.li383_low_res",
    "cth_like_fixed_bdy": "input.cth_like_fixed_bdy",
    "nfp2_QA_finite_beta": "input.nfp2_QA_finite_beta",
    "nfp4_QH_finite_beta": "input.nfp4_QH_finite_beta",
}

# scalar wout variables to digest (present in both VMEC2000 and vmex wout).
SCALARS = ("wb", "wp", "aspect", "volume_p", "betatotal", "b0",
           "betapol", "betator", "rmax_surf", "rmin_surf")


def digest_wout(path: Path) -> dict:
    d = Dataset(str(path))
    out: dict[str, float] = {}
    for k in SCALARS:
        if k in d.variables:
            out[k] = float(np.asarray(d.variables[k][...]))
    for k in ("iotaf", "presf"):
        if k in d.variables:
            v = np.asarray(d.variables[k][:], dtype=float)
            out[f"{k}_axis"] = float(v[0])
            out[f"{k}_edge"] = float(v[-1])
    # A rotation/scale-robust geometry checksum: RMS of the boundary R,Z
    # spectrum (rmnc/zmns last surface), which pins the converged shape.
    for name in ("rmnc", "zmns"):
        if name in d.variables:
            arr = np.asarray(d.variables[name][:], dtype=float)
            out[f"{name}_bdy_rms"] = float(np.sqrt(np.mean(arr[-1] ** 2)))
    out["ns"] = int(np.asarray(d.variables["ns"][...]))
    return out


def main() -> int:
    gd = resolve_golden_dir()
    if gd is None:
        print("golden dir unavailable; set VMEX_GOLDEN_DIR")
        return 1
    digests = {}
    for case in CASES:
        hits = glob.glob(f"{gd}/{case}/wout*.nc")
        if not hits:
            print(f"  (no golden wout for {case}; skipping)")
            continue
        digests[case] = digest_wout(Path(hits[0]))
        print(f"  {case}: {len(digests[case])} scalars, ns={digests[case]['ns']}")
    out_path = REPO / "tests" / "golden_digests.json"
    out_path.write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, {len(digests)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
