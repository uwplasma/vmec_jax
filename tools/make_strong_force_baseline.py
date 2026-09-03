#!/usr/bin/env python
"""Record the shipped strong-force certificate values as exact float64 hex.

Writes ``tests/data/strong_force_certificate_baseline.json`` — the reference
``tests/test_strong_force.py`` compares against so that adding a *report* to
the certificate can never move a *number* the README, the docs, the committed
benchmark artifacts, and the polish acceptance thresholds all quote.

Two cheap, solve-free states cover it: the analytic constant-toroidal-field
state and the lifted initial guess of ``input.solovev``.  Scalars are stored
as ``float.hex()`` and arrays as the SHA-256 of their raw little-endian
float64 bytes, because a decimal round trip would hide exactly the drift this
file exists to catch.

The values are measured on the **eager** (jit-disabled) lane, which is the
lane the unit suite runs; XLA promises nothing about fusion across programs or
platforms, and the same values shift by ~2e-13 between the eager and jitted
lanes.  The guard therefore compares at ``rtol=1e-12``, far tighter than any
behavioural change and loose enough to survive a different CPU.

Run this **only** when the certificate's arithmetic is deliberately changed —
never to make a failing guard pass.  A failure that you cannot explain is the
finding, not the noise::

    python tools/make_strong_force_baseline.py

Record the commit you measured at in the ``measured`` block it writes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", True)

from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.solver import (  # noqa: E402
    _initial_state,
    prepare_runtime,
    resolution_from_input,
)
from vmex.core.strong_force import (  # noqa: E402
    certify_strong_force,
    lift_high_order_state,
)

from test_strong_force import (  # noqa: E402
    _SHIPPED_CERTIFICATE_ARRAYS,
    _SHIPPED_CERTIFICATE_FIELDS,
    _constant_toroidal_field_state,
)

OUTPUT = ROOT / "tests" / "data" / "strong_force_certificate_baseline.json"


def _record(report) -> dict:
    entry = {
        name: float(np.asarray(getattr(report, name))).hex()
        for name in _SHIPPED_CERTIFICATE_FIELDS
    }
    entry["force_floor"] = float(report.force_floor).hex()
    entry["arrays"] = {}
    for name in _SHIPPED_CERTIFICATE_ARRAYS:
        values = np.ascontiguousarray(
            np.asarray(getattr(report, name), dtype=np.float64)
        )
        entry["arrays"][name] = {
            "size": int(values.size),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    return entry


def _solovev_initial_lift():
    inp = VmecInput.from_file(
        ROOT / "examples" / "data" / "input.solovev"
    ).change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    resolution = replace(resolution_from_input(inp), ns=11)
    runtime = prepare_runtime(inp, resolution)
    return lift_high_order_state(
        _initial_state(runtime.setup), runtime, degree=5
    )


def main() -> None:
    baseline = {
        "schema": "vmex.strong-force-certificate-baseline/1",
        "measured": {
            "vmex_commit": "UNRECORDED - fill in the commit you measured at",
            "note": (
                "Values of every StrongForceReport field that shipped before"
                " the volume-averaged normalizations were added, recorded as"
                " exact float64 hex on the eager (jit-disabled) lane."
            ),
            "jax": jax.__version__,
            "machine": platform.machine(),
            "jit_disabled": True,
        },
        "cases": {
            "constant_toroidal_field": _record(
                certify_strong_force(
                    _constant_toroidal_field_state(),
                    angular_multiplier=1,
                    radial_order_increment=0,
                )
            ),
            "solovev_initial_lift": _record(
                certify_strong_force(_solovev_initial_lift())
            ),
        },
    }
    OUTPUT.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print("set measured.vmex_commit to the commit this was measured at")


if __name__ == "__main__":
    main()
