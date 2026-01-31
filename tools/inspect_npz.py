#!/usr/bin/env python
"""Print a compact summary of arrays stored in a .npz file.

This is handy for quick debugging / regression checks without plotting.

Usage
-----
    python tools/inspect_npz.py path/to/file.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running without installing: add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str, help="Path to a .npz file")
    ap.add_argument("--max", type=int, default=200, help="Max entries to show for 1D arrays")
    args = ap.parse_args()

    from vmec_jax.diagnostics import print_summary, summarize_array

    p = Path(args.npz)
    if not p.exists():
        raise SystemExit(f"error: npz file not found: {p.resolve()}")

    d = np.load(p)
    keys = sorted(d.files)
    print(f"==== inspect_npz: {args.npz} ====")
    print(f"nkeys={len(keys)}")

    for k in keys:
        a = d[k]
        print_summary(summarize_array(k, a), indent="")
        if a.ndim == 1 and a.size <= args.max:
            print(f"  values: {a}")
    print("done")


if __name__ == "__main__":
    main()
