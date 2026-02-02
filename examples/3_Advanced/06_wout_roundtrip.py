"""Advanced example: write a VMEC-style wout file (roundtrip).

This script demonstrates that vmec_jax can *write* a minimal ``wout_*.nc`` file
containing the same subset of fields it can read via :func:`vmec_jax.wout.read_wout`.

Usage
-----
  python examples/3_Advanced/06_wout_roundtrip.py --wout examples/wout_circular_tokamak_reference.nc --out wout_roundtrip.nc

Requires:
- netCDF4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.wout import read_wout, write_wout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wout", type=str, required=True, help="Input wout_*.nc file")
    p.add_argument("--out", type=str, default="wout_roundtrip.nc", help="Output path")
    args = p.parse_args()

    w = read_wout(args.wout)
    out = Path(args.out)
    write_wout(out, w, overwrite=True)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()

