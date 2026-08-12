#!/usr/bin/env python3
"""Pack the released reference-asset tarballs from a populated checkout.

A bundle is exactly the set of git-ignored netCDF files under its roots --
the same invariant ``examples/data/README.md`` states ("large reference WOUT,
mgrid, Boozer, and JXB files are ignored by git and are fetched on demand").
Enumerating from ``git`` rather than a hand-kept list means a new reference
fixture joins the next release by being added and ignored, with nothing here
to update.

Two slimming rules apply, both measured to be lossless (2026-08-12):

1. MAKEGRID writes a vector potential (``ar_``/``ap_``/``az_``) alongside the
   field.  Neither code reads it back: VMEX's ``read_mgrid`` matches only
   ``^(br|bp|bz)_(\\d{3})$``, and ``xvmec2000`` on the stripped
   ``mgrid_cth_like.nc`` produced a byte-identical
   ``wout_cth_like_free_bdy.nc`` (sha256 644f597f..., EXECUTION TERMINATED
   NORMALLY).  It is exactly half of that file: 35.26 -> 17.63 MB.
2. ``examples/data/single_grid/`` was shipped as 16 byte-identical copies of
   its ``examples/data/`` siblings.  They are dropped here and re-created on
   extraction from the ``mirrors`` rule in ``assets/manifest.json``, which
   also guarantees the mirrored ``mgrid_cth_like_lasym_small.nc`` is the
   git-tracked file rather than a bundle copy that can drift from it.

Tarballs are byte-reproducible: members sorted, mtime/uid/gid zeroed, and a
fixed gzip header.  Re-running on the same inputs reproduces the recorded
SHA-256, so ``assets/manifest.json`` can be re-derived instead of trusted.

Usage::

    python tools/pack_reference_assets.py --outdir dist/assets
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: bundle name -> roots enumerated for git-ignored netCDF payload.
BUNDLE_ROOTS: dict[str, tuple[str, ...]] = {
    "reference-nc": ("examples/data",),
    "wout-fixtures": (
        "docs/_static/readme_best_cases",
        "docs/_static/qi_readme_cases",
    ),
}

#: Files re-created on extraction instead of shipped (see module docstring).
MIRROR_TARGET = "examples/data/single_grid"

_POTENTIAL_PREFIXES = ("ar_", "ap_", "az_")


def _ignored_netcdf(root: Path, source: Path) -> list[str]:
    """Return repository-relative git-ignored ``.nc`` paths under ``root``."""
    result = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--", str(root)],
        cwd=source, check=True, capture_output=True,
    )
    names = result.stdout.decode().rstrip("\0").split("\0")
    return sorted(name for name in names if name.endswith(".nc"))


def _strip_vector_potential(path: Path, into: Path) -> Path:
    """Copy an mgrid without its unread vector-potential arrays."""
    import netCDF4  # noqa: PLC0415 - optional heavy dependency

    out = into / path.name
    with netCDF4.Dataset(str(path)) as src:
        keep = [name for name in src.variables if not name.startswith(_POTENTIAL_PREFIXES)]
        if len(keep) == len(src.variables):
            return path
        used = {dim for name in keep for dim in src.variables[name].dimensions}
        with netCDF4.Dataset(str(out), "w", format="NETCDF3_CLASSIC") as dst:
            for name, dim in src.dimensions.items():
                if name in used:
                    dst.createDimension(name, len(dim))
            for name in keep:
                var = src.variables[name]
                dst.createVariable(name, var.dtype, var.dimensions)[:] = var[:]
    return out


def _member(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mode, info.uid, info.gid, info.mtime = 0o644, 0, 0, 0
    info.uname = info.gname = ""
    return info


def _pack(payload: Sequence[tuple[str, Path]], out: Path) -> tuple[int, str]:
    """Write a byte-reproducible ``.tar.gz`` and return its size and digest."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        for name, path in sorted(payload):
            info = _member(name)
            info.size = path.stat().st_size
            with path.open("rb") as handle:
                tf.addfile(info, handle)
    blob = gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    return len(blob), hashlib.sha256(blob).hexdigest()


def _revision(source: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True,
    )
    return result.stdout.decode().strip()


def build(bundle: str, source: Path, outdir: Path, scratch: Path) -> dict:
    """Pack one bundle and return the record to paste into the manifest."""
    names: list[str] = []
    for root in BUNDLE_ROOTS[bundle]:
        names.extend(_ignored_netcdf(Path(root), source))

    mirrored: list[str] = []
    payload: list[tuple[str, Path]] = []
    for name in names:
        path = source / name
        parent, filename = str(Path(name).parent), Path(name).name
        if parent == MIRROR_TARGET and (source / "examples" / "data" / filename).is_file():
            mirrored.append(filename)
            continue
        if filename.startswith("mgrid_"):
            path = _strip_vector_potential(path, scratch)
        payload.append((name, path))

    # ``mgrid_cth_like_lasym_small.nc`` is git-tracked under examples/data, so
    # mirroring it (rather than shipping the single_grid copy) is what keeps
    # the two in step -- a stale bundle copy overwriting the tracked file is
    # what poisoned the nightly free-boundary golden on 2026-07-12.
    tracked_small = "mgrid_cth_like_lasym_small.nc"
    if bundle == "reference-nc" and (source / "examples" / "data" / tracked_small).is_file():
        mirrored.append(tracked_small)
    mirrored = sorted(set(mirrored))

    out = outdir / f"vmex_assets_{bundle.replace('-', '_')}.tar.gz"
    size, digest = _pack(payload, out)
    record = {
        "name": bundle,
        "path": str(out),
        "size_bytes": size,
        "sha256": digest,
        "generator_revision": _revision(source),
        "members": [name for name, _ in sorted(payload)],
        "mirrors": (
            [{"source": "examples/data", "target": MIRROR_TARGET, "names": mirrored}]
            if mirrored else []
        ),
    }
    return record


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=REPO_ROOT,
                   help="checkout with the reference assets already installed")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "dist" / "assets")
    p.add_argument("--bundle", action="append", choices=tuple(BUNDLE_ROOTS),
                   help="bundle to pack; repeatable, defaults to all")
    args = p.parse_args(list(argv) if argv is not None else None)

    source = args.source.expanduser().resolve()
    bundles = args.bundle or list(BUNDLE_ROOTS)
    with tempfile.TemporaryDirectory() as tmp:
        records = [build(name, source, args.outdir.resolve(), Path(tmp)) for name in bundles]

    for record in records:
        print(f"\n{record['name']}: {len(record['members'])} members, "
              f"{record['size_bytes'] / 1e6:.1f} MB")
        print(f"  {record['path']}")
        print(f"  sha256 {record['sha256']}")
        if record["mirrors"]:
            print(f"  mirrors {len(record['mirrors'][0]['names'])} files into {MIRROR_TARGET}")
    print("\nManifest fields:")
    print(json.dumps(
        {r["name"]: {k: r[k] for k in ("sha256", "size_bytes", "generator_revision", "mirrors")}
         for r in records},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
