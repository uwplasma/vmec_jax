#!/usr/bin/env python3
"""Download large example/reference netCDF assets.

Generated optimization figures are intentionally not part of this asset bundle:
rerun the optimization renderers when report-quality panels are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_TAG = "assets-20260316-nc"
# NOTE: the published release tarball keeps the pre-rename "vmec_jax_" prefix --
# it was uploaded before the vmec_jax -> vmex repo rename, and only the repo and
# its code were renamed, not the already-published GitHub release assets. Keep
# this in sync with the actual asset filename, not the current package name.
REFERENCE_ASSET_NAME = "vmec_jax_assets_20260316_nc_only.tar.gz"
REFERENCE_URL = (
    "https://github.com/uwplasma/vmex/releases/download/"
    f"{REFERENCE_TAG}/{REFERENCE_ASSET_NAME}"
)
REFERENCE_SHA256 = "3344fc2401fffed240ee57ae741ec521594c592627c76dae203503f485e4c0d8"

WOUT_FIXTURES_TAG = "assets-20260526-wout-fixtures"
# Same pre-rename asset-filename caveat as REFERENCE_ASSET_NAME above.
WOUT_FIXTURES_ASSET_NAME = "vmec_jax_wout_fixtures_20260526.tar.gz"
WOUT_FIXTURES_URL = (
    "https://github.com/uwplasma/vmex/releases/download/"
    f"{WOUT_FIXTURES_TAG}/{WOUT_FIXTURES_ASSET_NAME}"
)
WOUT_FIXTURES_SHA256 = "e9fd844a4ebed043576eec8ac2b17f7ff3e2a0e0a1b5adfc89742139101e00f9"

REFERENCE_ASSET_PATHS = (
    "examples/data/mgrid_cth_like.nc",
    "examples/data/mgrid_d3d_ef.nc",
    "examples/data/wout_*_reference.nc",
    "examples/data/single_grid/mgrid_cth_like.nc",
    "examples/data/single_grid/mgrid_d3d_ef.nc",
    "examples/data/single_grid/wout_*_reference.nc",
)

WOUT_FIXTURE_PATHS = (
    "examples/data/wout_*.nc",
    "examples/data/single_grid/wout_*.nc",
    "docs/_static/readme_best_cases/*/wout_*.nc",
    "docs/_static/qi_readme_cases/*/wout_*.nc",
)

ASSET_PATH_REWRITES = (
    ("examples_single_grid/data", "examples/data/single_grid"),
)


@dataclass(frozen=True)
class AssetBundle:
    name: str
    url: str
    sha256: str
    common_paths: tuple[str, ...]

    @property
    def marker_name(self) -> str:
        return f".assets_installed_{self.name}.txt"


DEFAULT_BUNDLES = (
    AssetBundle(
        name="reference-nc",
        url=REFERENCE_URL,
        sha256=REFERENCE_SHA256,
        common_paths=REFERENCE_ASSET_PATHS,
    ),
    AssetBundle(
        name="wout-fixtures",
        url=WOUT_FIXTURES_URL,
        sha256=WOUT_FIXTURES_SHA256,
        common_paths=WOUT_FIXTURE_PATHS,
    ),
)
BUNDLES_BY_NAME = {bundle.name: bundle for bundle in DEFAULT_BUNDLES}


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _safe_extract(tf: tarfile.TarFile, dest: Path, members=None) -> None:
    """Extract ``tf`` under ``dest`` without allowing path traversal.

    ``members`` restricts extraction to the given tar members (used by the
    no-clobber default of :func:`_download_and_extract_bundle`).
    """
    dest_resolved = dest.resolve()
    selected = tf.getmembers() if members is None else members
    for member in selected:
        target = (dest_resolved / member.name).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise SystemExit(f"Refusing to extract path outside destination: {member.name}")
    tf.extractall(dest_resolved, members=selected)


def _print_bundle_info(bundles: Sequence[AssetBundle]) -> None:
    print("Asset bundles:")
    for bundle in bundles:
        print(f"- {bundle.name}")
        print(f"  URL:             {bundle.url}")
        print(f"  Expected SHA256: {bundle.sha256 or '(not checked)'}")
        print("  Common installed paths:")
        for path in bundle.common_paths:
            print(f"    {path}")


def _migrate_release_asset_paths(dest: Path) -> None:
    """Map files from older release tarball paths into the current layout."""
    for old_rel, new_rel in ASSET_PATH_REWRITES:
        old_dir = dest / old_rel
        if not old_dir.exists():
            continue
        new_dir = dest / new_rel
        new_dir.mkdir(parents=True, exist_ok=True)
        for old_path in old_dir.iterdir():
            if not old_path.is_file():
                continue
            new_path = new_dir / old_path.name
            if not new_path.exists():
                shutil.copy2(old_path, new_path)


def _selected_default_bundles(names: Sequence[str] | None) -> tuple[AssetBundle, ...]:
    if not names or "all" in names:
        return DEFAULT_BUNDLES
    selected: list[AssetBundle] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        selected.append(BUNDLES_BY_NAME[name])
    return tuple(selected)


def _download_and_extract_bundle(bundle: AssetBundle, *, dest: Path, force: bool) -> None:
    marker = dest / bundle.marker_name
    if marker.exists() and not force:
        print(f"Assets already installed for bundle {bundle.name!r} at {dest}. Use --force to re-download.")
        _migrate_release_asset_paths(dest)
        return

    print(f"Downloading {bundle.name} assets from: {bundle.url}")
    with urllib.request.urlopen(bundle.url) as resp:
        data = resp.read()

    digest = _sha256(data)
    if bundle.sha256 and digest != bundle.sha256:
        raise SystemExit(f"SHA256 mismatch for {bundle.name}: expected {bundle.sha256}, got {digest}")

    print(f"Extracting {bundle.name} assets...")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        if not force:
            # Never clobber files that already exist — some bundle paths
            # (e.g. examples/data/mgrid_cth_like_lasym_small.nc) are ALSO
            # git-tracked, and a stale bundle copy overwriting the tracked
            # one poisoned the nightly free-boundary golden test
            # (edge zmns 17.8% error, 2026-07-12).  --force restores the
            # old overwrite-everything behavior.
            members = [m for m in tf.getmembers()
                       if not (dest / m.name).exists()]
            skipped = len(tf.getmembers()) - len(members)
            if skipped:
                print(f"  (skipping {skipped} already-present files; "
                      "--force overwrites)")
            _safe_extract(tf, dest, members=members)
        else:
            _safe_extract(tf, dest)

    _migrate_release_asset_paths(dest)
    marker.write_text(f"{bundle.url}\n{digest}\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle",
        action="append",
        choices=("all", *BUNDLES_BY_NAME.keys()),
        help="Default asset bundle to install. May be repeated. Defaults to all bundles.",
    )
    p.add_argument("--url", type=str, default="", help="Custom asset tarball URL. Overrides --bundle.")
    p.add_argument("--sha256", type=str, default="", help="Expected SHA256 for --url.")
    p.add_argument("--dest", type=str, default=str(REPO_ROOT), help="Destination repo root.")
    p.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    p.add_argument("--list", action="store_true", help="Print the default bundle location and common paths.")
    p.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without fetching it.")
    args = p.parse_args(argv)

    if args.url:
        bundles = (
            AssetBundle(
                name="custom",
                url=args.url,
                sha256=args.sha256,
                common_paths=(),
            ),
        )
    else:
        bundles = _selected_default_bundles(args.bundle)

    if args.list or args.dry_run:
        _print_bundle_info(bundles)
        if args.dry_run:
            print("Dry run: no files downloaded or extracted.")
        return 0

    dest = Path(args.dest).expanduser().resolve()
    for bundle in bundles:
        _download_and_extract_bundle(bundle, dest=dest, force=bool(args.force))
    _migrate_release_asset_paths(dest)
    print("Assets installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
