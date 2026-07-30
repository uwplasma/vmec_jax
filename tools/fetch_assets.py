#!/usr/bin/env python3
"""Download large example/reference netCDF assets.

Generated optimization figures are intentionally not part of this asset bundle:
rerun the optimization renderers when report-quality panels are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "assets" / "manifest.json"

ASSET_PATH_REWRITES = (("examples_single_grid/data", "examples/data/single_grid"),)


@dataclass(frozen=True)
class AssetBundle:
    name: str
    url: str
    sha256: str
    size_bytes: int
    source: str
    license: str
    generator_revision: str
    destination: str
    default: bool
    common_paths: tuple[str, ...]

    @property
    def marker_name(self) -> str:
        return f".assets_installed_{self.name}.txt"


def _load_bundles() -> tuple[AssetBundle, ...]:
    data = json.loads(MANIFEST_PATH.read_text())
    if data.get("schema") != "vmex.release-assets/1":
        raise RuntimeError(f"unsupported asset manifest schema in {MANIFEST_PATH}")
    bundles = tuple(
        AssetBundle(**{**record, "common_paths": tuple(record["common_paths"])}) for record in data["bundles"]
    )
    if any(bundle.destination not in {"cache", "repository"} for bundle in bundles):
        raise RuntimeError(f"invalid asset destination in {MANIFEST_PATH}")
    return bundles


ALL_BUNDLES = _load_bundles()
DEFAULT_BUNDLES = tuple(bundle for bundle in ALL_BUNDLES if bundle.default)
BUNDLES_BY_NAME = {bundle.name: bundle for bundle in ALL_BUNDLES}


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
        if member.issym() or member.islnk():
            raise SystemExit(f"Refusing to extract archive link: {member.name}")
        target = (dest_resolved / member.name).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise SystemExit(f"Refusing to extract path outside destination: {member.name}")
    if hasattr(tarfile, "data_filter"):
        tf.extractall(dest_resolved, members=selected, filter="data")
    else:  # Python 3.10 and 3.11
        tf.extractall(dest_resolved, members=selected)


def _print_bundle_info(bundles: Sequence[AssetBundle]) -> None:
    print("Asset bundles:")
    for bundle in bundles:
        print(f"- {bundle.name}")
        print(f"  URL:             {bundle.url}")
        print(f"  Expected bytes:  {bundle.size_bytes}")
        print(f"  Expected SHA256: {bundle.sha256 or '(not checked)'}")
        print(f"  Source:          {bundle.source}")
        print(f"  License:         {bundle.license}")
        print(f"  Generator:       {bundle.generator_revision}")
        print(f"  Destination:     {bundle.destination}")
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
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / bundle.marker_name
    if marker.exists() and not force:
        if marker.read_text().splitlines() == [bundle.url, bundle.sha256]:
            print(f"Assets already installed for bundle {bundle.name!r} at {dest}. Use --force to re-download.")
            _migrate_release_asset_paths(dest)
            return
        print(f"Replacing stale marker for bundle {bundle.name!r}")

    print(f"Downloading {bundle.name} assets from: {bundle.url}")
    with urllib.request.urlopen(bundle.url) as resp:
        data = resp.read()

    if bundle.size_bytes and len(data) != bundle.size_bytes:
        raise SystemExit(f"Size mismatch for {bundle.name}: expected {bundle.size_bytes}, got {len(data)}")
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
            members = [m for m in tf.getmembers() if not (dest / m.name).exists()]
            skipped = len(tf.getmembers()) - len(members)
            if skipped:
                print(f"  (skipping {skipped} already-present files; --force overwrites)")
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
    p.add_argument(
        "--dest",
        type=str,
        help="Override the manifest destination (repository or user cache).",
    )
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
                size_bytes=0,
                source="custom URL",
                license="user supplied",
                generator_revision="user supplied",
                destination="repository",
                default=False,
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

    dest_override = Path(args.dest).expanduser().resolve() if args.dest else None
    for bundle in bundles:
        dest = dest_override
        if dest is None:
            dest = Path.home() / ".cache" / "vmex" / bundle.name if bundle.destination == "cache" else REPO_ROOT
        _download_and_extract_bundle(bundle, dest=dest, force=bool(args.force))
    print("Assets installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
