from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "fetch_assets.py"
MANIFEST = ROOT / "assets" / "manifest.json"


def _load_fetch_assets():
    """Load tools/fetch_assets.py as a module (script contract test)."""
    spec = importlib.util.spec_from_file_location("fetch_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fetch_assets_dry_run_lists_bundle(capsys) -> None:
    module = _load_fetch_assets()

    assert module.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "Asset bundles:" in out
    assert "reference-nc" in out
    assert "wout-fixtures" in out
    assert "golden-v1" not in out
    assert "Expected bytes:" in out
    assert "Expected SHA256:" in out
    assert "examples/data/mgrid_cth_like.nc" in out
    assert "examples/data/wout_*.nc" in out
    assert "Dry run: no files downloaded or extracted." in out


def test_fetch_assets_can_select_one_default_bundle(capsys) -> None:
    module = _load_fetch_assets()

    assert module.main(["--bundle", "wout-fixtures", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "wout-fixtures" in out
    assert "reference-nc" not in out
    assert "docs/_static/readme_best_cases/*/wout_*.nc" in out


def test_asset_manifest_has_complete_provenance() -> None:
    data = json.loads(MANIFEST.read_text())
    assert data["schema"] == "vmex.release-assets/1"
    bundles = data["bundles"]
    assert {bundle["name"] for bundle in bundles} == {
        "golden-v1",
        "reference-nc",
        "wout-fixtures",
    }
    for bundle in bundles:
        assert bundle["url"].startswith("https://github.com/uwplasma/vmex/")
        assert len(bundle["sha256"]) == 64
        assert bundle["size_bytes"] > 0
        assert bundle["source"] and bundle["license"]
        assert len(bundle["generator_revision"]) == 40


def test_golden_bundle_targets_the_user_cache(capsys) -> None:
    module = _load_fetch_assets()

    assert module.main(["--bundle", "golden-v1", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "golden-v1" in out
    assert "Destination:     cache" in out


def test_asset_download_rejects_wrong_size(tmp_path, monkeypatch) -> None:
    module = _load_fetch_assets()
    bundle = module.AssetBundle(
        name="small",
        url="https://example.invalid/small.tar.gz",
        sha256="0" * 64,
        size_bytes=4,
        source="test",
        license="test",
        generator_revision="0" * 40,
        destination="repository",
        default=False,
        common_paths=(),
    )
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: io.BytesIO(b"bad"))

    with pytest.raises(SystemExit, match="Size mismatch"):
        module._download_and_extract_bundle(bundle, dest=tmp_path, force=False)


def test_fetch_assets_safe_extract_rejects_path_traversal(tmp_path) -> None:
    module = _load_fetch_assets()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tf:
        data = b"bad"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    payload.seek(0)

    with tarfile.open(fileobj=payload, mode="r:gz") as tf:
        with pytest.raises(SystemExit, match="outside destination"):
            module._safe_extract(tf, tmp_path)

    assert not (tmp_path.parent / "outside.txt").exists()


def test_fetch_assets_safe_extract_rejects_links(tmp_path) -> None:
    module = _load_fetch_assets()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside.txt"
        tf.addfile(info)
    payload.seek(0)

    with tarfile.open(fileobj=payload, mode="r:gz") as tf:
        with pytest.raises(SystemExit, match="archive link"):
            module._safe_extract(tf, tmp_path)


def test_no_tracked_file_exceeds_one_mib() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("tracked-file gate requires a git checkout")
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = [ROOT / path for path in result.stdout.decode().rstrip("\0").split("\0") if path]
    oversized = [path.relative_to(ROOT) for path in paths if path.stat().st_size > 2**20]
    assert not oversized
