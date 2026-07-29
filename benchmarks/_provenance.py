"""Reproducible, path-redacted provenance helpers for benchmark scripts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def git_state(repo: Path) -> dict[str, object]:
    """Return the exact source revision and dirty state."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {
        "measurement_commit": revision,
        "measurement_dirty": dirty,
        "input_data_embedded": False,
    }


def assert_repo_vmex(module_file: str, repo: Path) -> str:
    """Require the imported VMEX module to come from this checkout.

    The returned path is repository-relative so benchmark JSON never records
    a user name, home directory, or private checkout location.
    """
    module = Path(module_file).resolve()
    try:
        relative = module.relative_to(repo.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "benchmark imported VMEX outside the repository checkout; run the script from its source tree"
        ) from exc
    if not relative.parts or relative.parts[0] != "vmex":
        raise RuntimeError(f"unexpected in-tree VMEX module: {relative}")
    return relative.as_posix()


def file_sha256(path: Path) -> str:
    """Hash an executable/reference artifact without exposing its path."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
