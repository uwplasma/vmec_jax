from __future__ import annotations

"""Pytest configuration.

Allows running tests directly from the repo without requiring an editable
install, silences XLA/absl C++ noise, disables jit globally (unit tests cover
correctness on small arrays; compilation dominates runtime — tests that need
the jit lane re-enable it explicitly), and gates ``full``-marked tests behind
``RUN_FULL=1``.
"""

import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Keep the test suite fast: avoid JAX compilation in unit tests.
try:  # pragma: no cover
    import jax

    jax.config.update("jax_disable_jit", True)
except Exception:  # pragma: no cover
    pass


def pytest_addoption(parser):
    group = parser.getgroup("vmex integration")
    group.addoption(
        "--run-vmec2000",
        action="store_true",
        help="run live integration tests against a local VMEC2000 executable",
    )
    group.addoption(
        "--vmec2000-executable",
        default="",
        metavar="PATH",
        help="VMEC2000 executable for --run-vmec2000 (otherwise auto-discover)",
    )


def pytest_collection_modifyitems(config, items):
    run_full = os.environ.get("RUN_FULL", "") == "1"
    run_vmec2000 = bool(config.getoption("--run-vmec2000"))
    for item in items:
        if item.get_closest_marker("full") is not None and not run_full:
            item.add_marker(pytest.mark.skip(reason="Full tests disabled. Set RUN_FULL=1."))
        if (
            item.get_closest_marker("vmec2000_live") is not None
            and not run_vmec2000
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="Live VMEC2000 integration disabled; use --run-vmec2000"
                )
            )


# ---------------------------------------------------------------------------
# Golden VMEC2000 parity fixtures (stdout, threed1, wout, timings for the
# benchmark decks).  Resolution order:
#   1. VMEX_GOLDEN_DIR environment variable (explicit override),
#   2. ~/vmex_notes/golden (local development snapshot),
#   3. ~/.cache/vmex/golden-v1 (downloaded once from the golden-v1 release).
# ---------------------------------------------------------------------------
# Keep the owner/repo as ``vmec_jax`` (the pre-rename name): the release asset
# lives there, and after the repo is renamed to VMEX GitHub redirects the old
# path to the new one, so this URL resolves both before and after the rename.
# Pointing it at ``VMEX`` directly 404s until the rename lands.
GOLDEN_URL = "https://github.com/uwplasma/vmec_jax/releases/download/golden-v1/vmec-jax-golden-v1.tar.gz"
GOLDEN_SHA256 = "85b1de372066d1dd0c57b1a9ffb569ccc1276bb67dec81e7bf15a5a943ca05d7"


def _download_golden(cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    tarball = cache_root / "vmec-jax-golden-v1.tar.gz"
    if not tarball.exists():
        urllib.request.urlretrieve(GOLDEN_URL, tarball)  # noqa: S310 - fixed https URL
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != GOLDEN_SHA256:
        tarball.unlink()
        raise RuntimeError(f"golden bundle checksum mismatch: {digest}")
    outdir = cache_root / "golden"
    if not outdir.exists():
        with tarfile.open(tarball) as tf:
            tf.extractall(cache_root, filter="data")
    return outdir


def resolve_golden_dir() -> Path | None:
    env = os.environ.get("VMEX_GOLDEN_DIR")
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    local = Path.home() / "vmex_notes" / "golden"
    if local.is_dir():
        return local
    try:
        return _download_golden(Path.home() / ".cache" / "vmex" / "golden-v1")
    except Exception:
        return None


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    path = resolve_golden_dir()
    if path is None:
        pytest.skip("golden VMEC2000 fixtures unavailable (offline?)")
    return path


@pytest.fixture(scope="module")
def _module_jit_enabled():
    """Run a whole module jitted (this conftest disables jit globally).

    Solver-heavy modules opt in with
    ``pytestmark = pytest.mark.usefixtures("_module_jit_enabled")`` — a
    usefixtures mark is instantiated before same-scope fixtures declared in
    the module, so module-scoped solve fixtures run jitted too.  Full solves
    are 5-40x faster jitted (e.g. solovev ns=11: 26 s interpreted vs 3.5 s
    cold / 0.03 s warm jitted); without this the suite's runtime depended on
    which xdist worker had previously run a test that re-enabled jit.
    """
    import jax

    prev = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", prev)
