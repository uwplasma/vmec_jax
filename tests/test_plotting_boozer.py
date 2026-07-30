"""Tests for ``vmex.core.plotting`` and ``vmex.core.boozer``.

Uses the golden VMEC2000 ``wout`` fixtures (see ``conftest.resolve_golden_dir``):

- ``plot_wout`` on the symmetric ``solovev`` deck and the ``lasym``
  ``up_down_asymmetric_tokamak`` deck: every requested figure is written,
  each under 400 kB, with no exceptions;
- ``run_booz_xform`` on ``cth_like_fixed_bdy``: the ``boozmn`` file loads,
  ``bmnc_b`` is finite, and surface selection works (spectrum parity with
  the legacy driver was A/B-proven before the legacy tree retired);
- ``plot_boozmn`` produces its figure set from that ``boozmn`` file.

All outputs go to ``tmp_path`` only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("matplotlib")

from vmex.core.boozer import run_booz_xform  # noqa: E402
from vmex.core.plotting import plot_boozmn, plot_wout  # noqa: E402

from tests.conftest import resolve_golden_dir

GOLDEN_DIR = resolve_golden_dir()
pytestmark = pytest.mark.skipif(
    GOLDEN_DIR is None, reason="golden VMEC2000 fixtures unavailable (offline?)"
)

MAX_FIGURE_BYTES = 400 * 1024
WOUT_KEYS = ("summary", "surfaces", "modB", "profiles", "3d")


def _golden_wout(case: str) -> Path:
    assert GOLDEN_DIR is not None
    path = GOLDEN_DIR / case / f"wout_{case}.nc"
    if not path.exists():
        pytest.skip(f"golden wout missing for {case}")
    return path


def _check_figures(paths: dict[str, Path], expected_keys) -> None:
    assert set(paths) == set(expected_keys)
    for key, path in paths.items():
        assert path.exists(), f"figure {key} not written: {path}"
        size = path.stat().st_size
        assert 0 < size < MAX_FIGURE_BYTES, f"figure {key} is {size} bytes: {path}"


# ==========================================================================
# plot_wout
# ==========================================================================

@pytest.mark.parametrize("case", ["solovev", "up_down_asymmetric_tokamak"])
def test_plot_wout_golden(case: str, tmp_path: Path) -> None:
    """All five figures render from golden wouts (sym and lasym) under 400 kB."""
    wout_path = _golden_wout(case)
    outdir = tmp_path / case
    paths = plot_wout(wout_path, outdir, which=WOUT_KEYS)
    _check_figures(paths, WOUT_KEYS)
    for path in paths.values():
        assert Path(path).parent == outdir


def test_plot_wout_accepts_woutdata_and_subset(tmp_path: Path) -> None:
    """plot_wout takes an in-memory WoutData and honors ``which`` subsets."""
    from vmex.core.wout import read_wout

    data = read_wout(str(_golden_wout("solovev")))
    paths = plot_wout(data, tmp_path, which=("profiles", "modB"), name="solovev_mem")
    _check_figures(paths, ("profiles", "modB"))
    assert paths["profiles"].name == "solovev_mem_profiles.png"


def test_plot_wout_rejects_unknown_figure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown figure"):
        plot_wout(_golden_wout("solovev"), tmp_path, which=("summary", "bogus"))


# ==========================================================================
# run_booz_xform + plot_boozmn
# ==========================================================================

@pytest.fixture(scope="module")
def booz_case() -> str:
    return "cth_like_fixed_bdy"


@pytest.fixture(scope="module")
def boozmn_new(booz_case: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("booz_xform_jax")
    wout_path = GOLDEN_DIR / booz_case / f"wout_{booz_case}.nc"
    if not wout_path.exists():
        pytest.skip(f"golden wout missing for {booz_case}")
    outdir = tmp_path_factory.mktemp("booz_new")
    return run_booz_xform(wout_path, mbooz=24, nbooz=24, outdir=outdir)


def test_run_booz_xform_output_loads_and_is_finite(boozmn_new: Path, booz_case: str) -> None:
    assert boozmn_new.exists()
    assert boozmn_new.name == f"boozmn_{booz_case}.nc"
    with netCDF4.Dataset(boozmn_new) as ds:
        bmnc_b = np.asarray(ds.variables["bmnc_b"][:], dtype=float)
        ns_b = int(ds.variables["ns_b"][...])
    assert bmnc_b.size > 0
    assert np.all(np.isfinite(bmnc_b))
    assert np.max(np.abs(bmnc_b)) > 0.0
    assert ns_b >= 1


def test_run_booz_xform_surface_selection(booz_case: str, tmp_path: Path) -> None:
    """Requesting s-values transforms only the matching surfaces."""
    pytest.importorskip("booz_xform_jax")
    wout_path = GOLDEN_DIR / booz_case / f"wout_{booz_case}.nc"
    out = run_booz_xform(
        wout_path, mbooz=16, nbooz=16, surfaces=(0.5, 1.0),
        output_path=tmp_path / "boozmn_subset.nc",
    )
    with netCDF4.Dataset(out) as ds:
        bmnc_b = np.asarray(ds.variables["bmnc_b"][:], dtype=float)
    assert min(bmnc_b.shape) == 2  # exactly the two requested surfaces
    assert np.all(np.isfinite(bmnc_b))


def test_plot_boozmn_figures(boozmn_new: Path, tmp_path: Path) -> None:
    keys = ("modB", "mode_profiles", "spectrum")
    paths = plot_boozmn(boozmn_new, tmp_path, which=keys)
    _check_figures(paths, keys)
    for path in paths.values():
        assert Path(path).parent == tmp_path
