"""Panel-inventory and style smoke tests for the ``--plot`` summary figure.

One in-process solve of the bundled ``cth_like_fixed_bdy`` deck feeds every
check, so the module needs no golden fixtures and stays network-free:

- the summary figure carries the full required panel set (iota full-mesh,
  pressure, ``<J.B>``, Mercier + Glasser ``D_R``, magnetic well, ``J(alpha, s)``
  invariant map, two Boozer ``|B|`` panels, scalar card);
- style invariants are pinned: every ``|B|`` contour set is non-filled and
  jet-mapped, the 3-D surface colormap constant is jet, all text is >= 11 pt,
  every drawn text artist stays inside the canvas, saved PNGs are >= 200 dpi;
- the wout-based Glasser ``D_R`` reconstruction used by the stability panel
  must agree with the traceable :func:`vmex.core.stability.glasser_d_r_state`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("booz_xform_jax")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from vmex.core import optimize as opt  # noqa: E402
from vmex.core import plotting  # noqa: E402
from vmex.core import stability as stab  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.wout import wout_from_state  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
DECK = "cth_like_fixed_bdy"

EXPECTED_PANELS = {
    "iota", "pressure", "jdotb", "stability", "well",
    "j_invariant", "card", "boozer_mid", "boozer_lcfs",
}


@pytest.fixture(scope="module")
def solved_case():
    """Solve the deck once; return ``(eq, WoutData)``."""
    inp = VmecInput.from_file(DATA_DIR / f"input.{DECK}")
    eq = opt.solve_equilibrium(inp)
    wout = wout_from_state(inp=inp, state=eq.state, fsqr=0.0, fsqz=0.0, fsql=0.0)
    return eq, wout


@pytest.fixture(scope="module")
def summary_figure(solved_case):
    """Rendered summary figure + meta; closed after the module finishes."""
    import matplotlib.pyplot as plt

    _, wout = solved_case
    fig, meta = plotting._summary_figure(wout)
    fig.canvas.draw()
    yield fig, meta
    plt.close(fig)


def _drawn_tick_labels(axis):
    """Tick labels matplotlib actually draws (tick within the view interval)."""
    lo, hi = sorted(axis.get_view_interval())
    for tick in axis.get_major_ticks():
        if lo <= tick.get_loc() <= hi:
            yield tick.label1


def _text_artists(fig):
    for ax in fig.axes:
        items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        items += list(_drawn_tick_labels(ax.xaxis))
        items += list(_drawn_tick_labels(ax.yaxis))
        items += list(ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            items += list(legend.get_texts())
        for text in items:
            if text.get_visible() and text.get_text().strip():
                yield text


def _contour_sets(ax):
    from matplotlib.contour import QuadContourSet

    return [c for c in getattr(ax, "collections", []) if isinstance(c, QuadContourSet)]


def test_summary_panel_inventory(summary_figure):
    """All nine required panels exist and are populated."""
    _, meta = summary_figure
    assert set(meta["axes"]) == EXPECTED_PANELS
    for name, ax in meta["axes"].items():
        assert ax.lines or ax.collections or ax.texts, f"panel {name!r} is empty"
        if name != "card":
            assert ax.get_xlabel().strip(), f"panel {name!r} lacks an x label"
            assert ax.get_ylabel().strip(), f"panel {name!r} lacks a y label"


def test_summary_contours_nonfilled_and_jet(summary_figure):
    """J and Boozer |B| contour sets are non-filled; |B| panels use jet."""
    _, meta = summary_figure
    for name in ("j_invariant", "boozer_mid", "boozer_lcfs"):
        sets = _contour_sets(meta["axes"][name])
        assert sets, f"panel {name!r} has no contour set"
        for cs in sets:
            assert cs.filled is False, f"filled contour in {name!r}"
            if name.startswith("boozer"):
                assert cs.get_cmap().name == "jet"


def test_summary_typography_and_no_clipping(summary_figure):
    """Text >= 11 pt and every drawn text artist inside the canvas."""
    fig, _meta = summary_figure
    renderer = fig.canvas.get_renderer()
    bbox = fig.bbox
    for text in _text_artists(fig):
        assert text.get_fontsize() >= 10.9, f"{text.get_text()!r} is {text.get_fontsize()} pt"
        extent = text.get_window_extent(renderer=renderer)
        assert extent.x0 >= bbox.x0 - 2 and extent.x1 <= bbox.x1 + 2, text.get_text()
        assert extent.y0 >= bbox.y0 - 2 and extent.y1 <= bbox.y1 + 2, text.get_text()


def test_summary_field_line_and_j_map_present(summary_figure):
    """Boozer panels carry the iota field line; the J map spans surfaces."""
    _, meta = summary_figure
    for name in ("boozer_mid", "boozer_lcfs"):
        labels = [line.get_label() for line in meta["axes"][name].lines]
        assert any("field line" in label for label in labels), name
    j_map = meta["j_map"]["j_map"]
    assert np.isfinite(j_map).any()
    assert j_map.shape[0] >= 5  # radial spread of Boozer surfaces


def test_summary_style_constants():
    """dpi and 3-D colormap style pins."""
    assert plotting._DPI >= 200
    assert plotting._CMAP_3D == "jet"


def test_d_r_reconstruction_matches_traceable(solved_case):
    """wout-based Glasser D_R == traceable glasser_d_r_state on this deck."""
    eq, wout = solved_case
    recon = plotting._glasser_d_r_from_wout(wout)
    assert recon["valid"], recon["note"]
    reference = np.asarray(stab.glasser_d_r_state(eq.state, eq.runtime))
    interior = slice(2, -1)
    scale = float(np.max(np.abs(reference[interior])))
    assert scale > 0.0
    error = float(np.max(np.abs(recon["d_r"][interior] - reference[interior])))
    assert error <= 1.0e-4 * scale


def test_saved_summary_png_resolution(solved_case, tmp_path):
    """plot_wout writes the summary PNG at >= 200 dpi pixel dimensions."""
    import matplotlib.image as mpimg

    _, wout = solved_case
    paths = plotting.plot_wout(wout, tmp_path, which=("summary",), name=DECK)
    png = paths["summary"]
    assert png.exists()
    pixels = mpimg.imread(str(png))
    width_in, height_in = 15.0, 11.5  # _summary_figure figsize
    assert pixels.shape[1] >= 0.95 * width_in * plotting._DPI
    assert pixels.shape[0] >= 0.95 * height_in * plotting._DPI


# ==========================================================================
# Degenerate inputs and fallback panels
# ==========================================================================

def test_d_r_guards_reject_degenerate_wouts():
    """D_R reconstruction flags too-few-surface and vanishing-phip inputs."""
    tiny = SimpleNamespace(lasym=False, ns=3)
    info = plotting._glasser_d_r_from_wout(tiny)
    assert info["valid"] is False and "too few surfaces" in info["note"]

    ns = 5
    flat = SimpleNamespace(
        lasym=False, ns=ns, nfp=1, signgs=-1,
        xm_nyq=np.array([0.0, 1.0]), xn_nyq=np.array([0.0, 0.0]),
        xm=np.array([0.0, 1.0]), xn=np.array([0.0, 0.0]),
        pres=np.zeros(ns), phips=np.zeros(ns), vp=np.ones(ns),
        iotas=np.ones(ns), buco=np.zeros(ns), jdotb=np.zeros(ns),
        bdotb=np.ones(ns), DMerc=np.zeros(ns),
    )
    info = plotting._glasser_d_r_from_wout(flat)
    assert info["valid"] is False and "vanishing phip" in info["note"]


def test_d_r_self_check_rejects_inconsistent_dmerc(solved_case):
    """A stored DMerc the integrals cannot reproduce invalidates the curve."""
    _, wout = solved_case
    tampered = dataclasses.replace(wout, DMerc=np.zeros_like(np.asarray(wout.DMerc)))
    info = plotting._glasser_d_r_from_wout(tampered)
    assert info["valid"] is False
    assert "self-check failed" in info["note"]
    assert info["d_r"] is None


def test_j_invariant_map_rejects_degenerate_field():
    """A constant Boozer |B| cannot define a trapped-particle pitch."""
    booz = {
        "bmnc_b": np.array([[1.0]]), "bmns_b": None,
        "xm_b": np.array([0]), "xn_b": np.array([0]),
        "nfp": 1, "iota_b": np.array([1.0]),
    }
    with pytest.raises(ValueError, match="degenerate"):
        plotting._j_invariant_map(booz)


def test_magnetic_well_panel_handles_zero_axis_vprime():
    """V'(0) = 0 draws the explanatory note instead of dividing by zero."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fake = SimpleNamespace(ns=5, vp=np.array([0.0, 1.0, 3.0, 2.0, 2.0]))
    plotting._magnetic_well_panel(ax, fake)
    assert any("V'(0) = 0" in t.get_text() for t in ax.texts)
    assert not ax.lines
    plt.close(fig)


def test_summary_survives_boozer_failure(solved_case, monkeypatch):
    """Boozer-transform failure leaves annotated placeholder panels."""
    import matplotlib.pyplot as plt

    _, wout = solved_case

    def _broken(_wout, **_kwargs):
        raise RuntimeError("synthetic boozer failure")

    monkeypatch.setattr(plotting, "_boozer_summary_data", _broken)
    fig, meta = plotting._summary_figure(wout)
    try:
        for name in ("j_invariant", "boozer_mid", "boozer_lcfs"):
            ax = meta["axes"][name]
            assert any("Boozer transform unavailable" in t.get_text() for t in ax.texts), name
            assert ax.get_title().strip() and ax.get_xlabel().strip()
    finally:
        plt.close(fig)


def test_summary_survives_j_map_failure(solved_case, monkeypatch):
    """J-map failure annotates its panel; Boozer |B| panels still render."""
    import matplotlib.pyplot as plt

    _, wout = solved_case

    def _broken(_booz, **_kwargs):
        raise RuntimeError("synthetic bounce failure")

    monkeypatch.setattr(plotting, "_j_invariant_map", _broken)
    fig, meta = plotting._summary_figure(wout)
    try:
        ax = meta["axes"]["j_invariant"]
        assert any("J map unavailable" in t.get_text() for t in ax.texts)
        assert ax.get_title() == "second adiabatic invariant"
        for name in ("boozer_mid", "boozer_lcfs"):
            assert _contour_sets(meta["axes"][name]), name
    finally:
        plt.close(fig)


def test_plot_surfaces_pads_unused_axes(solved_case, tmp_path):
    """A slice count off the grid ends with blank axes, not an IndexError."""
    _, wout = solved_case
    path = plotting.plot_surfaces(
        wout, tmp_path / "surfaces.png", nzeta=5, nradii=4, ntheta=48,
    )
    assert path.exists() and path.stat().st_size > 0


def test_plot_profiles_without_fsqt_history(solved_case, tmp_path):
    """An all-zero fsqt history draws the no-history note panel."""
    _, wout = solved_case
    assert not np.any(np.asarray(wout.fsqt) > 0.0)  # in-memory wout: no history
    path = plotting.plot_profiles(wout, tmp_path / "profiles.png")
    assert path.exists() and path.stat().st_size > 0
