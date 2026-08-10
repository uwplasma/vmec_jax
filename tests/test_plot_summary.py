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

from pathlib import Path

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
