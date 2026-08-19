"""Panel-inventory and style smoke tests for the ``--plot`` summary figure.

One in-process solve of the bundled ``cth_like_fixed_bdy`` deck feeds every
check, so the module needs no golden fixtures and stays network-free:

- the summary figure carries the full required panel set (iota full-mesh,
  pressure, ``<J.B>``, combined Mercier/Glasser/well profiles, 3-D LCFS,
  polar ``J(alpha, s)``, two Boozer ``|B|`` panels, scalar card);
- style invariants are pinned: every ``|B|`` contour set is non-filled and
  jet-mapped, the 3-D surface colormap constant is jet, all text is >= 11 pt,
  every drawn text artist stays inside the canvas, saved PNGs are >= 200 dpi;
- the wout-based Glasser ``D_R`` reconstruction and frozen-pressure response
  must recover the traceable equilibrium values at the stored beta.
"""

from __future__ import annotations

import dataclasses
import inspect
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
    "iota", "pressure", "jdotb", "stability", "boundary_3d",
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
        if name not in ("card", "boundary_3d"):
            assert ax.get_xlabel().strip(), f"panel {name!r} lacks an x label"
            assert ax.get_ylabel().strip(), f"panel {name!r} lacks a y label"


def test_summary_contours_and_colormaps(summary_figure):
    """The J disk is filled; Boozer |B| contours are unfilled and jet-mapped."""
    _, meta = summary_figure
    j_sets = _contour_sets(meta["axes"]["j_invariant"])
    assert any(cs.filled for cs in j_sets)
    for name in ("boozer_mid", "boozer_lcfs"):
        sets = _contour_sets(meta["axes"][name])
        assert sets, f"panel {name!r} has no contour set"
        for cs in sets:
            assert cs.filled is False, f"filled contour in {name!r}"
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
    j_axis = meta["axes"]["j_invariant"]
    assert j_axis.get_xlabel() == r"$s\cos\alpha$"
    assert j_axis.get_ylabel() == r"$s\sin\alpha$"
    assert j_axis.get_aspect() == 1.0


def test_summary_combines_stability_and_well(summary_figure):
    """DMerc, dashed D_R, and dash-dot V'' share zeroes; legend stays below."""
    fig, meta = summary_figure
    stability = meta["axes"]["stability"]
    well = meta["well_axis"]
    assert {line.get_linestyle() for line in stability.lines} >= {"-", "--"}
    assert any(line.get_linestyle() == "-." for line in well.lines)
    assert well.yaxis.label.get_color() == plotting._LINE_COLORS[2]
    assert "V''" in well.get_ylabel()
    labels = [text.get_text() for text in stability.get_legend().get_texts()]
    assert any("V''" in label and "well" in label for label in labels)
    fig.canvas.draw()
    assert stability.transData.transform((0.0, 0.0))[1] == pytest.approx(
        well.transData.transform((0.0, 0.0))[1])
    renderer = fig.canvas.get_renderer()
    assert stability.get_legend().get_window_extent(renderer).y1 <= (
        stability.get_window_extent(renderer).y0 + 2)


def test_summary_style_constants():
    """CLI plots keep publication resolution, smooth 3-D grids, and jet |B|."""
    assert plotting._DPI >= 200
    assert plotting._CMAP_3D == "jet"
    assert plotting._CMAP_MODB == "jet"
    signature = inspect.signature(plotting.plot_boundary_3d)
    assert signature.parameters["ntheta"].default >= 120


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


def test_frozen_pressure_scan_recovers_wout(solved_case):
    """The frozen pressure scan exactly returns the WOUT at its stored beta."""
    _, wout = solved_case
    info = plotting._glasser_d_r_from_wout(wout)
    scan = plotting._frozen_pressure_scan_from_wout(
        wout, info, np.array([0.0, float(wout.betatotal)]))
    interior = slice(2, -1)
    assert scan["valid"] and scan["note"] == "WOUT pressure shape"
    for key, reference in (
        ("dwell", wout.DWell), ("dmerc", wout.DMerc), ("d_r", info["d_r"]),
    ):
        np.testing.assert_allclose(
            scan[key][1, interior], np.asarray(reference)[interior], rtol=2.0e-7)
    np.testing.assert_allclose(
        scan["dmerc"][0], np.asarray(wout.DMerc) - np.asarray(wout.DWell))
    np.testing.assert_allclose(
        scan["d_r"][0], np.asarray(info["d_r"]) + np.asarray(wout.DWell))
    # -D_R = DMerc minus a non-negative GGJ correction. Nearly coincident
    # ideal and resistive margins are therefore physical, not duplicated data.
    assert np.all(-scan["d_r"][:, interior] <= scan["dmerc"][:, interior] + 1e-12)


def test_frozen_pressure_scan_uses_explicit_vacuum_seed(solved_case):
    """A vacuum WOUT uses the documented linear pressure seed, not a floor."""
    _, wout = solved_case
    info = plotting._glasser_d_r_from_wout(wout)
    dwell = np.asarray(wout.DWell)
    vacuum = dataclasses.replace(
        wout, pres=np.zeros_like(wout.pres), presf=np.zeros_like(wout.presf),
        DWell=np.zeros_like(dwell), DMerc=np.asarray(wout.DMerc) - dwell,
        betatotal=0.0)
    vacuum_info = {**info, "d_r": np.asarray(info["d_r"]) + dwell}
    scan = plotting._frozen_pressure_scan_from_wout(
        vacuum, vacuum_info, np.array([0.0, 0.01]))
    assert scan["valid"] and "vacuum seed" in scan["note"]
    np.testing.assert_allclose(scan["dmerc"][0], vacuum.DMerc)
    np.testing.assert_allclose(scan["d_r"][0], vacuum_info["d_r"])
    assert np.isfinite(scan["dmerc"]).all()
    assert np.isfinite(scan["d_r"]).all()


def test_frozen_pressure_scan_guards(solved_case, tmp_path):
    """Malformed beta grids and unavailable beta normalization fail clearly."""
    _, wout = solved_case
    info = plotting._glasser_d_r_from_wout(wout)
    with pytest.raises(ValueError, match="nonnegative 1-D"):
        plotting._frozen_pressure_scan_from_wout(wout, info, np.array([-0.01, 0.0]))
    unavailable = plotting._frozen_pressure_scan_from_wout(
        dataclasses.replace(wout, wb=0.0), info, np.array([0.0, 0.01]))
    assert unavailable == {
        "valid": False, "note": "pressure-to-beta normalization unavailable"}
    with pytest.raises(ValueError, match="beta_max must be positive"):
        plotting.plot_stability(wout, tmp_path / "invalid.png", beta_max=0.0)


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


def test_saved_stability_png(solved_case, tmp_path):
    """The detailed stability figure renders both diagnostic panels."""
    _, wout = solved_case
    path = plotting.plot_stability(wout, tmp_path / "stability.png")
    assert path.exists() and path.stat().st_size > 0


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


def test_j_invariant_map_uses_one_physical_pitch_on_every_surface(monkeypatch):
    """A radial maximum-J diagnostic holds physical pitch fixed."""
    import vmex.core.bounce as bounce

    pitches = []

    def _fake_bounce(*, alpha, pitch, **_kwargs):
        pitches.append(float(np.asarray(pitch)[0]))
        shape = (1, len(alpha), 1, 1)
        return {"action": np.ones(shape), "usable_mask": np.ones(shape, dtype=bool)}

    monkeypatch.setattr(bounce, "bounce_action_from_boozer", _fake_bounce)
    booz = {
        "bmnc_b": np.array([[1.0, 0.2], [1.1, 0.2]]), "bmns_b": None,
        "xm_b": np.array([0, 0]), "xn_b": np.array([0, 1]), "nfp": 1,
        "iota_b": np.array([0.5, 0.5]), "G_b": np.ones(2), "I_b": np.zeros(2),
        "s_b": np.array([0.25, 0.75]),
    }
    result = plotting._j_invariant_map(booz, pitch_fraction=0.5, nalpha=4)
    np.testing.assert_allclose(pitches, [1.0 / 1.05, 1.0 / 1.05], rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(result["pitch"], pitches[0])

    pitches.clear()
    result = plotting._j_invariant_map(booz, pitch=1.0 / 1.05, nalpha=4)
    np.testing.assert_allclose(pitches, [1.0 / 1.05, 1.0 / 1.05])
    np.testing.assert_allclose(result["pitch_inverse"], 1.05)

    pitches.clear()
    result = plotting._j_invariant_map(booz, pitch=1.0 / 0.85, nalpha=4)
    np.testing.assert_allclose(pitches, [1.0 / 0.85])
    np.testing.assert_array_equal(result["trapped_surface"], [True, False])
    assert np.all(np.isfinite(result["j_map"][0]))
    assert np.all(np.isnan(result["j_map"][1]))
    with pytest.raises(ValueError, match="not trapped"):
        plotting._j_invariant_map(booz, pitch=0.5, nalpha=4)


def test_volume_second_derivative_of_linear_vprime():
    """The plotted V'' recovers a linear physical V'(s) profile."""
    ns = 7
    s_half = plotting._half_mesh_s(ns)
    slope, intercept = -2.5, 8.0
    vp = np.concatenate(([0.0], (intercept + slope * s_half) / (2.0 * np.pi) ** 2))
    s, vpp = plotting._volume_second_derivative(SimpleNamespace(ns=ns, vp=vp))
    np.testing.assert_allclose(s, s_half)
    np.testing.assert_allclose(vpp, slope, atol=2.0e-14)


def test_vacuum_stability_panel_is_labeled_as_a_limit():
    """A vacuum curve must not be presented as a pressure-stability certificate."""
    import matplotlib.pyplot as plt

    wout = SimpleNamespace(
        ns=7, DMerc=np.ones(7), betatotal=0.0, vp=np.arange(7, dtype=float))
    figure, axis = plt.subplots()
    plotting._stability_panel(
        axis, wout, {"valid": False, "note": "not sampled"}, s_plot_ignore=0.0)
    assert "vacuum-limit" in axis.lines[0].get_label()
    assert "not finite-pressure" in axis.get_title()
    plt.close(figure)


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


def test_summary_plots_the_effective_ripple_profile_when_neo_is_available(
    solved_case, monkeypatch,
):
    """With NEO_JAX present the pressure panel gains an eps_eff^(3/2) twin axis.

    ``epsilon_eff^(3/2)`` (Nemov PoP 6, 4622 (1999)) spans decades across the
    minor radius, so the diagnostic overlay must be logarithmic and share the
    pressure panel's legend rather than replace the pressure curve.
    """
    import matplotlib.pyplot as plt

    from vmex.core import neoclassical

    _, wout = solved_case
    surfaces = np.linspace(0.15, 0.95, 5)
    values = np.geomspace(1.0e-6, 1.0e-3, 5)
    monkeypatch.setattr(neoclassical, "diagnostic_neo_config", lambda: None)
    monkeypatch.setattr(
        neoclassical, "epsilon_effective_from_wout",
        lambda _wout, **_kwargs: (surfaces, values))
    saved = dict(plotting._EPSILON_EFFECTIVE_CACHE)
    plotting._EPSILON_EFFECTIVE_CACHE.clear()

    fig, meta = plotting._summary_figure(wout)
    try:
        info = meta["epsilon_effective"]
        assert info["valid"] and info["note"] == "diagnostic resolution"
        axis = meta["epsilon_axis"]
        assert axis.get_yscale() == "log"
        np.testing.assert_allclose(axis.lines[0].get_ydata(), values)
        labels = [t.get_text() for t in meta["axes"]["pressure"].get_legend().get_texts()]
        assert len(labels) == 2 and any("epsilon" in t or r"\epsilon" in t for t in labels)
    finally:
        plt.close(fig)
        plotting._EPSILON_EFFECTIVE_CACHE.clear()
        plotting._EPSILON_EFFECTIVE_CACHE.update(saved)


def test_epsilon_effective_panel_resolves_a_sub_decade_profile(
    solved_case, monkeypatch,
):
    """A ripple profile flatter than one decade gets a readable linear axis.

    An optimized configuration is exactly the case where eps_eff^(3/2) varies
    by a factor of a few rather than by decades, and there the log autoscale
    snaps to powers of ten: the curve flattens against a limit, the radial
    minimum stops being visible, and the axis carries a single tick label.
    The minimum is the feature the panel exists to show.
    """
    import matplotlib.pyplot as plt

    from vmex.core import neoclassical

    _, wout = solved_case
    surfaces = np.linspace(0.15, 0.95, 5)
    values = np.array([4.4e-3, 2.6e-3, 1.6e-3, 2.1e-3, 3.9e-3])  # 2.7x span
    monkeypatch.setattr(neoclassical, "diagnostic_neo_config", lambda: None)
    monkeypatch.setattr(
        neoclassical, "epsilon_effective_from_wout",
        lambda _wout, **_kwargs: (surfaces, values))
    saved = dict(plotting._EPSILON_EFFECTIVE_CACHE)
    plotting._EPSILON_EFFECTIVE_CACHE.clear()

    fig, meta = plotting._summary_figure(wout)
    try:
        axis = meta["epsilon_axis"]
        assert axis.get_yscale() == "linear"
        low, high = axis.get_ylim()
        assert low < values.min() and values.max() < high
        assert len([t for t in axis.get_yticks() if low <= t <= high]) >= 4
    finally:
        plt.close(fig)
        plotting._EPSILON_EFFECTIVE_CACHE.clear()
        plotting._EPSILON_EFFECTIVE_CACHE.update(saved)


def test_epsilon_effective_summary_tolerates_an_unreferenceable_wout(monkeypatch):
    """A missing backend or an unhashable wout never breaks the summary."""
    from vmex.core import neoclassical

    def unavailable(_wout, **_kwargs):
        raise ImportError("effective ripple requires NEO_JAX")

    monkeypatch.setattr(neoclassical, "epsilon_effective_from_wout", unavailable)
    monkeypatch.setattr(neoclassical, "diagnostic_neo_config", lambda: None)
    before = dict(plotting._EPSILON_EFFECTIVE_CACHE)
    info = plotting._epsilon_effective_summary({"ns": 3})  # dict: no weak reference
    assert info["valid"] is False and "ImportError" in info["note"]
    assert plotting._EPSILON_EFFECTIVE_CACHE == before


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
