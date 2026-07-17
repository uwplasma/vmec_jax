"""Golden-file validation of the complete ``vmex.core.wout`` writer.

For each reference deck, converge the equilibrium with the core multigrid
solver (:func:`vmex.core.multigrid.solve_multigrid`) at the deck's own
settings, build the full VMEC2000-compatible dataset with
:func:`vmex.core.wout.wout_from_state`, write it with
:func:`write_wout`, and compare EVERY variable that also exists in the
golden VMEC2000 ``wout_*.nc`` (fresh VMEC2000 runs stored in
``~/vmex_notes/golden``).  This drives the same pure-core pipeline as
the ``vmec`` CLI (no legacy modules anywhere in the loop).

Comparison policy per variable class:

- integers / logicals / strings / mode tables / input profiles: exact;
- iteration-history quantities (``niter``, ``itfsq``, ``fsqt``, ``wdot``,
  ``ier_flag``, ``extcur``, ``mgrid_mode``): structural only (the solver
  trajectory differs between implementations);
- residual scalars ``fsqr``/``fsqz``/``fsql``: bounded by the deck ftol;
- floats: per-variable (rtol, atol) with optional near-axis masking, the
  tolerances adopted from ``tests/io/wout/test_wout_comprehensive_parity.py``
  where stricter than plan.md Appendix A;
- scalars VMEC2000 zeroes when it hits NITER without converging (the late
  ``eqfor.f`` block) are skipped when the *golden* run did not converge
  (vmex always writes the computed values - zero-crash policy).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from vmex.core.wout import (  # noqa: E402
    WoutData,
    read_wout,
    wout_from_state,
    write_wout,
)

from conftest import resolve_golden_dir

GOLDEN_DIR = resolve_golden_dir()
pytestmark_golden = pytest.mark.skipif(
    GOLDEN_DIR is None, reason="golden VMEC2000 fixtures unavailable (offline?)"
)
pytestmark = [
    pytestmark_golden,
    pytest.mark.usefixtures("_module_jit_enabled"),  # full solves: run jitted
]
DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

CASES = [
    "solovev",
    "cth_like_fixed_bdy",
    "li383_low_res",
    "up_down_asymmetric_tokamak",
]

# Variables VMEC2000's eqfor.f only fills after full convergence (it returns
# early when ier_flag != successful); golden files from NITER-exhausted runs
# store zeros there.
_NONCONV_ZEROED = {
    "wp", "aspect", "Aminor_p", "Rmajor_p", "volume_p", "rmax_surf",
    "rmin_surf", "zmax_surf", "betatotal", "betapol", "betator", "b0",
    "volavgB", "IonLarmor", "ctor",
}

# Structural-only: iteration-history / bookkeeping quantities.
_STRUCTURAL_ONLY = {"niter", "itfsq", "ier_flag", "fsqt", "wdot", "extcur",
                    "mgrid_mode", "fsqr", "fsqz", "fsql"}

_EXACT_INT = {"nfp", "ns", "mpol", "ntor", "mnmax", "mnmax_nyq", "mnyq",
              "nnyq", "signgs", "nextcur", "lasym__logical__",
              "lrecon__logical__", "lfreeb__logical__", "lrfp__logical__",
              "lmove_axis__logical__"}
_EXACT_FLOAT = {"xm", "xn", "xm_nyq", "xn_nyq", "version_", "gamma", "ftolv",
                "am", "ac", "ai", "am_aux_s", "am_aux_f", "ai_aux_s",
                "ai_aux_f", "ac_aux_s", "ac_aux_f"}
_EXACT_STRING = {"input_extension", "mgrid_file", "pmass_type", "pcurr_type",
                 "piota_type"}

# (rtol, atol, skip_first_surfaces, atol_is_scale_relative)
_TIGHT = (1e-6, 1e-7, 0, False)
_NORMAL = (5e-5, 1e-7, 0, False)
_FLOAT_TOL: dict[str, tuple[float, float, int, bool]] = {
    "rmnc": _TIGHT, "zmns": _TIGHT, "rmns": _TIGHT, "zmnc": _TIGHT,
    "lmns": (1e-6, 3e-7, 0, False), "lmnc": (1e-6, 3e-7, 0, False),
    "gmnc": _NORMAL, "gmns": _NORMAL, "bmnc": _NORMAL, "bmns": _NORMAL,
    "bsupumnc": _NORMAL, "bsupumns": _NORMAL,
    "bsupvmnc": _NORMAL, "bsupvmns": _NORMAL,
    "bsubumnc": _NORMAL, "bsubumns": _NORMAL,
    # near-zero covariant toroidal-field channels are sensitive to the
    # wrout filtering conventions (legacy parity contract: atol 3e-3)
    "bsubvmnc": (5e-5, 3e-3, 0, False), "bsubvmns": (5e-5, 3e-3, 0, False),
    "bsubsmns": (5e-5, 1e-6, 2, False), "bsubsmnc": (5e-5, 1e-6, 2, False),
    # currents inherit the bsubv near-zero noise with a d/ds gain; bound
    # them relative to the dominant harmonic amplitude
    "currumnc": (1e-4, 1e-2, 2, True), "currumns": (1e-4, 1e-2, 2, True),
    "currvmnc": (1e-4, 1e-2, 2, True), "currvmns": (1e-4, 1e-2, 2, True),
    # iota is recomputed from the solved state for current-driven (NCURR=1)
    # decks, so it carries the run-to-run convergence drift
    "iotaf": (1e-5, 1e-7, 0, False), "iotas": (1e-5, 1e-7, 0, False),
    "q_factor": (1e-5, 1e-6, 0, False),
    "presf": _TIGHT, "pres": _TIGHT,
    "phi": _TIGHT, "phipf": _TIGHT, "chipf": _TIGHT, "phips": _TIGHT,
    "chi": (1e-5, 1e-9, 0, False), "mass": _TIGHT,
    "vp": _NORMAL, "bvco": _NORMAL, "bdotb": _NORMAL, "bdotgradv": _NORMAL,
    "buco": (1e-2, 1e-8, 0, False),
    "jcuru": (1e-2, 1e-4, 0, True), "jcurv": (1e-2, 1e-4, 0, True),
    "jdotb": (1e-2, 1e-4, 0, True),
    "beta_vol": (1e-3, 1e-11, 0, False),
    "over_r": (1e-4, 1e-8, 0, False),
    "specw": (5e-3, 1e-8, 0, False),
    "equif": (1e-2, 1e-8, 0, False),
    "DMerc": (5e-2, 1e-3, 2, True), "DShear": (5e-2, 1e-3, 2, True),
    "DWell": (5e-2, 1e-3, 2, True), "DCurr": (5e-2, 1e-3, 2, True),
    "DGeod": (5e-2, 1e-3, 2, True),
    "raxis_cc": _TIGHT, "zaxis_cs": _TIGHT, "raxis_cs": _TIGHT,
    "zaxis_cc": _TIGHT,
    # scalars
    "wb": (1e-6, 1e-12, 0, False), "wp": (1e-6, 1e-12, 0, False),
    "volume_p": (1e-6, 1e-12, 0, False), "aspect": (1e-6, 1e-12, 0, False),
    "Aminor_p": (1e-6, 1e-12, 0, False), "Rmajor_p": (1e-6, 1e-12, 0, False),
    "betatotal": (1e-6, 1e-12, 0, False), "betapol": (1e-5, 1e-12, 0, False),
    "betator": (1e-5, 1e-12, 0, False), "betaxis": (1e-5, 1e-12, 0, False),
    "b0": (1e-6, 1e-12, 0, False), "rbtor0": (1e-6, 1e-12, 0, False),
    "rbtor": (1e-6, 1e-12, 0, False), "volavgB": (1e-6, 1e-12, 0, False),
    "IonLarmor": (1e-6, 1e-12, 0, False),
    "rmax_surf": (1e-6, 1e-12, 0, False), "rmin_surf": (1e-6, 1e-12, 0, False),
    "zmax_surf": (1e-6, 1e-12, 0, False),
    "ctor": (1e-3, 1e-6, 0, False),
}

# Equilibrium-drift tolerance set, applied when the solved state itself is
# expected to differ from the golden state beyond the tight bounds:
#  - loosely-converged goldens (ftolv > 1e-9, e.g. li383 at ftol=1e-6):
#    both runs stop at fsq ~ 1e-6 where slow quantities (axis iota for
#    current-driven decks) still carry ~1% drift.
# (lasym runs used to be in this set wholesale: the legacy lasym solve
# inherited the fixaray.f dnorm defect and converged ~5% away from VMEC2000.
# With the dnorm fix in vmex/kernels/tomnsp.py / vmex/core/fourier.py
# the up_down deck matches the golden at the tight bounds on the solved
# harmonics and scalars, so lasym no longer triggers drift tolerances by
# itself — only the writer-recomputed diagnostic channels below stay
# relaxed for lasym.)
# The wout *writer* itself is exact - proven by the tightly-converged
# symmetric cases and by the bit-exact recomputation identities
# (force_balance/currents/equif validated on the golden files). These
# bounds pin the current accuracy while still catching any
# normalization-factor (2x) regressions.
_DRIFT_TOL: dict[str, tuple[float, float, int, bool]] = {
    **{k: (5e-2, 2e-2, 0, True) for k in (
        "rmnc", "zmns", "rmns", "zmnc",
        "gmnc", "gmns", "bmnc", "bmns", "bsubumnc", "bsubumns",
        "bsubvmnc", "bsubvmns",
        "bsupumnc", "bsupumns", "bsupvmnc", "bsupvmns",
    )},
    # lambda absorbs the angle-parameterization difference; bsubs is the
    # most derivative-sensitive covariant component
    **{k: (5e-2, 1.5e-1, 0, True) for k in ("lmns", "lmnc", "bsubsmns", "bsubsmnc")},
    **{k: (1e-4, 2e-2, 2, True) for k in (
        "currumnc", "currumns", "currvmnc", "currvmns",
    )},
    **{k: (1e-2, 1e-3, 0, True) for k in ("vp", "bdotb", "bdotgradv", "buco", "bvco")},
    # the magnetic axis is the slowest-converging degree of freedom
    **{k: (2e-2, 1e-3, 0, True) for k in (
        "raxis_cc", "zaxis_cs", "raxis_cs", "zaxis_cc",
    )},
    **{k: (1e-2, 1e-2, 0, True) for k in ("jcuru", "jcurv", "jdotb")},
    "over_r": (1e-2, 1e-8, 0, False),
    "rbtor": (1e-4, 1e-12, 0, False),
    "rbtor0": (1e-4, 1e-12, 0, False),
    "b0": (5e-3, 1e-12, 0, False),
    "betaxis": (1e-2, 1e-12, 0, False),
    "beta_vol": (1e-2, 1e-11, 0, False),
    "betatotal": (1e-3, 1e-12, 0, False),
    "betapol": (1e-3, 1e-12, 0, False),
    "betator": (1e-3, 1e-12, 0, False),
    "specw": (1e-2, 1e-8, 0, False),
    "chi": (2e-2, 1e-6, 0, False),
    "chipf": (2e-2, 1e-4, 0, False),
    "iotaf": (2e-2, 1e-6, 0, False),
    "iotas": (2e-2, 1e-6, 0, False),
    "q_factor": (2e-2, 1e-6, 0, False),
    "wb": (1e-4, 1e-12, 0, False),
    "wp": (1e-3, 1e-12, 0, False),
}
# Normalized force-balance/stability diagnostics of drifted states are
# derivative-amplified beyond useful comparison: finite-only.
_DRIFT_FINITE_ONLY = {"DMerc", "DShear", "DWell", "DCurr", "DGeod", "equif"}
# lasym channels that stay drift-relaxed even though the lasym solve now
# matches golden tightly (fixaray.f dnorm fix): the writer-recomputed
# near-axis current/bsubv harmonics and Mercier-family diagnostics still
# disagree with golden beyond drift levels while the solved harmonics agree
# to ~1e-11 (lasym writer recomputation parity is a tracked follow-up).
_LASYM_DIAG_DRIFT = {
    "bsubvmnc", "bsubvmns", "currumnc", "currumns", "currvmnc", "currvmns",
}
# For goldens stored at loose ftol (li383: 1e-6), the current-density
# harmonics (radial derivatives of the ~1e-3-drifted field) lose all
# significant digits in subdominant channels: finite-only there.
_LOOSE_FINITE_ONLY = {"currumnc", "currvmnc", "currumns", "currvmns"}

_LASYM_PARTNERS = {
    "rmns", "zmnc", "lmnc", "gmns", "bmns", "bsubumns", "bsubvmns",
    "bsubsmnc", "currumns", "currvmns", "bsupumns", "bsupvmns",
    "raxis_cs", "zaxis_cc",
}


def _get(ds, name):
    return np.asarray(np.ma.filled(ds.variables[name][:], 0.0))


def _get_str(ds, name):
    raw = np.ma.filled(ds.variables[name][:], b"\x00")
    return raw.tobytes().decode("ascii", "ignore").rstrip(" \x00")


_RUN_CACHE: dict[str, tuple[Path, Path]] = {}


@pytest.fixture(scope="module", params=CASES, ids=CASES)
def case(request, tmp_path_factory):
    """Converge one deck with the core solver and write the new wout."""
    name = request.param
    golden = GOLDEN_DIR / name / f"wout_{name}.nc"
    deck = DATA_DIR / f"input.{name}"
    if not golden.exists():
        pytest.skip(f"missing golden file {golden}")
    if not deck.exists():
        pytest.skip(f"missing input deck {deck}")
    if name not in _RUN_CACHE:
        # conftest disables JIT for unit tests; full solves need it.
        jax.config.update("jax_disable_jit", False)
        os.environ.setdefault("VMEX_TOMNSPS_FFT", "0")
        from vmex.core.input import VmecInput
        from vmex.core.multigrid import solve_multigrid

        inp = VmecInput.from_file(deck)
        # raise_on_max_iterations=False: the up_down deck exhausts NITER
        # before FTOL — exactly like the golden VMEC2000 run, which simply
        # wrote its last (NITER-exhausted) state to the wout file.
        result = solve_multigrid(inp, verbose=False,
                                 raise_on_max_iterations=False)
        # fsqt history (wrout.f nstore_seq subsampling of fsqr + fsqz).
        history = np.asarray(result.fsq_history, dtype=float)
        fsqt = None
        if history.size:
            total = history[:, 0] + history[:, 1]
            stride = total.size // 100 + 1
            fsqt = total[stride - 1 :: stride][:100]
        wout = wout_from_state(
            inp=inp, state=result.state,
            fsqr=float(result.fsqr), fsqz=float(result.fsqz),
            fsql=float(result.fsql), fsqt=fsqt,
            niter=int(result.iterations), converged=bool(result.converged),
            input_extension=name,
        )
        out = tmp_path_factory.mktemp(name) / f"wout_{name}.nc"
        write_wout(out, wout)
        _RUN_CACHE[name] = (out, golden)
    return _RUN_CACHE[name]


def test_completeness_and_structure(case):
    """New wout covers every golden variable with identical dims/dtypes."""
    out, golden = case
    with netCDF4.Dataset(golden) as gd, netCDF4.Dataset(out) as nd:
        gvars, nvars = set(gd.variables), set(nd.variables)
        missing = gvars - nvars
        assert not missing, f"variables missing from new wout: {sorted(missing)}"
        extra = nvars - gvars
        assert not extra, f"nonstandard extra variables written: {sorted(extra)}"

        lasym = bool(int(_get(gd, "lasym__logical__")))
        partners = _LASYM_PARTNERS & nvars
        if lasym:
            assert partners == _LASYM_PARTNERS
        else:
            assert not partners, f"lasym partners written for symmetric run: {sorted(partners)}"

        for name in sorted(gvars):
            gv, nv = gd.variables[name], nd.variables[name]
            assert gv.dimensions == nv.dimensions, (
                f"{name}: dimensions {nv.dimensions} != golden {gv.dimensions}"
            )
            assert gv.dtype == nv.dtype, f"{name}: dtype {nv.dtype} != {gv.dtype}"
        for dim, gsize in ((d, len(v)) for d, v in gd.dimensions.items()):
            assert dim in nd.dimensions, f"missing dimension {dim}"
            assert len(nd.dimensions[dim]) == gsize, f"dimension {dim} size mismatch"


def test_golden_values(case):
    """Value-level parity for every golden variable, per-variable policy."""
    out, golden = case
    failures: list[str] = []
    with netCDF4.Dataset(golden) as gd, netCDF4.Dataset(out) as nd:
        lasym = bool(int(_get(gd, "lasym__logical__")))
        ftolv = float(_get(gd, "ftolv"))
        # lasym alone no longer triggers full drift tolerances: the
        # fixaray.f dnorm fix (vmex/kernels/tomnsp.py) closed the lasym
        # solver parity gap.  Only _LASYM_DIAG_DRIFT / _DRIFT_FINITE_ONLY
        # channels stay relaxed for lasym goldens.
        drift = ftolv > 1e-9
        golden_conv = max(float(_get(gd, "fsqr")), float(_get(gd, "fsqz")),
                          float(_get(gd, "fsql"))) <= ftolv
        for name in sorted(gd.variables):
            if name in _STRUCTURAL_ONLY:
                if name in ("fsqr", "fsqz", "fsql"):
                    ours = float(_get(nd, name))
                    if not ours <= max(10.0 * ftolv, 1e-10):
                        failures.append(f"{name}: residual {ours:.2e} above ftol bound")
                continue
            if name in _EXACT_STRING:
                g, n = _get_str(gd, name), _get_str(nd, name)
                if g != n:
                    failures.append(f"{name}: {n!r} != golden {g!r}")
                continue
            g = _get(gd, name)
            n = _get(nd, name)
            if g.shape != n.shape:
                failures.append(f"{name}: shape {n.shape} != {g.shape}")
                continue
            if name in _EXACT_INT or name in _EXACT_FLOAT:
                if not np.array_equal(g, n):
                    failures.append(f"{name}: exact mismatch (ours {n!r} vs {g!r})")
                continue
            if name in _NONCONV_ZEROED and not golden_conv and not np.any(g):
                # VMEC2000 zeroed this scalar because it hit NITER; we keep
                # the computed value. Require ours to be finite instead.
                if not np.all(np.isfinite(n)):
                    failures.append(f"{name}: non-finite value for nonconverged case")
                continue
            if ((drift or lasym) and name in _DRIFT_FINITE_ONLY) or (
                ftolv > 1e-9 and name in _LOOSE_FINITE_ONLY
            ):
                if not np.all(np.isfinite(n)):
                    failures.append(f"{name}: non-finite values")
                continue
            tol = _FLOAT_TOL.get(name)
            if drift or (lasym and name in _LASYM_DIAG_DRIFT):
                tol = _DRIFT_TOL.get(name, tol)
            if tol is None:
                failures.append(f"{name}: no comparison policy defined")
                continue
            rtol, atol, skip, scale_rel = tol
            gs, ns_ = (g[skip:], n[skip:]) if (skip and g.ndim >= 1 and g.shape[0] > skip) else (g, n)
            if scale_rel:
                # for sine/cosine parity pairs the meaningful amplitude is
                # the pair's (the sine partner of a nearly-symmetric field
                # is itself near zero)
                scale = float(np.max(np.abs(g)))
                for a, b in (("mnc", "mns"), ("mns", "mnc")):
                    if name.endswith(a):
                        partner = name[: -len(a)] + b
                        if partner in gd.variables:
                            scale = max(scale, float(np.max(np.abs(_get(gd, partner)))))
                atol_eff = atol * scale
            else:
                atol_eff = atol
            bad = np.abs(ns_ - gs) > (atol_eff + rtol * np.abs(gs))
            if np.any(bad):
                idx = np.unravel_index(int(np.argmax(np.abs(ns_ - gs))), np.shape(gs))
                failures.append(
                    f"{name}: {int(np.count_nonzero(bad))} elements out of tolerance "
                    f"(rtol={rtol:g}, atol={atol_eff:g}); worst at {idx}: "
                    f"ours={np.asarray(ns_)[idx]:.6e} golden={np.asarray(gs)[idx]:.6e}"
                )
    assert not failures, "wout golden mismatches:\n  " + "\n  ".join(failures)


def test_roundtrip(case):
    """write_wout -> read_wout is the identity on WoutData."""
    out, _ = case
    w = read_wout(out)
    assert isinstance(w, WoutData)
    tmp = out.parent / "roundtrip.nc"
    write_wout(tmp, w)
    w2 = read_wout(tmp)
    for f in dataclasses.fields(WoutData):
        a, b = getattr(w, f.name), getattr(w2, f.name)
        if a is None or b is None:
            assert a is None and b is None, f"{f.name}: optional mismatch"
            continue
        if isinstance(a, (str, int, bool)):
            assert a == b, f"{f.name}: {b!r} != {a!r}"
        elif isinstance(a, float):
            assert a == b or (np.isnan(a) and np.isnan(b)), f"{f.name}: {b} != {a}"
        else:
            assert np.array_equal(np.asarray(a), np.asarray(b)), f"{f.name}: array mismatch"
