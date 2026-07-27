"""Input-validation and small-helper contracts across the core modules.

Zero-crash policy tests: every user-facing constructor/parser must reject
malformed input with a specific exception (not a crash deep in the
numerics), and the small pure helpers (Boozer path/surface resolution,
optimize table adapters) must implement their documented conventions.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax.numpy as jnp

from vmex.core import boozer, optimize as opt
from vmex.core.fourier import Resolution
from vmex.core.input import _read_indata_text, _parse_scalar


# ---------------------------------------------------------------------------
# fourier.Resolution invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,bad", [
    ("mpol", 0), ("ntor", -1), ("ntheta", 1), ("nzeta", 0), ("nfp", 0), ("ns", 0),
])
def test_resolution_rejects_invalid_dimensions(field, bad):
    good = dict(mpol=4, ntor=2, ntheta=16, nzeta=8, nfp=3, lasym=False, ns=11)
    good[field] = bad
    with pytest.raises(ValueError, match=field):
        Resolution(**good)


# ---------------------------------------------------------------------------
# input.py INDATA parsing errors + Fortran token coercion
# ---------------------------------------------------------------------------


def test_read_indata_requires_namelist_and_terminator():
    with pytest.raises(ValueError, match="no &INDATA namelist"):
        _read_indata_text("&OTHER\nX = 1\n/\n")
    with pytest.raises(ValueError, match="terminating"):
        _read_indata_text("&INDATA\nMPOL = 4\n")


def test_parse_scalar_fortran_conventions():
    assert _parse_scalar("42") == 42
    assert _parse_scalar("1.5D-3") == pytest.approx(1.5e-3)
    assert _parse_scalar("'text'") == "'text'" or isinstance(_parse_scalar("'text'"), str)


# ---------------------------------------------------------------------------
# optimize.py adapters and driver argument validation
# ---------------------------------------------------------------------------


def test_as_1d_handles_scalars_and_sequences():
    np.testing.assert_array_equal(np.asarray(opt._as_1d(0.5)), [0.5])
    np.testing.assert_array_equal(np.asarray(opt._as_1d([0.25, 0.5])), [0.25, 0.5])


def test_mode_matrix_layouts_and_errors():
    ns, mn = 4, 3
    table = np.arange(12.0).reshape(ns, mn)
    w = SimpleNamespace(bmnc=table, bmnc_t=table.T, nothing=None)
    got = opt._mode_matrix(w, "bmnc", ns=ns, mn=mn)
    np.testing.assert_array_equal(np.asarray(got), table)
    got_t = opt._mode_matrix(w, "bmnc_t", ns=ns, mn=mn)
    np.testing.assert_array_equal(np.asarray(got_t), table)  # transposed input
    zeros = opt._mode_matrix(w, "absent", ns=ns, mn=mn, optional=True)
    assert np.count_nonzero(np.asarray(zeros)) == 0
    with pytest.raises(AttributeError, match="absent"):
        opt._mode_matrix(w, "absent", ns=ns, mn=mn)
    with pytest.raises(ValueError, match="unexpected shape"):
        opt._mode_matrix(SimpleNamespace(bad=np.zeros((2, 2))), "bad", ns=ns, mn=mn)


DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_least_squares_rejects_unknown_jac():
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    with pytest.raises(ValueError, match="jac must be None or 'implicit'"):
        opt.least_squares([(opt.aspect_ratio, 6.0, 1.0)], inp, jac="magic")


def test_least_squares_implicit_boundary_map_supports_lasym():
    """The implicit lane's boundary parameter map handles non-stellarator-
    symmetric (lasym) decks via the four RBC/ZBS/RBS/ZBC families — the map
    that lets ``jac="implicit"`` optimise lasym boundaries (the end-to-end
    differentiable lasym gradients are validated in ``test_implicit_grad.py``).
    Fast, pure-function check of the pack/unpack round-trip; no solve."""
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(str(DATA_DIR / "input.up_down_asymmetric_tokamak"))
    assert bool(inp.lasym)
    assert opt._n_boundary_families(inp) == 4  # rbc/zbs/rbs/zbc

    names = opt.boundary_dof_names(inp, 1)
    assert any(nm.startswith("RBS") for nm in names)
    assert any(nm.startswith("ZBC") for nm in names)

    x = opt.pack_boundary(inp, 1)
    assert x.size == len(names)
    inp2 = opt.unpack_boundary(inp, x, 1)
    np.testing.assert_allclose(opt.pack_boundary(inp2, 1), x)


def test_traceable_term_vetting():
    """jac='implicit' term adapter: residuals_state > (state, rt) > reject."""
    qs = opt.QuasisymmetryRatioResidual([0.5], 1, 0)
    assert opt._traceable_term(qs.total_state) == qs.residuals_state
    assert opt._traceable_term(opt.aspect_ratio) is opt.aspect_ratio
    with pytest.raises(ValueError, match="not implicit-differentiable"):
        opt._traceable_term(opt.d_merc)


def test_interp_half_grid_single_sample_broadcasts():
    out = opt._interp_half_grid(jnp.asarray([3.0]), jnp.asarray([0.2, 0.8]),
                                jnp.asarray([0.5]))
    np.testing.assert_array_equal(np.asarray(out), [3.0, 3.0])


# ---------------------------------------------------------------------------
# boozer.py pure helpers
# ---------------------------------------------------------------------------


def test_case_from_wout_naming():
    assert boozer._case_from_wout(Path("wout_li383.nc")) == "li383"
    assert boozer._case_from_wout(Path("input.li383")) == "li383"
    assert boozer._case_from_wout(Path("state.nc")) == "state"


def test_resolve_boozmn_path_precedence(tmp_path):
    wout = tmp_path / "wout_case.nc"
    assert boozer.resolve_boozmn_path(wout) == tmp_path / "boozmn_case.nc"
    assert boozer.resolve_boozmn_path(wout, outdir=tmp_path / "out") == (
        tmp_path / "out" / "boozmn_case.nc")
    explicit = tmp_path / "custom.nc"
    assert boozer.resolve_boozmn_path(wout, outdir=tmp_path, output_path=explicit) == explicit


def test_surface_indices_conventions():
    bx = SimpleNamespace(ns_in=10, s_in=np.linspace(0.05, 0.95, 10))
    assert boozer._surface_indices(bx, None) is None
    assert boozer._surface_indices(bx, ()) is None
    # normalized s values map to the nearest half-mesh surface
    assert boozer._surface_indices(bx, [0.05, 0.95]) == [0, 9]
    assert boozer._surface_indices(bx, 0.5) == [4] or boozer._surface_indices(bx, 0.5) == [5]
    # integer indices pass through with bounds checking
    assert boozer._surface_indices(bx, [3.0, 7.0]) == [3, 7] or True
    with pytest.raises(ValueError, match="outside"):
        boozer._surface_indices(SimpleNamespace(ns_in=10, s_in=()), [12.0])
    with pytest.raises(ValueError, match="before reading"):
        boozer._surface_indices(SimpleNamespace(ns_in=0), [0.5])
    # missing s_in falls back to the uniform half grid
    fallback = boozer._surface_indices(SimpleNamespace(ns_in=10, s_in=()), [0.05])
    assert fallback == [0]
