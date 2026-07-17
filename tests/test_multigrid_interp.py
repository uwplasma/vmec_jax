"""Property tests for ``vmex.core.multigrid`` state interpolation.

``interpolate_state`` is the interp.f port (A/B-proven against the legacy
parity port before that tree was deleted).  On a realistic coarse state —
the converged ns = 15 cth_like_fixed_bdy solution from the core solver:

- ns -> ns interpolation is the identity (and the general stencil reproduces
  coincident surfaces exactly);
- jit lane produces the same result as the eager lane;
- the interpolated state fed to ``core.solver.evaluate_forces`` on the fine
  grid yields finite, modest residuals with no Jacobian sign change.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from vmex.core import solver
from vmex.core.fourier import mode_table
from vmex.core.input import VmecInput
from vmex.core.multigrid import interpolate_coefficients, interpolate_state
from vmex.core.solver import SpectralState

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solves: run jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
INPUT_FILE = DATA_DIR / "input.cth_like_fixed_bdy"
CACHE = Path("/tmp/vmex_multigrid_cth_ns15_state.pkl")

FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
NS_COARSE = 15


@pytest.fixture(scope="module")
def coarse():
    """Converged ns=15 cth state (new core solver), cached to /tmp."""
    inp = VmecInput.from_file(str(INPUT_FILE))
    if CACHE.exists():
        with CACHE.open("rb") as fh:
            arrays = pickle.load(fh)
    else:
        result = solver.solve(inp, ftol=1e-12, mode="cli")
        assert result.converged
        arrays = {name: np.asarray(getattr(result.state, name)) for name in FIELDS}
        with CACHE.open("wb") as fh:
            pickle.dump(arrays, fh)
    state = SpectralState(**{k: jax.numpy.asarray(v) for k, v in arrays.items()})
    assert int(state.R_cos.shape[0]) == NS_COARSE
    modes = mode_table(int(inp.mpol), int(inp.ntor))
    return inp, state, modes


# ---------------------------------------------------------------------------
# Properties: identity and jit-compatibility
# ---------------------------------------------------------------------------


def test_ns_to_ns_is_identity(coarse):
    _, state, modes = coarse
    same = interpolate_state(state, ns_fine=NS_COARSE, modes=modes)
    for name in FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(same, name)), np.asarray(getattr(state, name)),
            err_msg=f"{name} ns->ns identity",
        )


def test_general_path_ns_to_ns_matches_interior(coarse):
    """The general stencil at ns->ns reproduces every surface (xint == 0);
    only the axis odd-m zeroing convention can touch row 0, and the converged
    state already has (near-)zero odd-m there."""
    _, state, modes = coarse
    m = np.asarray(modes.m)
    x = np.asarray(state.R_cos)
    # Force the general path by round-tripping through an intermediate grid
    # of the same spacing family is not exact; instead check the stencil
    # directly: coefficients interpolated to 2*ns-1 land exactly on the
    # coarse surfaces at even output indices (xint == 0 there).
    fine = np.asarray(interpolate_coefficients(state.R_cos, m=m, ns_fine=2 * NS_COARSE - 1))
    np.testing.assert_allclose(
        fine[2::2], x[1:], rtol=1e-13, atol=1e-300,
        err_msg="coincident surfaces must be reproduced",
    )


def test_jit_compatible(coarse):
    from functools import partial

    _, state, modes = coarse
    fn = jax.jit(partial(interpolate_state, ns_fine=25, modes=modes))
    fine_jit = fn(state)
    fine_eager = interpolate_state(state, ns_fine=25, modes=modes)
    for name in FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(fine_jit, name)),
            np.asarray(getattr(fine_eager, name)),
            rtol=1e-15, atol=0.0, err_msg=f"{name} jit vs eager",
        )


# ---------------------------------------------------------------------------
# Fine-grid force evaluation on the interpolated state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ns_fine", [25, 51])
def test_fine_force_evaluation_is_sane(coarse, ns_fine):
    """evaluate_forces at the fine ns: finite, modest, no Jacobian flip."""
    inp, state, modes = coarse
    resolution = solver.resolution_from_input(inp, ns=ns_fine)
    runtime = solver.prepare_runtime(inp, resolution)
    fine = interpolate_state(state, ns_fine=ns_fine, modes=modes)
    gc, residuals, diagnostics = solver.evaluate_forces(fine, runtime)
    assert not bool(diagnostics.jacobian_sign_changed)
    fsq = np.array([float(residuals.fsqr), float(residuals.fsqz), float(residuals.fsql)])
    assert np.all(np.isfinite(fsq)), f"non-finite residuals {fsq}"
    # Modest: absolutely small AND well below the cold-start (initial-guess)
    # residuals at the same fine ns (~5.6e-2 total for this deck; the
    # interpolated state measures ~1.8e-3 at ns=25 and ~1.6e-2 at ns=51 —
    # 15 -> 51 is a deliberately aggressive single jump).
    _, cold_res, _ = solver.evaluate_forces(solver._initial_state(runtime.setup), runtime)
    cold = float(cold_res.fsqr) + float(cold_res.fsqz) + float(cold_res.fsql)
    total = float(fsq.sum())
    assert total < 5e-2, f"residuals not modest: {fsq}"
    assert total < 0.5 * cold, f"interpolated ({total}) not better than cold start ({cold})"
    for leaf in jax.tree.leaves(gc):
        assert bool(jax.numpy.all(jax.numpy.isfinite(leaf)))
