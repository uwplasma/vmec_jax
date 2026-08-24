"""The ESSOS field handoff and alpha tracing (``vmex.core.tracing``).

Small and honest: 8 particles over ``tmax = 1e-5`` s on the solovev quick
case.  Gates: ``essos_vmec_field`` builds the same field from a wout path
and from an in-memory equilibrium, the trace runs on the released ESSOS
surface, the counts are mutually consistent, the loss fraction is a
fraction, the in-memory equilibrium route (temporary-wout hop) reproduces
the file route, and ``vmex --trace`` writes the four tracing figures end to
end.  Skips cleanly without ESSOS.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
jax = pytest.importorskip("jax")
pytest.importorskip("essos")

jax.config.update("jax_enable_x64", True)

from vmex.core import cli
from vmex.core.tracing import essos_vmec_field, trace_alphas
from vmex.core.wout import read_wout

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
SOLOVEV_DECK = DATA_DIR / "input.solovev"

TRACE_KWARGS = dict(
    tmax=1e-5, nparticles=8, s=0.25, seed=1, timestep=5e-7, times_to_trace=12,
)


@pytest.fixture(autouse=True)
def _enable_jit():
    """Tracing needs JIT (the repo conftest disables it for unit tests)."""
    jax.config.update("jax_disable_jit", False)
    yield


@pytest.fixture(scope="module")
def solovev_wout(tmp_path_factory) -> Path:
    """One quiet CLI solve of the solovev deck, shared by the tests below."""
    jax.config.update("jax_disable_jit", False)
    outdir = tmp_path_factory.mktemp("trace_wout")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = cli.main([str(SOLOVEV_DECK), "--outdir", str(outdir), "--quiet"])
    assert rc == 0, buffer.getvalue()
    return outdir / "wout_solovev.nc"


@pytest.fixture(scope="module")
def traced(solovev_wout):
    jax.config.update("jax_disable_jit", False)
    return trace_alphas(solovev_wout, **TRACE_KWARGS)


def test_counts_are_consistent(traced):
    n = traced.nparticles
    assert n == 8
    lost = int(np.sum(traced.lost_times >= 0.0))
    confined = int(np.sum(traced.lost_times < 0.0))
    assert lost == traced.particles_lost
    assert traced.particles_lost + confined == n
    # Failed orbits (non-finite without a recorded loss) are a subset of the
    # not-lost population, so lost + (confined - failed) + failed = n.
    assert 0 <= traced.particles_failed <= confined
    assert 0 <= traced.particles_unresolved <= n


def test_loss_fraction_is_a_fraction(traced):
    assert 0.0 <= traced.loss_fraction <= 1.0
    assert traced.loss_fraction == pytest.approx(
        traced.particles_lost / traced.nparticles)
    assert traced.loss_fraction == pytest.approx(float(traced.loss_fractions[-1]))
    assert np.all(np.diff(traced.loss_fractions) >= 0.0)  # cumulative


def test_shapes_times_and_birth_energy(traced):
    assert traced.trajectories.shape == (8, 12, 4)
    assert traced.trajectories_xyz.shape == (8, 12, 3)
    assert traced.times.shape == (12,)
    assert traced.loss_fractions.shape == (12,)
    assert traced.lost_times.shape == (8,)
    assert traced.energies.shape == (8, 12)
    assert traced.times[0] == 0.0
    assert traced.times[-1] == pytest.approx(TRACE_KWARGS["tmax"])
    # E(t=0) is the fusion-alpha birth energy by construction of mu.
    np.testing.assert_allclose(
        traced.energies[:, 0], traced.particle_energy, rtol=1e-12)
    assert traced.particle_energy > 0.0
    assert traced.total_speed > 0.0
    assert traced.wall_time_s > 0.0


def test_essos_field_handoff_matches_the_file_route(solovev_wout):
    """The field seam itself: both sources build the same ESSOS field."""
    from_file = essos_vmec_field(solovev_wout)
    from_memory = essos_vmec_field(read_wout(solovev_wout))
    assert int(from_memory.nfp) == int(from_file.nfp)
    assert int(from_memory.ns) == int(from_file.ns)
    points = np.array([[0.3, 0.4, 0.1], [0.7, 2.2, 0.9], [0.9, 5.0, 3.0]])
    for point in points:
        # The temporary wout is already deleted here; ESSOS read its tables
        # eagerly, so the field stays usable.
        np.testing.assert_allclose(
            float(from_memory.AbsB(point)), float(from_file.AbsB(point)), rtol=0.0)
        np.testing.assert_allclose(
            np.asarray(from_memory.to_xyz(point)),
            np.asarray(from_file.to_xyz(point)), rtol=0.0)


def test_in_memory_equilibrium_matches_the_file_route(traced, solovev_wout):
    result = trace_alphas(read_wout(solovev_wout), **TRACE_KWARGS)
    assert result.particles_lost == traced.particles_lost
    assert result.loss_fraction == pytest.approx(traced.loss_fraction)
    np.testing.assert_allclose(result.trajectories, traced.trajectories)


def test_cli_trace_writes_summary_and_figures(solovev_wout, tmp_path):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = cli.main([
            str(solovev_wout), "--trace", "--outdir", str(tmp_path),
            "--trace-particles", "8", "--trace-tmax", "1e-5",
            "--trace-times", "12", "--trace-seed", "1",
        ])
    stdout = buffer.getvalue()
    assert rc == 0, stdout
    assert "Loss fraction:" in stdout
    assert "Axis terminations:" in stdout
    assert "Solver failures:" in stdout
    for suffix in (
        "trace_trajectories", "trace_vparallel",
        "trace_loss_fraction", "trace_energy_error",
    ):
        assert (tmp_path / f"solovev_{suffix}.png").exists(), suffix
