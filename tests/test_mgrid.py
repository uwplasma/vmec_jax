"""Tests for ``vmex.core.mgrid`` (netCDF IO + interpolated field).

Covers (plan.md §8):

- netCDF round-trip (read -> write -> read) equality on the bundled
  ``mgrid_cth_like_lasym_small.nc`` fixture,
- extcur-scaling linearity of the interpolated field,
- jit equivalence and grad of ``|B|^2`` w.r.t. extcur,
- cross-read consistency with ESSOS's unmerged ``feature/mgrid-from-coils``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmex.core.errors import MgridNotFoundError  # noqa: E402
from vmex.core.extender import MagneticField, VmecExtender  # noqa: E402
from vmex.core.mgrid import (  # noqa: E402
    MgridData,
    MgridField,
    read_mgrid,
    tabulate_cartesian_field,
    write_mgrid,
)

REPO = Path(__file__).resolve().parents[1]
MGRID_PATH = REPO / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"

assert MGRID_PATH.is_file(), f"missing fixture {MGRID_PATH}"


@pytest.fixture(scope="module")
def data() -> MgridData:
    return read_mgrid(MGRID_PATH)


def _random_points(data: MgridData, n: int = 200, seed: int = 1234):
    """Random strictly-in-domain cylindrical points, one full torus in phi."""

    rng = np.random.default_rng(seed)
    eps_r = 1e-6 * (data.rmax - data.rmin)
    eps_z = 1e-6 * (data.zmax - data.zmin)
    r = rng.uniform(data.rmin + eps_r, data.rmax - eps_r, size=n)
    z = rng.uniform(data.zmin + eps_z, data.zmax - eps_z, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return r, phi, z


def _linear_vacuum_field(points):
    """Curl-free, divergence-free field B = (2x, -2y, 1)."""
    points = jnp.asarray(points)
    return jnp.stack(
        (2.0 * points[:, 0], -2.0 * points[:, 1], jnp.ones(points.shape[0])),
        axis=-1,
    )


def test_magnetic_field_interface_and_vacuum_extender_are_exact():
    points = jnp.array([[1.8, 0.2, -0.1], [2.0, -0.3, 0.4]])
    expected_grad = jnp.broadcast_to(
        jnp.diag(jnp.array([2.0, -2.0, 0.0])), (2, 3, 3)
    )
    field = MagneticField(_linear_vacuum_field).set_points(points)

    np.testing.assert_allclose(field.B(), _linear_vacuum_field(points))
    np.testing.assert_allclose(field.gradB(), expected_grad)
    np.testing.assert_allclose(field.dB_by_dX(), expected_grad)
    np.testing.assert_allclose(field.AbsB(), field.absB()[:, None])
    expected_grad_absB = jnp.einsum(
        "...i,...ij->...j", field.B(), expected_grad
    ) / field.absB()[:, None]
    np.testing.assert_allclose(field.GradAbsB(), expected_grad_absB)

    vacuum_wout = type(
        "VacuumWout",
        (),
        {"betatotal": 0.0, "wp": 0.0, "ctor": 0.0, "mgrid_file": ""},
    )()
    extender = VmecExtender.from_wout(
        vacuum_wout, external_field=_linear_vacuum_field
    ).set_points(points)
    assert not extender.uses_virtual_casing
    np.testing.assert_allclose(extender.B(), field.B())
    np.testing.assert_allclose(extender.gradB(), expected_grad)

    general = MagneticField(
        lambda p: jnp.stack(
            (p[:, 0] + 2 * p[:, 1], 3 * p[:, 0] - p[:, 1], p[:, 2]),
            axis=-1,
        )
    )
    component_first = jnp.array([[1.0, 2.0, 0.0], [3.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    expected = jnp.broadcast_to(component_first, (len(points), 3, 3))
    np.testing.assert_allclose(general.gradB(points), expected)
    np.testing.assert_allclose(general.dB_by_dX(points), jnp.swapaxes(expected, -1, -2))


def test_magnetic_field_cylindrical_points_round_trip():
    field = MagneticField(_linear_vacuum_field)
    points = jnp.array([[1.8, 0.25, -0.1]])

    assert field.set_points_cyl(points) is field
    np.testing.assert_allclose(field.get_points_cyl(), points)
    np.testing.assert_allclose(
        field.B_cyl(), field.B_cyl(points), rtol=1.0e-14, atol=1.0e-14
    )


# ---------------------------------------------------------------------------
# Read + round-trip
# ---------------------------------------------------------------------------


def test_round_trip_read_write_read(data: MgridData, tmp_path: Path) -> None:
    out = tmp_path / "mgrid_roundtrip.nc"
    write_mgrid(out, data)
    back = read_mgrid(out)

    assert (back.ir, back.jz, back.kp) == (data.ir, data.jz, data.kp)
    assert (back.nfp, back.nextcur) == (data.nfp, data.nextcur)
    assert (back.rmin, back.rmax, back.zmin, back.zmax) == (
        data.rmin,
        data.rmax,
        data.zmin,
        data.zmax,
    )
    assert back.mgrid_mode == data.mgrid_mode
    assert back.coil_groups == data.coil_groups
    assert back.raw_coil_cur == data.raw_coil_cur
    np.testing.assert_array_equal(back.br, data.br)
    np.testing.assert_array_equal(back.bp, data.bp)
    np.testing.assert_array_equal(back.bz, data.bz)


def test_write_mgrid_has_no_numpy_deprecation(data: MgridData, tmp_path: Path) -> None:
    """netCDF4's internal NumPy-2.5 reshape warning stays locally isolated."""
    out = tmp_path / "mgrid_warning_free.nc"
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        write_mgrid(out, data)
    assert out.is_file()


def test_missing_file_raises_mgrid_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_mgrid.nc"
    with pytest.raises(MgridNotFoundError):
        read_mgrid(missing)
    with pytest.raises(MgridNotFoundError):
        MgridField.from_file(missing)


# ---------------------------------------------------------------------------
# Interpolated field properties
# ---------------------------------------------------------------------------


def test_extcur_scaling_is_linear(data: MgridData) -> None:
    r, phi, z = _random_points(data, n=50, seed=7)
    base = 1.0 + np.arange(data.nextcur, dtype=float)
    f1 = MgridField.from_mgrid_data(data, extcur=base)
    f3 = MgridField.from_mgrid_data(data, extcur=3.0 * base)
    for a, b in zip(f1.b_cyl(r, phi, z), f3.b_cyl(r, phi, z)):
        np.testing.assert_allclose(3.0 * np.asarray(a), np.asarray(b), rtol=1e-13, atol=0.0)


def test_jit_equivalence(data: MgridData) -> None:
    r, phi, z = _random_points(data, n=100, seed=42)
    field = MgridField.from_mgrid_data(data)  # extcur defaults to raw currents

    @jax.jit
    def eval_field(f: MgridField, rr, pp, zz):
        return f.b_cyl(rr, pp, zz)

    eager = field.b_cyl(r, phi, z)
    jitted = eval_field(field, r, phi, z)
    for a, b in zip(eager, jitted):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-14, atol=0.0)


def test_grad_wrt_extcur_finite_nonzero(data: MgridData) -> None:
    r, phi, z = _random_points(data, n=64, seed=3)
    field = MgridField.from_mgrid_data(data)

    def bsq_sum(extcur):
        f = MgridField.from_mgrid_data(data, extcur=extcur)
        br, bp, bz = f.b_cyl(r, phi, z)
        return jnp.sum(br**2 + bp**2 + bz**2)

    g = jax.grad(bsq_sum)(jnp.asarray(field.extcur))
    g_np = np.asarray(g)
    assert g_np.shape == (data.nextcur,)
    assert np.all(np.isfinite(g_np))
    assert np.max(np.abs(g_np)) > 0.0


def test_tabulate_cartesian_callable_and_cylindrical_conversion() -> None:
    def field(points):
        p = np.asarray(points)
        return np.stack((2.0 + 0.1 * p[:, 0], -3.0 + 0.2 * p[:, 1], 4.0 + 0.3 * p[:, 2]), axis=-1)

    data = tabulate_cartesian_field(
        field,
        rmin=0.5,
        rmax=1.5,
        zmin=-0.4,
        zmax=0.4,
        ir=5,
        jz=4,
        kp=12,
        nfp=2,
    )
    sampled = MgridField.from_mgrid_data(data, extcur=[1.7])
    # Test exact grid points: no interpolation error obscures the Cartesian
    # -> cylindrical convention.
    phi = np.arange(data.kp) * 2.0 * np.pi / (data.nfp * data.kp)
    r = np.full_like(phi, 1.0)
    z = np.zeros_like(phi)
    xyz = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1)
    direct = 1.7 * field(xyz)
    br, bp, bz = (np.asarray(v) for v in sampled.b_cyl(r, phi, z))
    np.testing.assert_allclose(br, direct[:, 0] * np.cos(phi) + direct[:, 1] * np.sin(phi))
    np.testing.assert_allclose(bp, -direct[:, 0] * np.sin(phi) + direct[:, 1] * np.cos(phi))
    np.testing.assert_allclose(bz, direct[:, 2])


def test_tabulate_simsopt_set_points_protocol() -> None:
    class FakeSimsoptField:
        def set_points(self, points):
            self.points = np.asarray(points)

        def B(self):
            return np.column_stack(
                (self.points[:, 0] * 0 + 1.0, self.points[:, 1] * 0 + 2.0, self.points[:, 2] * 0 + 3.0)
            )

    data = tabulate_cartesian_field(
        FakeSimsoptField(),
        rmin=0.4,
        rmax=1.0,
        zmin=-0.2,
        zmax=0.2,
        ir=3,
        jz=3,
        kp=5,
        nfp=1,
    )
    assert data.br.shape == (1, 5, 3, 3)
    assert np.all(np.isfinite(data.br))
    assert np.all(np.isfinite(data.bp))
    np.testing.assert_allclose(data.bz, 3.0)


def test_tabulate_actual_essos_biot_savart() -> None:
    pytest.importorskip("essos")
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart

    dofs = np.zeros((2, 3, 3))
    for i, phi0 in enumerate((0.2, 0.8)):
        dofs[i, 0, 0], dofs[i, 0, 2] = 0.8 * np.cos(phi0), 0.25 * np.cos(phi0)
        dofs[i, 1, 0], dofs[i, 1, 2] = 0.8 * np.sin(phi0), 0.25 * np.sin(phi0)
        dofs[i, 2, 1] = 0.25
    bs = BiotSavart(Coils(Curves(jnp.asarray(dofs), 32, 1, False), jnp.asarray([1.0e5, -0.7e5])))
    data = tabulate_cartesian_field(
        bs,
        rmin=0.25,
        rmax=0.55,
        zmin=-0.15,
        zmax=0.15,
        ir=3,
        jz=3,
        kp=4,
        nfp=1,
    )
    assert np.all(np.isfinite(data.br))
    # At table nodes, cylindrical components must reconstruct ESSOS' direct
    # Cartesian field to roundoff.
    k, j, i = 1, 1, 1
    phi = k * 2.0 * np.pi / data.kp
    r = np.linspace(data.rmin, data.rmax, data.ir)[i]
    z = np.linspace(data.zmin, data.zmax, data.jz)[j]
    direct = np.asarray(bs.B(jnp.asarray([r * np.cos(phi), r * np.sin(phi), z])))
    reconstructed = np.asarray(
        [
            data.br[0, k, j, i] * np.cos(phi) - data.bp[0, k, j, i] * np.sin(phi),
            data.br[0, k, j, i] * np.sin(phi) + data.bp[0, k, j, i] * np.cos(phi),
            data.bz[0, k, j, i],
        ]
    )
    np.testing.assert_allclose(reconstructed, direct, rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# ESSOS cross-read
# ---------------------------------------------------------------------------


def test_essos_reads_same_grid_and_fields(data: MgridData) -> None:
    essos_mgrid = pytest.importorskip(
        "essos.mgrid", reason="requires ESSOS feature/mgrid-from-coils"
    )
    eg = essos_mgrid.MGrid.from_file(MGRID_PATH)

    # ESSOS naming: nr/nz/nphi == ir/jz/kp; same extents and nfp.
    assert (eg.nr, eg.nz, eg.nphi, eg.nfp) == (data.ir, data.jz, data.kp, data.nfp)
    assert (eg.rmin, eg.rmax, eg.zmin, eg.zmax) == (
        data.rmin,
        data.rmax,
        data.zmin,
        data.zmax,
    )
    assert eg.n_ext_cur == data.nextcur
    assert eg.mode == data.mgrid_mode
    np.testing.assert_array_equal(np.asarray(eg.raw_coil_current), np.asarray(data.raw_coil_cur))
    # ESSOS strips via _unpack (whitespace only) — same convention as ours.
    assert tuple(eg.coil_names) == data.coil_groups

    # Per-group field tables: ESSOS stores a list of (nphi, nz, nr) arrays,
    # ours is stacked (nextcur, kp, jz, ir) — identical per-group content.
    for i in range(data.nextcur):
        np.testing.assert_array_equal(np.asarray(eg.br_arr[i]), data.br[i])
        np.testing.assert_array_equal(np.asarray(eg.bp_arr[i]), data.bp[i])
        np.testing.assert_array_equal(np.asarray(eg.bz_arr[i]), data.bz[i])


def test_essos_reads_our_written_file(data: MgridData, tmp_path: Path) -> None:
    essos_mgrid = pytest.importorskip(
        "essos.mgrid", reason="requires ESSOS feature/mgrid-from-coils"
    )
    out = tmp_path / "mgrid_for_essos.nc"
    write_mgrid(out, data)
    eg = essos_mgrid.MGrid.from_file(out)
    assert (eg.nr, eg.nz, eg.nphi, eg.nfp) == (data.ir, data.jz, data.kp, data.nfp)
    assert eg.n_ext_cur == data.nextcur
    for i in range(data.nextcur):
        np.testing.assert_array_equal(np.asarray(eg.br_arr[i]), data.br[i])
        np.testing.assert_array_equal(np.asarray(eg.bp_arr[i]), data.bp[i])
        np.testing.assert_array_equal(np.asarray(eg.bz_arr[i]), data.bz[i])
