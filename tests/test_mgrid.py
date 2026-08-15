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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmex.core.errors import MgridNotFoundError  # noqa: E402
from vmex.core import extender as ext  # noqa: E402
from vmex.core.extender import MagneticField, VmecExtender, VmecInteriorField  # noqa: E402
from vmex.core.mgrid import (  # noqa: E402
    MgridData,
    MgridField,
    read_mgrid,
    tabulate_cartesian_field,
    write_mgrid,
)
from vmex.core.optimize import Equilibrium  # noqa: E402

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


def test_high_spatial_derivatives_and_parameter_vjps_are_exact():
    parameters = jnp.array([1.2, -0.7])
    points = jnp.array([[0.4, -0.2, 0.3], [0.8, 0.1, -0.5]])

    def parameterized_field(p, xyz):
        x, y, z = xyz.T
        return jnp.stack((p[0] * x**3 + p[1] * y,
                          p[0] * x * y**2 + p[1] * z**2,
                          p[0] * z + p[1] * x**2 * y), axis=-1)

    field = MagneticField(
        lambda xyz: parameterized_field(parameters, xyz),
        parameters=parameters, parameterized_B_fn=parameterized_field,
        dof_names=("p0", "p1")).set_points(points)
    with pytest.raises(ValueError, match="provided together"):
        MagneticField(
            lambda xyz: parameterized_field(parameters, xyz), parameters=parameters,
            parameter_data_fn=lambda p: p)
    with pytest.raises(ValueError, match="not both"):
        MagneticField(
            lambda xyz: parameterized_field(parameters, xyz), parameters=parameters,
            parameterized_B_fn=parameterized_field, parameter_data_fn=lambda p: p,
            B_from_data=parameterized_field)
    with pytest.raises(ValueError, match="dof_names must match"):
        MagneticField(
            lambda xyz: parameterized_field(parameters, xyz),
            parameters=parameters, parameterized_B_fn=parameterized_field,
            dof_names=("p0",))
    factored = MagneticField(
        lambda xyz: parameterized_field(parameters, xyz), parameters=parameters,
        parameter_data_fn=lambda p: {"coefficients": p},
        B_from_data=lambda data, xyz: parameterized_field(data["coefficients"], xyz),
        dof_names=("p0", "p1")).set_points(points)
    def point_field(point):
        return parameterized_field(parameters, point[None])[0]
    expected_second = jax.vmap(jax.jacfwd(jax.jacfwd(point_field)))(points)
    expected_third = jax.vmap(
        jax.jacfwd(jax.jacfwd(jax.jacfwd(point_field))))(points)
    np.testing.assert_allclose(field.gradgradB(), expected_second)
    np.testing.assert_allclose(field.gradgradgradB(), expected_third)

    quantities = [field.B(), field.gradB(), field.gradgradB(), field.gradgradgradB()]
    vjps = [field.B_vjp, field.gradB_vjp, field.gradgradB_vjp,
            field.gradgradgradB_vjp]
    for order, (value, method) in enumerate(zip(quantities, vjps)):
        cotangent = jnp.arange(value.size, dtype=value.dtype).reshape(value.shape) / value.size

        def quantity(p):
            def one_point(point):
                return parameterized_field(p, point[None])[0]
            function = one_point
            for _ in range(order):
                function = jax.jacfwd(function)
            return jax.vmap(function)(points)

        expected = jax.vjp(quantity, parameters)[1](cotangent)[0]
        np.testing.assert_allclose(method(cotangent), expected, rtol=2e-13, atol=2e-13)
        factored_method = (factored.B_vjp, factored.gradB_vjp,
                           factored.gradgradB_vjp, factored.gradgradgradB_vjp)[order]
        np.testing.assert_allclose(
            factored_method(cotangent), expected, rtol=2e-13, atol=2e-13)
    with pytest.raises(ValueError, match="cotangent has shape"):
        factored.B_vjp(jnp.ones((1, 3)))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        field.B_contravariant(jnp.ones(2))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        field.to_xyz(jnp.ones(2))
    new_points = points[:1] + 0.15
    factored.set_points_xyz(new_points)
    new_value = factored.B()
    expected_new = jax.vjp(
        lambda p: parameterized_field(p, new_points), parameters
    )[1](jnp.ones_like(new_value))[0]
    np.testing.assert_allclose(
        factored.B_vjp(jnp.ones_like(new_value)), expected_new,
        rtol=2e-13, atol=2e-13)
    assert field.dof_names == ("p0", "p1")

    equilibrium = Equilibrium(
        inp=None, state=None, runtime=None, result=None,
        field_factory=lambda: field)
    final_equilibrium = equilibrium.set_points_xyz(points)
    values = [final_equilibrium.B(), final_equilibrium.gradB(),
              final_equilibrium.gradgradB(), final_equilibrium.gradgradgradB()]
    methods = [final_equilibrium.B_vjp, final_equilibrium.gradB_vjp,
               final_equilibrium.gradgradB_vjp,
               final_equilibrium.gradgradgradB_vjp]
    np.testing.assert_allclose(final_equilibrium.absB(), jnp.linalg.norm(values[0], axis=1))
    for value, method, expected_method in zip(values, methods, vjps):
        cotangent = jnp.ones_like(value)
        np.testing.assert_allclose(method(cotangent), expected_method(cotangent))


def test_exterior_vjp_combines_plasma_and_external_dofs(monkeypatch):
    @dataclass(frozen=True)
    class SurfaceData:
        gamma: object
        B_total: object
        normal: object
        area_vector: object

    def surface_data(parameters):
        value = parameters[0] * jnp.ones((3, 2, 2))
        return SurfaceData(value, value, value, value)

    monkeypatch.setattr(
        VmecExtender, "from_surface_data", classmethod(
            lambda cls, data, external_field=None, **kwargs: cls(external_field)))
    field = VmecExtender.from_parameterized_surface_data(
        surface_data, jnp.array([2.0]), external_parameters=jnp.array([3.0]),
        external_field_from_parameters=lambda current:
            lambda xyz: current[0] * xyz,
        dof_names=("boundary",), external_dof_names=("coil current",))
    field.set_points_xyz([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(field.B(), [[3.0, 6.0, 9.0]])
    np.testing.assert_allclose(field.B_vjp(jnp.ones((1, 3))), [0.0, 6.0])
    assert field.dof_names == ("boundary", "coil current")

    with pytest.raises(ValueError, match="provided together"):
        VmecExtender.from_parameterized_surface_data(
            surface_data, jnp.array([2.0]), external_parameters=jnp.array([3.0]))
    with pytest.raises(ValueError, match="external_dof_names require"):
        VmecExtender.from_parameterized_surface_data(
            surface_data, jnp.array([2.0]), external_dof_names=("current",))
    with pytest.raises(ValueError, match="do not provide both"):
        VmecExtender.from_parameterized_surface_data(
            surface_data, jnp.array([2.0]), external_field=lambda xyz: xyz,
            external_parameters=jnp.array([3.0]),
            external_field_from_parameters=lambda current: lambda xyz: current[0] * xyz)
    with pytest.raises(ValueError, match="must match external_parameters"):
        VmecExtender.from_parameterized_surface_data(
            surface_data, jnp.array([2.0]), external_parameters=jnp.array([3.0]),
            external_field_from_parameters=lambda current: lambda xyz: current[0] * xyz,
            external_dof_names=("one", "two"))
    automatically_named = VmecExtender.from_parameterized_surface_data(
        surface_data, jnp.array([2.0]), external_parameters=jnp.array([3.0]),
        external_field_from_parameters=lambda current: lambda xyz: current[0] * xyz,
        dof_names=("boundary",))
    assert automatically_named.dof_names == ("boundary", "external[0]")


def test_interior_field_inverts_flux_coordinates_and_recovers_B():
    ns, major_radius, minor_radius = 7, 1.0, 0.3
    s_mesh = jnp.linspace(0.0, 1.0, ns)
    spectra = {
        "nfp": 1, "ns": ns,
        "xm": jnp.array([0.0, 1.0]), "xn": jnp.array([0.0, 0.0]),
        "xmn": jnp.array([0.0]), "xnn": jnp.array([0.0]),
        "rmnc": jnp.stack((jnp.full(ns, major_radius), minor_radius * s_mesh), axis=1),
        "zmns": jnp.stack((jnp.zeros(ns), minor_radius * s_mesh), axis=1),
        "rmns": None, "zmnc": None,
        "bsupu": jnp.zeros((ns, 1)), "bsupv": jnp.ones((ns, 1)),
        "bsupu_s": None, "bsupv_s": None, "lasym": False, "signgs": -1,
    }
    coordinates = jnp.array([[0.4, 0.7, 0.3], [0.8, 4.1, 1.2]])
    s, theta, phi = coordinates.T
    radius = major_radius + minor_radius * s * jnp.cos(theta)
    points = jnp.stack((radius * jnp.cos(phi), radius * jnp.sin(phi),
                        minor_radius * s * jnp.sin(theta)), axis=1)
    field = VmecInteriorField(spectra).set_points(points)

    got_coordinates = field.flux_coordinates()
    np.testing.assert_allclose(got_coordinates[:, 0], s, rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        jnp.mod(got_coordinates[:, 1] - theta + jnp.pi, 2 * jnp.pi) - jnp.pi,
        0.0, rtol=0, atol=2e-12)
    expected_B = jnp.stack((-points[:, 1], points[:, 0], jnp.zeros(2)), axis=1)
    expected_grad = jnp.broadcast_to(
        jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        (2, 3, 3))
    np.testing.assert_allclose(field.B(), expected_B, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(field.gradB(), expected_grad, rtol=0, atol=2e-10)
    np.testing.assert_allclose(field.gradgradB(), 0.0, rtol=0, atol=2e-8)

    flux_field = VmecInteriorField(spectra).set_points_flux(coordinates)
    np.testing.assert_allclose(flux_field.get_points_cart(), points, rtol=0, atol=2e-14)
    np.testing.assert_allclose(flux_field.get_points_flux(), coordinates, rtol=0, atol=0)
    np.testing.assert_allclose(flux_field.B(), expected_B, rtol=2e-12, atol=2e-12)
    tracing_field = flux_field.field_in_flux_coordinates()
    np.testing.assert_allclose(
        jax.vmap(tracing_field.B_contravariant)(coordinates),
        jnp.tile(jnp.array([0.0, 0.0, 1.0]), (2, 1)))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        tracing_field.B_contravariant(jnp.ones(2))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        tracing_field.to_xyz(jnp.ones(2))

    equilibrium = Equilibrium(
        inp=None, state=None, runtime=None, result=None,
        field_factory=lambda: flux_field)
    assert equilibrium.set_points(points) is equilibrium
    cylindrical = jnp.stack((jnp.hypot(points[:, 0], points[:, 1]), phi, points[:, 2]), axis=1)
    assert equilibrium.set_points_cyl(cylindrical) is equilibrium
    assert equilibrium.set_points_flux(coordinates) is equilibrium
    assert equilibrium.field_in_flux_coordinates().spectra is spectra
    np.testing.assert_allclose(
        jax.vmap(tracing_field.to_xyz)(coordinates), points, rtol=0, atol=2e-14)
    mapped_B = jax.vmap(
        lambda point: jax.jacfwd(tracing_field.to_xyz)(point)
        @ tracing_field.B_contravariant(point))(coordinates)
    np.testing.assert_allclose(mapped_B, expected_B, rtol=2e-12, atol=2e-12)


def test_magnetic_field_cylindrical_points_round_trip():
    field = MagneticField(_linear_vacuum_field)
    points = jnp.array([[1.8, 0.25, -0.1]])

    assert field.set_points_cyl(points) is field
    np.testing.assert_allclose(field.get_points_cyl(), points)
    np.testing.assert_allclose(
        field.B_cyl(), field.B_cyl(points), rtol=1.0e-14, atol=1.0e-14
    )
    xyz = jnp.array([[1.0, 2.0, 3.0]])
    assert field.set_points_xyz(xyz) is field
    np.testing.assert_allclose(field.get_points_cart(), xyz)
    np.testing.assert_allclose(field.B_contravariant(xyz[0]), field.B(xyz)[0])
    np.testing.assert_allclose(field.to_xyz(xyz[0]), xyz[0])


def test_field_api_validation_and_constructor_routing(monkeypatch, tmp_path):
    points = jnp.array([[2.0, 0.0, 0.1]])
    with pytest.raises(ValueError, match="shape"):
        MagneticField(_linear_vacuum_field).set_points([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="provided together"):
        MagneticField(_linear_vacuum_field, parameters=jnp.ones(1))
    with pytest.raises(RuntimeError, match="set_points"):
        MagneticField(_linear_vacuum_field).B()
    with pytest.raises(ValueError, match="field returned shape"):
        MagneticField(lambda xyz: xyz[:, :2]).B(points)
    with pytest.raises(RuntimeError, match="optimizable parameters"):
        MagneticField(_linear_vacuum_field).set_points(points).B_vjp(jnp.ones((1, 3)))

    expected = (points.shape + (3,), points.shape + (3, 3), points.shape + (3, 3, 3))
    supplied = MagneticField(
        _linear_vacuum_field,
        gradB_fn=lambda xyz: jnp.zeros(xyz.shape + (3,)),
        gradgradB_fn=lambda xyz: jnp.zeros(xyz.shape + (3, 3)),
        gradgradgradB_fn=lambda xyz: jnp.zeros(xyz.shape + (3, 3, 3)),
    )
    assert (supplied.gradB(points).shape, supplied.gradgradB(points).shape,
            supplied.gradgradgradB(points).shape) == expected
    for name, function in (
        ("gradient", lambda xyz: jnp.zeros(xyz.shape)),
        ("second", lambda xyz: jnp.zeros(xyz.shape + (3,))),
        ("third", lambda xyz: jnp.zeros(xyz.shape + (3, 3))),
    ):
        kwargs = {"gradB_fn": function} if name == "gradient" else {
            "gradgradB_fn" if name == "second" else "gradgradgradB_fn": function}
        method = {"gradient": "gradB", "second": "gradgradB",
                  "third": "gradgradgradB"}[name]
        with pytest.raises(ValueError, match=name):
            getattr(MagneticField(_linear_vacuum_field, **kwargs), method)(points)

    parameterized = MagneticField(
        _linear_vacuum_field, parameters=jnp.ones(1),
        parameterized_B_fn=lambda p, xyz: p[0] * _linear_vacuum_field(xyz),
    ).set_points(points)
    with pytest.raises(ValueError, match="cotangent"):
        parameterized.B_vjp(jnp.ones((2, 3)))

    field = MagneticField(_linear_vacuum_field).set_points(points)
    np.testing.assert_allclose(field.get_points_cart(), points)
    np.testing.assert_allclose(field.get_points_cyl(), [[2.0, 0.0, 0.1]])

    class CylindricalField:
        @staticmethod
        def b_cyl(r, phi, z):
            return r, 2.0 * phi, z

    assert VmecExtender(CylindricalField()).B(points).shape == points.shape
    with pytest.raises(ValueError, match="external field returned shape"):
        VmecExtender(lambda xyz: xyz[:, :2]).B(points)
    with pytest.raises(TypeError, match="external_field"):
        VmecExtender(object()).B(points)
    with pytest.raises(ValueError, match="at least one"):
        VmecExtender(None)

    assert ext._has_plasma_sources(SimpleNamespace(
        betatotal=0.0, wp=0.0, ctor=0.0, presf=np.array([0.0, 1.0])))
    data = SimpleNamespace(nextcur=2, mgrid_mode="R", raw_coil_cur=[2.0, 4.0])
    captured = {}
    monkeypatch.setattr(ext, "read_mgrid", lambda path: data)
    def from_mgrid_data(cls, source, extcur):
        captured["extcur"] = extcur
        return _linear_vacuum_field
    monkeypatch.setattr(ext.MgridField, "from_mgrid_data", classmethod(from_mgrid_data))
    wout = SimpleNamespace(betatotal=0.0, wp=0.0, ctor=0.0,
                           mgrid_file="mgrid.nc", extcur=[10.0])
    mgrid_field = VmecExtender.from_wout(wout, base_dir=tmp_path)
    assert mgrid_field.B(points).shape == points.shape
    np.testing.assert_allclose(captured["extcur"], [5.0, 0.0])
    with pytest.raises(ValueError, match="plasma must"):
        VmecExtender.from_wout(wout, plasma="bad")
    with pytest.raises(ValueError, match="vacuum extension"):
        VmecExtender.from_wout(SimpleNamespace(
            betatotal=0.0, wp=0.0, ctor=0.0, mgrid_file=""))

    from vmex.core import freeboundary_diff as fbd
    sentinel = object()
    spectra = {"marker": 1}
    monkeypatch.setattr(fbd, "_state_field_spectra", lambda *a, **k: spectra)
    assert VmecInteriorField.from_state(object(), object()).spectra is spectra
    interior = VmecInteriorField.from_parameterized_state(
        object(), lambda p: (object(), object()), jnp.ones(1), dof_names=("p",))
    assert interior.spectra is spectra and interior.dof_names == ("p",)
    monkeypatch.setattr(fbd, "surface_field_data_from_wout", lambda *a, **k: "surface")
    monkeypatch.setattr(fbd, "surface_field_data_from_state", lambda *a, **k: "state")
    monkeypatch.setattr(VmecExtender, "from_surface_data", classmethod(
        lambda cls, surface, **kwargs: sentinel))
    finite = SimpleNamespace(betatotal=0.01, wp=0.0, ctor=0.0, mgrid_file="")
    assert VmecExtender.from_wout(finite, external_field=_linear_vacuum_field) is sentinel
    assert VmecExtender.from_state(object(), object()) is sentinel

    from vmex.core import wout as wout_module
    monkeypatch.setattr(wout_module, "read_wout", lambda path: wout)
    assert VmecExtender.from_file(
        tmp_path / "wout.nc", external_field=_linear_vacuum_field).B(points).shape == points.shape

    monkeypatch.setattr(VmecExtender, "from_state", classmethod(
        lambda cls, *args, **kwargs: ("state", kwargs.get("external_field"))))
    monkeypatch.setattr(VmecExtender, "from_wout", classmethod(
        lambda cls, *args, **kwargs: ("wout", kwargs.get("external_field"))))
    from vmex.core import freeboundary
    monkeypatch.setattr(freeboundary, "_external_field_from_input", lambda inp: "coils")
    equilibrium = SimpleNamespace(inp=SimpleNamespace(lfreeb=True), state=object(), wout=finite)
    assert VmecExtender.from_equilibrium(equilibrium) == ("state", "coils")
    with pytest.raises(ValueError, match="plasma must"):
        VmecExtender.from_equilibrium(equilibrium, plasma="bad")
    equilibrium.wout = wout
    assert VmecExtender.from_equilibrium(equilibrium, plasma="vacuum") == ("wout", "coils")

    fallback = Equilibrium(inp="input", state="state", runtime="runtime", result=None)
    interior_field = MagneticField(_linear_vacuum_field)
    monkeypatch.setattr(VmecInteriorField, "from_state", classmethod(
        lambda cls, inp, state, **kwargs: interior_field))
    monkeypatch.setattr(VmecExtender, "from_equilibrium", classmethod(
        lambda cls, equilibrium, **kwargs: (equilibrium, kwargs)))
    assert fallback.field is interior_field
    assert fallback.exterior_field(plasma="vacuum") == (fallback, {"plasma": "vacuum"})


def test_volume_average_beta_is_plasma_to_magnetic_energy_ratio(monkeypatch):
    from vmex.core import statephysics

    energies = SimpleNamespace(wp=1.0, wb=4.0)
    monkeypatch.setattr(
        statephysics, "_field_chain", lambda state, runtime: (None,) * 4 + (energies,))
    assert float(statephysics.volume_average_beta(None, None)) == pytest.approx(0.25)
    energies.wb = 0.0
    assert float(statephysics.volume_average_beta(None, None)) == 0.0


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
