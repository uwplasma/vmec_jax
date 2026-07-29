"""Free-boundary tests: NESTOR operator properties + end-to-end golden run.

``vmex.core.vacuum`` is a cleaned port of the parity-proven JAX NESTOR
operator (A/B-proven to ~5e-12 before the legacy tree was deleted).  The
operator lane checks the scalpot.f skip branch and the first-call vacuum
diagnostics against the golden VMEC2000 stdout.

End-to-end lane: the golden ``cth_like_free_bdy_lasym_small`` run is only
partially converged (NITER=1000 exhausted, fsq ~ 1e-1, chaotic past vacuum
turn-on), so the golden comparison is structural + coarse: activation
iteration matches the golden stdout (53) within a few iterations, the
first-call diagnostics match the golden print block, final ``fsqr`` is
within 10x, and edge ``rmnc/zmns`` agree to a few percent of the dominant
coefficient (per-coefficient rtol is meaningless between two chaotic
unconverged trajectories; measured R ~1%, Z 0.01-0.10 by op ordering).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmex.core import freeboundary as FB  # noqa: E402
from vmex.core import vacuum as V  # noqa: E402
from vmex.core.errors import MgridNotFoundError, VmecJacobianError  # noqa: E402
from vmex.core.freeboundary_linear import linearize_nestor_coupling  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.mgrid import MgridField, read_mgrid  # noqa: E402
from vmex.core.solver import (  # noqa: E402
    _initial_state, prepare_runtime, resolution_from_input,
)

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # vacuum solves: run jitted

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy_lasym_small"
MGRID = REPO / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"
CASE = "cth_like_free_bdy_lasym_small"


# ---------------------------------------------------------------------------
# Shared fixtures: boundary + external-field inputs for the vacuum operator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ab_inputs():
    """Boundary geometry + bexni from the golden deck's initial state."""
    inp = VmecInput.from_file(DECK)
    res = resolution_from_input(inp)
    rt = prepare_runtime(inp, res)
    state = _initial_state(rt.setup)
    basis = V.vacuum_basis(
        mf=int(inp.mpol) + 1, nf=int(inp.ntor),
        ntheta3=res.ntheta3, nzeta=res.nzeta, nfp=res.nfp,
        lasym=res.lasym, wint=np.asarray(rt.trig.wint),
    )
    rmnc, zmns, rmns, zmnc = FB._edge_fourier(state, rt)
    boundary = FB.boundary_from_coefficients(
        rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc, modes=rt.modes, basis=basis
    )
    ctor, _rbtor, axis_r, axis_z, _bsq3, _pres = FB._vacuum_scalars(state, rt)
    field = MgridField.from_mgrid_data(
        read_mgrid(MGRID),
        extcur=np.asarray(inp.extcur, dtype=float)[: read_mgrid(MGRID).nextcur],
    )
    phi = (np.asarray(basis.zeta) * basis.onp).reshape(basis.ntheta3, basis.nzeta)
    br_c, bp_c, bz_c = field.b_cyl(np.asarray(boundary.R), phi, np.asarray(boundary.Z))
    br_a, bp_a, bz_a = FB.axis_current_field(
        R=np.asarray(boundary.R), Z=np.asarray(boundary.Z),
        axis_r=np.asarray(axis_r), axis_z=np.asarray(axis_z),
        nfp=res.nfp, plascur=float(ctor),
    )
    ext = FB.external_field_channels(
        boundary=boundary,
        br=np.asarray(br_c) + br_a, bp=np.asarray(bp_c) + bp_a,
        bz=np.asarray(bz_c) + bz_a,
        basis=basis, signgs=int(rt.setup.signgs),
    )
    return dict(inp=inp, res=res, rt=rt, basis=basis, boundary=boundary, ext=ext)


def test_nestor_skip_branch_matches_full_solve(ab_inputs):
    """scalpot.f skip branch: cached matrix + cached non-singular source with
    a freshly recomputed analytic source must reproduce the full solve when
    the geometry/source are unchanged."""
    basis: V.VacuumBasis = ab_inputs["basis"]
    boundary = ab_inputs["boundary"]
    rt = ab_inputs["rt"]
    bexni = np.asarray(ab_inputs["ext"]["bexni"], dtype=float)
    signgs = int(rt.setup.signgs)

    solver = V.make_vacuum_solver(basis, signgs=signgs)
    potvac, mode_matrix, bvec_nonsing, rhs, gsource, grpmn = solver.full(
        boundary, jnp.asarray(bexni)
    )
    assembled = solver.assemble(boundary, jnp.asarray(bexni))
    for actual, expected in zip(
        assembled, (mode_matrix, rhs, bvec_nonsing, gsource, grpmn), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    for name, arr in (("potvac", potvac), ("rhs", rhs), ("mode_matrix", mode_matrix),
                      ("grpmn", grpmn), ("gsource", gsource)):
        a = np.asarray(arr)
        assert np.all(np.isfinite(a)), f"{name} not finite"
    assert np.max(np.abs(np.asarray(potvac))) > 0.0

    potvac_skip, rhs_skip = solver.skip(
        boundary, jnp.asarray(bexni), bvec_nonsing, mode_matrix
    )
    np.testing.assert_allclose(np.asarray(potvac_skip), np.asarray(potvac), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(rhs_skip), np.asarray(rhs), rtol=1e-12, atol=1e-14)


def test_live_nestor_blocks_match_coupled_residual_jvp_and_vjp():
    """The bordered blocks come from a live, small LASYM NESTOR equation."""
    basis = V.vacuum_basis(
        mf=1, nf=0, ntheta3=6, nzeta=1, nfp=1, lasym=True,
        wint=np.full((6,), 1.0 / 6.0),
    )
    theta = jnp.asarray(basis.theta).reshape((6, 1))
    zero = jnp.zeros_like(theta)
    a = 0.3
    boundary0 = V.VacuumBoundary(
        R=2.0 + a * jnp.cos(theta) + 0.02 * jnp.sin(theta),
        Z=a * jnp.sin(theta) + 0.01 * jnp.cos(theta),
        Ru=-a * jnp.sin(theta) + 0.02 * jnp.cos(theta),
        Zu=a * jnp.cos(theta) - 0.01 * jnp.sin(theta),
        Rv=zero, Zv=zero,
        ruu=-a * jnp.cos(theta) - 0.02 * jnp.sin(theta),
        zuu=-a * jnp.sin(theta) - 0.01 * jnp.cos(theta),
        ruv=zero, zuv=zero, rvv=zero, zvv=zero,
    )
    solver = V.make_vacuum_solver(basis, signgs=-1)
    bexni = jnp.full((6, 1), 0.05)

    def boundary(x):
        scale, twist = 1.0 + 1e-4 * x[0], 1e-4 * x[1]
        fields = {}
        for r_name, z_name in (
            ("R", "Z"), ("Ru", "Zu"), ("Rv", "Zv"),
            ("ruu", "zuu"), ("ruv", "zuv"), ("rvv", "zvv"),
        ):
            r, z = getattr(boundary0, r_name), getattr(boundary0, z_name)
            fields[r_name] = scale * r + twist * z
            fields[z_name] = scale * z - twist * r
        return dataclasses.replace(boundary0, **fields)

    def vacuum_system(x):
        matrix, rhs, *_ = solver.assemble(boundary(x), bexni)
        return matrix, rhs

    def plasma_residual(x, q):
        b = boundary(x)
        guu = b.Ru * b.Ru + b.Zu * b.Zu
        guv = b.Ru * b.Rv + b.Zu * b.Zv
        gvv = b.Rv * b.Rv + b.Zv * b.Zv + b.R * b.R
        bsq, *_ = V.vacuum_channels(
            basis=basis, potvac=q, bexu=jnp.full_like(b.R, 0.02),
            bexv=0.1 * b.R, guu=guu, guv=guv, gvv=gvv,
        )
        return jnp.asarray([jnp.mean(bsq), jnp.mean(bsq * b.R)])

    x0 = jnp.zeros((2,), dtype=jnp.float64)
    q0, *_ = solver.full(boundary0, bexni)
    op = linearize_nestor_coupling(plasma_residual, vacuum_system, x0, q0)

    def coupled(value):
        x, q = value[:2], value[2:]
        matrix, rhs = vacuum_system(x)
        return jnp.concatenate((plasma_residual(x, q), matrix @ q - rhs))

    base = jnp.concatenate((x0, q0))
    tangent = jnp.linspace(-0.2, 0.3, base.size)
    cotangent = jnp.linspace(0.1, -0.1, base.size)
    expected = jax.jvp(coupled, (base,), (tangent,))[1]
    np.testing.assert_allclose(op(tangent), expected, rtol=2e-12, atol=2e-12)
    _, pullback = jax.vjp(coupled, base)
    np.testing.assert_allclose(
        op.transpose(cotangent), pullback(cotangent)[0],
        rtol=2e-12, atol=2e-12,
    )
    for value in (
        op.plasma(jnp.ones_like(x0)),
        op.vacuum_to_plasma(jnp.ones_like(q0)),
        op.plasma_to_vacuum(jnp.ones_like(x0)),
        op.vacuum(jnp.ones_like(q0)),
    ):
        assert float(jnp.linalg.norm(value)) > 0.0


def test_vacuum_first_call_diagnostics(ab_inputs):
    """vacuum.f first-call print block values against the golden stdout."""
    basis: V.VacuumBasis = ab_inputs["basis"]
    boundary = ab_inputs["boundary"]
    rt = ab_inputs["rt"]
    ext = ab_inputs["ext"]
    solver = V.make_vacuum_solver(basis, signgs=int(rt.setup.signgs))
    potvac, *_ = solver.full(boundary, jnp.asarray(ext["bexni"]))
    _bsq, bsubu_s, bsubv_s, _bu, _bv = V.vacuum_channels(
        basis=basis, potvac=potvac,
        bexu=jnp.asarray(ext["bexu"]), bexv=jnp.asarray(ext["bexv"]),
        guu=jnp.asarray(ext["guu"]), guv=jnp.asarray(ext["guv"]),
        gvv=jnp.asarray(ext["gvv"]),
    )
    wint2 = np.asarray(basis.wint).reshape(basis.ntheta3, basis.nzeta)
    fac = 1.0e-6 / FB.MU0
    bsubuvac = float(np.sum(np.asarray(bsubu_s) * wint2)) * float(rt.setup.signgs) * 2 * np.pi
    bsubvvac = float(np.sum(np.asarray(bsubv_s) * wint2))
    # Golden stdout: 4.69E-02, 4.32E-02, -4.59E-01 (first vacuum call).
    assert bsubuvac * fac == pytest.approx(4.69e-2, abs=3e-3)
    assert bsubvvac == pytest.approx(-4.59e-1, abs=5e-3)
    state = _initial_state(ab_inputs["rt"].setup)
    ctor, *_rest = FB._vacuum_scalars(state, ab_inputs["rt"])
    assert float(ctor) * fac == pytest.approx(4.32e-2, abs=2e-4)


def test_fused_vacuum_matches_reference(ab_inputs):
    """R15.2: the fused on-device vacuum update (``_make_fused_vacuum().full``,
    ONE jitted program) reproduces the parity-proven step-by-step host path
    to floating-point precision — the two differ only by op ordering.  Uses
    the LASYM fixture so both parities and the asym solve blocks run."""
    from dataclasses import replace

    inp = ab_inputs["inp"]
    res = ab_inputs["res"]
    rt = ab_inputs["rt"]
    basis: V.VacuumBasis = ab_inputs["basis"]
    ns = int(res.ns)
    dtype = rt.setup.s_full.dtype
    state = _initial_state(rt.setup)
    signgs = int(rt.setup.signgs)
    rt_freeb = replace(
        rt, lfreeb=True, jmax=ns,
        bsqvac_edge=jnp.zeros((basis.ntheta3, basis.nzeta), dtype=dtype),
        presf_ns_scale=jnp.asarray(FB._presf_ns_scale(inp, ns), dtype=dtype),
    )
    field = MgridField.from_mgrid_data(
        read_mgrid(MGRID),
        extcur=np.asarray(inp.extcur, dtype=float)[: read_mgrid(MGRID).nextcur],
    )
    solver = V.make_vacuum_solver(basis, signgs=signgs)

    # -- reference: step-by-step host path --
    ctor, _rb, axis_r, axis_z, _b3, _pr = FB._vacuum_scalars(state, rt_freeb)
    rmnc, zmns, rmns, zmnc = FB._edge_fourier(state, rt_freeb)
    boundary = FB.boundary_from_coefficients(
        rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc, modes=rt.modes, basis=basis
    )
    phi = (np.asarray(basis.zeta) * basis.onp).reshape(basis.ntheta3, basis.nzeta)
    br_c, bp_c, bz_c = field.b_cyl(np.asarray(boundary.R), phi, np.asarray(boundary.Z))
    br_a, bp_a, bz_a = FB.axis_current_field(
        R=np.asarray(boundary.R), Z=np.asarray(boundary.Z),
        axis_r=np.asarray(axis_r), axis_z=np.asarray(axis_z),
        nfp=basis.nfp, plascur=float(ctor),
    )
    ext = FB.external_field_channels(
        boundary=boundary, br=np.asarray(br_c) + br_a, bp=np.asarray(bp_c) + bp_a,
        bz=np.asarray(bz_c) + bz_a, basis=basis, signgs=signgs,
    )
    potvac_r, mm_r, bv_r, *_ = solver.full(boundary, jnp.asarray(ext["bexni"]))
    bsqvac_r, *_ = V.vacuum_channels(
        basis=basis, potvac=potvac_r,
        bexu=jnp.asarray(ext["bexu"]), bexv=jnp.asarray(ext["bexv"]),
        guu=jnp.asarray(ext["guu"]), guv=jnp.asarray(ext["guv"]),
        gvv=jnp.asarray(ext["gvv"]),
    )

    # -- fused: one jitted program --
    fused = FB._make_fused_vacuum(
        basis, modes=rt.modes, signgs=signgs, solver_vac=solver,
        axis_r0=axis_r, axis_z0=axis_z,
    )
    out = fused.full(state, rt_freeb, field)

    def _rel(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)

    assert _rel(out["bsqvac"], bsqvac_r) < 1e-10
    assert _rel(out["potvac"], potvac_r) < 1e-10
    assert _rel(out["mode_matrix"], mm_r) < 1e-10
    assert _rel(out["bvec_nonsing"], bv_r) < 1e-10
    assert float(out["ctor"]) == pytest.approx(float(ctor), rel=1e-12, abs=1e-14)
    skipped = fused.skip(
        state, rt_freeb, field, out["bvec_nonsing"],
        out["mode_factor"], out["mode_pivots"],
    )
    assert _rel(skipped["potvac"], out["potvac"]) < 1e-10
    assert _rel(skipped["bsqvac"], out["bsqvac"]) < 1e-10


def test_vacuum_lane_never_consumes_a_sign_changed_state():
    """funct3d.f parity: NESTOR evaluates ``xstore`` on a bad-Jacobian pass
    (Jacobian first; a sign-changed pass restarts and the re-evaluation runs
    the IVAC0 block at the restored state).  Feeding NESTOR the sign-changed
    state hands analyt.f a degenerate boundary — the poisoned ``bsqvac``
    (DEL-BSQ = NaN) behind the QI gate's NON-FINITE FORCE EVALUATION.  The
    steady lane must keep the vacuum response finite and exit through the
    ordinary restart."""
    from dataclasses import replace

    from vmex.core.errors import NONFINITE_FLAG

    inp = VmecInput.from_file(DECK)
    field = FB._external_field_from_input(inp, MGRID)
    res = FB.free_boundary_resolution(inp, field)
    rt = prepare_runtime(inp, res)
    ns = int(res.ns)
    dtype = rt.setup.s_full.dtype

    healthy = _initial_state(rt.setup)
    axis_r, axis_z = FB._vacuum_scalars(healthy, rt)[2:4]
    basis, fused, lane = FB._vacuum_executables(
        res, mf=int(inp.mpol) + 1, nf=int(inp.ntor),
        signgs=int(rt.setup.signgs),
        wint=np.asarray(rt.trig.wint, dtype=float),
        modes=rt.modes, axis_r0=axis_r, axis_z0=axis_z, use_fft=False,
    )
    rt_freeb = replace(
        rt, lfreeb=True, jmax=ns, max_iterations=5,
        bsqvac_edge=jnp.zeros((basis.ntheta3, basis.nzeta), dtype=dtype),
        presf_ns_scale=jnp.asarray(FB._presf_ns_scale(inp, ns), dtype=dtype),
    )

    # Collapse the edge row onto the magnetic axis: the boundary surface
    # degenerates (Ru = Zu = 0, analyt.f log argument -> 0) and, lying
    # inside the interior surfaces, flips the Jacobian sign.
    keep00 = ((np.asarray(rt.modes.m) == 0)
              & (np.asarray(rt.modes.n) == 0)).astype(float)
    bad = dataclasses.replace(
        healthy,
        R_cos=healthy.R_cos.at[-1].set(healthy.R_cos[-1] * keep00),
        R_sin=healthy.R_sin.at[-1].set(0.0),
        Z_cos=healthy.Z_cos.at[-1].set(0.0),
        Z_sin=healthy.Z_sin.at[-1].set(0.0),
    )
    assert bool(FB._jacobian_ok(healthy, rt))
    assert not bool(FB._jacobian_ok(bad, rt))

    # The mechanism: NESTOR on the sign-changed state IS non-finite — this
    # is exactly what the pre-gate lane consumed.
    seed = fused.full(healthy, rt_freeb, field)
    poisoned = fused.full(bad, rt_freeb, field)
    assert np.all(np.isfinite(np.asarray(seed["bsqvac"])))
    assert not np.all(np.isfinite(np.asarray(poisoned["bsqvac"])))

    int64 = lambda v: jnp.asarray(v, dtype=jnp.int64)  # noqa: E731
    carry = FB._initial_carry(bad, rt_freeb, ijacob=0,
                              residuals=(1.0e-4, 1.0e-4, 1.0e-4))
    # A steady-state pass: mid-run iteration, best state stored at xstore;
    # max_iterations=5 bounds the lane to this single pass.
    carry = dataclasses.replace(carry, xstore=healthy,
                                iteration=int64(5), iter1=int64(1))
    vc = FB._VacuumLoopCarry(
        carry=carry,
        rcon0=rt_freeb.rcon0, zcon0=rt_freeb.zcon0,
        bsqvac=seed["bsqvac"], rbsq=seed["rbsq"],
        mode_matrix=seed["mode_matrix"], mode_factor=seed["mode_factor"],
        mode_pivots=seed["mode_pivots"],
        bvec_nonsing=seed["bvec_nonsing"],
        potvac=seed["potvac"], surface_fields=seed["surface_fields"],
        ivac=int64(3), nvacskip=int64(1), nvskip0=int64(1),
        delbsq=jnp.asarray(1.0, dtype=dtype),
        delbsq_traj=jnp.full((5,), np.nan, dtype=dtype),
        ctor=seed["ctor"], rbtor=seed["rbtor"],
        vacuum_calls=int64(0), full_updates=int64(0),
    )
    out = lane(vc, rt_freeb, field)

    # NESTOR consumed the restored state: the vacuum response equals the
    # healthy evaluation and stays finite; the pass ends in the ordinary
    # restart path, not the non-finite failure.
    np.testing.assert_allclose(
        np.asarray(out.bsqvac), np.asarray(seed["bsqvac"]),
        rtol=1e-12, atol=1e-14)
    assert np.all(np.isfinite(np.asarray(out.carry.fsqr)))
    for leaf in jax.tree.leaves(out.carry.state):
        assert np.all(np.isfinite(np.asarray(leaf)))
    assert int(out.carry.ier) != NONFINITE_FLAG


# ---------------------------------------------------------------------------
# End-to-end golden run
# ---------------------------------------------------------------------------


def _golden_stdout(golden_dir: Path) -> str:
    path = golden_dir / CASE / "stdout.txt"
    if not path.is_file():
        pytest.skip(f"golden stdout missing: {path}")
    return path.read_text()


@pytest.mark.full
def test_free_boundary_end_to_end_golden(golden_dir):
    """Run the golden free-boundary deck with the core solver (structural).

    The golden run is unconverged (NITER exhausted, fsq ~ 1e-1, chaotic past
    turn-on), so this checks iteration *structure* and coarse values — see
    the module docstring for the tolerance rationale.
    """
    stdout_g = _golden_stdout(golden_dir)
    m = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)\s+ITERATIONS", stdout_g)
    assert m, "golden stdout lacks the vacuum banner"
    golden_turnon = int(m.group(1))
    final_line = [ln for ln in stdout_g.splitlines() if re.match(r"^\s*\d+\s+[\d.E+-]+", ln)][-1]
    golden_final_fsqr = float(final_line.split()[1])

    inp = VmecInput.from_file(DECK)
    lines: list[str] = []
    result = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, verbose=True,
        emit=lambda *a, **k: lines.append(a[0] if a else ""),
        error_on_no_convergence=False,
    )
    out = "".join(lines)

    m2 = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)\s+ITERATIONS", out)
    assert m2, "core run never activated the vacuum field"
    turnon = int(m2.group(1))
    # Same fixed-boundary physics up to activation: expect the same turn-on
    # iteration (53 in the golden stdout) modulo float-order jitter.
    assert abs(turnon - golden_turnon) <= 5

    assert "In VACUUM" in out
    assert result.iterations == int(inp.niter_array[0])
    assert not result.converged  # golden doesn't converge either
    # Final fsqr within 10x of the golden stdout's final printed value.
    assert result.fsqr <= 10.0 * golden_final_fsqr

    # edge rmnc/zmns vs golden wout (both trajectories unconverged)
    netCDF4 = pytest.importorskip("netCDF4")
    wout = golden_dir / CASE / f"wout_{CASE}.nc"
    with netCDF4.Dataset(wout) as ds:
        g_rmnc = np.asarray(ds.variables["rmnc"][:])[-1]
        g_zmns = np.asarray(ds.variables["zmns"][:])[-1]
        g_xm = np.asarray(ds.variables["xm"][:]).astype(int)
        g_xn = np.asarray(ds.variables["xn"][:]).astype(int)
    mine = {
        (int(m_), int(n_)): k for k, (m_, n_) in enumerate(zip(result.xm, result.xn))
    }
    idx = np.asarray([mine[(m_, n_)] for m_, n_ in zip(g_xm, g_xn)])
    r_err = np.abs(result.rmnc[-1][idx] - g_rmnc).max() / np.abs(g_rmnc).max()
    z_err = np.abs(result.zmns[-1][idx] - g_zmns).max() / np.abs(g_zmns).max()
    # Deliberately chaotic fixture: the endpoint sits on an attractor, so
    # even the machine-precision-equivalent fused vacuum path shifts it
    # (measured r 0.014-0.037, z 0.098 on CPU platforms, z 0.167 on CUDA);
    # bounds carry platform headroom.  The converged fixture below is the
    # pointwise-parity gate; only coarse structure is meaningful here.
    assert r_err < 0.08, f"edge rmnc scale-relative error {r_err}"
    assert z_err < 0.20, f"edge zmns scale-relative error {z_err}"


# ---------------------------------------------------------------------------
# Converged free-boundary golden (real mgrid, VMEC2000 terminates normally)
# ---------------------------------------------------------------------------

#: Deck + real MAKEGRID mgrid that VMEC2000 converges CLEANLY (476 iterations,
#: fsq < 1e-10, TERMINATED NORMALLY); the ``cth_like_free_bdy_lasym_small``
#: fixture above is a bounded LASYM smoke that neither code converges.
CONV_DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy"
CONV_MGRID = REPO / "examples" / "data" / "mgrid_cth_like.nc"
CONV_CASE = "cth_like_free_bdy"


@pytest.mark.full
def test_free_boundary_converged_golden(golden_dir):
    """Free boundary converges to VMEC2000's fsq level with wout parity.

    Regression guard for the NESTOR toroidal-phase fix (``xn*phi_geom`` in
    ``boundary_from_coefficients``): pre-fix, a mis-placed per-``nfp``
    ``bsqvac`` peak stalled the solve at NITER (fsqr ~ 9e-2).  Requires the
    real ``mgrid_cth_like.nc`` and the ``cth_like_free_bdy`` golden bundle;
    skips when either is unavailable.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    wout = golden_dir / CONV_CASE / f"wout_{CONV_CASE}.nc"
    if not CONV_MGRID.exists():
        pytest.skip("real mgrid_cth_like.nc unavailable (run tools/fetch_assets.py)")
    if not wout.exists():
        pytest.skip(f"converged golden bundle {CONV_CASE} unavailable")

    inp = VmecInput.from_file(CONV_DECK)
    lines: list[str] = []
    result = FB.solve_free_boundary(
        inp, mgrid_path=CONV_MGRID, verbose=True,
        emit=lambda *a, **k: lines.append(a[0] if a else ""),
        max_iterations=2500, error_on_no_convergence=False,
    )
    out = "".join(lines)

    # 1. Convergence gate: reaches VMEC2000's residual level (deck FTOL 1e-10).
    ftol = float(inp.ftol_array[-1])
    assert result.converged, f"free boundary did not converge (fsqr={result.fsqr:.2e})"
    assert result.fsqr <= ftol and result.fsqz <= ftol and result.fsql <= ftol

    # 2. Vacuum turn-on matches the golden stdout (53) modulo float jitter.
    m = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)\s+ITERATIONS", out)
    assert m, "vacuum never activated"
    stdout_g = (golden_dir / CONV_CASE / "stdout.txt").read_text()
    mg = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)\s+ITERATIONS", stdout_g)
    assert mg and abs(int(m.group(1)) - int(mg.group(1))) <= 3

    # 3. Per-variable wout parity vs the VMEC2000 golden (measured rmnc
    #    1.8e-5, zmns 1.2e-4, wb 2e-7, ctor 1e-15; gates leave headroom for
    #    the turn-on soft-restart timing difference).
    with netCDF4.Dataset(wout) as ds:
        g_wb = float(ds.variables["wb"][:])
        g_rmnc = np.asarray(ds.variables["rmnc"][:])
        g_zmns = np.asarray(ds.variables["zmns"][:])
        g_iotaf = np.asarray(ds.variables["iotaf"][:])
        g_xm = np.asarray(ds.variables["xm"][:]).astype(int)
        g_xn = np.asarray(ds.variables["xn"][:]).astype(int)

    assert abs(result.wb - g_wb) <= 1e-5 * abs(g_wb) + 1e-12, "wb parity"

    mine = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(result.xm, result.xn))}
    idx = np.asarray([mine[(a, b)] for a, b in zip(g_xm, g_xn)])
    r_err = np.abs(result.rmnc[:, idx] - g_rmnc).max() / np.abs(g_rmnc).max()
    z_err = np.abs(result.zmns[:, idx] - g_zmns).max() / np.abs(g_zmns).max()
    iota_err = np.abs(result.iotaf - g_iotaf).max() / np.abs(g_iotaf).max()
    assert r_err < 1e-3, f"rmnc scale-relative error {r_err}"
    assert z_err < 1e-3, f"zmns scale-relative error {z_err}"
    assert iota_err < 1e-3, f"iotaf scale-relative error {iota_err}"


# ---------------------------------------------------------------------------
# Missing-mgrid fallback policy
# ---------------------------------------------------------------------------


def test_missing_mgrid_raises(tmp_path):
    """solve_free_boundary surfaces MgridNotFoundError for a missing file."""
    inp = VmecInput.from_file(DECK)
    with pytest.raises(MgridNotFoundError):
        FB.solve_free_boundary(inp, mgrid_path=tmp_path / "mgrid_missing.nc")


def test_jac75_retry_rebuilds_vacuum_and_converges(capsys):
    """A recovered free-boundary stage rebuilds NESTOR at its checkpoint."""
    if not CONV_MGRID.exists():
        pytest.skip(
            "converged public CTH mgrid asset unavailable; run "
            "examples/data/fetch_assets.py"
        )
    inp = dataclasses.replace(
        VmecInput.from_file(CONV_DECK), delt=1.0e4,
    )
    with pytest.raises(VmecJacobianError) as exc:
        FB.solve_free_boundary(
            inp,
            mgrid_path=CONV_MGRID,
            max_iterations=2500,
            jacobian_retries=0,
        )
    assert exc.value.jacobian_resets == 75

    result = FB.solve_free_boundary(
        inp,
        mgrid_path=CONV_MGRID,
        max_iterations=2500,
        jacobian_retries=2,
        verbose=True,
    )
    output = capsys.readouterr().out
    assert "JACOBIAN RECOVERY RETRY" in output
    assert "VACUUM PRESSURE TURNED ON" in output
    assert result.converged
    assert max(result.fsqr, result.fsqz, result.fsql) <= 1.0e-10


def test_bad_supplied_axis_is_reguessed_before_fused_filament(capsys):
    """Free boundary must share fixed boundary's first-bad-axis recovery."""
    inp = VmecInput.from_file(DECK)
    raxis_c = inp.raxis_c.copy()
    raxis_c[0] = 2.0  # deliberately outside the CTH-like plasma boundary
    inp = dataclasses.replace(inp, raxis_c=raxis_c, niter_array=[2], nstep=1)
    result = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, max_iterations=2, verbose=True,
        error_on_no_convergence=False,
    )
    output = capsys.readouterr().out
    assert "TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS" in output
    assert not result.converged
    assert np.isfinite(result.fsqr)
    assert result.r00 < 1.0


def test_high_first_force_reguesses_valid_axis_before_vacuum(capsys):
    """LMOVE_AXIS also retries a valid-Jacobian axis when FSQ(1) > 1e2."""
    inp = VmecInput.from_file(DECK)
    raxis_c = inp.raxis_c.copy()
    raxis_c[0] = 0.81  # valid Jacobian, but raw first-force sum is ~2.8e2
    inp = dataclasses.replace(inp, raxis_c=raxis_c, niter_array=[2], nstep=1)
    result = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, max_iterations=2, verbose=True,
        error_on_no_convergence=False,
    )
    output = capsys.readouterr().out
    assert "INITIAL JACOBIAN CHANGED SIGN" not in output
    assert "TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS" in output
    assert result.fsq_history[0, :3].sum() < 1.0e2
    assert np.isfinite(result.fsqr)


def test_axis_reguess_rebuild_keeps_fft_vacuum_lane(capsys, monkeypatch):
    """The mid-flight axis-reguess rebuild must keep the resolved ``use_fft``.

    The LMOVE_AXIS transfer rebuilds the NESTOR executables for the improved
    axis; that rebuild once dropped the ``use_fft`` kwarg, silently swapping
    an FFT-selected run onto the dense-synthesis vacuum lane for the rest of
    the stage.  A ``_vacuum_executables`` spy records the kwarg every call
    receives during a forced first-force reguess under ``use_fft=True``.
    """
    seen: list[object] = []
    original = FB._vacuum_executables

    def recording(resolution, **kwargs):
        seen.append(kwargs.get("use_fft", "MISSING"))
        return original(resolution, **kwargs)

    monkeypatch.setattr(FB, "_vacuum_executables", recording)
    # fresh cache: the steady lane bakes use_fft into its traced body, so a
    # cached entry from another test would hide a wrong-kwarg rebuild
    monkeypatch.setattr(FB, "_VACUUM_EXECUTABLE_CACHE", {})

    inp = VmecInput.from_file(DECK)
    raxis_c = inp.raxis_c.copy()
    raxis_c[0] = 0.81  # valid Jacobian, first-force sum > 1e2 -> LMOVE_AXIS
    inp = dataclasses.replace(inp, raxis_c=raxis_c, niter_array=[2], nstep=1)
    FB.solve_free_boundary(
        inp, mgrid_path=MGRID, max_iterations=2, verbose=True,
        error_on_no_convergence=False, use_fft=True,
    )
    output = capsys.readouterr().out
    assert "TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS" in output
    assert len(seen) >= 2, "axis reguess never rebuilt the vacuum executables"
    assert seen == [True] * len(seen), (
        f"vacuum-executable builds saw use_fft={seen}; the reguess rebuild "
        "fell back to the dense-synthesis lane")


def test_cached_vacuum_executable_rechecks_dynamic_axis(monkeypatch):
    """A structural cache hit must still validate the current magnetic axis."""
    resolution = object()
    cached = (object(), object(), object())
    axis_r = jnp.asarray([1.0])
    axis_z = jnp.asarray([0.0])
    device = next(iter(axis_r.devices()))
    monkeypatch.setitem(
        FB._VACUUM_EXECUTABLE_CACHE,
        (resolution, 1, 2, 3, False, str(device), "None"),
        cached,
    )
    seen = []
    monkeypatch.setattr(
        FB, "_assert_static_filament_topology", lambda *args: seen.append(args),
    )
    result = FB._vacuum_executables(
        resolution, mf=2, nf=3, signgs=1, wint=None, modes=None,
        axis_r0=axis_r, axis_z0=axis_z,
    )

    assert result is cached
    assert len(seen) == 1
    assert seen[0][0] is cached[0]
    assert seen[0][1] is axis_r
    assert seen[0][2] is axis_z


def test_cli_missing_mgrid_fallback_warns(tmp_path):
    """CLI policy: missing mgrid -> fixed-boundary fallback warning (VMEC2000)."""
    import types

    from vmex.core.cli import _free_boundary_plan

    deck = tmp_path / "input.cth_like_free_bdy_lasym_small"
    deck.write_text(DECK.read_text())  # mgrid deliberately NOT copied
    inp = VmecInput.from_file(deck)
    messages: list[str] = []
    args = types.SimpleNamespace(coils=None)
    plan = _free_boundary_plan(args, inp, deck,
                               emit=lambda s, **k: messages.append(str(s)))
    assert plan is None  # fixed-boundary fallback
    assert any("FIXED-BOUNDARY" in msg for msg in messages)
    assert any("mgrid file not found" in msg for msg in messages)
