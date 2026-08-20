"""Implicit-differentiation gradient tests for ``vmex.core.implicit``.

Self-referential against central finite differences through the full host
solver: the traceable parameter map reproduces ``prepare_runtime`` exactly;
the residual vanishes at the fixed point; solovev (ns=11, ftol=1e-14)
gradients match central FD to rtol <= 1e-6; the adjoint GMRES converges
(< 1e-10) within ~300 matvecs on the preconditioned formulation but not on
raw force; li383 3D ``d(wb)/d(RBC(0,1))`` matches FD to rtol 2e-4 (measured
FD noise floor ~3e-5); the gradient is iteration-policy independent;
solver-sensitive metrics (iota_edge, DMerc, jdotb, D_R) match the
*frozen-path* FD where a naive re-solve FD sign-flips; typed errors relay
through the callback; and the multigrid lane is FD-validated directly.

FD steps: central differences with the same ftol/iteration policy on both
sides, disk-cached per (case, parameter, step, ftol); chosen so truncation
~ h^2 sits above solver-termination noise ~ eps_wb / h: solovev h = 3e-5
(boundary), 1e-5 (phiedge), 1e-4 (pres_scale); li383 h = 4e-4 (boundary).
"""

from __future__ import annotations

import dataclasses
import gc
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from vmex.core import implicit as im
from vmex.core import solver
from vmex.core import stability as stab
from vmex.core.input import VmecInput

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
FD_CACHE = Path(tempfile.gettempdir()) / "vmex_implicit_fd_cache.pkl"


@pytest.fixture(autouse=True, scope="module")
def _jit_enabled():
    """conftest disables jit globally; these full solves are ~40x slower
    interpreted (105-160 s/test in CI) — run them jitted."""
    prev = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", prev)


@pytest.fixture(autouse=True)
def _release_jax_caches_in_full_matrix():
    """Full-lane sweeps accumulate executable caches until the hosted CI
    worker is memory-evicted (happened twice); release after each test in
    the full matrix only — the ordinary lanes keep their compile reuse."""
    yield
    if os.environ.get("RUN_FULL"):
        jax.clear_caches()
        gc.collect()

CASES = {
    "solovev": dict(ftol=1e-14, max_iterations=2000),
    "li383_low_res": dict(ftol=1e-13, max_iterations=6000),
}


# ---------------------------------------------------------------------------
# helpers: cached FD solves through the host solver
# ---------------------------------------------------------------------------


def _load_cache() -> dict:
    if FD_CACHE.exists():
        try:
            with FD_CACHE.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            return {}
    return {}


def _store_cache(cache: dict) -> None:
    try:
        with FD_CACHE.open("wb") as fh:
            pickle.dump(cache, fh)
    except OSError:
        pass


def _perturb(p0: im.ImplicitParams, field: str, idx, h: float) -> im.ImplicitParams:
    value = getattr(p0, field)
    if idx is None:
        return dataclasses.replace(p0, **{field: value + h})
    arr = np.asarray(value).copy()
    arr[idx] += h
    return dataclasses.replace(p0, **{field: jnp.asarray(arr)})


def _outputs(name: str, inp: VmecInput, cfg, params) -> dict:
    """Host solve at ``params`` -> derived scalars (identical policy always)."""
    result = solver.solve(
        im.input_with_params(inp, params), cfg.resolution,
        ftol=cfg.ftol, max_iterations=cfg.max_iterations, mode="cli",
    )
    assert result.converged
    rt = im.runtime_from_params(params, cfg)
    wb, wp = im.mhd_energy(result.state, rt)
    return dict(wb=float(wb), wp=float(wp),
                aspect=float(im.aspect_ratio(result.state, rt)))


def _fd(name: str, inp: VmecInput, cfg, p0, field: str, idx, h: float) -> dict:
    """Central FD of every derived scalar, disk-cached per (case, dof, h)."""
    cache = _load_cache()
    out = {}
    for sign in (+1.0, -1.0):
        key = (name, field, None if idx is None else tuple(np.atleast_1d(idx)),
               float(sign * h), float(cfg.ftol), int(cfg.resolution.ns))
        if key not in cache:
            cache[key] = _outputs(name, inp, cfg, _perturb(p0, field, idx, sign * h))
            _store_cache(cache)
        out[sign] = cache[key]
    return {k: (out[+1.0][k] - out[-1.0][k]) / (2.0 * h) for k in out[+1.0]}


def _tnorm(tree) -> float:
    return float(np.sqrt(sum(float(jnp.vdot(a, a)) for a in jax.tree.leaves(tree))))


def _tree_dot(left, right) -> float:
    return float(sum(jnp.vdot(a, b) for a, b in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True)))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request):
    name = request.param
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    result = solver.solve(inp, cfg.resolution, ftol=cfg.ftol,
                          max_iterations=cfg.max_iterations, mode="cli")
    rt = im.runtime_from_params(p0, cfg)
    mask = im._dof_mask(result.state, rt, cfg)
    return name, inp, cfg, p0, result.state, rt, mask


@pytest.fixture(scope="module")
def solovev(request):
    name = "solovev"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    result = solver.solve(inp, cfg.resolution, ftol=cfg.ftol,
                          max_iterations=cfg.max_iterations, mode="cli")
    rt = im.runtime_from_params(p0, cfg)
    mask = im._dof_mask(result.state, rt, cfg)
    return name, inp, cfg, p0, result.state, rt, mask


# ---------------------------------------------------------------------------
# 1. the differentiable parameter map reproduces the host setup exactly
# ---------------------------------------------------------------------------


def test_runtime_from_params_matches_run_setup(case):
    name, inp, cfg, p0, _, rt_p, _ = case
    rt_ref = solver.prepare_runtime(inp, cfg.resolution, ftol=cfg.ftol,
                                    max_iterations=cfg.max_iterations)
    for f in dataclasses.fields(type(rt_ref.setup)):
        a = getattr(rt_ref.setup, f.name)
        b = getattr(rt_p.setup, f.name)
        if isinstance(a, (bool, int)):
            assert a == b, f"{name}: setup.{f.name}"
            continue
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=0.0, atol=1e-13,
            err_msg=f"{name}: setup.{f.name}")
    np.testing.assert_allclose(np.asarray(rt_ref.rcon0), np.asarray(rt_p.rcon0),
                               rtol=0.0, atol=1e-15, err_msg=f"{name}: rcon0")
    np.testing.assert_allclose(np.asarray(rt_ref.zcon0), np.asarray(rt_p.zcon0),
                               rtol=0.0, atol=1e-15, err_msg=f"{name}: zcon0")


@pytest.mark.parametrize("source", [0, 5, 9])
def test_raw_residual_jvp_is_exactly_nearest_neighbor(solovev, source):
    """A one-surface tangent has no raw-force response beyond adjacent rows."""
    _, _, cfg, p0, state, _, mask = solovev
    P = im._dof_projector(cfg, mask)
    z_star = P(state)
    tangent = jax.tree.map(
        lambda active: jnp.zeros_like(active).at[source].set(active[source]),
        mask,
    )
    F = im.residual_fn(cfg, state, mask, formulation="raw")
    response = jax.jvp(lambda z: F(z, p0), (z_star,), (tangent,))[1]
    keep = jnp.abs(jnp.arange(cfg.resolution.ns) - source) <= 1
    local_max = 0.0
    for value in jax.tree.leaves(response):
        assert float(jnp.max(jnp.abs(jnp.where(keep[:, None], 0.0, value)))) == 0.0
        local_max = max(local_max, float(jnp.max(jnp.abs(value[keep]))))
    assert local_max > 0.0


def test_three_surface_raw_kernel_matches_global_nonlinear_rows(case):
    """The local nonlinear chain reproduces every row with a complete halo."""
    _, _, cfg, p0, state, rt, mask = case
    ns = cfg.resolution.ns
    P = im._dof_projector(cfg, mask)
    edge = im._edge_mask(cfg)
    z_star = P(state)
    key = jax.random.key(13)
    noise = jax.tree.map(
        lambda a: 1e-5 * a * jax.random.normal(
            key, a.shape, dtype=jnp.asarray(a).dtype
        ),
        mask,
    )
    z = jax.tree.map(jnp.add, z_star, noise)
    x = im._assemble(z, rt, state, P, edge)
    full = im.residual_fn(cfg, state, mask, formulation="raw")(z, p0)

    segments = ((0, (0, 1)), (max(0, ns // 2 - 1), (1,)), (ns - 3, (1, 2)))
    for start, rows in segments:
        window = jax.tree.map(lambda a: a[start:start + 3], x)
        local = im._raw_residual_segment(
            window, rt, jnp.asarray(start), axis_closure=(start == 0)
        )
        embedded = jax.tree.map(
            lambda a, whole: jnp.zeros_like(whole).at[start:start + 3].set(a),
            local, full,
        )
        projected = P(embedded)
        for name, actual, expected in zip(
            im._STATE_FIELDS, jax.tree.leaves(projected),
            jax.tree.leaves(full), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(actual)[start + np.asarray(rows)],
                np.asarray(expected)[start + np.asarray(rows)],
                rtol=2e-12, atol=2e-12,
                err_msg=f"{name}, segment {start}, local rows {rows}",
            )


# ---------------------------------------------------------------------------
# 2. the implicit residual vanishes at the converged fixed point
# ---------------------------------------------------------------------------


def test_residual_zero_at_fixed_point(case):
    name, inp, cfg, p0, x_star, rt, mask = case
    P = im._dof_projector(cfg, mask)
    F = im.residual_fn(cfg, jax.lax.stop_gradient(x_star), mask)
    r0 = _tnorm(F(P(x_star), p0))

    # reference scale: the residual at a small but macroscopic perturbation
    delta = jax.tree.map(lambda a: a * (1.0 + 1e-3), x_star)
    r1 = _tnorm(F(P(delta), p0))
    assert r1 > 0.0
    assert r0 < 1e-4 * r1, f"{name}: |F(x*)| = {r0:.3e} vs |F(x*+d)| = {r1:.3e}"


# ---------------------------------------------------------------------------
# 2b. the dof mask is a structural invariant -> memoized by structural key
# ---------------------------------------------------------------------------


def _mask_bit_ident(a, b) -> bool:
    return all(np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)))
               for f in im._STATE_FIELDS)


def test_dof_mask_structural_invariance_and_cache(solovev):
    """The dof mask depends only on structure (resolution / lconm1 / ncurr),
    so ``_MASK_CACHE`` (keyed by ``_mask_cache_key``) reuses it across the
    fresh configs ``make_config``/``run`` mint: bit-identical under recompute
    (different seed, perturbed parameters) and one cache entry, no recompute,
    across two fresh configs of the same structure."""
    name, inp, cfg, p0, x_star, rt, mask = solovev

    # fresh configs of the same structure share one hashable structural key,
    # even though ImplicitConfig is eq=False (object-identity equality).
    cfg_a = im.make_config(inp, **CASES[name])
    cfg_b = im.make_config(inp, **CASES[name])
    assert cfg_a is not cfg_b and cfg_a != cfg_b
    key = im._mask_cache_key(cfg)
    assert im._mask_cache_key(cfg_a) == key == im._mask_cache_key(cfg_b)
    assert hash(im._mask_cache_key(cfg_a)) == hash(key)

    # recompute with a different seed and with perturbed parameters
    mask_seed = im._dof_mask(x_star, rt, cfg, seed=7)
    p1 = _perturb(p0, "rbc", (int(inp.ntor), 1), 0.01)
    rt1 = im.runtime_from_params(p1, cfg)
    mask_pert = im._dof_mask(x_star, rt1, cfg, seed=0)
    assert _mask_bit_ident(mask, mask_seed), f"{name}: mask not seed-invariant"
    assert _mask_bit_ident(mask, mask_pert), f"{name}: mask not parameter-invariant"

    # end-to-end: the module cache hits across two fresh configs (one entry).
    saved = dict(im._MASK_CACHE)
    try:
        im._MASK_CACHE.clear()
        _, m_a = im._host_solve_and_mask(cfg_a, p0)
        assert len(im._MASK_CACHE) == 1
        _, m_b = im._host_solve_and_mask(cfg_b, p0)  # fresh cfg -> cache HIT
        assert len(im._MASK_CACHE) == 1, "a second structural entry was minted"
        assert _mask_bit_ident(m_a, m_b)
        assert _mask_bit_ident(mask, m_a)
    finally:
        im._MASK_CACHE.clear()
        im._MASK_CACHE.update(saved)


# ---------------------------------------------------------------------------
# 3. solovev gradient table vs central FD (rtol <= 1e-6)
# ---------------------------------------------------------------------------


def test_solovev_gradients_vs_fd(solovev):
    name, inp, cfg, p0, _, _, _ = solovev
    ntor = int(inp.ntor)

    def outs(p):
        sol = im.run(inp, p, ftol=cfg.ftol, max_iterations=cfg.max_iterations)
        return jnp.stack([sol.wb, sol.wp, sol.aspect])

    jac = jax.jacrev(outs)(p0)  # one forward solve + one adjoint per output
    ad = {
        ("wb", "rbc"): float(np.asarray(jac.rbc)[0, ntor, 1]),
        ("aspect", "rbc"): float(np.asarray(jac.rbc)[2, ntor, 1]),
        ("wb", "phiedge"): float(np.asarray(jac.phiedge)[0]),
        ("wp", "pres_scale"): float(np.asarray(jac.pres_scale)[1]),
    }

    checks = [
        ("wb", "rbc", (ntor, 1), 3e-5),
        ("aspect", "rbc", (ntor, 1), 3e-5),
        ("wb", "phiedge", None, 1e-5),
        ("wp", "pres_scale", None, 1e-4),
    ]
    print(f"\n[{name}] gradient vs central FD (forward ftol = {cfg.ftol:g}):")
    for out, field, idx, h in checks:
        fd = _fd(name, inp, cfg, p0, field, idx, h)[out]
        a = ad[(out, field)]
        rel = abs(a / fd - 1.0)
        print(f"  d({out})/d({field}{'' if idx is None else idx}) "
              f"h={h:.0e}: AD={a:+.12e}  FD={fd:+.12e}  rel={rel:.2e}")
        assert rel <= 1e-6, f"{out}/{field}: rel error {rel:.3e}"


# ---------------------------------------------------------------------------
# 4. adjoint GMRES: preconditioned formulation converges, raw does not
# ---------------------------------------------------------------------------


def test_adjoint_gmres_preconditioner_value(solovev):
    name, inp, cfg, p0, x_star, rt, mask = solovev
    P = im._dof_projector(cfg, mask)
    gbar = jax.grad(lambda s: im.mhd_energy(s, rt)[0])(x_star)
    b = P(gbar)
    nb = _tnorm(b)
    assert nb > 0.0

    budgets = {}
    for formulation in ("preconditioned", "raw"):
        A = im.adjoint_matvec(cfg, p0, x_star, mask, formulation=formulation)
        lam, _ = jax.scipy.sparse.linalg.gmres(
            A, b, tol=1e-13, atol=0.0, restart=30, maxiter=10,
            solve_method="incremental",
        )  # <= 300 matvecs
        residual = jax.tree.map(lambda u, v: u - v, A(lam), b)
        budgets[formulation] = _tnorm(residual) / nb

    print(f"\n[{name}] adjoint GMRES relative residual after <= 300 matvecs "
          f"(restart=30, maxiter=10):")
    for formulation, rel in budgets.items():
        print(f"  {formulation:15s}: {rel:.3e}")

    # preconditioned-residual formulation: converged well below 1e-10
    assert budgets["preconditioned"] < 1e-10
    # raw force without the 1D preconditioner: stuck orders of magnitude away
    assert budgets["raw"] > 1e-6
    assert budgets["raw"] / budgets["preconditioned"] > 1e4


# ---------------------------------------------------------------------------
# 5. one 3D case: li383, d(wb)/d(boundary coefficient) vs FD
# ---------------------------------------------------------------------------


def test_li383_boundary_gradient_vs_fd():
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)

    grad = jax.grad(
        lambda p: im.run(inp, p, ftol=cfg.ftol,
                         max_iterations=cfg.max_iterations).wb)(p0)
    ad = float(np.asarray(grad.rbc)[ntor, 1])
    fd = _fd(name, inp, cfg, p0, "rbc", (ntor, 1), 4e-4)["wb"]
    rel = abs(ad / fd - 1.0)
    print(f"\n[{name}] d(wb)/d(RBC(0,1)) h=4e-4: AD={ad:+.10e} FD={fd:+.10e} "
          f"rel={rel:.2e} (FD noise floor ~3e-5)")
    assert rel <= 2e-4


# ---------------------------------------------------------------------------
# 6. iteration-policy independence + memory sanity (informational)
# ---------------------------------------------------------------------------


def test_gradient_independent_of_iteration_policy(solovev):
    """Only the fixed point defines the derivative; RSS prints are the
    O(1)-memory sanity (backward = a few residual linearizations, no tape)."""
    import resource

    name, inp, cfg, p0, _, _, _ = solovev
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # noqa: E731

    grads = {}
    r0 = rss()
    for cap in (500, 5000):
        grads[cap] = jax.grad(
            lambda p: im.run(inp, p, ftol=cfg.ftol, max_iterations=cap).wb)(p0)
        print(f"\n[{name}] grad(wb) with max_iterations={cap}: "
              f"peak RSS delta so far = {rss() - r0:.0f} MB")

    a = np.asarray(grads[500].rbc)
    b = np.asarray(grads[5000].rbc)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-14)
    assert np.all(np.isfinite(a))


# ---------------------------------------------------------------------------
# 7. solver-sensitive metric (iota_edge, ncurr=1): the implicit adjoint equals
#    the FROZEN-PATH FD; a naive full re-solve FD is NOT a valid reference.
# ---------------------------------------------------------------------------


def test_iota_edge_gradient_vs_frozen_path_fd():
    """``d(iota_edge)/d(boundary)`` on the 3D ``ncurr=1`` case: the implicit
    adjoint must equal the frozen-path central FD to solver accuracy.  A naive
    re-solve FD is deliberately NOT the reference (measured, ``h=1e-4``):

        RBC(n=-1,m=1):  adjoint -0.77343,  frozen-path FD -0.77343,  naive FD +0.04543  (sign flip)
        RBC(n=+1,m=1):  adjoint -1.26567,  frozen-path FD -1.26567,  naive FD -2.14135  (69% off)
    """
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)
    assert int(inp.ncurr) == 1  # derived-iota case where the effect is largest

    grad = jax.grad(
        lambda p: im.run(inp, p, ftol=cfg.ftol,
                         max_iterations=cfg.max_iterations).iota_edge)(p0)
    zero = jax.tree.map(jnp.zeros_like, p0)
    print(f"\n[{name}] d(iota_edge)/d(RBC): implicit adjoint vs frozen-path FD")
    for n in (ntor - 1, ntor + 1):          # m=1, n=-1 (naive-FD sign flip) and n=+1
        ad = float(np.asarray(grad.rbc)[n, 1])
        tangent = dataclasses.replace(zero, rbc=zero.rbc.at[n, 1].set(1.0))
        fd, info = im.frozen_path_directional_fd(
            p0, cfg, im.iota_edge, tangent, h=1e-4)
        res = max(info["newton_res"])
        rel = abs(ad / fd - 1.0) if fd else abs(ad - fd)
        print(f"  RBC(n={n - ntor:+d},m=1): AD={ad:+.9e}  frozen-FD={fd:+.9e}  "
              f"rel={rel:.2e}  (Newton res {res:.0e})")
        assert res < 1e-8, f"n={n - ntor}: frozen solve not converged (res {res:.1e})"
        assert rel <= 3e-4, (
            f"RBC(n={n - ntor},1): adjoint {ad:.5e} vs frozen-path FD {fd:.5e} "
            f"(rel {rel:.2e}) — the implicit gradient must match the frozen path")


# ---------------------------------------------------------------------------
# 8. typed errors through pure_callback (zero-crash policy, plan Item I.1)
# ---------------------------------------------------------------------------


def test_typed_error_through_pure_callback():
    """A failing host solve raises the SHORT typed exception, not callback
    noise: the ``_HOST_ERROR`` relay stashes the typed exception in the host
    callback and re-raises it at the ``pure_callback`` site with ``from
    None`` (previously a ~3.7 KB raw ``JaxRuntimeError``)."""
    from vmex.core.errors import VmecConvergenceError

    inp = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    with pytest.raises(VmecConvergenceError) as excinfo:
        im.run(inp, ftol=1e-14, max_iterations=3)
    exc = excinfo.value
    assert len(str(exc)) < 200, f"typed message must stay short: {len(str(exc))} chars"
    assert "MORE ITERATIONS REQUIRED" in str(exc)
    assert exc.__cause__ is None and exc.__suppress_context__  # noise killed
    assert exc.ftol == 1e-14  # diagnostics preserved

    # the custom-vjp forward rule call site relays the typed exception too
    p0 = im.params_from_input(inp)
    with pytest.raises(VmecConvergenceError):
        jax.grad(lambda p: im.run(inp, p, ftol=1e-14, max_iterations=3).wb)(p0)


# ---------------------------------------------------------------------------
# 9. multigrid implicit gradient vs frozen-path FD (plan Item I.4)
# ---------------------------------------------------------------------------


def test_multigrid_gradient_vs_frozen_path_fd():
    """``im.run(multigrid=True)`` through a genuine ns 5 -> 11 solovev ladder:
    the adjoint must match the frozen-path central FD of ``wb`` along
    ``RBC(0,1)`` (only the final fixed point defines the derivative)."""
    inp0 = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    inp = dataclasses.replace(
        inp0,
        ns_array=np.array([5, 11]),
        ftol_array=np.array([1e-10, 1e-14]),
        niter_array=np.array([1000, 2000]),
    )
    cfg = im.make_config(inp, multigrid=True, ftol=1e-14)
    assert cfg.multigrid and int(cfg.resolution.ns) == 11
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)

    grad = jax.grad(
        lambda p: im.run(inp, p, multigrid=True, ftol=1e-14).wb)(p0)
    ad = float(np.asarray(grad.rbc)[ntor, 1])

    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent = dataclasses.replace(zero, rbc=zero.rbc.at[ntor, 1].set(1.0))
    fd, info = im.frozen_path_directional_fd(p0, cfg, lambda s, rt:
                                             im.mhd_energy(s, rt)[0],
                                             tangent, h=3e-5)
    res = max(info["newton_res"])
    rel = abs(ad / fd - 1.0)
    print(f"\n[solovev multigrid ns=5->11] d(wb)/d(RBC(0,1)) h=3e-5: "
          f"AD={ad:+.10e}  frozen-FD={fd:+.10e}  rel={rel:.2e} "
          f"(Newton res {res:.0e})")
    assert res < 1e-8, f"frozen solve not converged (res {res:.1e})"
    assert rel <= 1e-6


def _assert_stability_gradients(
    case,
    *,
    include_profiles,
    metric_names=("DMerc",),
    direction_labels=None,
):
    name, inp, cfg, p0, _, _, _ = case
    ntor = int(inp.ntor)
    zero = jax.tree.map(jnp.zeros_like, p0)
    directions = [
        ("RBC(0,1)", dataclasses.replace(
            zero, rbc=zero.rbc.at[ntor, 1].set(1.0))),
        ("ZBS(0,1)", dataclasses.replace(
            zero, zbs=zero.zbs.at[ntor, 1].set(1.0))),
    ]
    if bool(inp.lasym):
        directions += [
            ("RBS(1,1)", dataclasses.replace(
                zero, rbs=zero.rbs.at[ntor + 1, 1].set(1.0))),
            ("ZBC(1,1)", dataclasses.replace(
                zero, zbc=zero.zbc.at[ntor + 1, 1].set(1.0))),
        ]
    if include_profiles == "pressure":
        directions += [
            ("RBC(0,2)", dataclasses.replace(
                zero, rbc=zero.rbc.at[ntor, 2].set(1.0))),
            ("pres_scale", dataclasses.replace(
                zero, pres_scale=jnp.ones_like(zero.pres_scale))),
        ]
    elif include_profiles == "current":
        directions.append(("curtor/|curtor|", dataclasses.replace(
            zero, curtor=jnp.asarray(abs(float(inp.curtor))))))

    if direction_labels is not None:
        directions = [
            item for item in directions if item[0] in direction_labels
        ]
    metric_functions = {
        "DMerc": lambda state, runtime: jnp.sum(
            stab.d_merc_state(state, runtime)[2:-1]
        ),
        "jdotb": lambda state, runtime: jnp.sum(
            stab.jdotb_state(state, runtime)[2:-1]
        ),
        "D_R": lambda state, runtime: jnp.sum(
            stab.glasser_d_r_state(
                state, runtime, shear_epsilon=1.0e-8
            )[2:-1]
        ),
    }
    for metric_name in metric_names:
        metric = metric_functions[metric_name]

        def objective(params):
            solution = im.run(
                inp,
                params,
                ftol=cfg.ftol,
                max_iterations=cfg.max_iterations,
            )
            return metric(solution.state, solution.runtime)

        grad = jax.grad(objective)(p0)
        for label, tangent in directions:
            ad = _tree_dot(grad, tangent)
            fd, info = im.frozen_path_directional_fd(
                p0, cfg, metric, tangent, h=1.0e-4
            )
            residual = max(info["newton_res"])
            scale = max(abs(ad), abs(fd), 1.0e-12)
            relative_error = abs(ad - fd) / scale
            print(
                f"\n[{name}] d(sum {metric_name}[2:-1])/d({label}): "
                f"AD={ad:+.10e} frozen-FD={fd:+.10e} "
                f"rel={relative_error:.2e} (Newton res {residual:.0e})"
            )
            assert residual < 1.0e-8
            assert relative_error <= 2.0e-3


@pytest.mark.parametrize("case", ["solovev"], indirect=True)
def test_dmerc_gradient_vs_frozen_path_fd(case):
    """Implicit DMerc shape/pressure derivatives on the axisymmetric case."""
    _assert_stability_gradients(case, include_profiles="pressure")


@pytest.mark.parametrize("case", ["solovev"], indirect=True)
def test_jdotb_and_glasser_gradients_vs_frozen_path_fd(case):
    """Implicit current and resistive-interchange shape/profile gradients."""
    _assert_stability_gradients(
        case,
        include_profiles="pressure",
        metric_names=("jdotb", "D_R"),
        direction_labels={"RBC(0,1)", "pres_scale"},
    )


@pytest.mark.full
@pytest.mark.parametrize("case", ["li383_low_res"], indirect=True)
def test_dmerc_3d_current_gradient_vs_frozen_path_fd(case):
    """Nightly: implicit 3-D DMerc shape/current derivatives."""
    _assert_stability_gradients(case, include_profiles="current")


@pytest.mark.full
@pytest.mark.parametrize("case", ["li383_low_res"], indirect=True)
def test_jdotb_and_glasser_3d_current_gradients_vs_frozen_path_fd(case):
    """Nightly: implicit 3-D current-diagnostic shape/current derivatives."""
    _assert_stability_gradients(
        case,
        include_profiles="current",
        metric_names=("jdotb", "D_R"),
        direction_labels={"RBC(0,1)", "curtor/|curtor|"},
    )


# ===========================================================================
# lasym (non-stellarator-symmetric) implicit-gradient lane
# ===========================================================================
#
# The traceable parameter map reproduces the readin.f ``delta``
# theta-normalization and the four internal-block families for lasym, so
# ``jax.grad`` through ``im.run`` differentiates rbs/zbc boundary dofs.
# The 2D lasym deck ``up_down_asymmetric_tokamak`` exhausts NITER at its
# native ns = 17; ns = 11 / ftol = 1e-12 converges and is used here.
# The m = 1 lasym constraint floors the *frozen* residual at ~1e-7, capping
# naive FD-vs-AD agreement of the full im.run gradient at ~1e-4, while the
# gradient itself is exact: the p -> boundary map FD-validates to <= 1e-6
# with no solver in the loop, and the analytic directional derivative equals
# the frozen-path FD to <= 5e-5 at a Newton-refined fixed point.

LASYM_CASE = dict(ns=11, ftol=1e-12, max_iterations=4000)


@pytest.fixture(scope="module")
def lasym():
    """Converged fixed-boundary 2D lasym case (up_down deck, ns = 11)."""
    inp0 = VmecInput.from_file(str(DATA_DIR / "input.up_down_asymmetric_tokamak"))
    inp = dataclasses.replace(
        inp0,
        ns_array=np.array([LASYM_CASE["ns"]]),
        ftol_array=np.array([LASYM_CASE["ftol"]]),
        niter_array=np.array([LASYM_CASE["max_iterations"]]),
    )
    assert bool(inp.lasym)
    cfg = im.make_config(inp, ftol=LASYM_CASE["ftol"],
                         max_iterations=LASYM_CASE["max_iterations"])
    p0 = im.params_from_input(inp)
    result = solver.solve(inp, cfg.resolution, ftol=cfg.ftol,
                          max_iterations=cfg.max_iterations, mode="cli")
    assert result.converged
    rt = im.runtime_from_params(p0, cfg)
    mask = im._dof_mask(result.state, rt, cfg)
    return "up_down_asymmetric_tokamak", inp, cfg, p0, result.state, rt, mask


def test_lasym_delta_rotation_traceable():
    """The traceable delta rotation reproduces ``setup._lasym_delta_rotation``
    to ~1e-12 and stays differentiable in the (0,1) coefficients."""
    from vmex.core import setup as setup_mod

    inp = VmecInput.from_file(str(DATA_DIR / "input.up_down_asymmetric_tokamak"))
    cfg = im.make_config(inp, ftol=1e-12, max_iterations=10)
    mpol, ntor = int(inp.mpol), int(inp.ntor)
    rbc = np.asarray(inp.rbc, dtype=float)
    rbs = np.asarray(inp.rbs, dtype=float)
    zbc = np.asarray(inp.zbc, dtype=float)
    zbs = np.asarray(inp.zbs, dtype=float)
    ref = setup_mod._lasym_delta_rotation(rbc, rbs, zbc, zbs, mpol=mpol, ntor=ntor)
    tr = im._lasym_delta_rotation_traceable(
        jnp.asarray(rbc), jnp.asarray(rbs), jnp.asarray(zbc), jnp.asarray(zbs),
        cfg, mpol=mpol, ntor=ntor)
    worst = 0.0
    for nm, a, b in zip(("rbc", "rbs", "zbc", "zbs"), tr, ref):
        err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        worst = max(worst, err)
        assert err <= 1e-12, f"delta rotation {nm}: {err:.2e}"
    print(f"\n[up_down] traceable delta rotation vs setup reference: "
          f"max abs err = {worst:.2e}")

    # delta is a smooth function of RBS(0,1)/ZBC(0,1) -> finite gradient
    def rot_scalar(rbs_a):
        out = im._lasym_delta_rotation_traceable(
            jnp.asarray(rbc), rbs_a, jnp.asarray(zbc), jnp.asarray(zbs),
            cfg, mpol=mpol, ntor=ntor)
        return jnp.sum(out[1])  # rotated rbs block
    g = jax.grad(rot_scalar)(jnp.asarray(rbs))
    assert np.all(np.isfinite(np.asarray(g)))


def test_lasym_delta_rotation_jacobian_at_zero_delta():
    """The rotation Jacobian is exact where ``delta == 0``.

    ``readin.f`` skips the rotation loop when ``delta`` vanishes, where
    ``cos(0)/sin(0)`` already make it the identity, so matching the value
    there says nothing about the derivative.  A deck with
    ``RBS(0, 1) == ZBC(0, 1)`` sits on that point and still carries a rank-one
    ``m*(partner)*d(delta)`` term along ``RBS(0, 1) - ZBC(0, 1)``.  Both
    shipped LASYM decks have ``delta != 0``, so this one is built on the spot.
    """
    from vmex.core import setup as setup_mod

    inp = VmecInput.from_file(str(DATA_DIR / "input.up_down_asymmetric_tokamak"))
    cfg = im.make_config(inp, ftol=1e-12, max_iterations=10)
    mpol, ntor = int(inp.mpol), int(inp.ntor)
    rbc = np.asarray(inp.rbc, dtype=float)
    zbc = np.asarray(inp.zbc, dtype=float)
    zbs = np.asarray(inp.zbs, dtype=float)
    rbs = np.asarray(inp.rbs, dtype=float).copy()
    rbs[ntor, 1] = zbc[ntor, 1]  # put the reference exactly on delta == 0
    denom0 = abs(rbc[ntor, 1]) + abs(zbs[ntor, 1])
    assert denom0 > 0.0
    assert float(np.arctan((rbs[ntor, 1] - zbc[ntor, 1]) / denom0)) == 0.0
    inp = dataclasses.replace(inp, rbs=rbs)
    cfg = dataclasses.replace(cfg, inp=inp)

    def traceable(rbs_a, zbc_a):
        out = im._lasym_delta_rotation_traceable(
            jnp.asarray(rbc), rbs_a, zbc_a, jnp.asarray(zbs),
            cfg, mpol=mpol, ntor=ntor)
        return jnp.concatenate([jnp.ravel(a) for a in out])

    def reference(rbs_a, zbc_a):
        out = setup_mod._lasym_delta_rotation(
            rbc, np.asarray(rbs_a), np.asarray(zbc_a), zbs, mpol=mpol, ntor=ntor)
        return np.concatenate([np.ravel(np.asarray(a)) for a in out])

    # Value agreement is necessary but not sufficient, so check it and then
    # the derivative along the antisymmetric channel.
    np.testing.assert_allclose(
        np.asarray(traceable(jnp.asarray(rbs), jnp.asarray(zbc))),
        reference(rbs, zbc), rtol=0.0, atol=1e-12)

    direction = np.zeros_like(rbs)
    direction[ntor, 1] = 1.0
    _, jvp = jax.jvp(traceable, (jnp.asarray(rbs), jnp.asarray(zbc)),
                     (jnp.asarray(direction), jnp.asarray(-direction)))
    h = 1e-6
    fd = (reference(rbs + h * direction, zbc - h * direction)
          - reference(rbs - h * direction, zbc + h * direction)) / (2.0 * h)
    scale = max(float(np.max(np.abs(fd))), 1e-30)
    err = float(np.max(np.abs(np.asarray(jvp) - fd))) / scale
    assert scale > 1e-3, f"the probe direction must move the rotation: {scale:.2e}"
    assert err <= 1e-7, f"delta-rotation Jacobian on RBS(0,1)-ZBC(0,1): {err:.2e}"


def test_lasym_runtime_from_params_matches_run_setup(lasym):
    """The lasym traceable map reproduces ``prepare_runtime`` at the base
    parameters (all four boundary families, rcon0/zcon0)."""
    name, inp, cfg, p0, _, rt_p, _ = lasym
    rt_ref = solver.prepare_runtime(inp, cfg.resolution, ftol=cfg.ftol,
                                    max_iterations=cfg.max_iterations)
    for f in dataclasses.fields(type(rt_ref.setup)):
        a = getattr(rt_ref.setup, f.name)
        b = getattr(rt_p.setup, f.name)
        if isinstance(a, (bool, int)):
            assert a == b, f"{name}: setup.{f.name}"
            continue
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=0.0, atol=1e-13,
            err_msg=f"{name}: setup.{f.name}")
    np.testing.assert_allclose(np.asarray(rt_ref.rcon0), np.asarray(rt_p.rcon0),
                               rtol=0.0, atol=1e-15, err_msg=f"{name}: rcon0")
    np.testing.assert_allclose(np.asarray(rt_ref.zcon0), np.asarray(rt_p.zcon0),
                               rtol=0.0, atol=1e-15, err_msg=f"{name}: zcon0")
    # the asymmetric edge families actually carry the RBS content of the deck
    assert float(np.max(np.abs(np.asarray(rt_p.setup.boundary_R_sin)))) > 0.0


def test_lasym_residual_zero_at_fixed_point(lasym):
    name, inp, cfg, p0, x_star, rt, mask = lasym
    P = im._dof_projector(cfg, mask)
    F = im.residual_fn(cfg, jax.lax.stop_gradient(x_star), mask)
    r0 = _tnorm(F(P(x_star), p0))
    delta = jax.tree.map(lambda a: a * (1.0 + 1e-3), x_star)
    r1 = _tnorm(F(P(delta), p0))
    assert r1 > 0.0
    assert r0 < 1e-4 * r1, f"{name}: |F(x*)| = {r0:.3e} vs |F(x*+d)| = {r1:.3e}"


def test_lasym_boundary_map_derivative_vs_fd(lasym):
    """The traceable ``p -> processed boundary`` map derivative FD-validates
    to <= 1e-6 for all four families (rbc/zbs/rbs/zbc), no solver in the
    loop — isolating the lasym readin.f map from any convergence floor."""
    name, inp, cfg, p0, _, _, _ = lasym
    ntor = int(inp.ntor)

    def bnd_scalar(p):
        s = im.runtime_from_params(p, cfg).setup
        wsum = lambda a, c: jnp.sum(a * jnp.arange(1, a.size + 1)) * c  # noqa: E731
        return (wsum(s.boundary_R_cos, 1.0) + wsum(s.boundary_R_sin, 1.3)
                + wsum(s.boundary_Z_cos, 0.7) + wsum(s.boundary_Z_sin, 1.9))

    g = jax.grad(bnd_scalar)(p0)
    print(f"\n[{name}] boundary-map derivative AD vs FD (no solver):")
    for field, idx in (("rbc", (ntor, 1)), ("zbs", (ntor, 2)),
                       ("rbs", (ntor, 1)), ("rbs", (ntor, 2)),
                       ("zbc", (ntor, 1)), ("zbc", (ntor, 2))):
        h = 1e-6
        fp = float(bnd_scalar(_perturb(p0, field, idx, h)))
        fm = float(bnd_scalar(_perturb(p0, field, idx, -h)))
        fd = (fp - fm) / (2.0 * h)
        ad = float(np.asarray(getattr(g, field))[idx])
        rel = abs(ad / fd - 1.0) if fd else abs(ad - fd)
        print(f"  d/d({field}{idx}): AD={ad:+.10e} FD={fd:+.10e} rel={rel:.2e}")
        assert rel <= 1e-6, f"{field}{idx}: boundary-map rel {rel:.2e}"


def test_lasym_wb_aspect_gradient_vs_fd(lasym):
    """``d(wb)/d`` and ``d(aspect)/d`` clean m = 2 rbs/zbc dofs vs naive
    central FD through the host solve; agreement is at the solver-limited
    floor of this m=1-constrained deck (~1e-4, printed)."""
    name, inp, cfg, p0, _, _, _ = lasym
    ntor = int(inp.ntor)

    def outs(p):
        sol = im.run(inp, p, ftol=cfg.ftol, max_iterations=cfg.max_iterations)
        return jnp.stack([sol.wb, sol.aspect])

    jac = jax.jacrev(outs)(p0)
    row = {"wb": 0, "aspect": 1}
    checks = [
        ("wb", "rbs", (ntor, 2), 3e-5, 2e-4),
        ("aspect", "rbs", (ntor, 2), 3e-5, 1e-3),
        ("wb", "zbc", (ntor, 2), 3e-5, 2e-4),
        ("aspect", "zbc", (ntor, 2), 3e-5, 5e-4),
    ]
    print(f"\n[{name}] lasym boundary gradient vs naive central FD "
          f"(forward ftol = {cfg.ftol:g}, solver-limited ~1e-4):")
    for out, field, idx, h, tol in checks:
        fd = _fd(name, inp, cfg, p0, field, idx, h)[out]
        ad = float(np.asarray(getattr(jac, field))[row[out], idx[0], idx[1]])
        rel = abs(ad / fd - 1.0)
        print(f"  d({out})/d({field}{idx}) h={h:.0e}: "
              f"AD={ad:+.10e} FD={fd:+.10e} rel={rel:.2e}")
        assert rel <= tol, f"{out}/{field}{idx}: rel {rel:.3e} > {tol:.0e}"


@pytest.mark.full
def test_lasym_adjoint_vs_frozen_path_fd(lasym):
    """Lasym-adjoint correctness lock: the analytic directional derivative and
    the central frozen-path FD, BOTH anchored at the same Newton-refined
    fixed point (frozen residual ~1e-14), agree to solver accuracy —
    isolating the linearization ``dz = -(dF/dz)^{-1} dF/dp t``."""
    name, inp, _cfg, p0, x_star, _, mask = lasym
    ntor = int(inp.ntor)
    # Tight adjoint budget so the frozen-Newton refinement reaches ~1e-14 (the
    # fixture cfg's optimizer-grade 1e-11 would floor it near 1e-10 and leave
    # ~1e-4 FD noise); the fixed point and the resolution are unchanged.
    cfg = im.make_config(inp, ftol=_cfg.ftol, max_iterations=_cfg.max_iterations,
                         adjoint_tol=1e-14, adjoint_maxiter=800)
    im._template_runtime(cfg)  # warm the host template before any trace below
    P = im._dof_projector(cfg, mask)
    edge_mask = im._edge_mask(cfg)
    metric = lambda s, rt: im.mhd_energy(s, rt)[0]  # noqa: E731

    def _norm(t):
        return float(jnp.sqrt(sum(jnp.vdot(v, v).real
                                  for v in jax.tree.leaves(t))))

    def _newton(z, p, F, steps=40, tol=1e-14):
        r0 = max(_norm(F(z, p)), 1.0)
        for _ in range(steps):
            fz = F(z, p)
            if _norm(fz) <= tol * r0:
                break
            _, jvp = jax.linearize(lambda zz: F(zz, p), z)
            step, _ = im._adjoint_solve(jvp, fz, cfg)
            z = jax.tree.map(lambda a, b: a - b, z, step)
        return z

    # Newton-refine the fixed point, then re-freeze F there so the analytic
    # anchor and the FD base point coincide exactly.
    frozen0 = jax.lax.stop_gradient(x_star)
    z_ref = _newton(P(x_star), p0, im.residual_fn(cfg, frozen0, mask))
    rt0 = im.runtime_from_params(p0, cfg)
    frozen = jax.lax.stop_gradient(im._assemble(z_ref, rt0, frozen0, P, edge_mask))
    F = im.residual_fn(cfg, frozen, mask)
    z0 = P(frozen)
    base_res = _norm(F(z0, p0))
    assert base_res < 1e-11, f"refined base residual {base_res:.1e}"

    def Fz(dz):
        return jax.jvp(lambda zz: F(zz, p0), (z0,), (dz,))[1]

    def G(zz, prm):
        rt = im.runtime_from_params(prm, cfg)
        return metric(im._assemble(zz, rt, frozen, P, edge_mask), rt)

    def analytic_dir(tangent):
        b = jax.jvp(lambda prm: F(z0, prm), (p0,), (tangent,))[1]
        dz, _ = im._adjoint_solve(Fz, jax.tree.map(jnp.negative, b), cfg)
        return float(jax.jvp(G, (z0, p0), (P(dz), tangent))[1])

    def frozen_fd(tangent, h=1e-5):
        vals = []
        for sign in (+1.0, -1.0):
            p_h = jax.tree.map(lambda a, d, s=sign: a + s * h * d, p0, tangent)
            zz = _newton(z0, p_h, F)
            rt = im.runtime_from_params(p_h, cfg)
            vals.append(float(metric(im._assemble(zz, rt, frozen, P, edge_mask), rt)))
        return (vals[0] - vals[1]) / (2.0 * h)

    zero = jax.tree.map(jnp.zeros_like, p0)
    print(f"\n[{name}] lasym adjoint vs frozen-path FD at refined fixed point "
          f"(base |F| = {base_res:.1e}):")
    directions = (
        # The zbc direction has the largest O(h^2) truncation coefficient on
        # JAX 0.6.2.  h=5e-6 keeps that error below the adjoint tolerance while
        # the other directions retain the less cancellation-sensitive 1e-5.
        ("zbc", (ntor, 2), 5e-6),
        ("rbs", (ntor, 2), 1e-5),
        ("zbs", (ntor, 1), 1e-5),
    )
    for field, idx, h in directions:
        tangent = dataclasses.replace(
            zero, **{field: getattr(zero, field).at[idx].set(1.0)})
        an = analytic_dir(tangent)
        fd = frozen_fd(tangent, h=h)
        rel = abs(an / fd - 1.0) if fd else abs(an - fd)
        print(f"  d(wb)/d({field}{idx}), h={h:.0e}: analytic={an:+.9e} "
              f"frozen-FD={fd:+.9e} rel={rel:.2e}")
        assert rel <= 5e-5, f"{field}{idx}: adjoint vs frozen-path FD rel {rel:.2e}"


# ===========================================================================
# 3D lasym (non-stellarator-symmetric AND toroidally varying, ntor > 0)
# ===========================================================================
#
# Same machinery as the 2D lane but with a genuinely toroidal boundary
# (``basic_non_stellsym_simsopt``: nfp = 1, mpol = 2, ntor = 2, converges
# fixed-boundary at ns = 11 / ftol = 1e-12).  ``wb``/``aspect`` match the
# frozen-path central FD on the n = 1 dofs to ~1e-6..1e-8 — the
# FD-validation that let the ``optimize`` ``ntor > 0`` guard be removed.

LASYM3D_CASE = dict(ns=11, ftol=1e-12, max_iterations=4000)


@pytest.fixture(scope="module")
def lasym_3d():
    """Converged fixed-boundary 3D lasym case (basic_non_stellsym_simsopt)."""
    inp0 = VmecInput.from_file(str(DATA_DIR / "input.basic_non_stellsym_simsopt"))
    inp = dataclasses.replace(
        inp0,
        ns_array=np.array([LASYM3D_CASE["ns"]]),
        ftol_array=np.array([LASYM3D_CASE["ftol"]]),
        niter_array=np.array([LASYM3D_CASE["max_iterations"]]),
    )
    assert bool(inp.lasym) and int(inp.ntor) > 0          # genuinely 3D lasym
    ntor = int(inp.ntor)
    # nonzero asymmetric families at a toroidal (n > 0) harmonic
    assert float(np.max(np.abs(np.asarray(inp.rbs)[ntor + 1:, :]))) > 0.0
    assert float(np.max(np.abs(np.asarray(inp.zbc)[ntor + 1:, :]))) > 0.0
    cfg = im.make_config(inp, ftol=LASYM3D_CASE["ftol"],
                         max_iterations=LASYM3D_CASE["max_iterations"])
    p0 = im.params_from_input(inp)
    result = solver.solve(inp, cfg.resolution, ftol=cfg.ftol,
                          max_iterations=cfg.max_iterations, mode="cli")
    assert result.converged
    rt = im.runtime_from_params(p0, cfg)
    mask = im._dof_mask(result.state, rt, cfg)
    return "basic_non_stellsym_simsopt", inp, cfg, p0, result.state, rt, mask


@pytest.mark.full
def test_lasym_3d_gradient_vs_frozen_path_fd(lasym_3d):
    """``jax.grad`` through ``im.run`` on a 3D lasym boundary vs frozen-path
    central FD: toroidal (n = 1) rbs/zbc dofs plus the symmetric rbc family
    as a regression guard; measured agreement ~1e-6..1e-8 (printed)."""
    name, inp, cfg, p0, _, _, _ = lasym_3d
    ntor = int(inp.ntor)

    def outs(p):
        sol = im.run(inp, p, ftol=cfg.ftol, max_iterations=cfg.max_iterations)
        return jnp.stack([sol.wb, sol.aspect])

    jac = jax.jacrev(outs)(p0)
    row = {"wb": 0, "aspect": 1}
    metric = {"wb": lambda s, rt: im.mhd_energy(s, rt)[0], "aspect": im.aspect_ratio}
    zero = jax.tree.map(jnp.zeros_like, p0)

    # (metric, family, (n_row, m), FD step, rel tolerance).  n_row = ntor + 1
    # is the toroidal n = +1 harmonic.  rbs/zbc are the asymmetric families
    # (the point of this lane); rbc is the symmetric-path regression guard.
    checks = [
        ("wb", "rbs", (ntor + 1, 1), 3e-5, 5e-6),
        ("aspect", "rbs", (ntor + 1, 1), 3e-5, 5e-6),
        ("wb", "zbc", (ntor + 1, 1), 3e-5, 5e-6),
        ("wb", "rbc", (ntor + 1, 1), 3e-5, 5e-5),
    ]
    print(f"\n[{name}] 3D lasym adjoint vs frozen-path FD "
          f"(ntor={ntor}, forward ftol={cfg.ftol:g}):")
    for out, field, idx, h, tol in checks:
        tangent = dataclasses.replace(
            zero, **{field: getattr(zero, field).at[idx].set(1.0)})
        fd, info = im.frozen_path_directional_fd(p0, cfg, metric[out], tangent, h=h)
        ad = float(np.asarray(getattr(jac, field))[row[out], idx[0], idx[1]])
        res = max(info["newton_res"])
        rel = abs(ad / fd - 1.0) if fd else abs(ad - fd)
        print(f"  d({out})/d({field}[n={idx[0] - ntor:+d},m={idx[1]}]) h={h:.0e}: "
              f"AD={ad:+.9e}  frozen-FD={fd:+.9e}  rel={rel:.2e}  "
              f"(Newton res {res:.0e})")
        assert res < 1e-8, f"{field}{idx}: frozen solve not converged (res {res:.1e})"
        assert rel <= tol, (
            f"{out}/{field}[n={idx[0] - ntor},m={idx[1]}]: 3D lasym adjoint vs "
            f"frozen-path FD rel {rel:.2e} > {tol:.0e}")


@pytest.mark.full
def test_lasym_3d_stability_gradients_vs_frozen_path_fd(lasym_3d):
    """LASYM Mercier/Glasser derivatives include the RBS/ZBC families."""
    _assert_stability_gradients(
        lasym_3d, include_profiles=False, metric_names=("DMerc", "D_R"),
        direction_labels={"RBS(1,1)", "ZBC(1,1)"})
