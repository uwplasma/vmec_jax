"""Implicit-differentiation gradient tests (``vmec_jax.core.implicit``, plan.md §6).

Validated here (no golden fixtures needed — everything is self-referential
against central finite differences through the full host solver):

1. the traceable parameter map ``runtime_from_params`` reproduces
   ``run_setup``/``prepare_runtime`` exactly at the base parameters
   (solovev 2D ncurr=0 and li383 3D ncurr=1/lconm1);
2. the implicit residual vanishes at the converged fixed point;
3. solovev (ns=11, ftol=1e-14) gradients vs central FD, rtol <= 1e-6:
   ``d(wb)/d(RBC(0,1))``, ``d(aspect)/d(RBC(0,1))``, ``d(wb)/d(phiedge)``,
   ``d(wp)/d(pres_scale)``;
4. the adjoint GMRES converges (< 1e-10 relative) within a ~300-matvec
   budget on the preconditioned-residual formulation and is orders of
   magnitude away on the raw-force formulation (informational print — this
   is the value of the 1D preconditioner);
5. one 3D case (li383_low_res, ns=16, forward ftol=1e-13):
   ``d(wb)/d(RBC(0,1))`` vs central FD.  The FD noise floor from the
   iterative forward solver (where exactly the ftol crossing lands) was
   measured at ~3e-5 relative across h in [2e-5, 1e-3]; the assertion uses
   rtol 2e-4 with the measured agreement (~1e-5) printed;
6. the gradient is independent of the iteration policy (max_iterations cap)
   — only the fixed point defines the derivative — with informational
   peak-RSS prints for the O(1)-memory claim (the backward pass costs a
   handful of residual linearizations, never a per-iteration tape);
7. a solver-sensitive metric (li383 ``ncurr=1`` ``d(iota_edge)/d(boundary)``):
   the implicit adjoint equals the *frozen-path* central FD
   (``frozen_path_directional_fd``) to solver accuracy on the m=1 modes,
   where a *naive* full re-solve FD sign-flips (adjoint -0.773 vs naive
   +0.045) — the adjoint is the correct frozen-logic gradient;
8. typed errors through the callback (plan Item I.1): an unconvergeable
   ``im.run`` raises the SHORT typed ``VmecConvergenceError`` (the
   ``_HOST_ERROR`` relay), not a multi-KB ``JaxRuntimeError``;
9. the multigrid lane directly (plan Item I.4): ``im.run(multigrid=True)``
   through a genuine ns 5 -> 11 ladder, ``d(wb)/d(RBC(0,1))`` vs the
   frozen-path FD.

FD steps (documented choices): central differences with the *same* ftol and
iteration policy on both sides, converged outputs cached on disk (``/tmp``)
keyed by (case, parameter, step, ftol).  Steps are chosen per parameter so
truncation ~ h^2 sits above the solver-termination noise ~ eps_wb / h:
solovev h = 3e-5 (boundary), 1e-5 (phiedge), 1e-4 (pres_scale); li383
h = 4e-4 (boundary).
"""

from __future__ import annotations

import dataclasses
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from vmec_jax.core import implicit as im
from vmec_jax.core import solver
from vmec_jax.core.input import VmecInput

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
FD_CACHE = Path(tempfile.gettempdir()) / "vmec_jax_implicit_fd_cache.pkl"


@pytest.fixture(autouse=True, scope="module")
def _jit_enabled():
    """tests/conftest.py disables jit globally for cheap unit tests; the
    implicit-gradient tests run full solves + adjoint GMRES and are ~40x
    slower interpreted (105-160 s/test in CI) — run them jitted."""
    prev = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", prev)

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
    """``d(iota_edge)/d(boundary)`` on the 3D ``ncurr=1`` case — the hard case
    from the collaborator AD-vs-FD feedback.  ``iota`` is derived from the
    current-constrained ``chips``, so the metric reads the converged solver
    state and is *solver-sensitive*: a naive re-solve FD at ``p ± h`` lets the
    convergence logic re-form and gives the wrong answer (measured, ``h=1e-4``):

        RBC(n=-1,m=1):  adjoint -0.77343,  frozen-path FD -0.77343,  naive FD +0.04543  (sign flip)
        RBC(n=+1,m=1):  adjoint -1.26567,  frozen-path FD -1.26567,  naive FD -2.14135  (69% off)

    The implicit adjoint linearizes the *frozen* fixed point; this test locks in
    that it reproduces the frozen-path central FD
    (:func:`im.frozen_path_directional_fd`, Newton-solving the frozen residual
    at ``p ± h``) to solver accuracy — i.e. the adjoint is the correct
    frozen-logic gradient, and the naive FD is deliberately not the reference.
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
    """A failing host solve raises the SHORT typed exception, not callback noise.

    Before the ``_HOST_ERROR`` relay (see the implicit module docstring,
    "Zero-crash typed errors through the callback") an unconvergeable
    ``im.run`` surfaced as a raw ``jax.errors.JaxRuntimeError`` with a
    ~3.7 KB message embedding the whole host traceback and the typed
    :class:`VmecConvergenceError` lost (``__cause__`` was ``None``).  Now the
    original typed exception is stashed by the host callback and re-raised at
    the ``pure_callback`` call site with ``from None``.
    """
    from vmec_jax.core.errors import VmecConvergenceError

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
    """``im.run(multigrid=True)`` gradients FD-validated directly.

    ``cfg.multigrid=True`` routes the host solve through ``solve_multigrid``
    (the lane ``optimize._least_squares_implicit`` hardcodes); only the final
    fixed point defines the derivative, so the adjoint through a genuine
    two-stage ladder (ns 5 -> 11 on solovev) must match the frozen-path
    central FD of ``wb`` along ``RBC(0,1)``.
    """
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
