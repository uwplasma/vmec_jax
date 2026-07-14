#!/usr/bin/env python
"""True single-stage plasma-boundary + ESSOS-coil optimization (coil-agnostic path).

vmec-jax is coil-agnostic: the coils live in ESSOS (:class:`essos.coils.Coils`)
and the differentiable free-boundary machinery consumes any ``xyz -> B`` field
callable -- no coil code inside vmec-jax at all.  This example runs the *full*
single-stage stellarator problem: it co-optimizes the plasma boundary Fourier
coefficients AND the ESSOS coil-group currents *simultaneously*, driven by ONE
exact gradient that threads through

  * the **implicit-differentiation adjoint** of the fixed-boundary equilibrium
    (boundary dofs -> converged VMEC state -> both the edge rotational transform
    and the virtual-casing plasma field on the *moving* boundary), and
  * a **coil-agnostic Biot-Savart callable** built straight from the ESSOS coil
    filaments (``essos.coils.Coils.gamma`` / ``gamma_dash``), differentiable in
    the coil-group currents,

at the same time.  ``jax.value_and_grad`` of the combined objective is exact and
finite-difference validated (the ``FD-check`` lines below, printed outside the
CI smoke budget): the boundary half comes out of the adjoint, the coil half out
of virtual casing + Biot-Savart, and the coupling -- the coil field is evaluated
on the boundary the plasma solve just produced -- is differentiated too.

Objective (a genuine single-stage functional)::

    J(boundary, currents) = W_BN   * < (B_plasma + B_coil) . n ^2 >   # coil<->plasma
                          + W_IOTA * (iota_edge - iota_*)^2            # a plasma target

Making the virtual-casing plasma field differentiable in the *boundary* (not just
the coils) needs its adaptive quadrature/patch precision frozen to static values
first -- :func:`~vmec_jax.core.freeboundary_diff.plan_vc_precision` selects it
once from the starting boundary; see that module and ``virtual_casing_jax``'s
``PrecisionPlan``.

Two cases are run from a truncated Landreman & Paul (2021) precise-QA deck held
by its 16 ESSOS modular coils: **(A) vacuum** (``am = 0``) and **(B) finite
beta** (parabolic pressure).  Each starts from a *detuned* base-coil current
(+15% on one coil group, so the coil half of the gradient has real leverage) and
asks for a shifted edge ``iota`` (so the boundary half does too); a short scaled
L-BFGS-B descent then decreases ``J`` in both.  The converged initial/final
wout, the boundary, and the fixed ESSOS coil geometry are written to
``output_single_stage_essos_coils_opt/`` so ``benchmarks/make_readme_figures.py``
can draw initial-vs-final without re-optimizing.

Requires the optional ``essos`` and ``virtual_casing_jax`` dependencies.  Honors
``VMEC_JAX_EXAMPLES_CI=1`` (tiny grid, one descent step) for the smoke test.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.optimize

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # exact gradients need float64

import vmec_jax as vj
from vmec_jax.core import freeboundary_diff as FBD
from vmec_jax.core import implicit as im

# --------------------------- parameters ------------------------------------
DATA = Path(__file__).resolve().parent / "data"
COILS_JSON = DATA / "ESSOS_biot_savart_LandremanPaulQA.json"  # ESSOS coil DOFs
INPUT_FILE = DATA / "input.LandremanPaul2021_QA_lowres"       # plasma seed deck
OUT_DIR = Path("output_single_stage_essos_coils_opt")

PHIEDGE = -0.025           # toroidal flux matching the ESSOS coil field [Wb]
MPOL, NTOR = 5, 5          # truncate the LP-QA deck to a compact spectral box
DIOTA = 0.03              # asked-for edge rotational-transform change
W_BN, W_IOTA = 300.0, 1.0  # objective weights (coil<->plasma vs iota target)
DETUNE = np.array([1.15, 1.0, 1.0, 1.0])  # +15% on base-coil group 0 at the start
BOUNDARY_MODES = [(1, 0), (2, 0)]         # (m, n) shaping dofs the optimizer moves
CASES = [("A_vacuum", 0.0), ("B_finite_beta", 1000.0)]  # (name, pres_scale)

CI = os.environ.get("VMEC_JAX_EXAMPLES_CI") == "1"
if CI:  # smoke budget: coarse grid, coarse ns, a couple of descent steps
    NS, NPHI, NTHETA, MAXITER = 12, 8, 8, 2
    SOLVE = dict(ftol=1e-9, max_iterations=4000)
else:
    NS, NPHI, NTHETA, MAXITER = 16, 16, 16, 12
    SOLVE = dict(ftol=1e-11, max_iterations=20000)


# --------------------- differentiable ESSOS coil field ----------------------
def make_coil_field(gamma, gamma_dash, nfp: int, stellsym: bool):
    """A differentiable ``base_currents -> (xyz(...,3) -> B(...,3))`` factory.

    Reproduces ``essos.coils.Coils.B`` (filamentary Biot-Savart, uniform-in-
    parameter quadrature) as a pure JAX callable off the fixed coil filaments
    ``gamma`` and their tangents ``gamma_dash``.  Returns a closure that maps the
    independent base-coil-group currents to the full-field ``xyz -> B`` callable
    the free-boundary objective consumes; the stellarator-symmetry current
    expansion is ESSOS's own, so ``B`` stays differentiable in the base currents.
    """
    from essos.coils import apply_symmetries_to_currents

    G = jnp.asarray(gamma)            # (n_coils, n_seg, 3) filament points
    Gd = jnp.asarray(gamma_dash)      # (n_coils, n_seg, 3) filament tangents

    def _bs_point(pt, cur):           # Biot-Savart B at one Cartesian point
        dR = (pt - G).T
        dB = jnp.cross(Gd.T, dR, axisa=0, axisb=0, axisc=0) / jnp.linalg.norm(dR, axis=0) ** 3
        return jnp.mean(jnp.einsum("i,bai", cur * 1e-7, dB, optimize="greedy"), axis=0)

    def coil_field(base_currents):
        cur = apply_symmetries_to_currents(base_currents, nfp, stellsym)
        return lambda P: jax.vmap(lambda q: _bs_point(q, cur))(P.reshape(-1, 3)).reshape(P.shape)

    return coil_field


# ------------------------------ plasma deck ---------------------------------
def build_input(pres_scale: float) -> vj.VmecInput:
    """Truncated LP-QA fixed-boundary deck; add parabolic pressure for finite beta."""
    inp = vj.VmecInput.from_file(str(INPUT_FILE))
    k = inp.ntor - NTOR
    kw = dict(
        mpol=MPOL, ntor=NTOR,
        rbc=inp.rbc[k:k + 2 * NTOR + 1, :MPOL], zbs=inp.zbs[k:k + 2 * NTOR + 1, :MPOL],
        rbs=inp.rbs[k:k + 2 * NTOR + 1, :MPOL], zbc=inp.zbc[k:k + 2 * NTOR + 1, :MPOL],
        raxis_c=inp.raxis_c[:NTOR + 1], zaxis_s=inp.zaxis_s[:NTOR + 1],
        raxis_s=inp.raxis_s[:NTOR + 1], zaxis_c=inp.zaxis_c[:NTOR + 1],
        phiedge=PHIEDGE, ns_array=[NS], niter_array=[SOLVE["max_iterations"]],
        ftol_array=[SOLVE["ftol"]], lfreeb=False)
    if pres_scale > 0:  # p(s) = pres_scale * (1 - s)
        kw.update(pmass_type="power_series", am=[1.0, -1.0] + [0.0] * 19, pres_scale=pres_scale)
    return dataclasses.replace(inp, **kw)


# ------------------------------- one case -----------------------------------
def run_case(name: str, pres_scale: float, coil_field, base0: np.ndarray, out_dir: Path) -> dict:
    """Single-stage boundary+coil optimization for one case; save wout + metadata."""
    print(f"\n{'=' * 72}\nCASE {name}  (pres_scale={pres_scale})\n{'=' * 72}")
    inp = build_input(pres_scale)
    p0 = im.params_from_input(inp)
    ntor, nb = int(inp.ntor), len(BOUNDARY_MODES)

    # Seed equilibrium: sets the iota target and freezes the VC precision plan.
    t0 = time.time()
    sol0 = im.run(inp, p0, **SOLVE)
    w_init = vj.wout_from_state(inp=inp, state=sol0.state, fsqr=0.0, fsqz=0.0, fsql=0.0, converged=True)
    beta0 = 100.0 * float(w_init.betatotal)
    sd0 = FBD.surface_field_data_from_state(inp, sol0.state, nphi=NPHI, ntheta=NTHETA)
    plan = FBD.plan_vc_precision(sd0, digits=4)
    iota_target = float(sol0.iota_edge) + DIOTA
    print(f"seed: iota_edge={float(sol0.iota_edge):.4f} -> target {iota_target:.4f}, "
          f"beta={beta0:.3f}%, aspect={float(w_init.aspect):.3f}  ({time.time() - t0:.1f}s)")

    # DOF packing:  x = [ R_cos(m,0), Z_sin(m,0) for m in modes,  base currents ].
    r0 = np.array([float(np.asarray(p0.rbc)[ntor, m]) for m, _ in BOUNDARY_MODES])
    z0 = np.array([float(np.asarray(p0.zbs)[ntor, m]) for m, _ in BOUNDARY_MODES])
    x0 = np.concatenate([r0, z0, base0 * DETUNE])

    def unpack(x):
        rbc, zbs = p0.rbc, p0.zbs
        for i, (m, _) in enumerate(BOUNDARY_MODES):
            rbc = rbc.at[ntor, m].set(x[i])
            zbs = zbs.at[ntor, m].set(x[nb + i])
        return dataclasses.replace(p0, rbc=rbc, zbs=zbs), jnp.asarray(x[2 * nb:])

    def obj_aux(x):
        params, base_cur = unpack(x)
        sol = im.run(inp, params, **SOLVE)
        sd = FBD.surface_field_data_from_state(inp, sol.state, nphi=NPHI, ntheta=NTHETA)
        prob = FBD.FreeBoundaryDiffProblem.from_surface_data(sd, digits=4, precision=plan)
        j_bn = prob.bnormal_objective(coil_field(base_cur))
        j_iota = (sol.iota_edge - iota_target) ** 2
        return W_BN * j_bn + W_IOTA * j_iota, (j_bn, j_iota, sol.iota_edge)

    value_and_grad = jax.value_and_grad(obj_aux, has_aux=True)

    def scalar_J(x):  # forward-only J (for the finite-difference gradient check)
        return float(obj_aux(jnp.asarray(x))[0])

    (J0, (jbn0, _jiota0, iota0)), g0 = value_and_grad(jnp.asarray(x0))
    print(f"start: J={float(J0):.4e}  W_BN*<(B.n)^2>={W_BN * float(jbn0):.4e}  "
          f"iota_edge={float(iota0):.4f}  |grad|={float(jnp.linalg.norm(g0)):.3e}")

    # Gradient sanity (skipped under the CI smoke budget): exact AD vs central FD
    # on one boundary mode and one coil current -- the two halves of the gradient.
    if not CI:
        for idx, h, lbl in [(0, 1e-4, "boundary R(1,0)"), (2 * nb, 20.0, "coil0 current")]:
            xp, xm = np.array(x0), np.array(x0)
            xp[idx] += h
            xm[idx] -= h
            fd = (scalar_J(xp) - scalar_J(xm)) / (2 * h)
            ad = float(g0[idx])
            print(f"  FD-check dJ/d[{lbl:15s}] AD={ad:+.6e} FD={fd:+.6e} "
                  f"rel_err={abs(ad - fd) / (abs(fd) + 1e-30):.2e}")

    # Scaled L-BFGS-B: boundary modes are O(0.01-1), currents O(1e5), so optimize
    # in scaled coordinates u (x = x0 + D*u) with a bounded step -- otherwise the
    # trial boundary self-intersects (VmecJacobianError).
    D = np.array([0.02] * nb + [0.02] * nb + [1.0e4] * len(base0))
    bounds = [(-3.0, 3.0)] * len(x0)
    hist: list[tuple[float, float, float, float]] = []

    def scipy_fun(u):
        x = x0 + D * u
        (J, aux), g = value_and_grad(jnp.asarray(x))
        hist.append((float(J), float(aux[0]), float(aux[1]), float(aux[2])))
        return float(J), np.asarray(g, dtype=float) * D  # chain rule dJ/du

    t1 = time.time()
    res = scipy.optimize.minimize(
        scipy_fun, np.zeros_like(x0), jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": MAXITER, "ftol": 1e-14, "gtol": 1e-12})
    wall = time.time() - t1

    xf = x0 + D * res.x
    (Jf, (jbnf, _jiotaf, iotaf)), _ = value_and_grad(jnp.asarray(xf))
    params_f, base_f = unpack(xf)
    sol_f = im.run(inp, params_f, **SOLVE)
    w_final = vj.wout_from_state(inp=inp, state=sol_f.state, fsqr=0.0, fsqz=0.0, fsql=0.0, converged=True)

    # -------------------------- before/after table --------------------------
    print(f"\n  {'metric':<16} {'initial':>13} {'final':>13}")
    print(f"  {'J':<16} {float(J0):>13.4e} {float(Jf):>13.4e}   "
          f"({float(J0) / max(float(Jf), 1e-30):.1f}x)")
    print(f"  {'<(B.n)^2>':<16} {float(jbn0):>13.3e} {float(jbnf):>13.3e}")
    print(f"  {'iota_edge':<16} {float(iota0):>13.4f} {float(iotaf):>13.4f}   "
          f"(target {iota_target:.4f})")
    print(f"  base currents [A] {np.asarray(base0 * DETUNE)} -> {np.asarray(base_f)}")
    print(f"[single_stage] {name}: J {float(J0):.4e} -> {float(Jf):.4e}  "
          f"({float(J0) / max(float(Jf), 1e-30):.1f}x)  beta={beta0:.2f}%  "
          f"in {res.nit} iters, {len(hist)} evals, {wall:.1f}s")

    # ------------------------------ save outputs ----------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    vj.write_wout(out_dir / f"wout_{name}_initial.nc", w_init)
    vj.write_wout(out_dir / f"wout_{name}_final.nc", w_final)
    return dict(
        name=name, pres_scale=pres_scale, beta=beta0,
        J0=float(J0), Jf=float(Jf), ratio=float(J0) / max(float(Jf), 1e-30),
        jbn0=float(jbn0), jbnf=float(jbnf),
        iota0=float(iota0), iotaf=float(iotaf), iota_target=iota_target,
        currents_initial=(base0 * DETUNE).tolist(), currents_final=np.asarray(base_f).tolist(),
        nit=int(res.nit), nev=len(hist), wall=wall)


def main() -> None:
    if not FBD.have_virtual_casing_jax():
        raise SystemExit("needs virtual_casing_jax (pip install -e /path/to/virtual_casing_jax)")
    try:
        from essos.coils import Coils_from_json
    except ImportError as exc:  # pragma: no cover - optional heavy dependency
        raise SystemExit("needs essos (pip install -e /path/to/ESSOS)") from exc
    if not COILS_JSON.exists():
        raise SystemExit(f"missing {COILS_JSON.name} in {DATA}")

    coils = Coils_from_json(str(COILS_JSON))
    coil_field = make_coil_field(coils.gamma, coils.gamma_dash, int(coils.nfp), bool(coils.stellsym))
    base0 = np.asarray(coils.dofs_currents, dtype=float) * float(coils.currents_scale)
    print(f"ESSOS LP-QA coils: {int(np.asarray(coils.currents).shape[0])} filaments "
          f"(nfp={int(coils.nfp)}, stellsym={bool(coils.stellsym)}), "
          f"{base0.size} independent base-coil currents ~ {np.abs(base0).mean():,.0f} A")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "coils_gamma.npy", np.asarray(coils.gamma))  # fixed filament geometry

    results = [run_case(name, ps, coil_field, base0, OUT_DIR) for name, ps in CASES]
    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'#' * 72}\nSUMMARY  (single exact gradient: adjoint boundary + coil virtual casing)\n{'#' * 72}")
    for r in results:
        print(f"  {r['name']:<14}: J {r['J0']:.3e} -> {r['Jf']:.3e}  (x{r['ratio']:.0f})  "
              f"beta={r['beta']:.2f}%  <(B.n)^2> {r['jbn0']:.2e} -> {r['jbnf']:.2e}")
    print(f"wrote {OUT_DIR}/  (initial+final wout, coil geometry, summary.json)")


if __name__ == "__main__":
    main()
