#!/usr/bin/env python
"""Restrict a free-boundary equilibrium and test its virtual-casing exterior.

The parent free-boundary solution fills ``0 <= s_free <= 1``. Its exact
``s_free=0.5`` surface becomes a new fixed boundary, with toroidal flux,
pressure gradient, and enclosed current remapped onto ``0 <= s_fixed <= 1``.
The matched ESSOS coil geometry is retained while its four independent
currents are refitted against the prescribed-interface virtual-casing
conditions. The final figure compares the common nested surfaces and the
fixed-boundary-plus-virtual-casing field with the larger free solution.

The exterior fields need not agree exactly: the larger equilibrium contains
plasma current in ``0.5 < s_free <= 1``, whereas the restricted problem treats
that region as vacuum. Boundary-normal matching on one interface also does
not uniquely reconstruct those removed volume currents. The comparison makes
that physical limitation measurable instead of treating the two problems as
mathematically identical.
"""

from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import profiles, virtual_casing as vc
from vmex.core.extender import VmecInteriorField
from vmex.core.plotting import surface_rz
from vmex.core.solver import prepare_runtime, resolution_from_input

from essos.coils import Coils
from essos.fields import BiotSavart

DATA = Path(__file__).resolve().parent / "data"
# The bundled coils were optimized at 2.5% beta. A short continuation reaches
# 2.625% while retaining a converged parent free boundary at full resolution;
# larger beta requires reoptimizing the coil field, not only scaling profiles.
BETA_SCALE = 1.05
S_FIXED, NS, MPOL, NTOR, NITER, FTOL = 0.5, 31, 5, 5, 5000, 1.0e-6
NPHI, NTHETA, VC_DIGITS, COIL_MAXITER = 24, 24, 4, 20
CHECK_LEVELS = ((128, 128), (256, 256), (512, 512))
NORMAL_WEIGHT, PRESSURE_WEIGHT, CURRENT_REGULARIZATION = 2.0e4, 5.0e3, 1.0e-3
CURRENT_SCALE, CURRENT_BOUND = 0.5, 1.0  # 0.5 ESSOS units = 50 kA per independent coil
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    NS, MPOL, NTOR, NITER, FTOL = 11, 3, 3, 3500, 1.0e-7
    NPHI, NTHETA, VC_DIGITS, COIL_MAXITER = 8, 8, 3, 1
    CHECK_LEVELS = ((64, 64), (128, 128), (256, 256))

print(f"Loading the QA target at about {2.5 * BETA_SCALE:.1f}% beta and its matched ESSOS coils...")
coils0 = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA_beta2p5_bootstrap.json"))

def coil_field(coils):
    biot_savart = BiotSavart(coils)
    return lambda points: jax.vmap(biot_savart.B)(
        points.reshape(-1, 3)).reshape(points.shape)

coil_B0 = jax.jit(coil_field(coils0))

def coil_B_numpy(points):
    return np.concatenate([np.asarray(coil_B0(block))
                           for block in np.array_split(np.asarray(points), 32)])

print("Tabulating the coil field in memory and solving the larger free boundary...")
mgrid = vj.MgridField.from_cartesian_field(
    coil_B_numpy, rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6,
    ir=48, jz=48, kp=16, nfp=coils0.nfp)
base_input = vj.VmecInput.from_file(
    DATA / "input.LandremanPaul2021_QA_beta2p5_bootstrap").change_resolution(
        mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=16)
base_input = replace(base_input, delt=0.35, ns_array=np.array([NS]),
    niter_array=np.array([NITER]), ftol_array=np.array([FTOL]))

def scaled_parent(scale, *, free):
    return replace(base_input, pres_scale=scale * base_input.pres_scale,
        curtor=scale * base_input.curtor, lfreeb=free,
        mgrid_file="ESSOS field (in memory)" if free else "NONE")

# A converged fixed-boundary state is a useful initial interior, but VMEX must
# still turn on NESTOR before accepting it as a free-boundary equilibrium.
fixed_parent = opt.solve_equilibrium(scaled_parent(1.0, free=False))
free_input = scaled_parent(1.0, free=True)
free_result = vj.solve_free_boundary_multigrid(
    free_input, external_field=mgrid, initial_state=fixed_parent.state, verbose=True)
if BETA_SCALE != 1.0:
    free_input = scaled_parent(BETA_SCALE, free=True)
    free_result = vj.solve_free_boundary_multigrid(
        free_input, external_field=mgrid, initial_state=free_result.state, verbose=True)
if not free_result.converged:
    raise RuntimeError("the parent free-boundary continuation did not converge")
free_wout = vj.wout_from_state(inp=free_input, state=free_result.state,
    fsqr=float(free_result.fsqr), fsqz=float(free_result.fsqz), fsql=float(free_result.fsql),
    niter=int(free_result.iterations), converged=bool(free_result.converged),
    vacuum_output=free_result.vacuum)

# Use the physical Fourier coefficients on s_free=0.5. VmecInput is immutable,
# so new coefficient arrays are filled explicitly before replace() constructs the deck.
surface_index = int(round(S_FIXED * (NS - 1)))
rbc, zbs = np.zeros_like(free_input.rbc), np.zeros_like(free_input.zbs)
rbs, zbc = np.zeros_like(free_input.rbs), np.zeros_like(free_input.zbc)
for mode, (m, xn) in enumerate(zip(free_wout.xm.astype(int), free_wout.xn)):
    n = int(round(xn / free_wout.nfp))
    if m < MPOL and abs(n) <= NTOR:
        rbc[n + NTOR, m] = free_wout.rmnc[surface_index, mode]
        zbs[n + NTOR, m] = free_wout.zmns[surface_index, mode]
        if free_input.lasym:
            rbs[n + NTOR, m] = free_wout.rmns[surface_index, mode]
            zbc[n + NTOR, m] = free_wout.zmnc[surface_index, mode]

# t=s_fixed maps to s_free=S_FIXED*t. Subtracting p(S_FIXED) preserves grad(p)
# in the retained plasma while making the artificial new LCFS pressure zero.
knots = np.linspace(0.0, 1.0, 25); source_s = S_FIXED * knots
pressure = np.asarray(profiles.pressure(free_input.pmass_type, free_input.am,
    free_input.am_aux_s, free_input.am_aux_f, source_s,
    pres_scale=free_input.pres_scale, spres_ped=free_input.spres_ped))
pressure = pressure - pressure[-1]
current = np.asarray(profiles.current(free_input.pcurr_type, free_input.ac,
    free_input.ac_aux_s, free_input.ac_aux_f, source_s))
current_edge = float(profiles.current(free_input.pcurr_type, free_input.ac,
    free_input.ac_aux_s, free_input.ac_aux_f, 1.0))
fixed_input = replace(free_input, lfreeb=False, mgrid_file="NONE", rbc=rbc, zbs=zbs,
    rbs=rbs, zbc=zbc, phiedge=float(free_wout.phi[surface_index]),
    pmass_type="cubic_spline", am_aux_s=knots, am_aux_f=pressure, pres_scale=1.0,
    pcurr_type="cubic_spline_i", ac_aux_s=knots, ac_aux_f=current,
    curtor=float(free_input.curtor * current[-1] / current_edge),
    raxis_c=np.asarray(free_wout.raxis_cc)[:NTOR + 1],
    zaxis_s=np.asarray(free_wout.zaxis_cs)[:NTOR + 1])
print(f"Solving the restricted fixed boundary at s_free={S_FIXED:.2f}...")
fixed_equilibrium = opt.solve_equilibrium(fixed_input, verbose=True)

print("Refitting the four independent ESSOS coil currents with virtual casing...")
surface_data = vc.surface_field_data_from_state(fixed_input, fixed_equilibrium.solution,
    runtime=fixed_equilibrium.solver_context, nphi=NPHI, ntheta=NTHETA)
precision = vc.plan_vc_precision(surface_data, digits=VC_DIGITS)
interface = vc.PlasmaVacuumInterface.from_surface_data(
    surface_data, digits=VC_DIGITS, precision=precision)
B_scale = jnp.sqrt(jnp.sum(interface.weights * jnp.sum(surface_data.B_total**2, axis=0)))
all_dofs0 = jnp.asarray(coils0.dofs); n_current = coils0.dofs_currents.size
current_dofs0 = all_dofs0[-n_current:]

def coils_from_u(u):
    dofs = all_dofs0.at[-n_current:].set(current_dofs0 + CURRENT_SCALE * jnp.asarray(u))
    return coils0.with_dofs(dofs)

def coil_objective(u):
    external = coil_field(coils_from_u(u))
    normal = jnp.sqrt(interface.weights) * interface.bnormal_residual(external) / B_scale
    pressure_jump = (jnp.sqrt(interface.weights) * interface.pressure_balance_residual(external)
                     / B_scale**2)
    return (0.5 * NORMAL_WEIGHT * jnp.vdot(normal, normal)
            + 0.5 * PRESSURE_WEIGHT * jnp.vdot(pressure_jump, pressure_jump)
            + 0.5 * CURRENT_REGULARIZATION * jnp.vdot(u, u))

coil_value_and_grad = jax.jit(jax.value_and_grad(coil_objective))
u0 = np.zeros(n_current); initial_coil_cost = float(coil_objective(u0))
coil_result = minimize(coil_value_and_grad, u0, jac=True, method="L-BFGS-B",
    bounds=[(-CURRENT_BOUND, CURRENT_BOUND)] * n_current,
    options={"maxiter": COIL_MAXITER, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8})
coils = coils_from_u(coil_result.x); external_field = jax.jit(coil_field(coils))
B_surface = interface.total_B_out(external_field); Bmag_surface = jnp.linalg.norm(B_surface, axis=0)
Bn_over_B = interface.bnormal_residual(external_field) / Bmag_surface
print(f"Coil-current cost {initial_coil_cost:.3e} -> {float(coil_result.fun):.3e}; "
      f"B.n/B RMS={100 * float(jnp.sqrt(jnp.sum(interface.weights * Bn_over_B**2))):.3f}%, "
      f"max={100 * float(jnp.max(jnp.abs(Bn_over_B))):.3f}%")

# The parent solution is the total field in the outer plasma-filled region. The
# restricted solution is its fixed interior plus virtual casing and refitted coils.
free_runtime = prepare_runtime(free_input, resolution_from_input(free_input, ns=NS))
free_field = VmecInteriorField.from_state(free_input, free_result.state, runtime=free_runtime)
sample_s = jnp.linspace(0.0, 1.0, 21)
sample_theta = jnp.linspace(0.0, 2 * jnp.pi, 13)[:-1]
sample_phi = jnp.array([0.0, jnp.pi / (2 * free_input.nfp)])
flux_points = jnp.array([[s, theta, phi] for s in sample_s
                         for phi in sample_phi for theta in sample_theta])
free_field.set_points_flux(flux_points); xyz = free_field.get_points_cart(); B_free = free_field.B()
points_per_surface = len(sample_theta) * len(sample_phi)
inner_s = sample_s[sample_s <= S_FIXED]
fixed_flux_points = jnp.array([[s / S_FIXED, theta, phi] for s in inner_s
                               for phi in sample_phi for theta in sample_theta])
fixed_equilibrium.set_points_flux(fixed_flux_points)
B_fixed_inside = fixed_equilibrium.B()
fixed_exterior = fixed_equilibrium.exterior_field(external_field=external_field,
    nphi=NPHI, ntheta=NTHETA, digits=VC_DIGITS)
fixed_exterior = fixed_exterior.with_near_surface_continuation(
    digits=VC_DIGITS, precision=precision, B_surface=interface.B_plasma)
outer_xyz = xyz[len(inner_s) * points_per_surface:]
print("Evaluating the virtual-casing continuation through the outer region...")
B_fixed_exterior = fixed_exterior.B(outer_xyz)
B_comparison = jnp.concatenate((B_fixed_inside, B_fixed_exterior))
direct_check = fixed_equilibrium.exterior_field(external_field=external_field,
    nphi=NPHI, ntheta=NTHETA, digits=VC_DIGITS, levels=CHECK_LEVELS)
B_direct_check = direct_check.B(outer_xyz[-1:])
continuation_check = B_fixed_exterior[-1:]
continuation_error = (jnp.linalg.norm(continuation_check - B_direct_check)
                      / jnp.linalg.norm(B_direct_check))
finite = jnp.all(jnp.isfinite(B_free) & jnp.isfinite(B_comparison), axis=1)
point_error = (jnp.linalg.norm(B_comparison - B_free, axis=1)
               / jnp.linalg.norm(B_free, axis=1))
point_error = jnp.where(finite, point_error, jnp.nan).reshape(len(sample_s), -1)
radial_field_error = jnp.sqrt(jnp.nanmean(point_error**2, axis=1))
print(f"Median |B| [T]: parent={float(jnp.nanmedian(jnp.linalg.norm(B_free, axis=1))):.3f}, "
      f"restricted={float(jnp.nanmedian(jnp.linalg.norm(B_comparison, axis=1))):.3f}, "
      f"coils={float(jnp.nanmedian(jnp.linalg.norm(external_field(xyz), axis=1))):.3f}")
print(f"Radial field-comparison RMS errors = {np.asarray(radial_field_error)}")
print(f"Continuation/direct-VC difference at the far check point = "
      f"{100 * float(continuation_error):.2f}%")

# Plot eleven parent surfaces and the corresponding restricted surfaces through
# s_free=0.5. The two black dash patterns keep coincident comparisons visible;
# outer contours mark a field-comparison region, not extra equilibrium surfaces.
theta = np.linspace(0.0, 2 * np.pi, 361); free_surfaces = np.linspace(0.0, 1.0, 11)
figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), width_ratios=(0.85, 1.15),
                           constrained_layout=True)
surface_errors = []; surface_curves = []
for s_free in free_surfaces:
    j_free = int(round(s_free * (NS - 1)))
    R_free, Z_free = surface_rz(free_wout, s_index=j_free, theta=theta, phi=np.array([0.0]))
    surface_curves.append((s_free, R_free, Z_free))
for s_free, R_free, Z_free in surface_curves:
    if s_free > S_FIXED:
        axes[0].plot(R_free[:, 0], Z_free[:, 0], color="k", ls=(0, (1.5, 1.2)),
                     lw=2.8, alpha=0.72)
    axes[0].plot(R_free[:, 0], Z_free[:, 0], color="#D62728", lw=1.1)
    if s_free <= S_FIXED + 1.0e-12:
        s_fixed = s_free / S_FIXED; j_fixed = int(round(s_fixed * (NS - 1)))
        R_fixed, Z_fixed = surface_rz(
            fixed_equilibrium.wout, s_index=j_fixed, theta=theta, phi=np.array([0.0]))
        axes[0].plot(R_fixed[:, 0], Z_fixed[:, 0], color="k", ls="--", lw=2.0)
        surface_errors.append(float(np.sqrt(np.mean(
            (R_fixed - R_free) ** 2 + (Z_fixed - Z_free) ** 2))))
axes[0].set(xlabel="R [m]", ylabel="Z [m]", title=r"Free solution and restricted $s=0.5$ plasma")
axes[0].set_aspect("equal"); axes[0].grid(alpha=0.25)
axes[0].legend(handles=[
    Line2D([], [], color="#D62728", lw=1.2, label=r"free: $0\leq s\leq1$"),
    Line2D([], [], color="k", lw=2.0, ls="--", label=r"fixed: $0\leq s_{free}\leq0.5$"),
    Line2D([], [], color="k", lw=2.8, ls=(0, (1.5, 1.2)),
           label="fixed+VC comparison region"),
], fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
axes[1].semilogy(np.asarray(sample_s), np.asarray(radial_field_error), "o-", color="#0072B2")
axes[1].set(xlabel=r"parent $s_{free}$", ylabel=r"RMS $|B_{fixed+VC}-B_{free}|/|B_{free}|$",
            title="Fixed interior and fixed+VC region", xlim=(0.0, 1.0))
axes[1].axvline(S_FIXED, color="0.35", linestyle="--", linewidth=1.0)
axes[1].grid(alpha=0.25, which="both")
figure.savefig("vmex_fixed_free_boundary_comparison.png", dpi=200)
plt.close(figure)
print(f"Common-surface RMS errors [m] = {np.asarray(surface_errors)}")
outer_error = point_error[np.asarray(sample_s) > S_FIXED]
print(f"Outer-region pointwise error: median={float(jnp.nanmedian(outer_error)):.3e}, "
      f"95th percentile={float(np.nanpercentile(np.asarray(outer_error), 95)):.3e}")
print("Wrote vmex_fixed_free_boundary_comparison.png")
