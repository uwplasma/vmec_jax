#!/usr/bin/env python
"""Print a privacy-preserving first-iteration diagnosis for one VMEC input deck.

Usage::

    python tools/diagnose_input.py path/to/input.case

This intentionally stops before a full equilibrium solve or NESTOR call.  Its
default output is safe to share: it omits the input path, every parsed input
value, resolution, profile name, axis value, coefficient, and force magnitude.
Use ``--details`` only for local diagnosis of a non-confidential deck.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

from vmex.core.geometry import half_mesh_jacobian
from vmex.core.input import UnsupportedInputModeError, VmecInput
from vmex.core.solver import (
    _geometry,
    _initial_state,
    evaluate_forces,
    prepare_runtime,
    reguess_initial_axis,
)


def _named_arrays(value: Any, prefix: str = ""):
    if is_dataclass(value):
        for field in fields(value):
            name = f"{prefix}.{field.name}" if prefix else field.name
            yield from _named_arrays(getattr(value, field.name), name)
        return
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        for name in value._fields:
            child = f"{prefix}.{name}" if prefix else name
            yield from _named_arrays(getattr(value, name), child)
        return
    try:
        array = np.asarray(value)
    except Exception:
        return
    if np.issubdtype(array.dtype, np.number):
        yield prefix, array


def _nonfinite_names(value: Any, prefix: str) -> list[str]:
    return [name for name, array in _named_arrays(value, prefix) if not np.isfinite(array).all()]


def _range(value: Any) -> str:
    array = np.asarray(value, dtype=float)
    if not array.size:
        return "empty"
    if not np.isfinite(array).all():
        return "non-finite"
    return f"[{array.min():.6e}, {array.max():.6e}]"


def _ok(value: Any) -> bool:
    """Convert a scalar JAX/NumPy health flag to a host boolean."""
    return bool(np.asarray(value))


def _print_header() -> None:
    print("VMEX first-iteration diagnostic (shareable, input details redacted)")
    print(
        "runtime: "
        f"jax={jax.__version__} backend={jax.default_backend()} "
        f"x64={bool(jax.config.jax_enable_x64)} "
        f"VMEX_FAST_COMPILE={os.environ.get('VMEX_FAST_COMPILE', '<default>')}"
    )


def diagnose(path: Path, *, details: bool = False) -> int:
    _print_header()
    try:
        inp = VmecInput.from_file(path)
    except UnsupportedInputModeError as exc:
        print("input parsing: PASS")
        print("input physics mode supported: FAIL")
        print(f"assessment: {exc.code}")
        return 1
    except ValueError:
        # Keep the default diagnostic shareable: parsing exceptions can quote
        # filenames, namelist designators, and input-derived values.
        print("input parsing: FAIL")
        print("assessment: D00C_INPUT_PARSE_ERROR")
        return 1

    print("input parsing: PASS")

    rt = prepare_runtime(inp)
    theta_floor = 2 * int(inp.mpol) + 6
    zeta_floor = 1 if int(inp.ntor) == 0 else 2 * int(inp.ntor) + 4
    angular_grid_ok = (
        (int(inp.ntheta) <= 0 or int(inp.ntheta) >= theta_floor)
        and (int(inp.nzeta) <= 0 or int(inp.nzeta) >= zeta_floor)
    )
    state = _initial_state(rt.setup)
    _, geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=rt.setup.s_full)
    gc, residuals, diagnostics = evaluate_forces(state, rt)
    initial_sqrt_g = np.asarray(jacobian.sqrt_g)[1:]
    initial_jacobian_good = (
        np.isfinite(initial_sqrt_g).all()
        and np.all(initial_sqrt_g != 0.0)
        and not bool(jacobian.jacobian_sign_changed)
    )
    initial_fsq = np.asarray(
        [residuals.fsqr, residuals.fsqz, residuals.fsql], dtype=float
    )
    high_first_force = (
        initial_jacobian_good
        and np.isfinite(initial_fsq).all()
        and float(np.sum(initial_fsq)) > 1.0e2
    )
    recovery_required = (
        not initial_jacobian_good
        or (high_first_force and bool(inp.lmove_axis))
    )
    if recovery_required and rt.resolution.ns >= 3:
        # Diagnose the effective first force that VMEC2000/VMEX actually
        # evolve after eqsolve.f's one allowed guess_axis transfer.  The
        # discarded zero-axis/high-force pass contains no shareable solver
        # result and must not hide a later normalization failure.
        rt, state, _ = reguess_initial_axis(rt, state)
        _, geometry = _geometry(state, rt)
        jacobian = half_mesh_jacobian(geometry, s=rt.setup.s_full)
        gc, residuals, diagnostics = evaluate_forces(state, rt)

    axis_supplied = any(
        np.any(np.asarray(value) != 0.0)
        for value in (inp.raxis_c, inp.raxis_s, inp.zaxis_c, inp.zaxis_s)
    )
    setup_bad = _nonfinite_names(rt.setup, "setup")
    raw_bad = _nonfinite_names(residuals, "residual")
    update_bad = (
        _nonfinite_names(gc, "force")
        + _nonfinite_names(diagnostics.preconditioned, "preconditioned")
        + _nonfinite_names(diagnostics.cache, "preconditioner_cache")
    )
    diagnostic_bad = _nonfinite_names(
        (diagnostics.wb, diagnostics.wp, diagnostics.r00, diagnostics.z00),
        "diagnostics",
    )
    force_bad = raw_bad + update_bad + diagnostic_bad
    fsq = (float(residuals.fsqr), float(residuals.fsqz), float(residuals.fsql))
    pre = diagnostics.preconditioned
    health = diagnostics.health

    print("input physics mode supported: PASS")
    print(
        "angular grid meets VMEC automatic-resolution floor: "
        f"{'PASS' if angular_grid_ok else 'FAIL'}"
    )
    sqrt_g = np.asarray(jacobian.sqrt_g)[1:]
    jacobian_finite = np.isfinite(sqrt_g).all()
    jacobian_nonzero = _ok(health.jacobian_nonzero)
    jacobian_good = (
        jacobian_finite
        and jacobian_nonzero
        and not bool(jacobian.jacobian_sign_changed)
    )
    rz_norm_good = (
        _ok(health.volume_valid)
        and _ok(health.energy_scale_valid)
        and _ok(health.force_norm_finite)
    )
    lambda_norm_good = (
        _ok(health.lambda_scale_valid) and _ok(health.lambda_norm_finite)
    )
    raw_sums_good = (
        _ok(health.raw_r_sum_finite)
        and _ok(health.raw_z_sum_finite)
        and _ok(health.raw_lambda_sum_finite)
    )
    print(f"setup arrays finite: {'PASS' if not setup_bad else 'FAIL'}")
    print(
        "initial Jacobian valid: "
        f"{'PASS' if initial_jacobian_good else 'FAIL'}"
    )
    axis_recovery = (
        "REQUIRED" if recovery_required
        else "DISABLED" if high_first_force and not inp.lmove_axis
        else "NOT_REQUIRED"
    )
    print(f"automatic first-pass axis recovery: {axis_recovery}")
    if recovery_required:
        print(
            "post-recovery Jacobian valid: "
            f"{'PASS' if jacobian_good else 'FAIL'}"
        )
    print(f"magnetic field assembly finite: {'PASS' if _ok(health.fields_finite) else 'FAIL'}")
    print(f"R/Z force normalization valid: {'PASS' if rz_norm_good else 'FAIL'}")
    print(f"lambda force normalization valid: {'PASS' if lambda_norm_good else 'FAIL'}")
    print(
        "real-space force kernels finite: "
        f"{'PASS' if _ok(health.pipeline.real_space_finite) else 'FAIL'}"
    )
    print(
        "spectral/scaled force finite: "
        f"{'PASS' if (_ok(health.pipeline.spectral_finite) and _ok(health.pipeline.scaled_finite)) else 'FAIL'}"
    )
    print(f"unnormalized force sums finite: {'PASS' if raw_sums_good else 'FAIL'}")
    print(
        "normalized residual channels finite: "
        f"R={'PASS' if _ok(health.raw_r_residual_finite) else 'FAIL'} "
        f"Z={'PASS' if _ok(health.raw_z_residual_finite) else 'FAIL'} "
        f"L={'PASS' if _ok(health.raw_lambda_residual_finite) else 'FAIL'}"
    )
    if (
        _ok(health.pipeline.radial_solve_finite)
        and _ok(health.pipeline.radial_preconditioner_safe)
    ):
        radial_status = "PASS"
    else:
        radial_status = "FAIL"
    print(
        "preconditioner stages finite: "
        f"cache={'PASS' if _ok(health.cache_finite) else 'FAIL'} "
        f"rhs={'PASS' if _ok(health.pipeline.rhs_finite) else 'FAIL'} "
        f"radial={radial_status} "
        f"lambda={'PASS' if _ok(health.pipeline.preconditioned_finite) else 'FAIL'}"
    )
    print(f"raw force residuals finite: {'PASS' if not raw_bad else 'FAIL'}")
    print(f"preconditioned update finite: {'PASS' if not update_bad else 'FAIL'}")
    print(f"other force diagnostics finite: {'PASS' if not diagnostic_bad else 'FAIL'}")

    if details:
        print("\nLOCAL DETAILS — do not share for a confidential deck")
        print(f"input: {path.resolve()}")
        print(
            "resolution: "
            f"nfp={inp.nfp} mpol={inp.mpol} ntor={inp.ntor} "
            f"ntheta={rt.resolution.ntheta} nzeta={rt.resolution.nzeta} "
            f"ns={rt.resolution.ns} lasym={inp.lasym} lfreeb={inp.lfreeb}"
        )
        print(
            "controls: "
            f"phiedge={inp.phiedge:.17g} delt={inp.delt:.17g} "
            f"gamma={inp.gamma:.17g} ncurr={inp.ncurr} curtor={inp.curtor:.17g} "
            f"precon_type={inp.precon_type!r} "
            f"prec2d_threshold={inp.prec2d_threshold:.3e}"
        )
        print(
            "profiles: "
            f"pmass={inp.pmass_type!r} pcurr={inp.pcurr_type!r} "
            f"piota={inp.piota_type!r} phips={_range(rt.setup.phips)} "
            f"lamscale={float(rt.setup.lamscale):.6e} mass={_range(rt.setup.mass)}"
        )
        print(
            "axis: "
            f"input={'supplied' if axis_supplied else 'missing -> VMEC first-pass recovery'} "
            f"R(0)={float(np.sum(np.asarray(rt.setup.raxis_c))):.6e} "
            f"Z-coeff-max={float(np.max(np.abs(np.asarray(rt.setup.zaxis_s)))):.6e}"
        )
        print(
            "jacobian: "
            f"sign_changed={bool(jacobian.jacobian_sign_changed)} "
            f"sqrt_g(interior)={_range(sqrt_g)} "
            f"zero_count={int(np.count_nonzero(sqrt_g == 0.0))}"
        )
        print(
            "residuals: "
            f"FSQR={fsq[0]:.6e} FSQZ={fsq[1]:.6e} FSQL={fsq[2]:.6e}; "
            f"pre=({float(pre.fsqr1):.6e}, {float(pre.fsqz1):.6e}, "
            f"{float(pre.fsql1):.6e})"
        )
        bad = setup_bad + force_bad
        print("nonfinite arrays: " + (", ".join(bad) if bad else "none"))

    if details:
        likely: list[str] = []
        if inp.phiedge == 0.0 or float(rt.setup.lamscale) == 0.0:
            likely.append(
                "zero effective toroidal flux: check PHIEDGE and APHI; "
                "the force/lambda normalizations are singular"
            )
        if bool(jacobian.jacobian_sign_changed):
            likely.append(
                "bad initial axis/boundary Jacobian: remove the supplied "
                "RAXIS_*/ZAXIS_* to trigger inference, or seed them from the "
                "VMEC2000 output"
            )
        if setup_bad:
            likely.append(
                "a parsed profile/setup array is already non-finite before iteration 1"
            )
        raw_finite = np.isfinite(np.asarray(fsq)).all()
        pre_finite = np.isfinite(np.asarray([
            pre.fsqr1, pre.fsqz1, pre.fsql1,
        ], dtype=float)).all() and not _nonfinite_names(gc, "force")
        if raw_finite and not pre_finite:
            likely.append(
                "raw force residuals are finite but the update is not: the 1D "
                "radial/lambda preconditioner is the first failing stage"
            )
        if inp.gamma == 1.0:
            likely.append("GAMMA=1 makes the printed MHD-energy expression singular")
        if not likely and force_bad:
            likely.append(
                "non-finite values first appear inside the "
                "geometry/field/preconditioner force pass"
            )
        assessment = "; ".join(likely) if likely else "first force pass is finite"
    elif setup_bad:
        assessment = "D01_SETUP_NONFINITE"
    elif not jacobian_good:
        assessment = "D02_INITIAL_JACOBIAN"
    elif not _ok(health.fields_finite):
        assessment = "D03A_MAGNETIC_FIELD_NONFINITE"
    elif not _ok(health.volume_valid):
        assessment = "D03B_DEGENERATE_VOLUME"
    elif not _ok(health.energy_scale_valid):
        assessment = "D03C_ZERO_ENERGY_SCALE"
    elif not _ok(health.force_norm_finite):
        assessment = "D03D_RZ_NORMALIZATION_NONFINITE"
    elif not _ok(health.lambda_scale_valid):
        assessment = "D03E_ZERO_LAMBDA_SCALE"
    elif not _ok(health.lambda_norm_finite):
        assessment = "D03F_LAMBDA_NORMALIZATION_NONFINITE"
    elif not _ok(health.pipeline.real_space_finite):
        assessment = "D03G_REAL_SPACE_FORCE_NONFINITE"
    elif not _ok(health.pipeline.spectral_finite):
        assessment = "D03H_SPECTRAL_FORCE_NONFINITE"
    elif not _ok(health.pipeline.scaled_finite):
        assessment = "D03I_SCALED_FORCE_NONFINITE"
    elif not raw_sums_good:
        assessment = "D03J_UNNORMALIZED_FORCE_SUM_NONFINITE"
    elif not _ok(health.raw_r_residual_finite):
        assessment = "D03K_R_RESIDUAL_NONFINITE"
    elif not _ok(health.raw_z_residual_finite):
        assessment = "D03L_Z_RESIDUAL_NONFINITE"
    elif not _ok(health.raw_lambda_residual_finite):
        assessment = "D03M_LAMBDA_RESIDUAL_NONFINITE"
    elif raw_bad:
        assessment = "D03_RAW_FORCE_NONFINITE"
    elif not _ok(health.cache_finite):
        assessment = "D04A_PRECONDITIONER_CACHE_NONFINITE"
    elif not _ok(health.pipeline.rhs_finite):
        assessment = "D04B_PRECONDITIONER_RHS_NONFINITE"
    elif (
        not _ok(health.pipeline.radial_preconditioner_safe)
    ):
        assessment = "D04E_RADIAL_PRECONDITIONER_REJECTED"
    elif not _ok(health.pipeline.radial_solve_finite):
        assessment = "D04C_RADIAL_PRECONDITIONER_NONFINITE"
    elif not _ok(health.pipeline.preconditioned_finite):
        assessment = "D04D_LAMBDA_PRECONDITIONER_NONFINITE"
    elif update_bad:
        assessment = "D04_PRECONDITIONED_UPDATE_NONFINITE"
    elif diagnostic_bad:
        assessment = "D05_FORCE_DIAGNOSTIC_NONFINITE"
    elif not angular_grid_ok:
        assessment = "W01_ANGULAR_GRID_BELOW_VMEC_DEFAULT"
    else:
        assessment = "OK_FIRST_FORCE_PASS_FINITE"
    print(f"assessment: {assessment}")
    return 1 if setup_bad or force_bad or not jacobian_good else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="VMEC &INDATA or VMEC++ JSON deck")
    parser.add_argument(
        "--details", action="store_true",
        help="print input-derived values for local use only; never share for a confidential deck",
    )
    args = parser.parse_args()
    try:
        return diagnose(args.input, details=args.details)
    except Exception as exc:
        print(f"diagnostic failed: {type(exc).__name__}")
        if args.details:
            print(f"local detail: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
