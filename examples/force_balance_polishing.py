#!/usr/bin/env python
"""Polish the bundled shaped tokamak and compare the exported WOUT files.

VMEX first converges the ordinary VMEC discretization.  The optional polish
then solves the higher-order strong force-balance residual and certifies the
result with independent force, radial-refinement, and positive-Jacobian
checks.  A certified equilibrium is exported by sampling the native state on
a denser radial mesh, where the WOUT reconstruction can carry the polish
gain; ``solve_file`` writes that file directly.
"""

from pathlib import Path

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "input.shaped_tokamak_pressure_polished"
)
OUT_DIR = Path("output_force_balance_polishing")

# --------------------------- solve -----------------------------------------
# solve_file reads the VMEX-only directive in the input deck; because the
# polish certifies, the WOUT it writes samples the native state on the dense
# export mesh.  VmecInput itself contains physics only, so solve_multigrid
# requires an explicit Python flag.
inp = vj.VmecInput.from_file(INPUT_FILE)
result = vj.solve_file(
    INPUT_FILE, write_wout=True, outdir=OUT_DIR, verbose=True
)
if result.polished_state is None or result.polish_report is None:
    raise RuntimeError("the input deck did not request force-balance polishing")

report = result.polish_report
if not report.converged:
    raise RuntimeError(
        f"the polish did not certify: {report.termination_reason}"
    )
# The same independent force oracle evaluated the legacy state and the
# certified polished state; these are the certificate numbers.
print(
    "\nindependent strong-force certificate: "
    f"{report.initial_normalized_l2:.3e} -> "
    f"{report.final_normalized_l2:.3e}"
)
print(
    f"polish work: {report.nonlinear_iterations} nonlinear iterations, "
    f"{report.solve_seconds:.2f} s"
)

# --------------------------- save ------------------------------------------
# solve_file already wrote the certified polished WOUT; export the ordinary
# VMEC state alongside it for the before/after comparison.
legacy_path = vj.write_wout(
    OUT_DIR / "wout_shaped_tokamak_before_polish.nc",
    vj.wout_from_state(
        inp=inp,
        state=result.state,
        fsqr=float(result.fsqr),
        fsqz=float(result.fsqz),
        fsql=float(result.fsql),
        niter=int(result.iterations),
        converged=bool(result.converged),
    ),
)
polished_path = OUT_DIR / "wout_shaped_tokamak_pressure_polished.nc"
print(f"wrote {legacy_path}\nusing {polished_path}")

# --------------------------- plot ------------------------------------------
# The summary's radial force-balance panel drops between the two stages.
for stage, path in (("before", legacy_path), ("after", polished_path)):
    stage_dir = OUT_DIR / stage
    for figure_path in vj.plot_wout(path, stage_dir).values():
        print(f"wrote {figure_path}")
