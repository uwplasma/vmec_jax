#!/usr/bin/env python
"""Polish a finite-beta stellarator and compare the VMEC-grid output.

VMEX first converges the ordinary VMEC discretization.  The optional polish
then solves the higher-order strong force-balance residual and projects its
correction back onto the VMEC mesh for a compatible WOUT file.
"""

from pathlib import Path

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "input.finite_beta_stellarator_polished"
)
OUT_DIR = Path("output_force_balance_polishing")

# --------------------------- solve -----------------------------------------
# solve_file reads the VMEX-only directive in the input deck. VmecInput itself
# contains physics only, so solve_multigrid requires an explicit Python flag.
inp = vj.VmecInput.from_file(INPUT_FILE)
result = vj.solve_file(INPUT_FILE, write_wout=False, verbose=True)
if result.polished_state is None or result.polish_report is None:
    raise RuntimeError("the input deck did not request force-balance polishing")

report = result.polish_report
print(
    "\nrelative strong-force error: "
    f"{report.initial_normalized_l2:.3e} -> "
    f"{report.final_normalized_l2:.3e}"
)
print(
    f"polish work: {report.nonlinear_iterations} nonlinear iterations, "
    f"{report.solve_seconds:.2f} s"
)

# --------------------------- save ------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _wout(state):
    return vj.wout_from_state(
        inp=inp,
        state=state,
        fsqr=float(result.fsqr),
        fsqz=float(result.fsqz),
        fsql=float(result.fsql),
        niter=int(result.iterations),
        converged=bool(result.converged),
    )


legacy_path = vj.write_wout(
    OUT_DIR / "wout_stellarator_before_polish.nc", _wout(result.state)
)
polished_path = vj.write_wout(
    OUT_DIR / "wout_stellarator_after_polish.nc", _wout(result.polished_state)
)
print(f"wrote {legacy_path}\nwrote {polished_path}")

# --------------------------- plot ------------------------------------------
for stage, path in (("before", legacy_path), ("after", polished_path)):
    stage_dir = OUT_DIR / stage
    for figure_path in vj.plot_wout(path, stage_dir).values():
        print(f"wrote {figure_path}")
