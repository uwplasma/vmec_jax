"""Regenerate the independent Pleiades two-coil mirror reference.

Set ``PLEIADES_ROOT`` below to a checkout of
``github.com/eepeterson/pleiades`` at commit
``0161abb3e9a1d85143c650f068ec524d672fc9ab``, then run this file directly.
The output belongs under ignored ``results/``; review it before deliberately
updating ``examples/data/pleiades_two_coil_beta_reference.csv``.
"""

import collections
import collections.abc
import contextlib
import io
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Inputs
PLEIADES_ROOT = Path("/path/to/pleiades")
RESOLUTIONS = ((31, 61), (41, 81), (51, 101))
BETAS = (0.01, 0.03, 0.10)
OUTPUT_DIR = Path("results/pleiades_mirror_reference")

if not (PLEIADES_ROOT / "pleiades" / "eq_solve.py").is_file():
    raise SystemExit("Set PLEIADES_ROOT at the top of this file to a Pleiades checkout")
sys.path.insert(0, str(PLEIADES_ROOT))
collections.Iterable = collections.abc.Iterable  # Pleiades 2021 compatibility with Python 3.10+

from pleiades import ArbitraryPoints, RectMesh, compute_equilibrium  # noqa: E402
from pleiades.analysis import get_gpsi  # noqa: E402
from pleiades.fields import compute_greens  # noqa: E402

MU0 = 4.0e-7 * np.pi
rows = []
for nr, nz in RESOLUTIONS:
    mesh = RectMesh(rmin=0.0, rmax=0.5, nr=nr, zmin=-0.8, zmax=0.8, nz=nz)
    radius, z = mesh.R, mesh.Z
    coils = ArbitraryPoints(np.asarray([[0.9, -1.0], [0.9, 1.0]]), current=2.0e5)
    coils.mesh = mesh
    vacuum_flux = np.asarray(coils.psi()).reshape(radius.shape)
    center = int(np.argmin(np.abs(z[:, 0])))
    vacuum_axis_field = float(np.asarray(coils.BZ()).reshape(radius.shape)[center, 0])
    with contextlib.redirect_stdout(io.StringIO()):
        plasma_green = get_gpsi(radius, z)
    for beta in BETAS:
        pressure0 = beta * vacuum_axis_field**2 / (2.0 * MU0)

        def pressure(radial_position, pressure0=pressure0):
            return pressure0 * (1.0 - (radial_position / 0.25) ** 2) if radial_position < 0.25 else 0.0

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _, plasma_currents, _ = compute_equilibrium(
                radius,
                z,
                pressure,
                vacuum_flux,
                plasma_green,
                tol=1.0e-10,
                maxiter=400,
                relax=0.9,
            )
        trace = output.getvalue().strip().splitlines()
        iterations, iteration_error = int(trace[-2]), float(trace[-1])
        current_loops = np.column_stack([radius.ravel(), z.ravel(), plasma_currents.ravel()])
        plasma_axis_field = float(compute_greens(current_loops, np.asarray([[0.0, 0.0]]))[2][0])
        axis_field = vacuum_axis_field + plasma_axis_field
        rows.append((nr, nz, beta, iterations, iteration_error, vacuum_axis_field, axis_field, axis_field / vacuum_axis_field))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
data = np.asarray(rows)
header = "nr,nz,beta,iterations,iteration_error,vacuum_axis_field_T,axis_field_T,field_ratio"
np.savetxt(OUTPUT_DIR / "pleiades_two_coil_beta_reference.csv", data, delimiter=",", header=header, comments="")
fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
for nr, nz in RESOLUTIONS:
    selected = data[(data[:, 0] == nr) & (data[:, 1] == nz)]
    ax.plot(100 * selected[:, 2], selected[:, -1], "o-", lw=1.8, label=f"{nr}x{nz}")
ax.plot(100 * data[: len(BETAS), 2], np.sqrt(1.0 - data[: len(BETAS), 2]), "k--", label=r"$\sqrt{1-\beta}$")
ax.set(xlabel="Central beta [%]", ylabel=r"$B(\beta)/B_{vac}$", title="Pleiades two-coil reference convergence")
ax.grid(alpha=0.22)
ax.legend()
fig.savefig(OUTPUT_DIR / "pleiades_two_coil_beta_reference.png", dpi=180)
print(data)
