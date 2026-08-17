#!/usr/bin/env python
"""Calculate effective ripple from a VMEX equilibrium with NEO_JAX."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from neo_jax import NeoConfig

import vmex as vj
from vmex import optimize as opt

DATA = Path(__file__).resolve().parent / "data" / "input.LandremanPaul2021_QA_lowres"
SURFACES = np.linspace(0.15, 0.95, 5)
# Increase these NEO controls for publication calculations; this compact set
# keeps the example responsive while retaining the radial trend. Set
# max_rational_field_periods=0 only when an unlimited exact rational correction
# is required; a near-rational surface can otherwise take a very long time.
NEO_CONFIG = NeoConfig(theta_n=24, phi_n=24, npart=12, multra=1, no_bins=20,
    nstep_per=6, nstep_min=30, nstep_max=60, acc_req=0.1,
    max_rational_field_periods=100000)

print("Solving the VMEX equilibrium...")
equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(DATA))
print("Transforming to Boozer coordinates and calculating epsilon_eff^(3/2)...")
s, epsilon_effective_3_2 = vj.epsilon_effective_from_wout(
    equilibrium.wout, surfaces=SURFACES, config=NEO_CONFIG)
print("s =", np.asarray(s))
print("epsilon_eff^(3/2) =", np.asarray(epsilon_effective_3_2))

figure, axis = plt.subplots(figsize=(5.2, 3.8))
axis.semilogy(s, epsilon_effective_3_2, "o-")
axis.set(xlabel=r"$s=\psi/\psi_b$", ylabel=r"$\epsilon_{\rm eff}^{3/2}$")
axis.grid(alpha=0.25, which="both")
figure.tight_layout(); figure.savefig("epsilon_effective.png", dpi=200); plt.close(figure)
print("Wrote epsilon_effective.png")
