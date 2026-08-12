import os
number_of_processors_to_use = 8 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import Vmec
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.dynamics import Tracing, Particles
import numpy as np

# Input parameters
tmax = 3e-4
timestep = 5.e-7
times_to_trace=200
nparticles_per_core=25
nparticles = number_of_processors_to_use*nparticles_per_core
n_particles_to_plot = 4
s = 0.25 # s-coordinate: flux surface label
energy=FUSION_ALPHA_PARTICLE_ENERGY
seed = 42
theta_key, phi_key, pitch_key = random.split(random.PRNGKey(seed), 3)
theta = random.uniform(theta_key, (nparticles,), minval=0, maxval=2*jnp.pi)
phi = random.uniform(phi_key, (nparticles,), minval=0, maxval=2*jnp.pi/2/3)
initial_vparallel_over_v = random.uniform(
    pitch_key, (nparticles,), minval=-1, maxval=1
)

# Load coils and field
# Run `vmex --scale wout_QP_optimized.nc` first.
wout_file = os.path.join(os.path.dirname(__file__),
                          # "wout_nfp3_final_scaled.nc")
                          #"wout_QI_stel_11771_A150_final_post_optimization.nc")
                          # "wout_nfp2_circular_final_scaled.nc"
                          "wout_QP_optimized_scaled.nc"
                          )
vmec = Vmec(wout_file)

# Initialize particles
initial_xyz=jnp.array([[s]*nparticles, theta, phi]).T
particles = Particles(initial_xyz=initial_xyz, mass=ALPHA_PARTICLE_MASS,
                      charge=ALPHA_PARTICLE_CHARGE, energy=energy,
                      initial_vparallel_over_v=initial_vparallel_over_v)

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=vmec, particles=particles, maxtime=tmax,
                  timestep=timestep, times_to_trace=times_to_trace,
                  model='GuidingCenter',
                  )
print(f"ESSOS tracing of {nparticles} particles during {tmax}s took {time()-time0:.2f} seconds")
print(f"Final loss fraction: {tracing.loss_fractions[-1]*100:.2f}%")
print(f"Particles lost: {int(tracing.total_particles_lost)}")
print(f"Axis terminations: {int(tracing.total_particles_unresolved)}")
print(f"Solver failures: {int(tracing.total_particles_failed)}")
trajectories = tracing.trajectories

# Plot trajectories, velocity parallel to the magnetic field, loss fractions and/or energy error
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# Plot random particles
## Plot trajectories in 3D
vmec.surface.plot(ax=ax1, show=False, alpha=0.4)
tracing.plot(ax=ax1, show=False, n_trajectories_plot=n_particles_to_plot)
energies = tracing.energy()
rng = np.random.default_rng(seed)
for i in rng.choice(nparticles, size=n_particles_to_plot, replace=False):
    trajectory = trajectories[i]
    ## Plot energy error
    ax2.plot(tracing.times[2:], jnp.abs(energies[i][2:]-particles.energy)/particles.energy, label=f'Particle {i+1}')
    ## Plot velocity parallel to the magnetic field
    ax3.plot(tracing.times, trajectory[:, 3]/particles.total_speed, label=f'Particle {i+1}')
    ## Plot s-coordinate
    ax4.plot(tracing.times, trajectory[:,0], label=f'Particle {i+1}')
    # ax4.set_ylabel(r'$s=\psi/\psi_b$')
## Plot loss fractions
#ax4.plot(tracing.times, tracing.loss_fractions)
#ax4.set_ylabel('Loss Fraction');ax4.set_ylim(0, 1);ax4.set_xscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel('Relative Energy Error')
ax2.set_xlabel('Time (s)')
ax3.set_ylim(-1, 1)
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax3.set_xlabel('Time (s)')
ax4.set_xlabel('Time (s)')
# plt.tight_layout()
plt.show()
# fig.savefig('tracing_results.png', dpi=300)

# # Save results in vtk format to analyze in Paraview
# vmec.surface.to_vtk('surface')
# tracing.to_vtk('trajectories')
