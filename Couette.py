
'''   
    6  2  5
     \ | /
    3--0--1
     / | \
    7  4  8
'''

from lbm_core import *
from boundaries import apply_boundary_conditions_couette, enforce_velocity_boundaries_couette   
from error import compute_error
from visualization import visualize_fields
import matplotlib.pyplot as plt
import os

# Simulation parameters
Nx, Ny = 100, 100
tau = 0.6
steps = 90001
plot_every = 2000
ux_top_plate = 0.2
uy_top_plate = 0



# Initialize fields
F, rho, ux, uy = initialize(Nx, Ny)

# Track error history and previous midline velocity
error_history = []
ux_mid_prev = np.zeros(Nx)
uy_mid_prev = np.zeros(Ny)

# Plotting setup
plt.ion()
fig, ax = plt.subplots()

# Main loop
for it in range(steps):
    # Collision and streaming
    Feq = compute_equilibrium(rho, ux, uy)
    F = collision(F, Feq, tau)
    F = streaming(F)

    # Apply boundaries
    F = apply_boundary_conditions_couette(F, ux_top_plate, uy_top_plate)
    rho, ux, uy = update_macroscopic(F)
    ux, uy = enforce_velocity_boundaries_couette(ux, uy, ux_top_plate)
   
    # Compute and store error
    err, ux_mid_prev, uy_mid_prev = compute_error(ux, uy, ux_mid_prev, uy_mid_prev, Ny, Nx)
    error_history.append(err)
    
    if it % plot_every == 0:
        visualize_fields(it, ux, uy, ux_top_plate, tau, Nx, Ny)

    '''   
    # Plot convergence
    if it % 100 == 0:
        ax.clear()
        ax.plot(error_history, 'r-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max centerlines velocity error')
        ax.set_title(f'Step {it}, Error={err:.2e}')
        plt.pause(0.001)
    '''   
    print('step =', it)
# Finish
plt.ioff()
plt.show()
print('Done')
