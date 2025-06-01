# IMPORTANT:
# Only one set of functions should be used at a time:
# - Use `apply_boundary_conditions` and `enforce_velocity_boundaries` for single moving lid (top only).
# - Use `apply_boundary_conditions2` and `enforce_velocity_boundaries2` for two moving lids (top and bottom).
# NEVER use both sets in the same simulation loop — they will override each other’s results and lead to incorrect boundary behavior.


from lbm_core import *
from boundaries import apply_boundary_conditions, enforce_velocity_boundaries    # for top lid moving
from boundaries import apply_boundary_conditions2, enforce_velocity_boundaries2    # for top and bottom lid moving
from error import compute_error
from visualization import visualize_fields
import matplotlib.pyplot as plt
import os

# Simulation parameters
Nx, Ny = 300, 300
tau = 0.6
steps = 50001
plot_every = 500
ux_top = 0.2
uy_top = 0
ux_bottom = 1*ux_top
uy_bottom = 0


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
    #F = apply_boundary_conditions(F, ux_top, uy_top)
    F = apply_boundary_conditions2(F, ux_top, ux_bottom, uy_top, uy_bottom)
    rho, ux, uy = update_macroscopic(F)
    #ux, uy = enforce_velocity_boundaries(ux, uy, ux_top)
    ux, uy = enforce_velocity_boundaries2(ux, uy, ux_top, ux_bottom)
    
    # Compute and store error
    err, ux_mid_prev, uy_mid_prev = compute_error(ux, uy, ux_mid_prev, uy_mid_prev, Ny, Nx)
    error_history.append(err)
    
    if it % plot_every == 0:
        visualize_fields(it, ux, uy, ux_top, tau, Nx, Ny)

        
    # Plot convergence
    if it % 100 == 0:
        ax.clear()
        ax.plot(error_history, 'r-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max centerlines velocity error')
        ax.set_title(f'Step {it}, Error={err:.2e}')
        plt.pause(0.001)
    
    print('step =', it)
# Finish
plt.ioff()
plt.show()
print('Done')
