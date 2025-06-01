
'''   
    6  2  5
     \ | /
    3--0--1
     / | \
    7  4  8
'''

from lbm_core import *
from boundaries import apply_boundary_conditions, enforce_velocity_boundaries    # for top lid moving
from error import compute_error
from visualization import visualize_fields
import matplotlib.pyplot as plt
import os

# Simulation parameters
Nx, Ny = 200, 200
tau = 0.6
steps = 10001
plot_every = 100
ux_top = 0.2
uy_top = 0
opposite_indices = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # Opposite directions for bounce-back

# Initialize fields
F, rho, ux, uy = initialize(Nx, Ny)

# Track error history and previous midline velocity
error_history = []
ux_mid_prev = np.zeros(Nx)
uy_mid_prev = np.zeros(Ny)

# Create obstacle mask (circle at 1/2 of Nx and Ny)
obstacle_radius = Nx // 8              # Radius of the circular obstacle
obstacle_x = Nx // 2                   # X center position (1/4 of domain width)
obstacle_y = Ny // 2                   # Y center position (1/4 of domain height)
obstacle = np.zeros((Ny, Nx), dtype=bool)                             # Initialize boolean mask
# Create grid of coordinates
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y)

# Define circular region: distance from center <= radius
obstacle[((X - obstacle_x)**2 + (Y - obstacle_y)**2) <= obstacle_radius**2] = True

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
    F = apply_boundary_conditions(F, ux_top, uy_top)
    
    # Bounce-back for circular obstacle
    for i in range(9):
        F[obstacle, i] = F[obstacle, opposite_indices[i]]
    
    
    rho, ux, uy = update_macroscopic(F)
    
    ux, uy = enforce_velocity_boundaries(ux, uy, ux_top)
    ux[obstacle] = 0  
    uy[obstacle] = 0
    
    
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
