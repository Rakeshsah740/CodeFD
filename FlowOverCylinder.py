
'''   
    6  2  5
     \ | /
    3--0--1
     / | \
    7  4  8
'''

from lbm_core import *
from boundaries import apply_boundary_conditions_cylinder,enforce_velocity_boundaries_cylinder
from error import compute_error
from visualizationCylinder import visualize_fields
import matplotlib.pyplot as plt
import os

# Simulation parameters
Nx, Ny = 400, 100
tau = 0.6
steps = 35001
plot_every = 200
ux_inlet = 0.2
uy_inlet = 0
opposite_indices = [0, 3, 4, 1, 2, 7, 8, 5, 6]






# Initialize fields
F, rho, ux, uy = initialize(Nx, Ny)



# Track error history and previous midline velocity
error_history = []
ux_mid_prev = np.zeros(Nx)
uy_mid_prev = np.zeros(Ny)

# Plotting setup
plt.ion()
fig, ax = plt.subplots()


# Create obstacle mask (circle at 1/2 of Nx and Ny)
obstacle_radius = Ny // 8               # Radius of the circular obstacle
obstacle_x = Nx // 5                   # X center position 
obstacle_y = Ny // 2                   # Y center position 
obstacle = np.zeros((Ny, Nx), dtype=bool)                             # Initialize boolean mask
# Create grid of coordinates
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y)

# Define circular region: distance from center <= radius
for y in range(Ny):
    for x in range(Nx):
        dx = x - obstacle_x
        dy = y - obstacle_y
        if dx**2 + dy**2 <= obstacle_radius**2:
            obstacle[y, x] = True
        else:
            obstacle[y, x] = False




# Main loop
for it in range(steps):
    # Collision and streaming
    Feq = compute_equilibrium(rho, ux, uy)
    F = collision(F, Feq, tau)
    F = streaming(F)
    
    
    
    # Apply boundaries
    F = apply_boundary_conditions_cylinder(F, ux_inlet,uy_inlet)
    for i in range(9):
        F[obstacle, i] = F[obstacle, opposite_indices[i]]
    rho, ux, uy = update_macroscopic(F)
    
    ux[obstacle] = 0  
    uy[obstacle] = 0
    ux, uy = enforce_velocity_boundaries_cylinder(ux, uy,ux_inlet)
    
    

    
    # Compute and store error
    err, ux_mid_prev, uy_mid_prev = compute_error(ux, uy, ux_mid_prev, uy_mid_prev, Ny, Nx)
    error_history.append(err)
    

  



    
    
    if it % plot_every == 0:
        visualize_fields(it, ux, uy,ux_inlet, tau, Nx, Ny)

   
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
