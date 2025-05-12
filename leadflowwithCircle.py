"""
    8   1   2
      \ | /
    7 - 0 - 3
      / | \
    6   5   4 
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
Nx, Ny = 400, 400  # Square domain
tau = 0.6          # Relaxation time
steps = 40001       # Number of iterations
plot_every = 500    # Plotting interval
u_top = 0.01       # Lid velocity

# Lattice setup (D2Q9)
cxs = np.array([0, 0, 1, 1, 1,   0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
opposite_indices = [0, 5, 6, 7, 8, 1, 2, 3, 4]  # Opposite directions for bounce-back

# Initialize distribution functions
F = np.ones((Ny, Nx, 9)) + 0.01 * np.random.randn(Ny, Nx, 9)
rho = np.ones((Ny, Nx))  # Initial density
ux = np.zeros((Ny, Nx))  # Initial x-velocity
uy = np.zeros((Ny, Nx))  # Initial y-velocity


# Create circular obstacle
obstacle_radius = Nx // 8  # Radius of the circular obstacle
obstacle_x = Nx // 2
obstacle_y = Ny // 2

# Create grid of coordinates
x, y = np.arange(Nx), np.arange(Ny)
X, Y = np.meshgrid(x, y)

# Calculate distance from center
distance = np.sqrt((X - obstacle_x)**2 + (Y - obstacle_y)**2)
obstacle = distance <= obstacle_radius

# Create visualization figure
plt.figure(figsize=(18, 6))

for it in range(steps):
    # Store pre-collision state for visualization and bounce-back
    F_pre_collision = F.copy()
    
    # --- Streaming step ---
    for i in range(9):
        F[:, :, i] = np.roll(F[:, :, i], cxs[i], axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cys[i], axis=0)
    
    # --- Boundary conditions ---
    # Moving lid (top boundary) - Zou/He boundary condition
    rho_lid = (F[-1, :, 0] + F[-1, :, 1] + F[-1, :, 3] + 
               2*(F[-1, :, 2] + F[-1, :, 5] + F[-1, :, 6])) / (1 - u_top)
    
    F[-1, :, 4] = F[-1, :, 2] - (2/3) * rho_lid * u_top
    F[-1, :, 7] = F[-1, :, 5] + (1/6) * rho_lid * u_top
    F[-1, :, 8] = F[-1, :, 6] + (1/6) * rho_lid * u_top
    
    # Bounce-back for stationary walls (left, right, bottom)
    # Bottom wall (y = 0)
    F[0, :, 1] = F[0, :, 5]  # North (1) = South (5)
    F[0, :, 2] = F[0, :, 6]  # Northeast (2) = Southwest (6)
    F[0, :, 8] = F[0, :, 4]  # Northwest (8) = Southeast (4)

    # Left wall (x = 0)
    F[:, 0, 3] = F[:, 0, 7]  # East (3) = West (7)
    F[:, 0, 2] = F[:, 0, 6]  # Northeast (2) = Southwest (6)
    F[:, 0, 4] = F[:, 0, 8]  # Southeast (4) = Northwest (8)

    # Right wall (x = Nx-1)
    F[:, Nx-1, 7] = F[:, Nx-1, 3]  # West (7) = East (3)
    F[:, Nx-1, 6] = F[:, Nx-1, 2]  # Southwest (6) = Northeast (2)
    F[:, Nx-1, 8] = F[:, Nx-1, 4]  # Northwest (8) = Southeast (4)
    
    # Bounce-back for circular obstacle
    for i in range(9):
        F[obstacle, i] = F_pre_collision[obstacle, opposite_indices[i]]

    
    # --- Macroscopic variables ---
    rho = np.sum(F, 2)
    ux = np.sum(F * cxs, 2) / rho
    uy = np.sum(F * cys, 2) / rho
    
    # Enforce boundary velocities
    ux[0, :] = 0    # Bottom wall
    uy[0, :] = 0
    ux[:, 0] = 0    # Left wall
    uy[:, 0] = 0
    ux[:, Nx-1] = 0   # Right wall
    uy[:, Nx-1] = 0
    ux[Ny-1, :] = u_top  # Top wall (moving lid)
    uy[Ny-1, :] = 0
    ux[obstacle] = 0  # Obstacle velocity
    uy[obstacle] = 0
    
    
    # --- Collision step ---
    F_pre_collision = F.copy()  # Store pre-collision state for visualization
    Feq = np.zeros_like(F)
    for i in range(9):
        cu = cxs[i]*ux + cys[i]*uy
        Feq[:, :, i] = rho * weights[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*(ux**2 + uy**2))
    F += -(1/tau) * (F - Feq)
    
    import os
    os.makedirs('cavityWithCircle', exist_ok=True)  # Create folder only once at start
    
    # --- Visualization ---
    if it % plot_every == 0:
        plt.clf()
        print("step =", it)
        
        # Calculate vorticity
        duy_dx = np.gradient(uy, axis=1)
        dux_dy = np.gradient(ux, axis=0)
        vorticity = duy_dx - dux_dy
        
        Re = int(u_top * Nx / ((2 * tau - 1) / 6))
        
        # --- Save X-Velocity field (u_x)
        plt.figure(figsize=(8,5))
        plt.imshow(ux, cmap='jet', origin='lower')
        plt.colorbar(label='X-velocity (u_x)')
        plt.title(f'X-Velocity Field (Step {it}, Re={Re})')
        plt.savefig(f'cavityWithCircle/u_x_field_step_{it:05d}.png', dpi=300)
        plt.close()

        # --- Save Y-Velocity field (u_y)
        plt.figure(figsize=(8,5))
        plt.imshow(uy, cmap='jet', origin='lower')
        plt.colorbar(label='Y-velocity (u_y)')
        plt.title(f'Y-Velocity Field (Step {it}, Re={Re})')
        plt.savefig(f'cavityWithCircle/u_y_field_step_{it:05d}.png', dpi=300)
        plt.close()

        # --- Save Velocity Magnitude field
        plt.figure(figsize=(8,5))
        vel_mag = np.sqrt(ux**2 + uy**2)
        plt.imshow(vel_mag, cmap='jet', origin='lower')
        plt.colorbar(label='Velocity Magnitude |u|')
        plt.title(f'Velocity Magnitude (Step {it}, Re={Re})')
        plt.savefig(f'cavityWithCircle/velocity_magnitude_step_{it:05d}.png', dpi=300)
        plt.close()

        # --- Save Vorticity field
        plt.figure(figsize=(8,5))
        plt.imshow(vorticity, cmap='bwr', origin='lower', vmin=-0.2, vmax=0.2)
        plt.colorbar(label='Vorticity')
        plt.title(f'Vorticity Field (Step {it}, Re={Re})')
        plt.savefig(f'cavityWithCircle/vorticity_field_step_{it:05d}.png', dpi=300)
        plt.close()

        # --- Save Temperature (density rho)
        plt.figure(figsize=(8,5))
        plt.imshow(rho, cmap='inferno', origin='lower')
        plt.colorbar(label='Temperature (ρ)')
        plt.title(f'Temperature Field (Step {it}, Re={Re})')
        plt.savefig(f'cavityWithCircle/temperature_step_{it:05d}.png', dpi=300)
        plt.close()

        # --- Save X-velocity profile at mid-height
        y_pos = Ny // 2  # Middle y-location
        plt.figure(figsize=(8,5))
        plt.plot(range(Nx), ux[y_pos, :], 'b-')
        plt.xlabel('x (horizontal)')
        plt.ylabel('u_x (velocity in x)')
        plt.title(f'Profile of u_x at y={y_pos} (Step {it}, Re={Re})')
        plt.grid(True)
        plt.savefig(f'cavityWithCircle/ux_profile_step_{it:05d}.png', dpi=300)
        plt.close()
        
        # Create coordinate grid based on actual domain size
        x = np.arange(Nx)
        y = np.arange(Ny)
        X, Y = np.meshgrid(x, y)
        
        plt.figure(figsize=(6, 6))
        plt.streamplot(
            X, Y,
            ux, uy,
            density=2,
            color='blue',
            linewidth=1.2,
            arrowsize=1.5,
            minlength=0.2,
            integration_direction='both'
        )
        plt.gca().set_aspect('equal')
        plt.xlim(0, Nx)
        plt.ylim(0, Ny)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Stream Trace for Re = {Re}')
        plt.savefig(f'cavityWithCircle/streamlines_step_{it:05d}.png', dpi=300, bbox_inches='tight')
        plt.close()
        