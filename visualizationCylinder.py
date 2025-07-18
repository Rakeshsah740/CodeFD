'''   
    6  2  5
     \ | /
    3--0--1
     / | \
    7  4  8
'''
import numpy as np
import matplotlib.pyplot as plt
import os

def create_output_folder(folder_name="plot"):
    """
    Create output folder if it doesn't exist.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def compute_vorticity(ux, uy):
    """
    Compute vorticity as curl of velocity field.
    """
    duy_dx = np.gradient(uy, axis=1)
    dux_dy = np.gradient(ux, axis=0)
    return duy_dx - dux_dy

def visualize_fields(it, ux, uy, ux_top, tau, Nx, Ny, folder="plot"):
    """
    Save figures of velocity fields, vorticity, profiles, and streamlines.
    """
    create_output_folder(folder)
    
    # Derived fields
    vel_mag = np.sqrt(ux**2 + uy**2)
    vorticity = compute_vorticity(ux, uy)
    vorticity_physical = vorticity * (ux_top / Nx)
    
    Re = int(ux_top * Nx / ((2 * tau - 1) / 6))  # Reynolds number

      
    # --- Save X-Velocity field ---
    plt.figure(figsize=(8, 5))
    plt.imshow(ux, cmap='jet', origin='lower')
    plt.colorbar(label='X-velocity (u_x)')
    plt.title(f'X-Velocity Field (Step {it}, Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{folder}/u_x_field_step_{it:04d}.png', dpi=250)
    plt.close()

    # --- Save Y-Velocity field ---
    plt.figure(figsize=(8, 5))
    plt.imshow(uy, cmap='jet', origin='lower')
    plt.colorbar(label='Y-velocity (u_y)')
    plt.title(f'Y-Velocity Field (Step {it}, Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{folder}/u_y_field_step_{it:04d}.png', dpi=250)
    plt.close()

    # --- Save Velocity Magnitude field ---
    plt.figure(figsize=(8, 5))
    plt.imshow(vel_mag, cmap='jet', origin='lower')
    plt.colorbar(label='Velocity Magnitude |u|')
    plt.title(f'Velocity Magnitude (Step {it}, Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{folder}/velocity_magnitude_step_{it:04d}.png', dpi=250)
    plt.close()

    # --- Save Vorticity field ---
    vmax = np.percentile(np.abs(vorticity_physical), 95)
    vmin = -vmax
    plt.figure(figsize=(8, 5))
    plt.imshow(vorticity_physical, cmap='bwr', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Vorticity')
    plt.title(f'Vorticity Field (Step {it}, Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{folder}/vorticity_field_step_{it:04d}.png', dpi=250)
    plt.close()
    
   
    # --- X-Velocity Profile at Mid-height (Horizontal Line: y = Ny/2) ---
    y_pos = Ny // 2  # Mid-height
    x_normalized = np.linspace(0, 1, Nx)  # Normalize x-axis to match ux.shape[1]

    plt.figure(figsize=(8, 5))
    plt.plot(x_normalized, ux[y_pos, :] / ux_top, 'b-')  # u_x across x, at mid-y
    plt.title(f'Normalized $u_x$ Profile at y={y_pos} (Step {it}, Re={Re})')
    plt.xlabel('x')
    plt.ylabel('$u_x / u_{{top}}$')
    plt.grid(True)
    plt.savefig(f'{folder}/ux_profile_horizontal_step_{it:04d}.png', dpi=250)
    plt.close()

    # --- X-Velocity Profile at Mid-width (Vertical Line: x = Nx/2) ---
    x_pos = Nx // 2  # Mid-width
    y_normalized = np.linspace(0, 1, Ny)  # Normalize y-axis to match ux.shape[0]

    plt.figure(figsize=(8, 5))
    plt.plot(ux[:, x_pos] / ux_top, y_normalized, 'b-')  # u_x along y, at mid-x
    plt.title(f'Normalized $u_x$ Profile at x={x_pos} (Step {it}, Re={Re})')
    plt.xlabel('$u_x / u_{{top}}$')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig(f'{folder}/ux_profile_vertical_step_{it:04d}.png', dpi=250)
    plt.close()

    # --- Streamlines ---
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 5))
    plt.streamplot(
        X, Y, ux, uy,
        density=2, color='blue', linewidth=1.2, arrowsize=1.5
    )
    plt.title(f'Streamlines (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.xlim(0, Nx)
    plt.ylim(0, Ny)
    plt.savefig(f'{folder}/streamlines_step_{it:04d}.png', dpi=100, bbox_inches='tight')
    plt.close()