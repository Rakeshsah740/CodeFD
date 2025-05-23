def apply_boundary_conditions(F, ux_top, uy_top, Nx, Ny):
    """
    Apply Zou/He velocity boundary at the top and bounce-back at other walls.
    """
    # Top wall (moving lid)
    rho_lid = (F[-1, :, 0] + F[-1, :, 1] + F[-1, :, 3] + 
               2*(F[-1, :, 2] + F[-1, :, 5] + F[-1, :, 6])) / (1 + uy_top)

    # Zou/He adjustments
    F[-1, :, 4] = F[-1, :, 2] - (2/3) * rho_lid * uy_top
    F[-1, :, 7] = F[-1, :, 5] + 0.5*(F[-1, :, 1] - F[-1, :, 3]) - (1/6) * rho_lid * (3*ux_top + uy_top)
    F[-1, :, 8] = F[-1, :, 6] + 0.5*(F[-1, :, 3] - F[-1, :, 1]) + (1/6) * rho_lid * (3*ux_top - uy_top)

    # Bottom wall (bounce-back)
    F[0, :, 2] = F[0, :, 4]
    F[0, :, 5] = F[0, :, 7]
    F[0, :, 6] = F[0, :, 8]

    # Left wall (bounce-back)
    F[:, 0, 1] = F[:, 0, 3]
    F[:, 0, 5] = F[:, 0, 7]
    F[:, 0, 8] = F[:, 0, 6]

    # Right wall (bounce-back)
    F[:, -1, 3] = F[:, -1, 1]
    F[:, -1, 6] = F[:, -1, 8]
    F[:, -1, 7] = F[:, -1, 5]

    return F

def enforce_velocity_boundaries(ux, uy, ux_top, Nx, Ny):
    """
    Set velocity boundary values explicitly.
    """
    # Bottom
    ux[0, :] = uy[0, :] = 0

    # Left
    ux[:, 0] = uy[:, 0] = 0

    # Right
    ux[:, -1] = uy[:, -1] = 0

    # Top (moving lid)
    ux[-1, :] = ux_top
    uy[-1, :] = 0

    return ux, uy
