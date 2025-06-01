def apply_boundary_conditions(F, ux_top, uy_top):
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

def enforce_velocity_boundaries(ux, uy, ux_top):
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




def apply_boundary_conditions2(F, ux_top, ux_bottom, uy_top, uy_bottom):
    """
    Apply Zou/He velocity boundary at the top and bounce-back at other walls.
    """
    # Top wall (moving lid)
    rho_lid1 = (F[-1, :, 0] + F[-1, :, 1] + F[-1, :, 3] + 
               2*(F[-1, :, 2] + F[-1, :, 5] + F[-1, :, 6])) / (1 + uy_top)

    # Zou/He adjustments
    F[-1, :, 4] = F[-1, :, 2] - (2/3) * rho_lid1 * uy_top
    F[-1, :, 7] = F[-1, :, 5] + 0.5*(F[-1, :, 1] - F[-1, :, 3]) - (1/6) * rho_lid1 * (3*ux_top + uy_top)
    F[-1, :, 8] = F[-1, :, 6] + 0.5*(F[-1, :, 3] - F[-1, :, 1]) + (1/6) * rho_lid1 * (3*ux_top - uy_top)
    
    
    ###########################################################################################################
    # Bottom wall (moving lid)
    rho_lid2 = (F[0, :, 0] + F[0, :, 1] + F[0, :, 3] + 
               2*(F[0, :, 4] + F[0, :, 7] + F[0, :, 8])) / (1 + uy_bottom)

    # Zou/He adjustments/ Bottom wall (bounce-back)
    F[0, :, 2] = F[0, :, 4] + (2/3) * rho_lid2 * uy_bottom
    F[0, :, 5] = F[0, :, 7] - 0.5*(F[0, :, 1] - F[0, :, 3]) + (1/6) * rho_lid2 * (3*ux_bottom + uy_bottom)
    F[0, :, 6] = F[0, :, 8] - 0.5*(F[0, :, 3] - F[0, :, 1]) - (1/6) * rho_lid2 * (3*ux_bottom - uy_bottom)


    # Left wall (bounce-back)
    F[:, 0, 1] = F[:, 0, 3]
    F[:, 0, 5] = F[:, 0, 7]
    F[:, 0, 8] = F[:, 0, 6]

    # Right wall (bounce-back)
    F[:, -1, 3] = F[:, -1, 1]
    F[:, -1, 6] = F[:, -1, 8]
    F[:, -1, 7] = F[:, -1, 5]

    return F

def enforce_velocity_boundaries2(ux, uy, ux_top,ux_bottom):
    """
    Set velocity boundary values explicitly.
    """
    # Bottom
    ux[0, :] = ux_top
    uy[0, :] = 0

    # Left
    ux[:, 0] = uy[:, 0] = 0

    # Right
    ux[:, -1] = uy[:, -1] = 0

    # Top (moving lid)
    ux[-1, :] = ux_bottom
    uy[-1, :] = 0

    return ux, uy
