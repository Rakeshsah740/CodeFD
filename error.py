import numpy as np

def compute_error(ux, uy, ux_prev, uy_prev, Ny, Nx):
    """
    Compute maximum difference in velocity at center lines.
    """
    ux_mid = ux[Ny // 2, :]
    uy_mid = uy[:, Nx // 2]

    error_ux = np.max(np.abs(ux_mid - ux_prev))
    error_uy = np.max(np.abs(uy_mid - uy_prev))
    total_error = max(error_ux, error_uy)

    return total_error, ux_mid.copy(), uy_mid.copy()
