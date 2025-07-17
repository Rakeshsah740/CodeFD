
import numpy as np

def compute_error(ux, uy, ux_prev, uy_prev, Ny, Nx, eps=1e-12):
    """
    Compute maximum relative error in velocity at center lines.
    """
    ux_mid = ux[Ny // 2, :]
    uy_mid = uy[:, Nx // 2]

    # Avoid division by zero by adding eps
    error_ux_rel = np.max(np.abs(ux_mid - ux_prev) / (np.abs(ux_prev) + eps))
    error_uy_rel = np.max(np.abs(uy_mid - uy_prev) / (np.abs(uy_prev) + eps))
    total_error_rel = max(error_ux_rel, error_uy_rel)
    error_ux_abs = np.max(np.abs(ux_mid - ux_prev))
    error_uy_abs = np.max(np.abs(uy_mid - uy_prev))
    total_error_abs = max(error_ux_abs, error_uy_abs)

    return total_error_abs, total_error_rel, ux_mid.copy(), uy_mid.copy()

