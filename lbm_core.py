import numpy as np

# Lattice setup (D2Q9)
               #0 ,1 ,2,  3, 4, 5, 6,  7, 8
cxs = np.array([0, 1, 0, -1, 0, 1,-1, -1, 1])
cys = np.array([0, 0, 1,  0,-1, 1, 1, -1,-1])
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
opposite_indices = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # Opposite directions for bounce-back

def initialize(Nx, Ny):
    """
    Initialize the distribution functions and macroscopic variables.
    """
    F = np.ones((Ny, Nx, 9))      # Distribution functions
    rho = np.ones((Ny, Nx))       # Density field
    ux = np.zeros((Ny, Nx))       # X-velocity
    uy = np.zeros((Ny, Nx))       # Y-velocity
    return F, rho, ux, uy

def compute_equilibrium(rho, ux, uy):
    """
    Compute the equilibrium distribution function Feq.
    """
    Feq = np.zeros((rho.shape[0], rho.shape[1], 9))
    for i in range(9):
        cu = cxs[i]*ux + cys[i]*uy
        Feq[:, :, i] = rho * weights[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*(ux**2 + uy**2))
    return Feq

def collision(F, Feq, tau):
    """
    Apply the BGK collision operator.
    """
    return F - (1/tau) * (F - Feq)

def streaming(F):
    """
    Perform streaming step using numpy roll.
    """
    for i in range(9):
        F[:, :, i] = np.roll(F[:, :, i], cxs[i], axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cys[i], axis=0)
    return F

def update_macroscopic(F):
    """
    Compute macroscopic variables from F.
    """
    rho = np.sum(F, axis=2)
    ux = np.sum(F * cxs, axis=2) / rho
    uy = np.sum(F * cys, axis=2) / rho
    return rho, ux, uy
