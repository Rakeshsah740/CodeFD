import numpy as np
import vtk
from vtkmodules.util import numpy_support
import os

# ======================
# Simulation Parameters
# ======================
Nx, Ny, Nz = 60, 60, 60      # Grid size
tau = 0.55                   # Relaxation time
steps = 15001                   # Number of time steps
plot_every = 100              # PNG every N steps
u_top = 0.12                  # Lid velocity

# D3Q19 Lattice Constants   4  5  6  7   8  9 10   11 12  13 14  15 16 17 18
cxs = np.array([0, 1,-1, 0, 0, 0, 0, 1, -1, 1,-1,  0, 0,  1, -1,  1, -1, 0, 0])
cys = np.array([0, 0, 0, 1,-1, 0, 0, 1, -1, 0, 0,  1,-1, -1,  1,  0,  0, 1,-1])
czs = np.array([0, 0, 0, 0, 0, 1,-1, 0,  0, 1,-1,  1,-1,  0,  0,  -1, 1,-1, 1])
opposite = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9 , 12, 11, 14, 13, 16, 15, 18, 17]  # Opposite directions for bounce-back
weights = np.array([1/3] + [1/18]*6 + [1/36]*12)


# Initialize Fields
F   = np.ones((Nz, Ny, Nx, 19))   # distribution functions
rho = np.ones((Nz, Ny, Nx))       # density
ux  = np.zeros((Nz, Ny, Nx))      # velocity components
uy  = np.zeros((Nz, Ny, Nx))
uz  = np.zeros((Nz, Ny, Nx))

os.makedirs('vtk_output', exist_ok=True)


# VTK Export Function
def export_step(grid, it):
    """
    Write both a .vti file (for ParaView) and
    — if desired — a PNG snapshot of isosurfaces + glyphs.
    """
    # --- 1) Export the raw grid to .vti ---
    vti_writer = vtk.vtkXMLImageDataWriter()
    vti_writer.SetFileName(f'vtk_output/step_{it:04d}.vti')
    vti_writer.SetInputData(grid)
    vti_writer.Write()

    # --- 2) (Optional) PNG visualization ---
    # Contour + glyph setup
    contour = vtk.vtkContourFilter()
    contour.SetInputData(grid)
    contour.GenerateValues(5, 0.0, u_top * 1.5)

    cmap = vtk.vtkPolyDataMapper()
    cmap.SetInputConnection(contour.GetOutputPort())
    cmap.SetScalarRange(0.0, u_top * 1.5)

    actor = vtk.vtkActor()
    actor.SetMapper(cmap)
    actor.GetProperty().SetOpacity(0.6)

    # Glyph arrows
    mask = vtk.vtkMaskPoints()
    mask.SetInputData(grid)
    mask.SetOnRatio(8)

    arrow = vtk.vtkArrowSource()
    arrow.SetTipResolution(8); arrow.SetShaftResolution(8)

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputConnection(mask.GetOutputPort())
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(0.5)

    gmapper = vtk.vtkPolyDataMapper()
    gmapper.SetInputConnection(glyph.GetOutputPort())

    gactor = vtk.vtkActor()
    gactor.SetMapper(gmapper)
    gactor.GetProperty().SetColor(0,0,0)

    # Text annotation
    text = vtk.vtkTextActor()
    text.SetInput(f"Step {it}, Re≈{int(u_top*Nx/((2*tau-1)/6))}")
    text.GetTextProperty().SetFontSize(24)
    text.GetTextProperty().SetColor(0,0,0)
    text.SetDisplayPosition(20,20)

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1,1,1)
    renderer.AddActor(actor)
    renderer.AddActor(gactor)
    renderer.AddActor2D(text)

    rw = vtk.vtkRenderWindow()
    rw.AddRenderer(renderer)
    rw.SetSize(1200,900)
    rw.OffScreenRenderingOn()
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    png = vtk.vtkPNGWriter()
    png.SetFileName(f'vtk_output/step_{it:04d}.png')
    png.SetInputConnection(w2i.GetOutputPort())
    png.Write()


# Main LBM Loop
for it in range(steps):
    print(f"Step {it+1}/{steps}")

    # 1) Streaming
    for i in range(19):
        F[...,i] = np.roll(F[...,i], cxs[i], axis=2)
        F[...,i] = np.roll(F[...,i], cys[i], axis=1)
        F[...,i] = np.roll(F[...,i], czs[i], axis=0)

    # 2) Boundary Conditions
    # Moving lid at z = Nz-1
    ux_lid = u_top
    uy_lid = 0
    uz_lid = 0
    
    rho_lid = (
        F[-1, :, :, 0]
        + F[-1, :, :, 1] + F[-1, :, :, 2] + F[-1, :, :, 3]
        + F[-1, :, :, 4] + F[-1, :, :, 5] + F[-1, :, :, 6]
        + 2 * (F[-1, :, :, 9] + F[-1, :, :, 10] + F[-1, :, :, 11])
    ) / (1 + uz_lid)

    # Reconstruct unknown populations at z = Nz-1
    F[-1, :, :, 12] = F[-1, :, :, 9] - (2 / 3) * rho_lid * ux_lid
    F[-1, :, :, 13] = F[-1, :, :, 10] + (1 / 6) * rho_lid * ux_lid
    F[-1, :, :, 14] = F[-1, :, :, 11] + (1 / 6) * rho_lid * ux_lid


    # Bounce-back on all other walls
    #Bottom wall (Z = 0)
    for i in [4, 8, 12, 13, 18]:
        F[0, :, :, i] = F[0, :, :, opposite[i]]

    #Front wall (X = 0)
    for i in [5, 9, 11, 16, 18]:
        F[:, :, 0, i] = F[:, :, 0, opposite[i]]
    
    #Back wall (X = Nx–1)
    for i in [6, 10, 12, 15, 17]:
        F[:, :, -1, i] = F[:, :, -1, opposite[i]]
        
    #Left wall (Y = 0)
    for i in [2, 8, 10, 14, 16]:
        F[:, 0, :, i] = F[:, 0, :, opposite[i]]

    #Right wall (Y = Ny–1)
    for i in [1, 7, 9, 13, 15]:
        F[:, -1, :, i] = F[:, -1, :, opposite[i]]


    # 3) Compute macroscopic vars
    rho = F.sum(axis=-1)
    ux  = (F * cxs).sum(axis=-1)/rho
    uy  = (F * cys).sum(axis=-1)/rho
    uz  = (F * czs).sum(axis=-1)/rho

    # Re-enforce wall velocities
    # Bottom wall
    ux[0, :, :] = uy[0, :, :] = uz[0, :, :] = 0

    # Top wall (moving lid)
    ux[-1, :, :] = u_top
    uy[-1, :, :] = uz[-1, :, :] = 0

    # Left and Right walls (X = 0, Nx–1)
    ux[:, :, 0] = uy[:, :, 0] = uz[:, :, 0] = 0
    ux[:, :, -1] = uy[:, :, -1] = uz[:, :, -1] = 0

    # Front and Back walls (Y = 0, Ny–1)
    ux[:, 0, :] = uy[:, 0, :] = uz[:, 0, :] = 0
    ux[:, -1, :] = uy[:, -1, :] = uz[:, -1, :] = 0


    # 4) Collision (BGK)
    cu  = np.zeros_like(F)
    for i in range(19):
        cu[...,i] = cxs[i]*ux + cys[i]*uy + czs[i]*uz

    Feq = np.zeros_like(F)
    usq = ux**2 + uy**2 + uz**2
    for i in range(19):
        Feq[...,i] = rho * weights[i] * (
            1 + 3*cu[...,i] + 4.5*cu[...,i]**2 - 1.5*usq
        )

    F += -(1/tau)*(F - Feq)

    # 5) Export for ParaView 
    if it % plot_every == 0 or it == steps-1:
        # Build a VTK grid with scalars & vectors
        grid = vtk.vtkImageData()
        grid.SetDimensions(Nx, Ny, Nz)
        grid.SetSpacing(1,1,1)

        # Scalars
        velmag = np.sqrt(ux**2 + uy**2 + uz**2).ravel()
        vtk_s = numpy_support.numpy_to_vtk(velmag, deep=True)
        vtk_s.SetName("VelocityMagnitude")
        grid.GetPointData().SetScalars(vtk_s)

        # Vectors
        vecs = np.stack([ux, uy, uz], axis=-1).reshape(-1,3)
        vtk_v = numpy_support.numpy_to_vtk(vecs, deep=True)
        vtk_v.SetName("VelocityVector")
        vtk_v.SetNumberOfComponents(3)
        grid.GetPointData().SetVectors(vtk_v)

        export_step(grid, it)

print("Done!")
