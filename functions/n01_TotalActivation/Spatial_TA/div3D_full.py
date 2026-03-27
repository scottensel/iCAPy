import numpy as np


def div3D_full(dx, dy, dz, wx=None, wy=None, wz=None):
    """
    Computes the 3D divergence of the vector field [dx, dy, dz].
    Particularly suitable for fMRI data where the 4th dimension of d*
    refers to time; a 3D algorithm is applied to the first 3 components.

    Inputs:
        dx, dy, dz - 3D or 4D arrays representing the gradient components
                     of the vector field; if 4D, divergence is computed
                     along the first three spatial dimensions
        wx, wy, wz - optional weight matrices along the X, Y and Z
                     directions; applied element-wise to dx, dy, dz before
                     computing the divergence

    Outputs:
        V - divergence of the input vector field

    Implemented by Younes Farouj, 12.03.2016
    Weight option added by Younes Farouj, 28.04.2016
    """
    dx = np.array(dx)
    dy = np.array(dy)
    dz = np.array(dz)

    # Apply weights to each gradient component if provided
    if wx is not None:
        dx *= np.array(wx)
    if wy is not None:
        dy *= np.array(wy)
    if wz is not None:
        dz *= np.array(wz)

    if dx.ndim == 3:
        # X term: padarray(dx(1:end-1,:,:), [1 0 0], 'post')
        #       - padarray(dx(1:end-1,:,:), [1 0 0], 'pre')
        Ex      = dx[:-1, :, :]
        Ex_post = np.pad(Ex, ((0, 1), (0, 0), (0, 0)), mode='constant')
        Ex_pre  = np.pad(Ex, ((1, 0), (0, 0), (0, 0)), mode='constant')
        div_x   = Ex_post - Ex_pre

        # Y term: padarray(dy(:,1:end-1,:), [0 1 0], 'post')
        #       - padarray(dy(:,1:end-1,:), [0 1 0], 'pre')
        Ey      = dy[:, :-1, :]
        Ey_post = np.pad(Ey, ((0, 0), (0, 1), (0, 0)), mode='constant')
        Ey_pre  = np.pad(Ey, ((0, 0), (1, 0), (0, 0)), mode='constant')
        div_y   = Ey_post - Ey_pre

        # Z term: padarray(dz(:,:,1:end-1), [0 0 1], 'post')
        #       - padarray(dz(:,:,1:end-1), [0 0 1], 'pre')
        Ez      = dz[:, :, :-1]
        Ez_post = np.pad(Ez, ((0, 0), (0, 0), (0, 1)), mode='constant')
        Ez_pre  = np.pad(Ez, ((0, 0), (0, 0), (1, 0)), mode='constant')
        div_z   = Ez_post - Ez_pre

    elif dx.ndim == 4:
        # X term (4D version — time is the 4th dimension)
        Ex      = dx[:-1, :, :, :]
        Ex_post = np.pad(Ex, ((0, 1), (0, 0), (0, 0), (0, 0)), mode='constant')
        Ex_pre  = np.pad(Ex, ((1, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        div_x   = Ex_post - Ex_pre

        # Y term
        Ey      = dy[:, :-1, :, :]
        Ey_post = np.pad(Ey, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant')
        Ey_pre  = np.pad(Ey, ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant')
        div_y   = Ey_post - Ey_pre

        # Z term
        Ez      = dz[:, :, :-1, :]
        Ez_post = np.pad(Ez, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')
        Ez_pre  = np.pad(Ez, ((0, 0), (0, 0), (1, 0), (0, 0)), mode='constant')
        div_z   = Ez_post - Ez_pre

    else:
        raise ValueError("Input arrays dx, dy, and dz must be 3D or 4D.")

    # Sum the three components to obtain the divergence
    V = div_x + div_y + div_z

    return V
