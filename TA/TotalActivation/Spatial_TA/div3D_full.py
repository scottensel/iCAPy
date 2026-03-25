import numpy as np
try:
    import cupy as cp  # Import CuPy if available
except ImportError:
    cp = None  # Set to None if CuPy is not available
def div3D_full(dx, dy, dz, wx=None, wy=None, wz=None, use_cuda=False):
    """
    Compute the divergence of a 3D or 4D gradient field.

    Parameters:
    - dx, dy, dz: 3D or 4D numpy arrays representing the gradient components.
    - wx, wy, wz: Optional weights for the x, y, and z components.

    Returns:
    - V: The divergence of the input gradient field.

            V = padarray( dx(1:end-1,:,:,:), [1 0 0 0], 'post' ) - ...
            padarray( dx(1:end-1,:,:,:), [1 0 0 0], 'pre' ) + ...
            padarray( dy(:,1:end-1,:,:), [0 1 0 0], 'post' ) - ...
            padarray( dy(:,1:end-1,:,:), [0 1 0 0], 'pre' )  + ...
            padarray( dz(:,:,1:end-1,:), [0 0 1 0], 'post' ) - ...
            padarray( dz(:,:,1:end-1,:), [0 0 1 0], 'pre' );

    """
    # Choose backend
    xp = cp if use_cuda and cp is not None else np

    # Convert to backend arrays
    dx = xp.array(dx)
    dy = xp.array(dy)
    dz = xp.array(dz)

    # Apply weights if provided
    if wx is not None:
        dx *= xp.array(wx)
    if wy is not None:
        dy *= xp.array(wy)
    if wz is not None:
        dz *= xp.array(wz)

    if dx.ndim == 3:
        # ---- X term: dx(1:end-1, :, :) ----
        Ex = dx[:-1, :, :]  # dx(1:end-1,:,:)
        Ex_post = xp.pad(Ex, ((0, 1), (0, 0), (0, 0)), mode='constant')  # 'post' along axis 0
        Ex_pre = xp.pad(Ex, ((1, 0), (0, 0), (0, 0)), mode='constant')  # 'pre'  along axis 0
        div_x = Ex_post - Ex_pre

        # ---- Y term: dy(:, 1:end-1, :) ----
        Ey = dy[:, :-1, :]  # dy(:,1:end-1,:)
        Ey_post = xp.pad(Ey, ((0, 0), (0, 1), (0, 0)), mode='constant')  # 'post' along axis 1
        Ey_pre = xp.pad(Ey, ((0, 0), (1, 0), (0, 0)), mode='constant')  # 'pre'  along axis 1
        div_y = Ey_post - Ey_pre

        # ---- Z term: dz(:, :, 1:end-1) ----
        Ez = dz[:, :, :-1]  # dz(:,:,1:end-1)
        Ez_post = xp.pad(Ez, ((0, 0), (0, 0), (0, 1)), mode='constant')  # 'post' along axis 2
        Ez_pre = xp.pad(Ez, ((0, 0), (0, 0), (1, 0)), mode='constant')  # 'pre'  along axis 2
        div_z = Ez_post - Ez_pre

    elif dx.ndim == 4:
        # ---- X term: dx(1:end-1, :, :, :) ----
        Ex = dx[:-1, :, :, :]  # dx(1:end-1,:,:,:)
        Ex_post = xp.pad(Ex, ((0, 1), (0, 0), (0, 0), (0, 0)), mode='constant')  # 'post' along axis 0
        Ex_pre = xp.pad(Ex, ((1, 0), (0, 0), (0, 0), (0, 0)), mode='constant')  # 'pre'  along axis 0
        div_x = Ex_post - Ex_pre

        # ---- Y term: dy(:, 1:end-1, :, :) ----
        Ey = dy[:, :-1, :, :]  # dy(:,1:end-1,:,:)
        Ey_post = xp.pad(Ey, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant')  # 'post' along axis 1
        Ey_pre = xp.pad(Ey, ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant')  # 'pre'  along axis 1
        div_y = Ey_post - Ey_pre

        # ---- Z term: dz(:, :, 1:end-1, :) ----
        Ez = dz[:, :, :-1, :]  # dz(:,:,1:end-1,:)
        Ez_post = xp.pad(Ez, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')  # 'post' along axis 2
        Ez_pre = xp.pad(Ez, ((0, 0), (0, 0), (1, 0), (0, 0)), mode='constant')  # 'pre'  along axis 2
        div_z = Ez_post - Ez_pre

    else:
        raise ValueError("Input arrays dx, dy, and dz must be 3D or 4D.")

    # Sum components to get the divergence
    V = div_x + div_y + div_z

    return V