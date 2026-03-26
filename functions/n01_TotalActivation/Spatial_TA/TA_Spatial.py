import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full
from functions.n01_TotalActivation.Spatial_TA.div3D_full import div3D_full
from functions.n01_TotalActivation.Spatial_TA.evaluate_3D_TV import evaluate_3D_TV
from functions.n01_TotalActivation.Spatial_TA.MyProx import MyProx


# def ta_spatial(y, param):
#
#     # Choose backend based on the 'cuda' flag
#
#     # Set default values if fields are missing in param
#     # if 'NitSpat' not in param:
#     #     param['NitSpat'] = 400
#     # if 'LambdaSpat' not in param:
#     #     param['LambdaSpat'] = 2
#     # if 'Lip' not in param:
#     #     param['Lip'] = 12
#     # if 'tol' not in param:
#     #     param['tol'] = 1e-6
#     param.setdefault('NitSpat', 400)
#     param.setdefault('LambdaSpat', 2)
#     param.setdefault('Lip', 12)
#     param.setdefault('tol', 1e-6)
#
#     # Verify the dimensionality of the input
#     if len(param['Dimension']) != 4:
#         raise ValueError("Dimension must have 4 elements.")
#
#     # Set weights if not provided
#     if 'weight_x' not in param:
#         param['weight_x'] = np.ones(param['Dimension'])
#     if 'weight_y' not in param:
#         param['weight_y'] = np.ones(param['Dimension'])
#     if 'weight_z' not in param:
#         param['weight_z'] = np.ones(param['Dimension'])
#
#     # Initialize the output for the spatial regularization problem
#     x_out = np.zeros_like(y)
#
#     # Create a 4D volume based on VoxelIdx coordinates
#     y_vol = np.zeros(param['Dimension'])
#
#
#     for i in range(len(param['VoxelIdx'])):
#         x1, y1, z1 = param['VoxelIdx'][i]
#         y_vol[x1, y1, z1, :] = y[:, i]
#
#     # Define the gradient and adjoint operators with backend support
#     Op = lambda x: gradient3D_full(x, param['weight_x'], param['weight_y'], param['weight_z'])
#     Adj_Op = lambda u, v, w: -div3D_full(u, v, w, param['weight_x'], param['weight_y'], param['weight_z'])
#     evaluate_norm = lambda y: evaluate_3D_TV(y)
#
#     # Apply the MyProx function to solve the TV regularization
#     x_vol = MyProx(y_vol, Op, Adj_Op, evaluate_norm, param)
#
#     # Map the result from the 4D volume back to 2D output format
#     for i in range(len(param['VoxelIdx'])):
#         x, y, z = param['VoxelIdx'][i]
#         x_out[:, i] = x_vol[x, y, z, :]
#
#     return x_out


def ta_spatial(y, param):
    # defaults
    param.setdefault('NitSpat', 400)
    param.setdefault('LambdaSpat', 2)
    param.setdefault('Lip', 12)
    param.setdefault('tol', 1e-6)

    dim = param['Dimension']  # (X,Y,Z,T)
    x_out = np.zeros_like(y)

    # Cache voxel indices as 3 int arrays (once)
    if 'VoxelIdx_xyz' not in param:
        vox = np.asarray(param['VoxelIdx'], dtype=np.intp)  # (V,3)
        param['VoxelIdx_xyz'] = (vox[:, 0], vox[:, 1], vox[:, 2])
    xi, yi, zi = param['VoxelIdx_xyz']

    # Build 4D volume without Python loop
    y_vol = np.zeros(dim, dtype=y.dtype)
    y_vol[xi, yi, zi, :] = y.T  # y is (T,V) so y.T is (V,T)

    Op = lambda x: gradient3D_full(x, param['weight_x'], param['weight_y'], param['weight_z'])
    Adj_Op = lambda u, v, w: -div3D_full(u, v, w, param['weight_x'], param['weight_y'], param['weight_z'])
    evaluate_norm = lambda yy: evaluate_3D_TV(yy)

    x_vol = MyProx(y_vol, Op, Adj_Op, evaluate_norm, param)

    # Map back without loop
    x_out[:] = x_vol[xi, yi, zi, :].T

    return x_out