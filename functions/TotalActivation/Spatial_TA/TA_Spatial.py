import numpy as np
from functions.TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full
from functions.TotalActivation.Spatial_TA.div3D_full import div3D_full
from functions.TotalActivation.Spatial_TA.evaluate_3D_TV import evaluate_3D_TV
from functions.TotalActivation.Spatial_TA.MyProx import MyProx
try:
    import cupy as cp  # Import CuPy for GPU support if available
except ImportError:
    cp = None

def ta_spatial(y, param):

    # Choose backend based on the 'cuda' flag
    xp = cp if param.get('use_cuda', False) and cp is not None else np

    # Set default values if fields are missing in param
    if 'NitSpat' not in param:
        param['NitSpat'] = 400
    if 'LambdaSpat' not in param:
        param['LambdaSpat'] = 2
    if 'Lip' not in param:
        param['Lip'] = 400
    if 'tol' not in param:
        param['tol'] = 1e-6

    # Verify the dimensionality of the input
    if len(param['Dimension']) != 4:
        raise ValueError("Dimension must have 4 elements.")

    # Set weights if not provided
    if 'weight_x' not in param:
        param['weight_x'] = xp.ones(param['Dimension'])
    if 'weight_y' not in param:
        param['weight_y'] = xp.ones(param['Dimension'])
    if 'weight_z' not in param:
        param['weight_z'] = xp.ones(param['Dimension'])

    # Initialize the output for the spatial regularization problem
    x_out = xp.zeros_like(y)

    # Create a 4D volume based on VoxelIdx coordinates
    y_vol = xp.zeros(param['Dimension'])


    for i in range(len(param['VoxelIdx'])):
        x1, y1, z1 = param['VoxelIdx'][i]
        y_vol[x1, y1, z1, :] = y[:, i]

    # Define the gradient and adjoint operators with backend support
    Op = lambda x: gradient3D_full(x, param['weight_x'], param['weight_y'], param['weight_z'], use_cuda=(xp == cp))
    Adj_Op = lambda u, v, w: -div3D_full(u, v, w, param['weight_x'], param['weight_y'], param['weight_z'], use_cuda=(xp == cp))
    evaluate_norm = lambda y: evaluate_3D_TV(y, use_cuda=(xp == cp))

    # Apply the MyProx function to solve the TV regularization
    x_vol = MyProx(y_vol, Op, Adj_Op, evaluate_norm, param)

    # Map the result from the 4D volume back to 2D output format
    for i in range(len(param['VoxelIdx'])):
        x, y, z = param['VoxelIdx'][i]
        x_out[:, i] = x_vol[x, y, z, :]

    x_out = x_out.get() if (xp == cp) and cp is not None and isinstance(x_out, cp.ndarray) else x_out

    return x_out
