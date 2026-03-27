import numpy as np
from functions.n01_TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full
from functions.n01_TotalActivation.Spatial_TA.div3D_full import div3D_full
from functions.n01_TotalActivation.Spatial_TA.evaluate_3D_TV import evaluate_3D_TV
from functions.n01_TotalActivation.Spatial_TA.MyProx import MyProx


def ta_spatial(y, param):
    """
    Computes the TV regularization:
        F(x) = min ||y - x||^2 + lambda * ||TV{x}||_1
    using the 3D extension of FISTA:

        Beck, A., & Teboulle, M. (2009). A fast iterative
        shrinkage-thresholding algorithm for linear inverse problems.
        SIAM journal on imaging sciences, 2(1), 183-202.

    Inputs:
        y     - (n_time_points x n_ret_voxels) 2D matrix of data input
                to the spatial regularization
        param - dict containing relevant TA parameters:
            'NitSpat'    - int, number of iterations for spatial
                           regularization; default 400
            'LambdaSpat' - float, empirically tuned regularization
                           coefficient; default 2
            'Lip'        - float, Lipschitz constant of the gradient
                           operator; default 12
            'tol'        - float, tolerance threshold below which
                           iterations stop; default 1e-6
            'VoxelIdx'   - (n_ret_vox x 3) array of 3D coordinates of
                           the retained voxels used in TA
            'Dimension'  - 4-element list/array of X, Y, Z, T sizes
            'weight_x'   - (X x Y x Z x T) spatial weight along X;
                           set to ones if not provided
            'weight_y'   - spatial weight along Y
            'weight_z'   - spatial weight along Z

    Outputs:
        x_out - (n_time_points x n_ret_voxels) 2D matrix of spatially
                regularized output

    Implemented by Younes Farouj, 10.03.2016
    """

    # Set default parameter values if not provided
    param.setdefault('NitSpat',    400)
    param.setdefault('LambdaSpat', 2)
    param.setdefault('Lip',        12)
    param.setdefault('tol',        1e-6)

    dim   = param['Dimension']   # (X, Y, Z, T)

    # param['weight'] is 1 if neighbouring elements are from the same
    # tissue type, and less than 1 otherwise
    param.setdefault('weight_x', np.ones(dim))
    param.setdefault('weight_y', np.ones(dim))
    param.setdefault('weight_z', np.ones(dim))

    # This is going to be the output of the spatial regularization problem
    x_out = np.zeros_like(y)

    # Cache voxel index arrays as integer index tuples (computed once)
    if 'VoxelIdx_xyz' not in param:
        vox = np.asarray(param['VoxelIdx'], dtype=np.intp)   # (V, 3)
        param['VoxelIdx_xyz'] = (vox[:, 0], vox[:, 1], vox[:, 2])
    xi, yi, zi = param['VoxelIdx_xyz']

    # Convert the 2D input (T x V) into a 4D volume (X x Y x Z x T) by
    # placing each voxel's time course at its 3D coordinates
    y_vol          = np.zeros(dim, dtype=y.dtype)
    y_vol[xi, yi, zi, :] = y.T   # y is (T, V) so y.T is (V, T)

    # Gradient operator: Op(x) = gradient3D_full(x, wx, wy, wz)
    Op = lambda x: gradient3D_full(
        x, param['weight_x'], param['weight_y'], param['weight_z']
    )

    # Adjoint operator: minus divergence — <f, grad g> = -<div f, u>
    Adj_Op = lambda u, v, w: -div3D_full(
        u, v, w, param['weight_x'], param['weight_y'], param['weight_z']
    )

    # TV norm evaluator: evaluate_norm(y) = sum(||grad y||_2)
    evaluate_norm = lambda yy: evaluate_3D_TV(yy)

    # Apply the proximal operator to solve the TV regularization problem
    x_vol = MyProx(y_vol, Op, Adj_Op, evaluate_norm, param)

    # Map the regularized 4D volume back to the 2D output format
    x_out[:] = x_vol[xi, yi, zi, :].T

    return x_out
