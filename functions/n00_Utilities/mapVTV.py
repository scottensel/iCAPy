import numpy as np


def map_vtv(Vin, Vin_i, Vout_i):
    """
    Maps input data to a desired output dimension by resampling from the
    input image space into the output image space using affine transforms.

    For each voxel in the output space, its world-space position in mm is
    computed from the output affine matrix, then converted to the nearest
    voxel index in the input space. Out-of-bounds voxels are clamped to
    the input volume boundary.

    Inputs:
        Vin   - (X x Y x Z) 3D array with the data to map (input volume)
        Vin_i - dict with input image space information:
                  'dim' - 3-element list/array [X, Y, Z] of input dimensions
                  'mat' - (4 x 4) affine matrix mapping input voxel indices
                          to world-space mm coordinates
        Vout_i - dict with output image space information:
                  'dim' - 3-element list/array [X, Y, Z] of output dimensions
                  'mat' - (4 x 4) affine matrix for the output space

    Outputs:
        Vout - (X x Y x Z) 3D array in the output image space, with each
               voxel filled from the nearest voxel in Vin

    Implemented by Jonas Richiardi
    """
    # Initialise output volume to zero in the output image space
    Vout = np.zeros(Vout_i['dim'])

    # Generate all voxel coordinates in the output space (1-based, matching MATLAB)
    x1, x2, x3 = np.meshgrid(
        np.arange(1, Vout_i['dim'][0] + 1),
        np.arange(1, Vout_i['dim'][1] + 1),
        np.arange(1, Vout_i['dim'][2] + 1),
        indexing='ij'
    )
    idx = np.arange(Vout.size)   # map all voxels

    oob_list = []   # list of out-of-bound input voxels (for diagnostics)

    for i in idx:
        # Recover the world-space position (in mm) of this output voxel
        # from the output affine transform matrix
        mm = Vout_i['mat'] @ [x1.ravel()[i], x2.ravel()[i], x3.ravel()[i], 1]

        # Convert that world-space position into the nearest voxel index
        # in the input image space using the inverse of the input affine
        vx = np.round(np.linalg.solve(Vin_i['mat'], mm)).astype(int)

        # Clamp negative indices to 1 (minimum valid index)
        vx = np.clip(vx, 1, np.array(Vin_i['dim']))

        # Check if the mapped voxel falls outside the input volume bounds;
        # if so, record it and clamp to the boundary
        if not (1 <= vx[0] <= Vin_i['dim'][0] and
                1 <= vx[1] <= Vin_i['dim'][1] and
                1 <= vx[2] <= Vin_i['dim'][2]):
            oob_list.append(vx)

        # Assign the input voxel value to the output voxel (0-based indexing)
        Vout.ravel()[i] = Vin[vx[0] - 1, vx[1] - 1, vx[2] - 1]

        if Vout.ravel()[i] < 0:
            print(f"Warning: Negative voxel value at {i}")

    return Vout
