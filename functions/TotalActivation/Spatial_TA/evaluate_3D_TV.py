import numpy as np
from functions.TotalActivation.Spatial_TA.gradient3D_full import gradient3D_full

def evaluate_3D_TV(y):

    # Choose backend

    dx, dy, dz = gradient3D_full(y)

    amplitude = np.sqrt(abs(dx) ** 2 + abs(dy) ** 2 + abs(dz) ** 2)

    total_variation = np.sum(amplitude)

    return total_variation
