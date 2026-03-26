import numpy as np

def map_vtv(Vin, Vin_i, Vout_i):
    Vout = np.zeros(Vout_i['dim'])
    x1, x2, x3 = np.meshgrid(np.arange(1, Vout_i['dim'][0]+1), np.arange(1, Vout_i['dim'][1]+1), np.arange(1, Vout_i['dim'][2]+1), indexing='ij')
    idx = np.arange(Vout.size)
    oob_list = []
    for i in idx:
        oob = False
        mm = Vout_i['mat'] @ [x1.ravel()[i], x2.ravel()[i], x3.ravel()[i], 1]
        vx = np.round(np.linalg.solve(Vin_i['mat'], mm)).astype(int)
        vx = np.clip(vx, 1, np.array(Vin_i['dim']))
        if not (1 <= vx[0] <= Vin_i['dim'][0] and 1 <= vx[1] <= Vin_i['dim'][1] and 1 <= vx[2] <= Vin_i['dim'][2]):
            oob_list.append(vx)
            oob = True
        Vout.ravel()[i] = Vin[vx[0]-1, vx[1]-1, vx[2]-1]
        if Vout.ravel()[i] < 0:
            print(f"Warning: Negative voxel value at {i}")
    return Vout
