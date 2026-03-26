import numpy as np
from functions.n01_TotalActivation.Temporal_TA.filter_boundary import filter_boundary


def deconv_1d_gtv(y, n, lambda_val, Nit, maxeig, d=1, p_fig=0, noise_estimate=1e-4):
    stop_cri = 1e-6
    N = len(y)
    z, s = np.zeros(N), np.zeros(N)
    t = 1
    J, nv, Lambda = np.zeros(Nit), np.zeros(Nit), np.zeros(Nit)
    precision = noise_estimate / 100000

    for k in range(Nit):
        z_l = z.copy()
        z = 1 / (lambda_val * maxeig) * filter_boundary(n, d, y, 'normal') + \
            s - filter_boundary(n, d, filter_boundary(n, d, s, 'transpose'), 'normal') / maxeig
        z = np.clip(z, -1, 1)

        t_l, t = t, (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        s = z + (t_l - 1) / t * (z - z_l)

        x = y - lambda_val * filter_boundary(n, d, z, 'transpose')
        J[k] = np.sum((x - y) ** 2) + lambda_val * np.sum(np.abs(filter_boundary(n, d, x, 'normal')))
        nv[k] = np.sqrt(np.sum((x - y) ** 2) / N)

        if p_fig:
            import matplotlib.pyplot as plt
            plt.plot(y, '-g', label="Original")
            plt.plot(x, '-r', label="Filtered")
            plt.pause(0.2)
            plt.clf()

        if k > 2 and abs(J[k - 1] - J[k - 2]) < stop_cri:
            break

        if abs(nv[k] - noise_estimate) > precision:
            lambda_val = lambda_val * noise_estimate / nv[k]

        Lambda[k] = lambda_val

    return x, nv, J, Lambda
