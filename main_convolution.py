from matplotlib import use
use("Qt5Agg")
import matplotlib.pyplot as plt
plt.ion()

from time import time

import numpy as np

from src.operators.convolution_operator import ConvolutionOperator
from src.solvers.fw import FW


def add_snr(y0, snr, N):
    signal_power = np.mean(np.square(y0))
    mse_db = 10 * np.log10(signal_power) - snr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse), N)
    y = y0 + w
    return y


def add_psnr(y0, psnr, N):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse), N)
    y = y0 + w
    return y

n = 8
fwhm = 0.02
# N = 300

max_iter = 100
eps=1e-5
amplitude_thresh = 1e-4
certificate_tol = 1e-3

swarm_particles = 100

fig_path = "/home/jarret/Documents/EPFL/PhD/thesis/manuscript/figures/cd-pfw/reco1d"



if __name__ == '__main__':
    np.random.seed(100)

    x0 = np.random.uniform(0.1, 0.9, n)
    a0 = np.random.uniform(1, 3, n)

    # x0 = np.array([0.1, 0.25, 0.5, 0.7, 0.9])
    # a0 = np.array([1, 1.5, 3.5, 2, 5])
    # a0 = np.array([1, 1, 1, 1, 1])

    # x0 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    # a0 = np.array([3, 4.5, 1.5, 3, 4, 3, 1, 2.5, 1, 1.5, 1, 1, 1, 3, 1, 1])

    x_dim = 1
    bounds = np.array([0, 1])
    # bounds = np.array([-1, 1])

    # Full width at half maximum
    forward_op = ConvolutionOperator(x0, fwhm, bounds, n_measurements_per_gaussan=5)
    N = forward_op.n_measurements

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    lambdas = [0.002, 0.01, 0.02, 0.1]

    options = {"initialization": "smoothing", "polyatomic": False, "swarm": False, "sliding": True, "positivity_constraint": True,
               "max_iter": max_iter, "dual_certificate_tol": certificate_tol, "smooth_sigma": 5, "correction_eps": eps, "amplitude_threshold": amplitude_thresh}
    sfw_solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    sfw_solver.fit()
    print("Time: ", time() - t1)
    sfw_solver.time_results()
    sfw_solver.plot(x0, a0)
    sfw_solver.flat_norm_results(x0, a0, lambdas)
    sfw_solver.plot_solution(x0, a0)
    print("objective with SFW = ", sfw_solver.blasso_objective_val()[0])

    options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": False, "positivity_constraint": True,
               "max_iter": max_iter, "dual_certificate_tol": certificate_tol, "smooth_sigma": 5, "animation": False, "correction_eps": eps, "amplitude_threshold": amplitude_thresh}
    pfw_solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    pfw_solver.fit()
    print("Time: ", time() - t1)
    pfw_solver.time_results()
    pfw_solver.plot(x0, a0)
    pfw_solver.flat_norm_results(x0, a0, lambdas)
    pfw_solver.plot_solution(x0, a0)
    print("objective with PFW = ", pfw_solver.blasso_objective_val()[0])

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(y.shape[0])/y.shape[0], y, label="Interpolation", lw=1, alpha=.7, zorder=100)
    plt.scatter(np.arange(y.shape[0])/y.shape[0], y, label="Measurements", marker='+', zorder=101)
    for i, xpos in enumerate(x0):
        if i == 0:
            plt.axvline(xpos, color="gray", ls="--", label='Ground truth locations', alpha=.6)
        else:
            plt.axvline(xpos, color="gray", ls="--", alpha=.6)
    plt.axhline(0, color="gray", alpha=.6, lw=1)
    plt.xlim(bounds)
    plt.yticks([0])
    plt.legend()
    # plt.savefig(fig_path + "/" + "gaussian_measurements" + ".pdf")
    plt.show()

    sfw_sol = sfw_solver.solution()  # (x, a)
    pfw_sol = pfw_solver.solution()  # (x, a)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharey=True, sharex=True)
    for ax, sol, title in zip(axes.flat, [sfw_sol, pfw_sol], ["PFW", "SFW"]):
        x, a = sol
        ax.stem(x0, a0, linefmt='k.--', markerfmt='ko', basefmt=" ", label='Ground Truth')
        ax.stem(x, a, linefmt='r.--', markerfmt='rx', basefmt=" ", label='Reconstruction')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    plt.xlim(bounds)
    # plt.suptitle("Ground Truth vs Reconstruction")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(fig_path + "/" + "gaussian_reco" + ".pdf")
    plt.show()