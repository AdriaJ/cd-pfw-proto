from matplotlib import use
use("Qt5Agg")

from time import time

import matplotlib.pyplot as plt
import numpy as np

from src.operators.fourier_operator import FourierOperator
from src.solvers.fw import FW

def add_snr(y0, snr, N):
    signal_power = np.mean(np.square(y0))
    mse_db = 10 * np.log10(signal_power) - snr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse / 2), (N, 2))
    y = y0 + w
    return y


def add_psnr(y0, psnr, N):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse / 2), (N, 2))
    y = y0 + w
    return y

n = 8
N = 20 * n
# N = 300

fmax = 100

max_iter = 100
eps=1e-4
amplitude_thresh = 1e-4
certificate_tol = 1e-3

swarm_particles = 100
fig_path = "/home/jarret/Documents/EPFL/PhD/thesis/manuscript/figures/cd-pfw/reco1d"


if __name__ == '__main__':
    plt.ion()
    np.random.seed(10)

    x0 = np.random.uniform(0.05, 0.95, n)
    a0 = np.random.uniform(1, 3, n)

    # x0 = np.array([0.1, 0.25, 0.5, 0.7, 0.9])
    # a0 = np.array([1, 1, 1, 1, 1])

    # x0 = np.array([-.5,-.1,.1,.5])
    # a0 = np.array([0.8,0.8,0.8,0.8])
    # a0 = np.array([1, 1.5, 0.5, 2, 5])
    # a0 = np.array([1, 15, 0.5, -3, 5])

    # x0 = np.array([0.2, 0.5, 0.8])
    # a0 = np.array([1, 2, 1.5])

    # x0 = np.random.uniform(-0.95, 0.95, 30)
    # a0 = np.random.uniform(0.5, 3, 30)

    # x0 = np.array([0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    # a0 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    # a0 = np.array([-1, 0.5, 1, 1, 1, 3, 1, 1])

    # x0 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    # a0 = np.array([3, 4.5, -1.5, -3, 4, 3, 1, 2.5, -1, -1.5, 1, 1, 1, 3, 1, 1])
    # a0 = np.abs(a0)

    bounds = np.array([0, 1])
    # bounds = np.array([-1, 1])

    # N = 30 * len(x0)
    # freq_bounds = np.array([-10, 10])
    freq_bounds = np.array([-fmax, fmax])
    forward_op = FourierOperator.get_RandomFourierOperator(x0, N, freq_bounds)

    # x = np.random.normal(size=forward_op.dim_shape)
    # for i in range(1000):
    #     x = forward_op.adjoint(forward_op(x))
    #     x = x / np.linalg.norm(x)
    # np.linalg.norm(forward_op.adjoint(forward_op(x))) / np.linalg.norm(x)

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)
    # y = add_snr(y0, psnr, N)

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    x_dim = 1

    lambdas = [0.002, 0.01, 0.02, 0.1]

    options = {"swarm": False, "n_particles": swarm_particles, "initialization": "smoothing", "polyatomic": False, "sliding": True, "positivity_constraint": True,
               "max_iter": max_iter, "dual_certificate_tol": certificate_tol, "smooth_sigma": 4, "n_particles": 10, "correction_eps": eps, "amplitude_threshold": amplitude_thresh}
    sfw_solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    sfw_solver.fit()
    print("Time: ", time() - t1)
    sfw_solver.time_results()
    sfw_solver.plot(x0, a0)
    sfw_solver.flat_norm_results(x0, a0, lambdas)
    # sfw_solver.plot_solution(x0, a0)
    print("objective with SFW = ", sfw_solver.blasso_objective_val()[0])

    sfw_sol = sfw_solver.solution()  # (x, a)


    options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": False, "positivity_constraint": True,
               "max_iter": max_iter, "dual_certificate_tol": certificate_tol, "smooth_sigma": 4, "correction_eps": eps,
               "amplitude_threshold": amplitude_thresh}
    # options = {"initialization": "grid", "polyatomic": False, "swarm": False, "sliding": True,
    #            "positivity_constraint": False, "max_iter": 20, "dual_certificate_tol": 1e-2, "smooth_sigma": 4, "n_particles": 10}
    pfw_solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    pfw_solver.fit()
    print("Time: ", time() - t1)
    pfw_solver.time_results()
    pfw_solver.plot(x0, a0)
    pfw_solver.flat_norm_results(x0, a0, lambdas)
    # pfw_solver.plot_solution(x0, a0)

    pfw_sol = pfw_solver.solution()  # (x, a)

    df = pfw_solver.data_fid(pfw_sol[0])
    objective = df(pfw_sol[1]) + lambda_ * np.linalg.norm(pfw_sol[1], 1)
    # fop = forward_op.get_new_operator(pfw_sol[0])
    # df2 = .5 * np.linalg.norm(fop(pfw_sol[1]) - y) ** 2
    print("objective with PFW = ", pfw_solver.blasso_objective_val()[0])

    # Plots

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharey=True, sharex=True)
    for ax, sol, title in zip(axes.flat, [pfw_sol, sfw_sol], ["PFW", "SFW"]):
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
    # plt.savefig(fig_path + "/" + "fourier_meas" + ".pdf")
    plt.show()

    dirty_im_f = forward_op.adjoint_function(y)
    X = np.linspace(0, 1, 1000)
    dirty_im = dirty_im_f(X)
    plt.figure(figsize=(10, 4))
    plt.plot(X, dirty_im, zorder=100)
    for xpos in x0:
        plt.axvline(xpos, color="gray", ls="--", label='Ground truth locations', alpha=.6)
    plt.axhline(0, color="gray", alpha=.6, lw=1)
    plt.xlim(bounds)
    plt.yticks([0])
    # plt.grid(True)
    # plt.savefig(fig_path + "/" + "dirty_image" + ".pdf")
    plt.show()

