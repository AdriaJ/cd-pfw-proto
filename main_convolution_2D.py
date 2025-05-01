from time import time

from matplotlib import use
use("Qt5Agg")
import matplotlib.pyplot as plt
plt.ion()

import numpy as np

from src.operators.convolution_operator import ConvolutionOperator
from src.solvers.fw import FW


def add_psnr(y0, psnr, N):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse), N)
    y = y0 + w
    return y

correction_eps = 1e-4
amplitude_thresh = 1e-6
certificate_tol = 1e-2

fig_path = "/home/jarret/Documents/EPFL/PhD/thesis/manuscript/figures/cd-pfw/reco2d"


if __name__ == '__main__':
    np.random.seed(10)

    # x0 = np.array([[0.1, 0.1], [0.2, 0.5], [0.75, 0.25], [0.1, 0.9], [0.5, 0.5]])
    # a0 = np.array([1, 1.5, 2.5, 2, 3])

    n = 10
    x0 = np.random.uniform(0.15, 0.85, size=(n, 2))
    a0 = np.random.uniform(1, 3, n)

    x_dim = 2
    bounds = np.array([0, 1])

    fwhm = 0.1
    n_measurements_per_gaussan = 10
    forward_op = ConvolutionOperator(x0, fwhm, bounds, x_dim, n_measurements_per_gaussan)
    N = forward_op.n_measurements
    print("N = ", N)

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)

    # n = np.sqrt(N).astype(int)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    # im = ax1.imshow(np.rot90(y0.reshape(n, n)), extent=(0, 1, 0, 1))
    # plt.colorbar(im, ax=ax1)
    # ax1.scatter(x0[:, 0], x0[:, 1], s=np.abs(a0) * 20, marker="+", c='k', label='Ground Truth')
    # ax1.set_title("Ground Truth")
    #
    # im = ax2.imshow(np.rot90(y.reshape(n, n)), extent=(0, 1, 0, 1))
    # plt.colorbar(im, ax=ax2)
    # ax2.scatter(x0[:, 0], x0[:, 1], s=np.abs(a0) * 20, marker="+", c='k', label='Ground Truth')
    # ax2.set_title("Observed Noisy Signal")
    # plt.show()
    # exit(0)

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    lambdas = [0.001, 0.01, 0.02, 0.1]

    options = {"initialization": "smoothing", "polyatomic": False, "swarm": True, "sliding": True, "positivity_constraint": True,
               "max_iter": 100, "dual_certificate_tol": certificate_tol, "smooth_sigma": 2, "correction_eps": correction_eps,
               "amplitude_threshold": amplitude_thresh}
    sfw_solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=None, show_progress=False, options=options)
    t1 = time()
    sfw_solver.fit()
    print("Time: ", time() - t1)
    sfw_solver.time_results()
    # solver.plot(x0, a0)
    sfw_solver.flat_norm_results(x0, a0, lambdas)
    sfw_solver.plot_solution(x0, a0)
    print("objective with SFW = ", sfw_solver.blasso_objective_val()[0])

    options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": False, "positivity_constraint": True,
               "max_iter": 100, "dual_certificate_tol": certificate_tol, "smooth_sigma": 2, "correction_eps": correction_eps,
               "amplitude_threshold": amplitude_thresh}
    pfw_solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=None, show_progress=False, options=options)
    t1 = time()
    pfw_solver.fit()
    print("Time: ", time() - t1)
    pfw_solver.time_results()
    # solver.plot(x0, a0)
    pfw_solver.flat_norm_results(x0, a0, lambdas)
    pfw_solver.plot_solution(x0, a0)
    print("objective with PFW = ", pfw_solver.blasso_objective_val()[0])

    # ------------------------------------------
    # Plots
    # ------------------------------------------
    nside_plot = 100
    dirty_im_cont = forward_op.adjoint_function(y)
    grid1 = np.linspace(bounds[0], bounds[1], nside_plot)[1:-1]
    grid2 = np.linspace(bounds[0], bounds[1], nside_plot)[1:-1]
    xx, yy = np.meshgrid(grid1, grid2)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    eta = dirty_im_cont(grid)/lambda_

    print(np.abs(eta).max())

    yside = int(np.sqrt(y.shape[0]))
    measim = y.reshape((yside, yside)).T

    solvers = [pfw_solver, sfw_solver]
    names = ["PFW", "SFW"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 15), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        x, a = solvers[i].solution()
        # im_sh = ax.imshow(eta.reshape((nside_plot-2, nside_plot-2)), origin="lower", interpolation="none",
        #                   extent=(bounds[0], bounds[1], bounds[0], bounds[1]))
        im_sh = ax.imshow(measim, origin="lower", interpolation="none",
                          extent=(bounds[0], bounds[1], bounds[0], bounds[1]))
        ax.scatter(x0[:, 0], x0[:, 1], s=np.abs(a0) * 50, marker="+", c='k', label='Ground Truth')
        ax.scatter(x[:, 0], x[:, 1], s=np.abs(a) * 50, marker="x", c='r', label='Reconstruction')
        ax.grid(True)
        ax.set_title(names[i])
        ax.label_outer(True)
    cbar_ax = fig.add_axes([0.2, 0.92, 0.6, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(im_sh, cax=cbar_ax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')  # Move ticks to top
    cbar.ax.xaxis.set_label_position('top')
    # plt.tight_layout(rect=[0, 0, 1, 0.85])  # Leave space at top for legend
    plt.savefig(fig_path + "/" + "gaussian_reco" + ".pdf", bbox_inches='tight')
    plt.show()

    zoom_area = [0.55, 0.85, 0.55, 0.85]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        x, a = solvers[i].solution()
        # image = eta.reshape((nside_plot-2, nside_plot-2))[int(np.round(zoom_area[0] * (nside_plot-2))): int(np.round(zoom_area[1] * (nside_plot-2))),
        #         int(np.round(zoom_area[2] * (nside_plot-2))):int(np.round(zoom_area[3] * (nside_plot-2)))]
        image = measim[
                int(np.round(zoom_area[0] * yside)): int(np.round(zoom_area[1] * yside)),
                int(np.round(zoom_area[2] * yside)):int(np.round(zoom_area[3] * yside))]
        im_sh = ax.imshow(image, origin="lower", interpolation="none",
                          extent=zoom_area, vmin=measim.min(), vmax=measim.max())
        ax.scatter(x0[:, 0], x0[:, 1], s=np.abs(a0) * 50, marker="+", c='k', label='Ground Truth')
        ax.scatter(x[:, 0], x[:, 1], s=np.abs(a) * 50, marker="x", c='r', label='Reconstruction')
        ax.grid(True)
        ax.set_title(names[i])
        ax.label_outer(True)
        ax.set_xlim(zoom_area[0], zoom_area[1])
        ax.set_ylim(zoom_area[2], zoom_area[3])
    # cbar_ax = fig.add_axes([0.2, 0.92, 0.6, 0.015])  # [left, bottom, width, height]
    # cbar = fig.colorbar(im_sh, cax=cbar_ax, orientation='horizontal')
    # cbar.ax.xaxis.set_ticks_position('top')  # Move ticks to top
    # cbar.ax.xaxis.set_label_position('top')
    # plt.tight_layout(rect=[0, 0, 1, 0.85])  # Leave space at top for legend
    # plt.savefig(fig_path + "/" + "gaussian_reco_zoom" + ".pdf", bbox_inches='tight')
    plt.show()

    # ------------------------------------------
    # Tests
    # ------------------------------------------

    # from src.operators.dual_certificate import DualCertificate, SmoothDualCertificate
    # from scipy.optimize import minimize
    #
    # mst = sfw_solver._mstate
    # dual_cert = DualCertificate(mst["x"], mst["a"], sfw_solver.y,
    #                             sfw_solver.forward_op, sfw_solver.lambda_,
    #                             sfw_solver.positive_constraint, sfw_solver.x_dim)
    # xinit = mst['particles'][0]
    # opti = minimize(lambda x: -1. * dual_cert.apply(x)[0], xinit, method="BFGS",
    #                 jac= lambda x : -1. * dual_cert.grad(x).ravel())
    #
    # print(xinit, opti.x)
    # print(dual_cert.apply(xinit), dual_cert.apply(opti.x))
    #
    # from joblib  import Parallel, delayed, parallel_backend
    #
    # def obj_fun(x):
    #     return -1. * dual_cert.apply(x)[0]
    # def grad_fun(x):
    #     return -1. * dual_cert.grad(x).ravel()
    #
    # def f(xinit, x, a, y, lambda_, pos, dim, fwhm, bounds, n_measurements_per_gaussan):
    #     op = ConvolutionOperator(x, fwhm, bounds, dim, n_measurements_per_gaussan)
    #     dual_cert = DualCertificate(x, a, y, op, lambda_, pos, dim)
    #
    #     # def obj_fun(x):
    #     #     return -1. * dual_cert.apply(x)[0]
    #     #
    #     # def grad_fun(x):
    #     #     return -1. * dual_cert.grad(x).ravel()
    #     # opti = minimize(obj_fun, xinit, method="BFGS", jac=grad_fun)
    #     opti = minimize(lambda x: -1. * dual_cert.apply(x)[0], xinit, method="BFGS",
    #                     jac=lambda x: -1. * dual_cert.grad(x).ravel())
    #     xout = opti.x
    #     return xout
    #
    # outputs_par = Parallel(n_jobs=-1)(delayed(f)(xinit, mst["x"], mst["a"], sfw_solver.y,
    #                                              sfw_solver.lambda_,
    #                                              sfw_solver.positive_constraint, sfw_solver.x_dim,
    #                                              fwhm, bounds, n_measurements_per_gaussan) for xinit in mst["particles"])
    # print(np.array(outputs_par))
    # print(dual_cert.apply(np.array(outputs_par)))
    #
    # def f(xinit):
    #     # def obj_fun(x):
    #     #     return -1. * dual_cert.apply(x)[0]
    #     #
    #     # def grad_fun(x):
    #     #     return -1. * dual_cert.grad(x).ravel()
    #     # opti = minimize(obj_fun, xinit, method="BFGS", jac=grad_fun)
    #
    #     opti = minimize(lambda x: -1. * dual_cert.apply(x)[0], xinit, method="BFGS",
    #                     jac=lambda x: -1. * dual_cert.grad(x).ravel())
    #     xout = opti.x
    #     return xout
    # with parallel_backend('threading'):
    #     outputs_par = Parallel(n_jobs=-1)(
    #         delayed(f)(xinit) for xinit in mst["particles"]
    #     )
    # print(np.array(outputs_par))
    #
    # # mst = sfw_solver._mstate
    # # dual_cert = DualCertificate(mst["x"], mst["a"], sfw_solver.y,
    # #                             sfw_solver.forward_op, sfw_solver.lambda_,
    # #                             sfw_solver.positive_constraint, sfw_solver.x_dim)
    # # import multiprocessing
    # # multiprocessing.set_start_method('fork', force=True)
    # #
    # # def f(xinit):
    # #     def obj_fun(x):
    # #         return -1. * dual_cert.apply(x)[0]
    # #
    # #     def grad_fun(x):
    # #         return -1. * dual_cert.grad(x).ravel()
    # #     opti = minimize(obj_fun, xinit, method="BFGS", jac=grad_fun)
    # #     # opti = minimize(lambda x: -1. * dual_cert.apply(x)[0], xinit, method="BFGS",
    # #                     # jac=lambda x: -1. * dual_cert.grad(x).ravel())
    # #     xout = opti.x
    # #     return xout
    # # outputs_par = Parallel(n_jobs=-1)(delayed(f)(xinit) for xinit in mst["particles"])
    #

    # from src.operators.dual_certificate import DualCertificate, SmoothDualCertificate
    # mst = pfw_solver._mstate
    # n_grid = 200
    # grid1 = np.linspace(bounds[0], bounds[1], n_grid)
    # grid2 = np.linspace(bounds[0], bounds[1], n_grid)
    # xx, yy = np.meshgrid(grid1, grid2)
    # grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    # # smoothing_plot = 1 if self.initialization == "smoothing" and not self.swarm else 0
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # # Initial dual certificate
    # dual_cert = DualCertificate(mst["x"], mst["a"], y, forward_op, lambda_, True)
    # eta = dual_cert(grid).reshape(n_grid, n_grid)
    # i = 0
    # s1 = axs[0].scatter(mst["iter_candidates"][-1][:, 0], mst["iter_candidates"][-1][:, 1], marker="x",
    #                        s=np.ones_like(mst["iter_candidates"][-1][:, 0]) * 50, c='k', label='Candidates')
    # im = axs[0].imshow(eta, label='Dual Certificate', cmap='viridis', origin='lower',
    #                       extent=(bounds[0], bounds[1], bounds[0], bounds[1]))
    # # plt.colorbar(im)
    # axs[0].set_title(f"Candidates - Iteration {i + 1}")
    # axs[0].grid(True)
    #
    # g, z_smooth = mst["smooth_dual_certificate"][-1]
    # axs[-1].imshow(z_smooth, label='Smooth Dual Certificate', cmap='viridis', origin='lower',
    #                   extent=(bounds[0], bounds[1], bounds[0], bounds[1]))
    # axs[-1].scatter(mst["smooth_peaks"][-1][:, 0], mst["smooth_peaks"][-1][:, 1], marker="x",
    #                    c='r', label='Reconstruction')
    # axs[-1].set_title(f"Smooth Dual Certificate - Iteration {i + 1}")
    # axs[-1].grid(True)
    #
    # dual_cert = DualCertificate(mst["iter_x"][-1], mst["iter_a"][-1], y, forward_op, lambda_, True)
    # eta = dual_cert(grid).reshape(n_grid, n_grid)
    # im = axs[1].imshow(eta, label='Dual Certificate', cmap='viridis', origin='lower',
    #                       extent=(bounds[0], bounds[1], bounds[0], bounds[1]))
    # # plt.colorbar()
    # s2 = axs[1].scatter(x0[:, 0], x0[:, 1], marker="+", s=np.abs(a0) * 50, c='k', label='Ground Truth')
    # s3 = axs[1].scatter(mst["iter_x"][-1][:, 0], mst["iter_x"][-1][:, 1], marker="x",
    #                        s=np.abs(mst["iter_a"][-1]) * 50, c='r', label='Reconstruction')
    # axs[1].set_title(f"Correction - Iteration {i + 1}")
    # axs[1].grid(True)
    # axs[1].set_zorder(1)
    # axs[1].set_frame_on(False)
    # plt.show()

    # if need_extra_plot:
    #     idx += 1
    #
    #     dual_cert = DualCertificate(mst["iter_x"][idx], mst["iter_a"][idx], self.y, self.forward_op,
    #                                 self.lambda_, self.positive_constraint)
    #     eta = dual_cert(grid).reshape(n_grid, n_grid)
    #     im = axs[i, 2].imshow(eta, label='Dual Certificate', cmap='viridis', origin='lower',
    #                           extent=(self.bounds[0], self.bounds[1], self.bounds[0], self.bounds[1]))
    #     plt.colorbar(im, ax=axs[i, 2])
    #     axs[i, 2].scatter(x[:, 0], x[:, 1], s=np.abs(a) * 50, marker="+", c='k', label='Ground Truth')
    #     axs[i, 2].scatter(mst["iter_x"][idx][:, 0], mst["iter_x"][idx][:, 1], marker="x",
    #                       s=np.abs(mst["iter_a"][idx]) * 50, c='r', label='Reconstruction')

    # mst = sfw_solver._mstate
    # dual_cert = DualCertificate(mst["x"], mst["a"], sfw_solver.y,
    #                             sfw_solver.forward_op, sfw_solver.lambda_,
    #                             sfw_solver.positive_constraint, sfw_solver.x_dim)
    # grid1 = np.linspace(bounds[0], bounds[1], 200)[1:-1]
    # grid2 = np.linspace(bounds[0], bounds[1], 200)[1:-1]
    # xx, yy = np.meshgrid(grid1, grid2)
    # grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    # eta = dual_cert(grid)
    #
    # print(np.abs(eta).max())
    #
    # plt.figure()
    # plt.imshow(eta.reshape((198, 198)), origin="lower", interpolation="none")
    # plt.colorbar()
    # plt.show()