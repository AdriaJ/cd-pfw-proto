import gc
import os
from time import time

import numpy as np
from matplotlib import pyplot as plt

from experiments.n_spikes_metrics import npeaks_max
from src.operators.fourier_operator import FourierOperator
from src.solvers.fw import FW

wdir = "/home/jarret/PycharmProjects/cd-pfw-proto/scaling_exp"

reps = 6
npeaks_max = 80
# npeaks = np.arange(65, npeaks_max, 10)
npeaks = 5

N = 10 * npeaks_max
fmax = 100
freq_bounds = np.array([-fmax, fmax])
psnr = 20
x_dim = 1
bounds = np.array([-1, 1])

n_particles = 100
smooth_sigma = 2.5

flat_lambdas = [2e-3, 1e-2]


def add_psnr(y0, psnr):
    y0_max = np.max(np.linalg.norm(y0, axis=1))
    N = y0.shape[0]
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse / 2), (N, 2))
    y = y0 + w
    return y

def run_seed(seed, n, seed_path):
    np.random.seed(seed)

    x0 = np.random.uniform(-0.95, 0.95, n)
    a0 = np.random.uniform(1, 3, n)
    forward_op = FourierOperator.get_RandomFourierOperator(x0, N, freq_bounds)
    # Get measurements
    y0 = forward_op(a0)
    # add noise
    y = add_psnr(y0, psnr)
    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    lambda_ = 0.1 * lambda_max

    # save source: x, a, freqs, noisy measurments
    np.savez(os.path.join(seed_path, f"source.npz"),
             x0=x0, a0=a0, freqs=forward_op.w, y=y)

    list_options = [
        {"polyatomic": False, "swarm": False, "sliding": True, "initialization": "smoothing",
         "positivity_constraint": True, "max_iter": 200, "dual_certificate_tol": 1e-2, "smooth_sigma": smooth_sigma},
        {"polyatomic": False, "swarm": True, "sliding": True, "swarm_c1": 0.5, "swarm_c2": 0.75,
         "positivity_constraint": True, "max_iter": 200, "dual_certificate_tol": 1e-2, "n_particles": n_particles},
        {"polyatomic": True, "swarm": False, "sliding": False, "initialization": "smoothing",
         "positivity_constraint": True, "max_iter": 200, "dual_certificate_tol": 1e-2, "smooth_sigma": smooth_sigma},
        {"polyatomic": True, "swarm": False, "sliding": True, "initialization": "smoothing",
         "positivity_constraint": True, "max_iter": 200, "dual_certificate_tol": 1e-2, "smooth_sigma": smooth_sigma}
    ]
    names_options = ["SFW", "SFW_PSO", "PFW", "SlidingPFW"]

    for options, name in zip(list_options, names_options):
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        duration = time() - t1
        mst = solver._mstate
        candidates_duration = sum(mst['candidates_search_durations'])
        corrections_duration = sum(mst['correction_durations'])
        sliding_duration = sum(mst['sliding_durations']) if options['sliding'] else None
        x, a, = solver.solution()
        flat_costs = solver.get_flat_norm_values(x0, a0, flat_lambdas)
        loss = solver.blasso_objective_val()[0]
        np.savez(os.path.join(seed_path, f"{name}.npz"),
                 x=x, a=a, t=duration, flat_costs=flat_costs, loss=loss,
                 t_candidate=candidates_duration, t_correction=corrections_duration, t_sliding=sliding_duration)


if __name__ == "__main__":
    seeds = np.random.choice(1000, reps, replace=False)

    db_path = os.path.join(wdir, "database")
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    for n_spikes in npeaks:
        peak_path = os.path.join(db_path, str(n_spikes))
        if not os.path.exists(peak_path):
            os.makedirs(peak_path)

        for seed in seeds:
            seed_path = os.path.join(peak_path, str(seed))
            if not os.path.exists(seed_path):
                os.makedirs(seed_path)

            run_seed(seed, n_spikes, seed_path)

