import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.ls_multiple_inits import fit_ls
from utils.sampling.sampling_autocorr import sample_double_exp, sample_single_exp
from utils.seed import set_seed


def test_mle_deadband():

    set_seed(123)

    # Gaussian
    true_gamma_1 = 1
    true_gamma_2 = 6
    true_A1 = 0.3
    n = 2000
    time_lags = np.linspace(0, 10, n)

    # Single exponential decay noisy data
    data_single_decay = sample_single_exp(
        gamma=true_gamma_1, t=time_lags, noise_std=0.05
    )

    # Double exponential decay noisy data
    data_double_decay = sample_double_exp(
        A1=true_A1,
        gamma1=true_gamma_1,
        gamma2=true_gamma_2,
        t=time_lags,
        noise_std=0.05,
    )

    param_init = [1]

    result_single_exp = fit_ls(
        model="single_exp",
        x=time_lags,
        y=data_single_decay,
        p0_list=param_init,
        bounds=([1e-6], [np.inf]),
        verbose=True,
    )

    param_init = [[0.1, 1, 1]]

    result_double_exp = fit_ls(
        model="double_exp",
        x=time_lags,
        y=data_double_decay,
        p0_list=param_init,
        bounds=([1e-6, 1e-6, 1e-6], [0.999, np.inf, np.inf]),
        verbose=False,
    )

    gamma_fit = result_single_exp["gamma"]

    A1_fit = result_double_exp["A1"]
    gamma1_fit = result_double_exp["gamma1"]
    gamma2_fit = result_double_exp["gamma2"]

    # Converged to synthetic value
    assert np.abs(gamma_fit - true_gamma_1) / gamma_fit < 0.001
    assert np.abs(A1_fit - true_A1) / A1_fit < 0.01
    assert np.abs(gamma1_fit - true_gamma_1) / gamma1_fit < 0.001
    assert np.abs(gamma2_fit - true_gamma_2) / gamma2_fit < 0.1
