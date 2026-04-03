import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import numpy as np

from domain.distributions import GaussianTail, QGaussianTail
from src.mle_multiple_inits import mle_fit_multiple_inits
from utils.sampling.sampling_tails import sample_gaussian_tail, sample_qgaussian_tail
from utils.seed import set_seed

config_path = Path(__file__).resolve().parent.parent / "params"

with open(config_path / "mle_constraints.json", "r") as f:
    constraints = json.load(f)

constraints_gaussian_tail = {
    k: tuple(v) for k, v in constraints["constraints_gaussian_tail"].items()
}

constraints_qgaussian_tail = {
    k: tuple(v) for k, v in constraints["constraints_qgaussian_tail"].items()
}


def test_mle_deadband():

    set_seed(123)

    # Gaussian
    true_sigma = 1.0
    n = 5000

    true_dist = GaussianTail(sigma=true_sigma)
    data_gaussian = sample_gaussian_tail(dist=true_dist, n=n)

    # q-Gaussian: testing one value
    true_q_super = 1.3
    true_beta = 20

    true_dist_super = QGaussianTail(q=true_q_super, beta=true_beta)
    data_qgaussian_super = sample_qgaussian_tail(dist=true_dist_super, n=n)

    sigma_inits = [{"sigma": 0.1}]

    result = mle_fit_multiple_inits(
        data=data_gaussian,
        distribution="gaussian_tail",
        init_list=sigma_inits,
        constraints=constraints_gaussian_tail,
        max_iter=500,
        std_threshold=1,
        verbose=False,
    )

    qgaussian_inits = [{"q": 1.3, "beta": 5}]

    result_super = mle_fit_multiple_inits(
        data=data_qgaussian_super,
        distribution="qgaussian_tail",
        init_list=qgaussian_inits,
        constraints=constraints_qgaussian_tail,
        max_iter=500,
        std_threshold=1,
        verbose=False,
    )

    sigma_fit = result[0]["sigma"][0]
    q_fit_super = result_super[0]["q"][0]
    beta_fit_super = result_super[0]["beta"][0]

    # Converged to synthetic value
    assert np.abs(sigma_fit - true_sigma) / sigma_fit < 1e-3
    assert np.abs(q_fit_super - true_q_super) / q_fit_super < 0.2
    assert np.abs(beta_fit_super - true_beta) / beta_fit_super < 0.01
