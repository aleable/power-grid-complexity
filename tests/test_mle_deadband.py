import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import numpy as np

from domain.distributions import SinhDistribution
from src.mle_multiple_inits import mle_fit_multiple_inits
from utils.sampling.sampling_deadband import sample_sinh_distribution
from utils.seed import set_seed

config_path = Path(__file__).resolve().parent.parent / "params"

with open(config_path / "mle_constraints.json", "r") as f:
    constraints = json.load(f)

constraints_sinh = {k: tuple(v) for k, v in constraints["constraints_sinh"].items()}


def test_mle_deadband():

    set_seed(123)

    true_a = 0.2
    true_dist = SinhDistribution(true_a)
    data = sample_sinh_distribution(true_dist, n=10000)

    a_inits = [{"a": 0.1}]

    result = mle_fit_multiple_inits(
        data=data,
        distribution="sinh",
        init_list=a_inits,
        constraints=constraints_sinh,
        max_iter=500,
        std_threshold=1,
        verbose=False,
    )

    a_fit = result[0]["a"][0]

    # Converged to synthetic value
    assert np.abs(a_fit - true_a) / a_fit < 0.02
