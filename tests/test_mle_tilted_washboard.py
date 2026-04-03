import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import numpy as np

from domain.distributions import TiltedWashboardDistribution
from src.mle_multiple_inits import mle_fit_multiple_inits
from utils.sampling.sampling_tilted_washboard import (
    sample_tilted_washboard_distribution,
)
from utils.seed import set_seed

config_path = Path(__file__).resolve().parent.parent / "params"

with open(config_path / "mle_constraints.json", "r") as f:
    constraints = json.load(f)

constraints_tilted_washboard = {
    k: tuple(v) for k, v in constraints["constraints_tilted_washboard"].items()
}


def test_mle_autocorr():

    set_seed(123)

    true_a = 420
    true_b = 440
    x_min = np.arcsin(true_a / true_b)
    x_grid = np.linspace(x_min - 0.4, x_min + 0.4, 1000)

    dist = TiltedWashboardDistribution(a=true_a, b=true_b)
    data = sample_tilted_washboard_distribution(dist, n=1000, x_grid=x_grid)

    param_inits = [
        {"a": 400, "b": 420},
    ]

    result = mle_fit_multiple_inits(
        data=data,
        distribution="tilted_washboard",
        init_list=param_inits,
        constraints=constraints_tilted_washboard,
        max_iter=500,
        std_threshold=25,
        x_grid=x_grid,  # Grid for normalization of pdf (taken from sampling)
        verbose=False,
    )

    a_fit = result[0]["a"][0]
    b_fit = result[0]["b"][0]

    # Converged to synthetic value
    assert np.abs(a_fit - true_a) / a_fit < 0.005
    assert np.abs(b_fit - true_b) / b_fit < 0.005
