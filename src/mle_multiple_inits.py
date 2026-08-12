import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

from src.likelihood import neg_log_likelihood
from src.mle import mle_fit


def mle_fit_multiple_inits(
    data: npt.NDArray[np.float64],
    distribution: Literal[
        "sinh",
        "tilted_washboard",
        "gaussian_tail",
        "qgaussian_tail",
        "wrapped_gaussian",
    ],
    init_list: list[Dict[str, float]],
    constraints: Dict[str, Tuple[float, float]],
    max_iter: Optional[int] = 100,
    std_threshold: Optional[float] = 1.0,
    x_grid: Optional[npt.NDArray[np.float64]] = None,
    verbose: bool = False,
) -> tuple[Dict[str, float] | None, float]:
    """
    MLE with multiple params initializations

    :param data: data to fit
    :type data: npt.NDArray[np.float64]
    :param distribution: distribution type
    :type distribution: Literal["sinh", "tilted_washboard", "gaussian_tail", "qgaussian_tail", "wrapped_gaussian"]
    :param init_list: initializations
    :type init_list: list[Dict[str, float]]
    :param constraints: parameters validity bounds
    :type constraints: Dict[str, Tuple[float, float]]
    :param max_iter: maximum number of iterations
    :type max_iter: Optional[int]
    :param std_threshold: acceptance value for std (if too large reject fit result)
    :type std_threshold: Optional[float]
    :param x_grid: grid for normalization tilted washboard
    :type x_grid: Optional[npt.NDArray[np.float64]]
    :param verbose:
    :type verbose: bool
    :return:
    :rtype: tuple[Dict[str, float] | None, float]
    """

    best_params = None
    best_negloglik = float("inf")

    for i, init_params in enumerate(init_list):
        if verbose:
            logging.info(
                f"Running initialization: {i + 1}/{len(init_list)}: {init_params}"
            )

        try:
            # Fit.
            result = mle_fit(
                data=data,
                distribution=distribution,
                max_iter=max_iter,
                initial_params=init_params,
                constraints=constraints,
                x_grid=x_grid,
                verbose=verbose,
            )

            # Compute log likelihood.
            param_array = [result["result"][k][0] for k in init_params.keys()]

            current_nll = neg_log_likelihood(
                params_array=param_array,
                data=data,
                distribution=distribution,
                x_grid=x_grid,
            )

            # Check if best (remove fit with stds of parameter too large)
            stds = np.array([v[1] for v in result["result"].values()])

            if current_nll < best_negloglik and np.all(stds < std_threshold):
                best_negloglik = current_nll
                best_params = result["result"]

        except Exception as e:
            if verbose:
                logging.error(f"Initialization {i+1} failed with error: {e}")
            continue

    return best_params, best_negloglik
