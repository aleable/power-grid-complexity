import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from numdifftools import Hessian
from scipy.optimize import minimize

from src.likelihood import clipped_log_likelihood_hessian, neg_log_likelihood


def mle_fit(
    data: npt.NDArray[np.float64],
    distribution: Literal[
        "sinh", "tilted_washboard", "gaussian_tail", "qgaussian_tail"
    ],
    constraints: Dict[str, Tuple[float, float]],
    max_iter: Optional[int] = 100,
    initial_params: Optional[Dict[str, float]] = None,
    x_grid: Optional[npt.NDArray[np.float64]] = None,
    verbose: bool = False,
) -> Dict:
    """
    MLE for distribution parameters
    Parameters stds are computed inverting Hessian with iterative Tikhonov reg.

    :param data: data to fit
    :type data: npt.NDArray[np.float64]
    :param distribution: distribution type
    :type distribution: Literal["sinh", "tilted_washboard", "gaussian_tail", "qgaussian_tail"]
    :param constraints: parameters validity bounds
    :type constraints: Dict[str, Tuple[float, float]]
    :param max_iter:
    :type max_iter: Optional[int]
    :param initial_params:
    :type initial_params: Optional[Dict[str, float]]
    :param x_grid: grid for normalization tilted washboard
    :type x_grid: Optional[npt.NDArray[np.float64]]
    :param verbose:
    :type verbose: bool
    :return: Dict
    :rtype: fit result
    """

    def _invert_hessian_compute_std(
        hessian: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Invert Hessian and compute std

        :param hessian:
        :type hessian: npt.NDArray[np.float64]
        :return: std and condition number
        :rtype: Tuple[NDArray[float64], float]
        """

        cov = np.linalg.inv(hessian)
        std = np.sqrt(np.diag(cov))
        cond_num = np.linalg.cond(hessian)

        return (
            std,
            cond_num,
        )

    def _iterative_regularization(
        hessian: npt.NDArray[np.float64],
        verbose: bool = False,
        reg: float = 1e-12,
        max_attempts: int = 100,
    ) -> Tuple[npt.NDArray[np.float64], float, float] | None:
        """
        Iterative regularization for Hessian inversion

        :param hessian:
        :type hessian: npt.NDArray[np.float64]
        :param verbose:
        :type verbose: bool
        :param reg: regularization strength
        :type reg: float
        :param max_attempts:
        :type max_attempts: int
        :return: std, reg strength, condition number
        :rtype: Tuple[NDArray[float64], float, float] | None
        """

        for attempt in range(max_attempts):

            try:
                hessian_reg = hessian + reg * np.eye(hessian.shape[0])
                np.linalg.cholesky(hessian_reg)
                std, cond_num = _invert_hessian_compute_std(hessian=hessian_reg)

                if verbose:
                    logging.info(
                        f"Iterative regularization Hessian succeeded at iter {attempt} with regularization: {reg:.2e}"
                    )

                return std, reg, cond_num

            except np.linalg.LinAlgError:
                reg *= 10

        # All attempts failed.
        if verbose:
            logging.info(
                "Iterative regularization failed, cannot provide std estimate."
            )
            std, cond_num = _invert_hessian_compute_std(hessian=hessian)

        return std, reg, cond_num

    def _logging_neg_log_likelihood(params: List[float]) -> float:
        """
        Logging log-likelihood update

        :param params: parameters to update
        :type params: List[float]
        :return: float
        :rtype: current log likelihood
        """

        current_value = neg_log_likelihood(
            params_array=params, data=data, distribution=distribution, x_grid=x_grid
        )

        if last_params[0] is not None:
            delta_x = np.max(np.abs(params - last_params[0]))
            delta_f = np.abs(current_value - last_value[0])
            logging.info(
                f"neg_log_likelihood={current_value:.6f}, "
                f"delta_x={delta_x:.2e}, delta_f={delta_f:.2e}, params={params}"
            )

        last_params[0] = np.copy(params)
        last_value[0] = current_value

        return current_value

    if initial_params is None:
        if distribution == "sinh":
            initial_params = {"a": 0.1}
        elif distribution == "tilted_washboard":
            initial_params = {"a": 1.0, "b": 1.0}
        elif distribution == "gaussian_tail":
            initial_params = {"sigma": 1.0}
        elif distribution == "qgaussian_tail":
            initial_params = {"q": 1.0, "beta": 20}
        else:
            raise ValueError(f"Distribution not implemented: {distribution}")

    # Set up parameter bounds from constraints.
    param_names = list(initial_params.keys())
    x0 = [initial_params[name] for name in param_names]
    bounds = []

    for name in param_names:
        if constraints is not None and name in constraints:
            bounds.append(constraints[name])
        else:
            bounds.append((None, None))

    # Run optimization.
    if verbose:
        logging.info(f"Starting MLE optimization for {distribution} distribution")
        logging.info(f"Initial parameters: {initial_params}")

    last_params = [None]
    last_value = [None]

    result = minimize(
        _logging_neg_log_likelihood,
        x0,
        bounds=bounds,
        method="Nelder-Mead",
        options={
            "disp": verbose,
            "maxfev": max_iter,
            "xatol": 1e-8,  # 1e-5 -> value used in paper
            "fatol": 1e-8,  # 1e-5 -> value used in paper
        },
    )

    if not result.success:
        logging.warning(f"Optimization did not converge. Message: {result.message}")

    if result.success:

        # Compute standard deviations of MLE parameters.
        clipped_function = clipped_log_likelihood_hessian(
            neg_log_likelihood=neg_log_likelihood,
            data=data,
            distribution=distribution,
            param_names=param_names,
            constraints=constraints,
            x_grid=x_grid,
        )

        hessian_func = Hessian(clipped_function)
        hessian_matrix = hessian_func(result.x)

        # Computing standard deviations. Regularize if Hessian is not invertible.
        std, cond_num = _invert_hessian_compute_std(hessian=hessian_matrix)

        has_nan = np.any(np.isnan(std))
        if not has_nan:
            mle_std = std
            mle_cond_num = cond_num
            mle_reg = 0
            logging.info("No regularization for Hessian")
        else:
            mle_std, mle_reg, mle_cond_num = _iterative_regularization(
                hessian=hessian_matrix, verbose=verbose
            )

    estimated_params = {
        name: float(value) for name, value in zip(param_names, result.x)
    }

    # Truncate params with condition number rule of thumb: https://en.wikipedia.org/wiki/Condition_number
    lost_digits = np.log10(mle_cond_num)
    digits = max(0, int(np.floor(15 - lost_digits)))
    stds_mle = [round(s, digits) for s in mle_std]
    params_mle = {key: round(value, digits) for key, value in estimated_params.items()}

    optimization_results = {
        key: (params_mle[key], stds_mle[i])
        for i, key in enumerate(estimated_params.keys())
    }

    results = {
        "result": optimization_results,
        "hessian_reg": mle_reg,
        "hessian_cond_number": mle_cond_num,
    }

    if verbose:
        logging.info(f"MLE results: {results}")
        logging.info(f"Final negative log-likelihood: {result.fun}")

    return results
