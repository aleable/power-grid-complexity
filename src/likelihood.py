from typing import Callable, Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt

from domain.distributions import (
    GaussianTail,
    QGaussianTail,
    SinhDistribution,
    TiltedWashboardDistribution,
    WrappedGaussian,
)


def neg_log_likelihood(
    params_array: list[float],
    data: npt.NDArray[np.float64],
    distribution: Literal[
        "sinh",
        "tilted_washboard",
        "gaussian_tail",
        "qgaussian_tail",
        "wrapped_gaussian",
    ],
    x_grid: Optional[npt.NDArray[np.float64]] = None,
) -> float:
    """
    Compute negative log likelihood

    :param params_array: distribution parameters
    :type params_array: list[float]
    :param data: data to fit
    :type data: npt.NDArray[np.float64]
    :param distribution: distribution type
    :type distribution: Literal["sinh", "tilted_washboard", "gaussian_tail", "qgaussian_tail"]
    :param x_grid: grid for normalization tilted washboard
    :type x_grid: Optional[npt.NDArray[np.float64]]
    :return: negative log-likelihood
    :rtype: float
    """

    if distribution == "sinh":
        a = params_array[0]
        dist = SinhDistribution(a=a)
    elif distribution == "tilted_washboard":
        a, b = params_array
        dist = TiltedWashboardDistribution(a=a, b=b)
    elif distribution == "gaussian_tail":
        sigma = params_array[0]
        dist = GaussianTail(sigma=sigma)
    elif distribution == "qgaussian_tail":
        q, beta = params_array
        dist = QGaussianTail(q=q, beta=beta)
    elif distribution == "wrapped_gaussian":
        mu, sigma = params_array
        dist = WrappedGaussian(mu=mu, sigma=sigma)
    else:
        raise ValueError(f"Distribution not implemented: {distribution}")

    if distribution == "tilted_washboard":
        pdf_values = dist.pdf(x=data, x_grid=x_grid)
    else:
        pdf_values = dist.pdf(x=data)

    pdf_values = np.maximum(pdf_values, 1e-10)

    return -np.sum(np.log(pdf_values))


def clipped_log_likelihood_hessian(
    neg_log_likelihood: Callable[
        [
            list[float],
            npt.NDArray[np.float64],
            Literal[
                "sinh",
                "tilted_washboard",
                "gaussian_tail",
                "qgaussian_tail",
                "wrapped_gaussian",
            ],
        ],
        float,
    ],
    data: npt.NDArray[np.float64],
    distribution: Literal[
        "sinh",
        "tilted_washboard",
        "gaussian_tail",
        "qgaussian_tail",
        "wrapped_gaussian",
    ],
    param_names: List[str],
    constraints: Dict,
    x_grid: Optional[npt.NDArray[np.float64]] = None,
) -> Callable:
    """
    Clip parameters into using bounds for Hessian computation

    :param neg_log_likelihood: negative log-likelihood
    :type neg_log_likelihood: Callable
    :param data: data to fit
    :type data: npt.NDArray[np.float64]
    :param distribution: distribution type
    :type distribution: Literal["sinh", "tilted_washboard", "gaussian_tail", "qgaussian_tail", "wrapped_gaussian"]
    :param param_names: parameters string list: "a", "b", etc.
    :type param_names: List[str]
    :param constraints: parameters validity bounds
    :type constraints: Dict
    :param x_grid: grid for normalization tilted washboard
    :type x_grid: Optional[npt.NDArray[np.float64]]
    :return: log-likelihood callable with clipped parameters for Hessian
    :rtype: Callable[..., Any]
    """

    # Bounds for all parameters.
    lower_bounds = np.array([constraints[name][0] for name in param_names])
    upper_bounds = np.array([constraints[name][1] for name in param_names])

    def clipped_fun(params):
        clipped_params = np.clip(params, lower_bounds, upper_bounds)
        return neg_log_likelihood(
            params_array=clipped_params,
            data=data,
            distribution=distribution,
            x_grid=x_grid,
        )

    return clipped_fun
