import logging
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from domain.functions import double_exp, single_exp


def ssr(y_data: npt.NDArray[np.float64], y_model: npt.NDArray[np.float64]) -> float:
    """
    Calculate Sum of Squared Residuals

    :param y_data: y data to fit
    :type y_data: npt.NDArray[np.float64]
    :param y_model: y fit
    :type y_model: npt.NDArray[np.float64]
    :return: ssr
    :rtype: float
    """

    return float(np.sum((y_data - y_model) ** 2))


def fit_ls(
    model: Literal["single_exp", "double_exp"],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0_list: List[List[float]],
    bounds: Tuple[List[float], List[float]],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Nonlinear least-squares with multiple params initializations and return best (minimum SSR)

    :param model: model to fit
    :type model: Literal["single_exp", "double_exp"]
    :param x: x data to fit
    :type x: npt.NDArray[np.float64]
    :param y: y data to fit
    :type y: npt.NDArray[np.float64]
    :param p0_list: initial conditions
    :type p0_list: List[List[float]]
    :param bounds: bounds for parameters
    :type bounds: Tuple[List[float], List[float]]
    :param verbose:
    :type verbose: bool
    :return: fit result
    :rtype: Dict[str, float]
    """

    if model == "double_exp":
        fit_func = double_exp_fit_autocorr
    elif model == "single_exp":
        fit_func = single_exp_fit_autocorr
    else:
        raise ValueError(f"Unknown model: {model}")

    best_result = None
    best_ssr = float("inf")

    for i, p0 in enumerate(p0_list):

        try:
            result = fit_func(
                x=x,
                y=y,
                p0=p0,
                bounds=bounds,
                verbose=verbose,
            )
            if result["ssr"] < best_ssr:
                best_result = result
                best_ssr = result["ssr"]

            if verbose:
                logging.info(
                    f"[{model}] Initial guess {i+1}/{len(p0_list)}: {p0}, SSR={result['ssr']:.2e}"
                )

        except Exception as e:
            if verbose:
                logging.info(f"[{model}] Initial guess {i+1} failed: {e}")
            continue

    if best_result is None:
        raise RuntimeError(f"All initial guesses failed for {model}.")

    if verbose:
        logging.info(f"{model} - best result: {best_result}")

    return best_result


def fit_model(
    model_fn: Callable,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p0: List[List[float]],
    bounds: Tuple[List[float], List[float]],
    verbose: bool = True,
    param_names: List[str] = None,
) -> Dict[str, float]:
    """
    Verbose wrapper for SciPy curve_fit() with different models
    """

    def verbose_wrapper(x, *params):
        y_model = model_fn(x, *params)
        ssr_val = ssr(y, y_model)
        if verbose:
            logging.info(
                ", ".join([f"{n}={v:.4g}" for n, v in zip(param_names, params)])
                + f", SSR={ssr_val:.3e}"
            )
        return y_model

    popt, _ = curve_fit(
        verbose_wrapper,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )

    result = {name: float(val) for name, val in zip(param_names, popt)}

    y_fit = model_fn(x, *popt)
    result["ssr"] = ssr(y, y_fit)

    return result


def double_exp_fit_autocorr(x, y, p0, bounds, verbose=True) -> Dict[str, float]:
    """
    Fit tails with double exponential
    """

    return fit_model(
        double_exp,
        x,
        y,
        p0=p0,
        bounds=bounds,
        verbose=verbose,
        param_names=["A1", "gamma1", "gamma2"],
    )


def single_exp_fit_autocorr(x, y, p0, bounds, verbose=True) -> Dict[str, float]:
    """
    Fit tails with single exponential
    """

    return fit_model(
        single_exp,
        x,
        y,
        p0=p0,
        bounds=bounds,
        verbose=verbose,
        param_names=["gamma"],
    )
