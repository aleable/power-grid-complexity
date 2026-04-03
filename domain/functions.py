import numpy as np
import numpy.typing as npt


def double_exp(
    t: npt.NDArray[np.float64], A1: float, gamma1: float, gamma2: float
) -> npt.NDArray[np.float64]:
    """
    Double exponential

    :param t: time lag
    :type t: npt.NDArray[np.float64]
    :param A1: amplitude
    :type A1: float
    :param gamma1:
    :type gamma1: damping 1
    :param gamma2:
    :type gamma2: damping 2
    :return: autocorrelation
    :rtype: NDArray[float64]
    """

    return A1 * np.exp(-gamma1 * t) + (1 - A1) * np.exp(-gamma2 * t)


def single_exp(t: npt.NDArray[np.float64], gamma: float) -> npt.NDArray[np.float64]:
    """
    Single exponential

    :param t: time lag
    :type t: npt.NDArray[np.float64]
    :param gamma: damping
    :type gamma: float
    :return: autocorrelation
    :rtype: NDArray[float64]
    """

    return np.exp(-gamma * t)
