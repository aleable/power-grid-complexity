import numpy as np
import numpy.typing as npt


def sample_single_exp(
    gamma: float, t: npt.NDArray[np.float64], noise_std: float = 0.02
) -> npt.NDArray[np.float64]:
    """
    Generate synthetic data for single exponential decay: y = exp(-gamma * t) + |y|*noise

    :param gamma: decay rate
    :type gamma: float
    :param t: time lag
    :type t: npt.NDArray[np.float64]
    :param noise_std: Gaussian noise std
    :type noise_std: float
    :return: noisy data
    :rtype: NDArray[float64]
    """

    clean = np.exp(-gamma * t)
    noise = np.random.normal(0, noise_std, size=t.shape)

    return clean + np.abs(clean) * noise


def sample_double_exp(
    A1: float,
    gamma1: float,
    gamma2: float,
    t: npt.NDArray[np.float64],
    noise_std: float = 0.02,
) -> npt.NDArray[np.float64]:
    """
    Generate synthetic data for double exponential decay:
    y = A1*exp(-gamma1*t) + (1-A1)*exp(-gamma2*t) + |y|*noise

    :param A1: amplitude
    :type A1: float
    :param gamma1: damping 1
    :type gamma1: float
    :param gamma2: damping 2
    :type gamma2: float
    :param t: time_lag
    :type t: npt.NDArray[np.float64]
    :param noise_std: Gaussian noise std
    :type noise_std: float
    :return: noisy data
    :rtype: NDArray[float64]
    """

    clean = A1 * np.exp(-gamma1 * t) + (1 - A1) * np.exp(-gamma2 * t)
    noise = np.random.normal(0, noise_std, size=t.shape)

    return clean + np.abs(clean) * noise
