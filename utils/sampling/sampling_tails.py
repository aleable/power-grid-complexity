import numpy as np
import numpy.typing as npt

from domain.distributions import GaussianTail, QGaussianTail


def sample_gaussian_tail(
    dist: GaussianTail,
    n: int,
) -> npt.NDArray[np.float64]:
    """
    Rejection sampling for GaussianTail.

    :param dist:
    :type dist: GaussianTail
    :param n: number of data to sample
    :type n: int
    :return: samples
    :rtype: npt.NDArray[np.float64]
    """

    samples = []

    while len(samples) < n:
        x = np.random.uniform(1, 5)
        y = np.random.uniform(0, 1)
        if y <= dist.pdf(x):
            samples.append(x)

    return np.array(samples)


def sample_qgaussian_tail(
    dist: QGaussianTail,
    n: int,
) -> npt.NDArray[np.float64]:
    """
    Rejection sampling for QGaussianTail.

    :param dist:
    :type dist: QGaussianTail
    :param n: number of data to sample
    :type n: int
    :return: samples
    :rtype: npt.NDArray[np.float64]
    """

    samples = []

    while len(samples) < n:
        x = np.random.uniform(1, 10)
        y = np.random.uniform(0, 30)
        if y <= dist.pdf(x):
            samples.append(x)

    return np.array(samples)
