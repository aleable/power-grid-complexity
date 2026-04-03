import numpy as np
import numpy.typing as npt

from domain.distributions import SinhDistribution


def sample_sinh_distribution(dist: SinhDistribution, n: int) -> npt.NDArray[np.float64]:
    """
    Sampling synthetic data with rejection sampling.

    :param dist:
    :type dist: SinhDistribution
    :param n: number of data to sample
    :type n: int
    :return: samples
    :rtype: npt.NDArray[np.float64]
    """

    samples = []

    while len(samples) < n:
        x = np.random.uniform(-dist.omega_dz, dist.omega_dz)
        y = np.random.uniform(0, 100)
        if y <= dist.pdf(x):
            samples.append(x)

    return np.array(samples)
