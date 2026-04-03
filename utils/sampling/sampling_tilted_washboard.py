import numpy as np
import numpy.typing as npt

from domain.distributions import TiltedWashboardDistribution


def sample_tilted_washboard_distribution(
    dist: TiltedWashboardDistribution,
    n: int,
    x_grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Rejection sampling for TiltedWashboardDistribution.

    :param dist:
    :type dist: TiltedWashboardDistribution
    :param n: number of data to sample
    :type n: int
    :param x_grid: x grid for distribution normalization
    :type x_grid: npt.NDArray[np.float64]
    :return: samples
    :rtype: npt.NDArray[np.float64]
    """

    samples = []

    while len(samples) < n:
        x = np.random.uniform(x_grid.min(), x_grid.max())
        y = np.random.uniform(0, 10)
        if y <= dist.pdf(x, x_grid):
            samples.append(x)

    return np.array(samples)
