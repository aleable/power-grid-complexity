import numpy as np


def set_seed(seed: int = 123):
    """
    Set the random seed.

    :param seed:
    :type seed: int
    """

    np.random.seed(seed)
