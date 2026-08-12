from typing import Union, cast

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln, logsumexp
from scipy.stats import norm


class SinhDistribution:
    """
    Sinh distribution to fit in the deadband
    pdf(omega) = C(a, omega_dz) * sinh(omega/a)/omega

    :var Args: parameter a to fit
    """

    def __init__(self, a: float):
        self.a = a
        self.omega_dz = (
            0.5  # Deadband size. Arbitrary size in syntethic data: deaband width = 1.0
        )
        self._C = 1.0 / (2.0 * self._integral())  # Normalization

    def _integral(self) -> float:
        """
        Distribution normalization computed over >= 0 and x2 with numerical integration

        :return: normalization function divided by 2
        :rtype: float
        """

        omega = np.linspace(1e-6, self.omega_dz, 20000)
        integrand = np.sinh(omega / self.a) / omega

        return np.trapezoid(integrand, omega)

    def _unnormalized_pdf(
        self, omega: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Unnormalized PDF

        :param omega:
        :type omega: npt.NDArray[np.float64]
        :return: unnormalized PDF
        :rtype: npt.NDArray[np.float64]
        """

        unnorm_pdf = np.empty_like(omega, dtype=float)
        mask_small = np.abs(omega) < 1e-6  # Sinh discontinous at omega = 0
        mask_large = ~mask_small

        unnorm_pdf[mask_small] = 1.0 / self.a
        unnorm_pdf[mask_large] = np.sinh(omega[mask_large] / self.a) / omega[mask_large]

        return unnorm_pdf

    def pdf(
        self, x: Union[float, npt.NDArray[np.float64]]
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        PDF

        :param x: angular velocity
        :type x: Union[float, npt.NDArray[np.float64]]
        :return: PDF
        :rtype: float | Any
        """

        omega_arr = np.atleast_1d(np.asarray(x, dtype=float))
        mask_support = np.abs(omega_arr) <= self.omega_dz
        p = np.zeros_like(omega_arr, dtype=float)

        p[mask_support] = self._C * self._unnormalized_pdf(omega_arr[mask_support])

        return cast(
            Union[float, npt.NDArray[np.float64]],
            p if omega_arr.shape != () else p.item(),
        )


class TiltedWashboardDistribution:
    """
    Tilded washboard distribution to fit the phase difference x

    :var Args: parameter a and b to fit
    """

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def _negative_V(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Negative tilted washboard potential.

        :param x: phase difference
        :type omega: npt.NDArray[np.float64]
        :return: negative potential -V(x)
        :rtype: npt.NDArray[np.float64]
        """

        return self.a * x + self.b * np.cos(x)

    def pdf(
        self,
        x: Union[float, npt.NDArray[np.float64]],
        x_grid: npt.NDArray[np.float64],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        PDF

        :param x: phase difference
        :type x: Union[float, npt.NDArray[np.float64]]
        :param x_grid: x grid for distribution normalization
        :type x_grid: npt.NDArray[np.float64]
        :return: PDF
        :rtype: float | Any
        """

        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        # Evaluate unnormalized log-pdf
        log_unnormalized = self._negative_V(x_arr)

        # Normalize (logsumexp to avoid overflow)
        dx = x_grid[1] - x_grid[0]
        # log(sum_i f(x_i)dx) = log(sum_i f(x_i)) + log(dx)
        log_norm_const = logsumexp(self._negative_V(x_grid)) + np.log(dx)
        log_pdf = log_unnormalized - log_norm_const

        p = np.exp(log_pdf)

        return cast(
            Union[float, npt.NDArray[np.float64]], p if x_arr.shape != () else p.item()
        )


class GaussianTail:
    """
    Right tail of a Gaussian distribution centered at mu = cutoff
    Note: the cutoff for fitting is conventionally set to 1

    :var Args: parameter sigma to fit
    """

    def __init__(self, sigma: float):

        self.sigma = sigma
        self.cutoff = 1

    def pdf(
        self, x: Union[float, npt.NDArray[np.float64]]
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        PDF

        :param x: angular velocity
        :type x: Union[float, npt.NDArray[np.float64]]
        :return: PDF
        :rtype: Union[float, npt.NDArray[np.float64]]
        """

        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        # 0.5, keeping it for more general implementation where mu != cutoff
        norm_const = 1 - norm.cdf(self.cutoff, loc=self.cutoff, scale=self.sigma)

        pdf_vals = np.zeros_like(x_arr, dtype=float)
        mask = x_arr >= self.cutoff

        pdf_vals[mask] = (
            norm.pdf(x_arr[mask], loc=self.cutoff, scale=self.sigma) / norm_const
        )

        return cast(
            Union[float, npt.NDArray[np.float64]],
            pdf_vals if x_arr.shape != () else pdf_vals.item(),
        )


class Gaussian:
    """
    Gaussian distribution for degenerate q-Gaussian with q = 1.0 conventionally centered at mu = 1.0

    :var Args: parameter sigma
    """

    def __init__(self, sigma: float = 1.0):

        self.sigma = sigma
        self.mu = 1.0

    def pdf(
        self,
        x: Union[float, npt.NDArray[np.float64]],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Tail Gaussian PDF

        :param x: angular velocity
        :type x: Union[float, npt.NDArray[np.float64]]
        :return: PDF
        :rtype: Union[float, npt.NDArray[np.float64]]
        """

        result = norm.pdf(x, loc=self.mu, scale=self.sigma)

        return cast(Union[float, npt.NDArray[np.float64]], result)


class QGaussian:
    """
    q-Gaussian distribution conventionally centered at mu = 1.0 for tails fit.

    :var Args:
        * q: nonextensivity parameter (shape)

        * beta: inverse temperature parameter
    """

    def __init__(self, q: float = 1.0, beta: float = 0.5):

        self.q = q
        self.beta = beta
        self.mu = 1.0

    def pdf(
        self,
        x: Union[float, npt.NDArray[np.float64]],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        PDF

        :param x: angular velocity
        :type x: Union[float, npt.NDArray[np.float64]]
        :return: PDF
        :rtype: Union[float, npt.NDArray[np.float64]]
        """

        # q = 1.0, fall back to Gaussian.
        if np.isclose(self.q, 1.0):
            sigma = np.sqrt(1.0 / (2.0 * self.beta))
            gaussian = Gaussian(
                sigma=sigma
            )  # Conventionally centered at mu = 1.0 as well
            return gaussian.pdf(x=x)

        # Notation from: https://en.wikipedia.org/wiki/Q-Gaussian_distribution
        # Constants calculated in log space to avoid overflow of gamma function as q -> 1
        elif self.q < 1.0:

            log_num = np.log(2) + 0.5 * np.log(np.pi) + gammaln(1 / (1 - self.q))
            log_den = (
                np.log(3 - self.q)
                + 0.5 * np.log(1 - self.q)
                + gammaln((3 - self.q) / (2 * (1 - self.q)))
            )
            Cq = np.exp(log_num - log_den)

        elif self.q < 3.0:

            log_num = 0.5 * np.log(np.pi) + gammaln((3 - self.q) / (2 * (self.q - 1)))
            log_den = 0.5 * np.log(self.q - 1) + gammaln(1 / (self.q - 1))
            Cq = np.exp(log_num - log_den)

        else:
            raise Exception("q-Gassian: q > 3")

        # e_q
        eq_to_rectify = 1 + (1 - self.q) * (-self.beta * ((x - self.mu) ** 2))

        eq = np.where(
            eq_to_rectify > 0.0,
            eq_to_rectify,
            0,
        )
        result = (np.sqrt(self.beta) / Cq) * (eq ** (1 / (1 - self.q)))

        return cast(Union[float, npt.NDArray[np.float64]], result)


class QGaussianTail:
    """
    Right tail of a q-Gaussian distribution
    mean = cutoff for fitting is conventionally set to 1

    :var Args:
        * q: nonextensivity parameter (shape)

        * beta: inverse temperature parameter
    """

    def __init__(self, q: float = 1.0, beta: float = 0.5):

        self.q = q
        self.beta = beta
        self.cutoff = 1

        # Base q-Gaussian
        self._qg = QGaussian(q=q, beta=beta)

        # Compute normalization over tail
        self._norm_const = self._compute_tail_norm()

    def _compute_tail_norm(self) -> float:
        """
        Distribution normalization computed with numerical integration

        :return: normalization constant
        :rtype: float
        """

        # Numerical grid for normalization
        xmax = self.cutoff + 100
        grid = np.linspace(self.cutoff, xmax, 5000)
        pdf_vals = self._qg.pdf(grid)

        # Cutoff support when q < 1 and q-Gauss has compact support
        if self.q < 1.0:
            mu = self.cutoff
            support_max = mu + 1.0 / np.sqrt((1 - self.q) * self.beta)
            grid = grid[grid <= support_max]
            pdf_vals = pdf_vals[: len(grid)]

        # 0.5, keeping it for more general implementation where mu != cutoff
        return np.trapezoid(pdf_vals, grid)

    def pdf(
        self,
        x: Union[float, npt.NDArray[np.float64]],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Tail q-Gaussian PDF

        :param x: angular velocity
        :type x: Union[float, npt.NDArray[np.float64]]
        :return: PDF
        :rtype: Union[float, npt.NDArray[np.float64]]
        """

        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        pdf_vals = np.zeros_like(x_arr, dtype=float)

        mask = x_arr >= self.cutoff
        pdf_vals[mask] = self._qg.pdf(x_arr[mask]) / self._norm_const

        return cast(
            Union[float, npt.NDArray[np.float64]],
            pdf_vals if x_arr.shape != () else pdf_vals.item(),
        )


class GaussianCoreModel:
    """
    Gaussian-core distribution

        V(x) = x^2 / (2 sigma^2) + A exp(-(x / omega_zero)^2)
        p(x) = exp(-V(x))

    Args:
        * sigma Gaussian tail width
        * A barrier height
    """

    def __init__(self, sigma: float, A: float):
        self.sigma = sigma
        self.A = A
        self.omega_zero = 2 * np.pi * 0.015

    def _negative_V(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        -V(x)
        """
        return -0.5 * (x / self.sigma) ** 2 - self.A * np.exp(
            -((x / self.omega_zero) ** 2)
        )

    def pdf(
        self,
        x: Union[float, npt.NDArray[np.float64]],
        x_grid: npt.NDArray[np.float64],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Normalized PDF using numerical quadrature on x_grid
        """

        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        # log unnormalized density
        log_unnorm = self._negative_V(x_arr)

        # normalization
        dx = x_grid[1] - x_grid[0]
        log_Z = logsumexp(self._negative_V(x_grid)) + np.log(dx)

        log_pdf = log_unnorm - log_Z
        p = np.exp(log_pdf)

        return cast(
            Union[float, npt.NDArray[np.float64]],
            p if x_arr.shape != () else p.item(),
        )


class WrappedGaussian:
    """
    Wrapped Gaussian distribution

        pdf(x) = sum_k N(x - mu + 2 pi k; 0, sigma)

    :var Args:
        * mu: circular mean
        * sigma: std
        * n_windings: number of wrapping terms (range from -n_windings to +n_windings)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, n_windings: int = 5):

        self.mu = mu
        self.sigma = sigma
        self.n_windings = n_windings

    def pdf(
        self,
        x: Union[float, npt.NDArray[np.float64]],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        PDF

        :param x: phase difference
        :type x: Union[float, npt.NDArray[np.float64]]
        :return: PDF
        :rtype: Union[float, npt.NDArray[np.float64]]
        """

        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        ks = np.arange(-self.n_windings, self.n_windings + 1)

        # Shape (n_windings_total, len(x)): one Gaussian per winding
        shifted = x_arr[None, :] - self.mu + 2.0 * np.pi * ks[:, None]

        # logsumexp over windings
        log_terms = (
            -0.5 * (shifted / self.sigma) ** 2
            - np.log(self.sigma)
            - 0.5 * np.log(2.0 * np.pi)
        )

        p = np.exp(logsumexp(log_terms, axis=0))

        return cast(
            Union[float, npt.NDArray[np.float64]],
            p if x_arr.shape != () else p.item(),
        )
