<h1>
Code for: Understanding the complexity of frequency and phase angle fluctuations in power grids
</h1>

This repository contains the implementation of the Maximum Likelihood Estimation (MLE) and Least Squares (LS) routines, and links to open-source data deposited in <a href="https://zenodo.org/records/19397526" target="_blank">a Zenodo folder</a>.

<b>Understanding the complexity of frequency and phase angle fluctuations in power grids</b><br/>
Alessandro Lonardi, Jacques M. Maritz, Leonardo Rydin Gorjão, and Christian Beck<br/>
[<a href="https://arxiv.org/abs/2604.03133" target="_blank">arXiv</a>]<br/>

The code is made available for the public, if you make use of it please cite our work in the form of the reference above.

Below, we explain how to run the code. You can fit synthetic data (which are also used for tests) and visualize real-world data collected with the PMUs used in our study.

<h2>Code installation</h2>

The code was developed using Python 3.12 and can be downloaded and used locally as-is.

To install the necessary packages, you can follow these steps:

1. Install [Poetry](https://python-poetry.org/)
2. Clone this repository to your machine
3. Install the dependencies with Poetry using

```bash
poetry install --no-root
```

<h2>Numerical methods</h2>

The core folders and files implementing the numerical methods are the following:
* [`domain`](./domain/) contains all the [`distributions`](./domain/distributions.py) and [`functions`](./domain/functions.py) fitted via MLE and LS
* [`params`](./params/) contains a json file specifying the MLE parameters boundaries
* [`src`](./src/)
    * [`likelihood.py`](./src/likelihood.py) contains the negative log-likelihood function $\ell(\boldsymbol{\theta})$
    * [`mle.py`](./src/mle.py) contains the MLE routine. We use ```scipy.optimize.minimize()``` for parameters optimization with the [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) optimizer that clips $\boldsymbol{\theta}$ inside its boundaries. As the optimizer does not require gradients, we approximate standard deviations of $\boldsymbol{\theta}$ by calculating the Hessian of the negative log-likelihood and inverting it (if $\mathbf{H}$ is non-invertible we regularize it with Tikhonov regularization and invert the regularized Hessian):<br/>
    $\mathbf{H}(\boldsymbol{\theta}) = \frac{\partial^2 (\ell(\boldsymbol{\theta}))}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}^\top} 
    \quad \to \quad
    \mathrm{Cov}({\boldsymbol{\theta}}^\*) \approx \mathbf{H}{({\boldsymbol{\theta}}^\*)}^{-1}
    \quad \to \quad
    \mathrm{std}(\theta_i^\*) = \sqrt{[\mathrm{Cov}(\boldsymbol{\theta}^\*)]_{ii}}
    $
    * [`mle_multiple_inits.py`](./src/mle_multiple_inits.py) contains a function to run MLE sequentially for multiple initializations of $\boldsymbol{\theta}$ and choose the solution with the highest likelihood
    * [`ls_multiple_inits.py`](./src/ls_multiple_inits.py) contains the LS routine. The function is a wrapper of ```scipy.optimize.curve_fit()``` that runs the fit for multiple initializations and chooses the solution with the lowest sum of squared residuals.

<h2>Tests</h2>

We test the code on a series of synthetic datasets that reproduce the measurements collected by PMUs. Specifically, we use rejection sampling (see [`utils/sampling`](./utils/sampling/)) to generate:
* Data following the deadband angular velocity distribution $p(\omega) \propto \sinh (\omega / a) / \omega$ for $\omega \in [-\omega_0, \omega_0]$
* Tail data sampled from the right-hand side (values above the mean) of a Gaussian and $q$-Gaussian distribution
* Data following the tilted washboard phase-angle difference distribution $p(x) \propto \exp (a x + b \cos(x))$

We also generate two datasets following single- and double-exponential decay and perturb their values using Gaussian noise with a small amplitude.

We fit the data using the MLE and LS routines. Detailed examples on how to run the code on all synthetic datasets are in [`notebooks/synthetic_data`](./notebooks/synthetic_data/)

Synthetic data serve as a validation for our method. For a more robust check, we implement the fit discussed in the notebooks as pytests in [`tests`](./tests/)

To run the tests locally with Poetry, you can execute:
```bash
poetry run pytest tests/
```

<h2>Open-source data</h2>

We open-source measurements recorded from our PMUs over the month of August 2025. The data have a 0.1 s resolution and include:
* Frequency measurements recorded in London
* Phase angle measurements recorded in Stellenbosch
* Phase angle measurements recorded in Bloemfontein

**Important:** Due to their large size, the files are not directly uploaded on GitHub. You can find them deposited in <a href="https://zenodo.org/records/19397526" target="_blank">a Zenodo folder</a>.

Once you download the data, add them to the [`data`](./data/) folder as:
* data/frequency_london.csv
* data/phase_angle_stellenbosch.csv
* data/phase_angle_bloemfontein.csv

Doing so allows running the notebooks in [`notebooks/real_data`](./notebooks/real_data/) to reproduce useful paper plots using the open-sourced data.

<h2>Gaussian core model</h2>

We added the `GaussianCoreModel` class in [`distributions.py`](./domain/distributions.py), which is used to perform MLE on frequency data from the UK power grid, obtained from [NESO](https://www.neso.energy/data-portal/system-frequency-data):

<b> Stochastic Modeling of Power-Grid Frequency Fluctuations in Low-Inertia Systems via a Gaussian-Core Potential and Superstatistics </b><br/>
Wanru Hao, Alessandro Lonardi, and Christian Beck<br/>
[<a href="https://arxiv.org/abs/2605.13289" target="_blank">arXiv</a>]<br/>


