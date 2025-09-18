import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
from scipy.special import kv, gamma


def sq_exp_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, sigma_prior: float = 1.0
) -> np.ndarray:
    diff = x1.reshape((-1, 1)) - x2.reshape((1, -1))
    return sigma_prior**2 * np.exp(-(2 / (length_scale**2)) * diff**2)


def matern_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    length_scale: float = 1.0,
    sigma_prior: float = 1.0,
    nu: float = 1.5,
) -> np.ndarray:
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    dists = np.abs(x1.reshape(-1, 1) - x2.reshape(1, -1))

    if np.isclose(nu, 0.5):
        # exp kernel
        k = np.exp(-dists / length_scale)
    else:
        scaling = np.sqrt(2 * nu) * dists / length_scale
        # avoid nans
        scaling = np.where(scaling == 0.0, 1e-10, scaling)

        coeff = (2 ** (1 - nu)) / gamma(nu)
        k = coeff * (scaling**nu) * kv(nu, scaling)

    return sigma_prior**2 * k
