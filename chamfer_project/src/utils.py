"""Shared utilities for Chamfer distance experiments."""
import numpy as np
from scipy.spatial.distance import cdist

_METRIC_MAP = {"l2": "euclidean", "l1": "cityblock", "euclidean": "euclidean", "cityblock": "cityblock"}


def exact_nn_distances(A, B, metric="l2"):
    """Return array of min distances gamma_a = min_{b in B} dist(a, b) for each a in A."""
    dist_matrix = cdist(A, B, metric=_METRIC_MAP.get(metric, metric))
    return dist_matrix.min(axis=1)


def compute_true_stats(A, B, metric="l2"):
    """Compute full-data Chamfer statistics for logging, calibration, and oracles."""
    if metric not in _METRIC_MAP:
        raise ValueError(f"metric must be one of {sorted(_METRIC_MAP)}, got {metric!r}")
    gamma = exact_nn_distances(A, B, metric=metric)
    ch = float(gamma.sum())
    mu = float(gamma.mean()) if len(gamma) else 0.0
    sigma = float(gamma.std(ddof=0)) if len(gamma) else 0.0
    if mu == 0.0:
        cv = 0.0
        M = 1.0
    else:
        cv = sigma / mu
        M = float(gamma.max()) / mu
    return {
        "gamma": gamma,
        "ch": ch,
        "mu": mu,
        "sigma": sigma,
        "cv": cv,
        "M": M,
    }


def compute_cv(gamma):
    """Coefficient of variation: std / mean. Returns 0 if mean == 0."""
    mu = gamma.mean()
    if mu == 0:
        return 0.0
    return gamma.std() / mu


def compute_true_M(gamma):
    """Max-to-mean ratio M = max(gamma) / mean(gamma). Returns 1 if mean == 0."""
    mu = float(gamma.mean())
    if mu == 0:
        return 1.0
    return float(gamma.max()) / mu


def relative_error(estimate, true_value):
    """Relative error |estimate - true| / true."""
    if true_value == 0:
        return abs(estimate)
    return abs(estimate - true_value) / true_value


def bernstein_oracle(true_cv, true_M, n, eps, delta, C_IS=4.0, C_NN=1.0):
    """Bernstein cost-model oracle used only for experimental classification."""
    import math

    cost_uniform_raw = (
        2.0 * (true_cv ** 2 + true_M * eps / 3.0) * math.log(2.0 / delta)
        / (eps ** 2)
    )
    cost_uniform = min(cost_uniform_raw, n)
    cost_is = C_NN * math.log(n) + C_IS * math.log(n) / (eps ** 2)
    return {
        "cost_uniform_bernstein_raw": cost_uniform_raw,
        "cost_uniform_bernstein": cost_uniform,
        "cost_is": cost_is,
        "oracle_branch": "uniform" if cost_uniform <= cost_is else "is",
        "cost_uniform_chebyshev": min(true_cv ** 2 / (eps ** 2 * delta), n),
    }


BernsteinOracle = bernstein_oracle
