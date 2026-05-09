"""Core Chamfer distance algorithms.

Three estimators:
  1. brute_force_chamfer       — exact O(n^2 d)
  2. uniform_sampling_chamfer  — uniform IS, O(T * n * d)
  3. importance_sampling_chamfer — importance sampling with CrudeNN weights, O(T * n * d + CrudeNN)
"""
import numpy as np
from scipy.spatial.distance import cdist

_METRIC_MAP = {"l2": "euclidean", "l1": "cityblock", "euclidean": "euclidean", "cityblock": "cityblock"}


# ---------------------------------------------------------------------------
# Exact baseline
# ---------------------------------------------------------------------------

def brute_force_chamfer(A, B, metric="l2"):
    """Exact one-sided Chamfer distance CH(A, B) = sum_{a in A} min_{b in B} dist(a, b).

    Returns
    -------
    ch : float
        Exact CH(A, B).
    gamma : ndarray, shape (n_a,)
        Per-point NN distances gamma_a.
    """
    dist_matrix = cdist(A, B, metric=_METRIC_MAP.get(metric, metric))  # (n_a, n_b)
    gamma = dist_matrix.min(axis=1)
    return float(gamma.sum()), gamma


def ExactChamfer(A, B, metric="l2"):
    """Spec-shaped exact Chamfer baseline."""
    estimate, gamma = brute_force_chamfer(A, B, metric=metric)
    return {
        "estimate": estimate,
        "branch": "exact",
        "num_exact_nn_queries": len(A),
        "gamma": gamma,
    }


# ---------------------------------------------------------------------------
# Uniform sampling
# ---------------------------------------------------------------------------

def uniform_sampling_chamfer(A, B, T, metric="l2", rng=None, replace=False):
    """Uniform sampling estimator for CH(A, B).

    Samples T points uniformly from A, computes exact NN distance for each,
    and returns n * mean(gamma_sample) as the unbiased estimate.

    Parameters
    ----------
    T : int
        Number of samples.
    replace : bool
        Sample without replacement (default False per ALGORITHM_SPEC §1.1).

    Returns
    -------
    estimate : float
    std_error : float
        Standard error of the estimate = n_a * std(gamma_sample) / sqrt(T).
    gamma_sample : ndarray, shape (T,)
        NN distances of sampled points.
    sampled_idx : ndarray, shape (T,)
        Indices into A of sampled points.
    """
    rng = np.random.default_rng(rng)
    n_a = len(A)
    if T >= n_a and not replace:
        estimate, gamma_sample = brute_force_chamfer(A, B, metric=metric)
        sampled_idx = np.arange(n_a)
        std_error = float(n_a * gamma_sample.std(ddof=1) / np.sqrt(n_a)) if n_a > 1 else 0.0
        return estimate, std_error, gamma_sample, sampled_idx

    sampled_idx = rng.choice(n_a, size=T, replace=replace)
    A_sample = A[sampled_idx]

    gamma_sample = cdist(A_sample, B, metric=_METRIC_MAP.get(metric, metric)).min(axis=1)

    estimate = float(n_a * gamma_sample.mean())
    std_error = float(n_a * gamma_sample.std(ddof=1) / np.sqrt(T)) if T > 1 else 0.0

    return estimate, std_error, gamma_sample, sampled_idx


def UniformEstimate(A, B, T, metric="l2", rng=None, replace=False):
    """Spec-shaped uniform sampling estimator."""
    n_a = len(A)
    estimate, _, gamma_sample, sampled_idx = uniform_sampling_chamfer(
        A, B, T=T, metric=metric, rng=rng, replace=replace
    )
    actual_T = len(gamma_sample)
    return {
        "estimate": estimate,
        "branch": "uniform_exact" if actual_T == n_a and not replace else "uniform",
        "T": actual_T,
        "num_exact_nn_queries": actual_T,
        "sample_indices": sampled_idx,
    }


# ---------------------------------------------------------------------------
# Importance sampling
# ---------------------------------------------------------------------------

def importance_sampling_chamfer(A, B, D_a, T, metric="l2", rng=None):
    """Importance sampling estimator for CH(A, B) using CrudeNN weights D_a.

    Each iteration:
      x ~ Categorical(D_a / sum(D_a))
      eta_l = (D / D_x) * gamma_x    where D = sum(D_a)

    E[eta_l] = CH(A, B)  (unbiased).
    Var controlled by D / CH(A, B) ≈ O(log n) → need T = O(log n / eps^2) samples.

    Parameters
    ----------
    D_a : ndarray, shape (n_a,)
        Coarse NN estimates from CrudeNN (or oracle).
    T : int
        Number of importance samples.

    Returns
    -------
    estimate : float
    eta_samples : ndarray, shape (T,)
        Individual IS estimates (for variance diagnostics).
    sampled_idx : ndarray, shape (T,)
    """
    rng = np.random.default_rng(rng)
    n_a = len(A)
    D = float(D_a.sum())
    if D <= 0.0 or not np.isfinite(D):
        raise ValueError("D_a must be finite, nonnegative, and have positive sum")

    probs = D_a / D

    # Sample T indices according to importance weights
    sampled_idx = rng.choice(n_a, size=T, p=probs, replace=True)

    # Compute exact NN distance for each sampled point
    A_sample = A[sampled_idx]
    gamma_sample = cdist(A_sample, B, metric=_METRIC_MAP.get(metric, metric)).min(axis=1)

    eta_samples = (D / D_a[sampled_idx]) * gamma_sample

    return float(eta_samples.mean()), eta_samples, sampled_idx


def ChamferEstimate(A, B, T_IS, metric="l2", rng=None, num_levels=None):
    """Spec-shaped importance-sampling Chamfer estimator."""
    from crude_nn import crude_nn

    D_a = crude_nn(A, B, metric=metric, num_levels=num_levels, rng=rng)
    estimate, _, sampled_idx = importance_sampling_chamfer(
        A, B, D_a, T=T_IS, metric=metric, rng=rng
    )
    return {
        "estimate": estimate,
        "branch": "is",
        "T_IS": T_IS,
        "D_sum": float(D_a.sum()),
        "num_exact_nn_queries": T_IS,
        "sample_indices": sampled_idx,
    }
