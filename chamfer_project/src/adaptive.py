"""Pilot-adaptive Chamfer estimator.

Runtime routing is intentionally one-statistic: compare the pilot CV estimate
against tau = sqrt(log n) by default. M_hat is computed for diagnostics only.
"""
import math
import time

import numpy as np
from scipy.spatial.distance import cdist

from chamfer_core import brute_force_chamfer, importance_sampling_chamfer
from crude_nn import crude_nn

_METRIC_MAP = {
    "l2": "euclidean",
    "l1": "cityblock",
    "euclidean": "euclidean",
    "cityblock": "cityblock",
}

C_IS = 4.0


def _pilot_stats(gamma_pilot):
    mu_hat = float(gamma_pilot.mean()) if len(gamma_pilot) else 0.0
    sigma_hat = float(gamma_pilot.std(ddof=1)) if len(gamma_pilot) > 1 else 0.0
    if mu_hat == 0.0:
        return mu_hat, sigma_hat, 0.0, 1.0
    return mu_hat, sigma_hat, sigma_hat / mu_hat, float(gamma_pilot.max()) / mu_hat


def adaptive_chamfer(
    A,
    B,
    eps,
    pilot_size=50,
    tau=None,
    delta=0.01,
    total_budget=None,
    return_diagnostics=False,
    metric="l2",
    num_levels=3,
    rng=None,
    C_IS_value=C_IS,
):
    """Pilot-adaptive Chamfer estimator.

    The uniform branch uses T_U = min(n, max(s, ceil(cv_hat^2/(eps^2 delta)))).
    The IS branch uses T_IS = ceil(C_IS log(n) / eps^2); delta is not used there.
    """
    rng = np.random.default_rng(rng)
    n_a = len(A)
    metric_scipy = _METRIC_MAP.get(metric, metric)
    if tau is None:
        tau = math.sqrt(math.log(max(n_a, 2)))

    pilot_size = min(int(pilot_size), n_a)
    if pilot_size <= 0:
        raise ValueError("pilot_size must be positive")

    t0 = time.perf_counter()
    pilot_idx = rng.choice(n_a, size=pilot_size, replace=False)
    gamma_pilot = cdist(A[pilot_idx], B, metric=metric_scipy).min(axis=1)
    _, _, cv_hat, M_hat = _pilot_stats(gamma_pilot)
    pilot_time = time.perf_counter() - t0

    branch = "uniform" if cv_hat <= tau else "is"
    t1 = time.perf_counter()

    T_unif_raw = None
    T_unif = None
    T_is = None
    extra_idx = np.array([], dtype=int)

    if branch == "uniform":
        T_unif_raw = int(math.ceil(cv_hat ** 2 / (eps ** 2 * delta)))
        T_unif = min(n_a, max(pilot_size, T_unif_raw))
        if total_budget is not None:
            T_unif = min(T_unif, max(pilot_size, int(total_budget)))

        if T_unif == n_a:
            estimate, _ = brute_force_chamfer(A, B, metric=metric)
            branch_taken = "uniform_exact"
            T_used = n_a
        else:
            n_extra = T_unif - pilot_size
            if n_extra > 0:
                remaining = np.setdiff1d(np.arange(n_a), pilot_idx, assume_unique=False)
                extra_idx = rng.choice(remaining, size=n_extra, replace=False)
                gamma_extra = cdist(A[extra_idx], B, metric=metric_scipy).min(axis=1)
                gamma_all = np.concatenate([gamma_pilot, gamma_extra])
            else:
                gamma_all = gamma_pilot
            T_used = len(gamma_all)
            estimate = float(n_a * gamma_all.sum() / T_used)
            branch_taken = "uniform"
    else:
        D_a = crude_nn(A, B, metric=metric, num_levels=num_levels, rng=rng)
        T_is = max(2, int(math.ceil(C_IS_value * math.log(max(n_a, 2)) / (eps ** 2))))
        if total_budget is not None:
            T_is = min(T_is, max(2, int(total_budget)))
        estimate, _, _ = importance_sampling_chamfer(A, B, D_a, T=T_is, metric=metric, rng=rng)
        T_used = T_is
        branch_taken = "is"

    main_time = time.perf_counter() - t1

    if return_diagnostics:
        diag = {
            "branch_taken": branch_taken,
            "branch": branch_taken,
            "pilot_cv": cv_hat,
            "cv_hat": cv_hat,
            "M_hat": M_hat,
            "T_U_raw": T_unif_raw,
            "T_U": T_unif,
            "T_IS": T_is,
            "T_used": T_used,
            "num_exact_nn_queries": T_used,
            "pilot_indices": pilot_idx,
            "extra_indices": extra_idx,
            "pilot_time": pilot_time,
            "main_time": main_time,
        }
        return estimate, diag
    return estimate


def AdaptiveChamfer(A, B, eps, delta, s=50, tau=None, metric="l1", C_IS=4.0, rng=None):
    """Spec-shaped Adaptive-Chamfer entry point."""
    estimate, diag = adaptive_chamfer(
        A,
        B,
        eps=eps,
        pilot_size=s,
        tau=tau,
        delta=delta,
        return_diagnostics=True,
        metric=metric,
        rng=rng,
        C_IS_value=C_IS,
    )
    out = {
        "estimate": estimate,
        "branch": diag["branch"],
        "cv_hat": diag["cv_hat"],
        "M_hat": diag["M_hat"],
        "num_exact_nn_queries": diag["num_exact_nn_queries"],
        "pilot_indices": diag["pilot_indices"],
    }
    if diag["branch"] in {"uniform", "uniform_exact"}:
        out.update({
            "T_U_raw": diag["T_U_raw"],
            "T_U": diag["T_U"],
            "extra_indices": diag["extra_indices"],
        })
    else:
        out["T_IS"] = diag["T_IS"]
    return out
