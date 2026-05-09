"""Unit tests for adaptive_chamfer (Phase 3).

Three tests:
  1. Low-CV gaussian: should take 'uniform' branch, error within +-2*eps.
  2. Outlier dataset (CV >> 5): should take 'is' branch, error within +-2*eps.
  3. Borderline cluster data (CV ~ tau): either branch acceptable; check correctness.

eps = 0.1, pilot_size = 50, tau = sqrt(log(n)).
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from chamfer_core import brute_force_chamfer
from data_gen import make_gaussian_data, make_outlier_data, make_cluster_data
from adaptive import adaptive_chamfer
from utils import compute_cv, exact_nn_distances, relative_error


def run_one(name, A, B, expected_branch, eps=0.1, pilot_size=50, metric="l2", seed=0):
    n_a = len(A)
    tau = math.sqrt(math.log(n_a))

    ch_true, gamma = brute_force_chamfer(A, B, metric=metric)
    true_cv = compute_cv(gamma)

    est, diag = adaptive_chamfer(
        A, B, eps=eps, pilot_size=pilot_size, tau=tau,
        return_diagnostics=True, metric=metric, rng=seed,
    )

    rerr = relative_error(est, ch_true)

    branch_ok = (expected_branch is None) or (diag["branch_taken"] == expected_branch)
    err_ok = rerr <= 2 * eps
    passed = branch_ok and err_ok

    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"    n={n_a}, true_CV={true_cv:.3f}, tau={tau:.3f}")
    print(f"    branch={diag['branch_taken']} (expected {expected_branch}), pilot_CV={diag['pilot_cv']:.3f}, T_used={diag['T_used']}")
    print(f"    rel_error={rerr:.4f} (target <= {2*eps:.2f}), CH_true={ch_true:.3f}, CH_hat={est:.3f}")
    return passed


def main():
    rng = np.random.default_rng(42)
    eps = 0.1
    pilot_size = 50

    print("=" * 70)
    print("Adaptive-Chamfer Unit Tests (eps=%.2f, pilot_size=%d)" % (eps, pilot_size))
    print("=" * 70)

    n_pass = 0
    n_total = 0

    # --- Test 1: low-CV Gaussian ---
    A1, B1 = make_gaussian_data(1000, d=10, rng=0)
    n_total += 1
    if run_one("Test 1 [Low-CV Gaussian]", A1, B1, expected_branch="uniform",
               eps=eps, pilot_size=pilot_size, seed=1):
        n_pass += 1
    print()

    # --- Test 2: many-outlier (CV >> 5) ---
    # Use 10% outliers so a 50-sample pilot reliably detects the high CV.
    # (A single-outlier dataset is ~95% likely to be missed by a 50-pilot,
    #  which is a separate failure-mode experiment, not a unit test.)
    rng2 = np.random.default_rng(0)
    n2 = 1000
    A2 = rng2.standard_normal((n2, 20))
    B2 = rng2.standard_normal((n2, 20))
    n_out = int(0.1 * n2)
    A2[:n_out] = rng2.standard_normal((n_out, 20)) + 30.0
    n_total += 1
    if run_one("Test 2 [Outlier high-CV]", A2, B2, expected_branch="is",
               eps=eps, pilot_size=pilot_size, seed=2):
        n_pass += 1
    print()

    # --- Test 3: borderline cluster (CV ~ tau) ---
    # Cluster data with moderate separation gives CV in roughly [1, sqrt(log n)] range.
    A3, B3 = make_cluster_data(1000, d=10, k=5, sep=3.0, rng=0)
    n_total += 1
    if run_one("Test 3 [Borderline Cluster]", A3, B3, expected_branch=None,
               eps=eps, pilot_size=pilot_size, seed=3):
        n_pass += 1
    print()

    print("=" * 70)
    print(f"Summary: {n_pass}/{n_total} PASS")
    print("=" * 70)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
