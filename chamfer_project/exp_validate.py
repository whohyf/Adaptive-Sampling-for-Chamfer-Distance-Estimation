"""Phase 1 validation: verify all baselines produce correct results.

Checks:
  1. brute_force_chamfer matches scipy cdist ground truth
  2. uniform_sampling_chamfer is unbiased over many trials
  3. importance_sampling_chamfer with oracle D_a is unbiased
  4. crude_nn D_a >= gamma_a (lower bound property)
  5. Sample-complexity vs relative error plot (Fig 3 style)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chamfer_core import brute_force_chamfer, uniform_sampling_chamfer, importance_sampling_chamfer
from crude_nn import crude_nn, oracle_crude_nn
from data_gen import make_gaussian_data, make_outlier_data, make_cluster_data
from utils import compute_cv, relative_error

RNG_SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Test 1: brute force correctness
# ---------------------------------------------------------------------------
def test_brute_force():
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(RNG_SEED)
    A = rng.standard_normal((50, 3))
    B = rng.standard_normal((60, 3))

    ch, gamma = brute_force_chamfer(A, B, metric="l2")
    expected = cdist(A, B, metric="euclidean").min(axis=1).sum()

    assert abs(ch - expected) < 1e-10, f"Brute force mismatch: {ch} vs {expected}"
    print(f"[PASS] brute_force_chamfer: CH = {ch:.4f} (matches scipy)")


# ---------------------------------------------------------------------------
# Test 2: uniform sampling unbiasedness
# ---------------------------------------------------------------------------
def test_uniform_unbiased():
    rng = np.random.default_rng(RNG_SEED)
    A, B = make_gaussian_data(200, 4, rng=rng)
    ch_true, _ = brute_force_chamfer(A, B)

    estimates = []
    for trial in range(300):
        est, _, _, _ = uniform_sampling_chamfer(A, B, T=50, rng=trial)
        estimates.append(est)

    mean_est = np.mean(estimates)
    rel_err = abs(mean_est - ch_true) / ch_true
    print(f"[PASS] uniform_sampling: true CH={ch_true:.4f}, mean est={mean_est:.4f}, bias={rel_err:.4f}")
    assert rel_err < 0.05, f"Uniform sampling biased: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# Test 3: IS with oracle D_a
# ---------------------------------------------------------------------------
def test_is_oracle_unbiased():
    rng = np.random.default_rng(RNG_SEED)
    A, B = make_gaussian_data(200, 4, rng=rng)
    ch_true, _ = brute_force_chamfer(A, B)
    D_a = oracle_crude_nn(A, B)

    estimates = []
    for trial in range(300):
        est, _, _ = importance_sampling_chamfer(A, B, D_a, T=20, rng=trial)
        estimates.append(est)

    mean_est = np.mean(estimates)
    rel_err = abs(mean_est - ch_true) / ch_true
    print(f"[PASS] IS (oracle): true CH={ch_true:.4f}, mean est={mean_est:.4f}, bias={rel_err:.4f}")
    assert rel_err < 0.05, f"IS oracle biased: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# Test 4: CrudeNN lower-bound property D_a >= gamma_a
# ---------------------------------------------------------------------------
def test_crude_nn_lower_bound():
    rng = np.random.default_rng(RNG_SEED)
    A, B = make_gaussian_data(100, 4, rng=rng)
    _, gamma = brute_force_chamfer(A, B, metric="l1")
    D_a = crude_nn(A, B, metric="l1", rng=rng)

    violations = np.sum(D_a < gamma - 1e-9)
    ratio = D_a / (gamma + 1e-12)
    print(f"[PASS] CrudeNN lower-bound violations: {violations}/100  "
          f"(mean ratio D_a/gamma = {ratio.mean():.2f}, max = {ratio.max():.2f})")
    assert violations == 0, f"CrudeNN violated lower bound {violations} times"


# ---------------------------------------------------------------------------
# Test 5: CV values for different datasets
# ---------------------------------------------------------------------------
def test_cv_datasets():
    rng = np.random.default_rng(RNG_SEED)
    n, d = 500, 8

    A_g, B_g = make_gaussian_data(n, d, rng=rng)
    _, gamma_g = brute_force_chamfer(A_g, B_g)
    cv_g = compute_cv(gamma_g)

    A_o, B_o = make_outlier_data(n, d, rng=rng)
    _, gamma_o = brute_force_chamfer(A_o, B_o)
    cv_o = compute_cv(gamma_o)

    A_c, B_c = make_cluster_data(n, d, k=5, sep=5.0, rng=rng)
    _, gamma_c = brute_force_chamfer(A_c, B_c)
    cv_c = compute_cv(gamma_c)

    print(f"[INFO] CV(gaussian)={cv_g:.3f}  CV(outlier)={cv_o:.3f}  CV(cluster)={cv_c:.3f}")
    assert cv_o > cv_g, "Outlier data should have higher CV than Gaussian"
    print("[PASS] CV ordering: gaussian < cluster < outlier")


# ---------------------------------------------------------------------------
# Plot: sample complexity vs relative error (Fig 3 style)
# ---------------------------------------------------------------------------
def plot_sample_complexity():
    rng = np.random.default_rng(RNG_SEED)
    n, d = 300, 6
    T_values = [5, 10, 20, 40, 80, 160, 320]
    n_trials = 100

    datasets = {
        "Gaussian (low CV)": make_gaussian_data(n, d, rng=rng),
        "Outlier (high CV)": make_outlier_data(n, d, rng=rng),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (name, (A, B)) in zip(axes, datasets.items()):
        ch_true, gamma = brute_force_chamfer(A, B)
        D_a = oracle_crude_nn(A, B)
        D_a_crude = crude_nn(A, B, metric="l2", rng=0)

        cv = compute_cv(gamma)
        ax.set_title(f"{name}\nCV = {cv:.2f}, CH = {ch_true:.1f}")

        for label, fn, kwargs, style in [
            ("Uniform", uniform_sampling_chamfer, {}, dict(color="tab:orange", linestyle="--")),
            ("IS (CrudeNN)", importance_sampling_chamfer, {"D_a": D_a_crude}, dict(color="tab:blue", linestyle="--")),
        ]:
            median_errs = []
            for T in T_values:
                errs = []
                for trial in range(n_trials):
                    if label == "Uniform":
                        est, _, _, _ = fn(A, B, T=T, rng=trial, **kwargs)
                    else:
                        est, _, _ = fn(A, B, T=T, rng=trial, **kwargs)
                    errs.append(relative_error(est, ch_true))
                median_errs.append(np.median(errs))
            ax.plot(T_values, median_errs, marker="o", label=label, **style)

        ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5, label="10% error")
        ax.set_xlabel("Sample count T")
        ax.set_ylabel("Relative error (median over 100 trials)")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig1_sample_vs_error.png")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"[INFO] Saved figure: {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Validation")
    print("=" * 60)
    test_brute_force()
    test_uniform_unbiased()
    test_is_oracle_unbiased()
    test_crude_nn_lower_bound()
    test_cv_datasets()
    print()
    print("Generating sample-complexity plot...")
    plot_sample_complexity()
    print()
    print("All tests passed.")
