"""Phase 4 Task 5: End-to-end comparison.

For each of three datasets (ShapeNet-like, text-like, outlier), 20 trials at eps=0.1:
  - Data generated ONCE with fixed seed (seed 0 per dataset).
  - Each trial uses independent estimator seeds (seed = trial + 1).
  - Brute force                                  -> runtime, rel_err = 0
  - Uniform sampling (T = 50)                    -> runtime, rel_err
  - Importance sampling (T = ceil(4 log n / eps^2), CrudeNN built) -> runtime, rel_err
  - Adaptive (s = 50, tau = sqrt(log n))         -> runtime, rel_err, branch_taken

Produces a table to stdout and saves results/end_to_end.csv.
This populates Table 1 of the final report.

Usage:
  python exp_end_to_end.py [--seed 0] [--out results/]
"""
import sys, os, math, csv, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from chamfer_core import (brute_force_chamfer, uniform_sampling_chamfer,
                          importance_sampling_chamfer)
from crude_nn import crude_nn
from adaptive import adaptive_chamfer
from data_gen import make_shapenet_pair, make_real_text_pair, make_outlier_pair
from utils import compute_cv, compute_true_M, relative_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EPS = 0.10
DELTA = 0.01
T_UNIF = 50
S = 50
NUM_LEVELS = 3
N_TRIALS = 20  # was 25; changed to 20 per ALGORITHM_SPEC §3.3
C_IS_ORACLE = 4.0
C_NN_ORACLE = 1.0


def is_sample_budget(n, eps=EPS, C_IS=C_IS_ORACLE):
    return max(2, int(math.ceil(C_IS * math.log(max(n, 2)) / eps ** 2)))


def cost_oracle_bernstein(true_cv, true_M, n, eps=EPS, delta=DELTA):
    """Bernstein-based cost-model oracle. See ALGORITHM_SPEC v3 §2.2."""
    cost_unif = min(
        2 * (true_cv ** 2 + true_M * eps / 3) * math.log(2 / delta) / eps ** 2,
        n,
    )
    cost_is = C_NN_ORACLE * math.log(n) + C_IS_ORACLE * math.log(n) / eps ** 2
    return "uniform" if cost_unif <= cost_is else "is"

# Text-like uses high_cv=False (low-CV regime, CV ~0.5) per ALGORITHM_SPEC §3.1 (D-2).
DATASETS = [
    ("ShapeNet-like", "l2", make_shapenet_pair,  dict(n=8000, high_cv=True)),
    ("Text-like",     "l2", make_real_text_pair, dict(n_a=2500, n_b=1800, d=300, high_cv=False)),
    ("Outlier",       "l2", make_outlier_pair,   dict(n=1000, d=20)),
]


def make_seed(base, *parts):
    """Deterministic seed from base seed and integer/float parts."""
    seed = int(base)
    for p in parts:
        if isinstance(p, float):
            p = int(p * 1000)
        seed = (seed * 1_000_003 + int(p)) % (2**31 - 1)
    return seed


def run_dataset(name, metric, make_fn, kwargs, base_seed):
    print(f"\n=== {name} ===")
    rows = []
    branch_counts = {"uniform": 0, "is": 0, "uniform_exact": 0}

    # Generate data ONCE with fixed seed (isolates estimator variance).
    data_seed = make_seed(base_seed, 0)
    A, B = make_fn(**kwargs, rng=data_seed)
    n_a = len(A)
    tau = math.sqrt(math.log(n_a))

    # Compute ground truth once.
    t0 = time.perf_counter()
    ch_true, gamma = brute_force_chamfer(A, B, metric=metric)
    bf_time = time.perf_counter() - t0
    true_cv = compute_cv(gamma)
    true_M = compute_true_M(gamma)
    oracle_bernstein = cost_oracle_bernstein(true_cv, true_M, n_a)
    exact_cost = n_a
    uniform_cost = min(T_UNIF, n_a)
    T_is_dataset = is_sample_budget(n_a)
    is_cost = C_NN_ORACLE * math.log(n_a) + T_is_dataset
    print(f"  n={n_a}  true_CV={true_cv:.3f}  true_M={true_M:.2f}  tau={tau:.3f}  "
          f"oracle={oracle_bernstein}  bf={bf_time*1000:.1f}ms")

    for trial in range(N_TRIALS):
        # Estimator seeds vary across trials; data is fixed.
        estimator_seed = trial + 1

        u_seed = make_seed(base_seed, estimator_seed, 1)
        is_seed_crude = make_seed(base_seed, estimator_seed, 2)
        is_seed_is = make_seed(base_seed, estimator_seed, 3)
        ad_seed = make_seed(base_seed, estimator_seed, 4)

        # --- Uniform (T=50) ---
        t0 = time.perf_counter()
        u_est, _, _, _ = uniform_sampling_chamfer(A, B, T=T_UNIF, metric=metric, rng=u_seed)
        u_time = time.perf_counter() - t0
        u_err = relative_error(u_est, ch_true)

        # --- IS (original fixed-success-probability budget) ---
        t0 = time.perf_counter()
        D_a = crude_nn(A, B, metric=metric, num_levels=NUM_LEVELS, rng=is_seed_crude)
        is_est, _, _ = importance_sampling_chamfer(
            A, B, D_a, T=T_is_dataset, metric=metric, rng=is_seed_is
        )
        is_time = time.perf_counter() - t0
        is_err = relative_error(is_est, ch_true)

        # --- Adaptive ---
        t0 = time.perf_counter()
        ad_est, diag = adaptive_chamfer(
            A, B, eps=EPS, pilot_size=S, tau=tau,
            delta=DELTA,
            return_diagnostics=True, metric=metric,
            num_levels=NUM_LEVELS, rng=ad_seed,
        )
        ad_time = time.perf_counter() - t0
        ad_err = relative_error(ad_est, ch_true)
        branch_counts[diag["branch_taken"]] += 1

        ad_branch_class = "uniform" if diag["branch_taken"] == "uniform_exact" else diag["branch_taken"]
        bernstein_correct = int(ad_branch_class == oracle_bernstein)
        rows.append({
            "dataset": name,
            "trial": trial,
            "n_a": n_a,
            "true_cv": true_cv,
            "true_M": true_M,
            "tau": tau,
            "exact_nn_query_cost": exact_cost,
            "uniform_nn_query_cost": uniform_cost,
            "is_nn_query_cost": is_cost,
            "is_T_IS": T_is_dataset,
            "bf_time": bf_time,
            "u_time": u_time, "u_err": u_err,
            "is_time": is_time, "is_err": is_err,
            "ad_time": ad_time, "ad_err": ad_err,
            "ad_branch": diag["branch_taken"],
            "ad_pilot_cv": diag["pilot_cv"],
            "ad_M_hat": diag.get("M_hat", float("nan")),
            "ad_T_used": diag["T_used"],
            "bernstein_oracle": oracle_bernstein,
            "bernstein_correct": bernstein_correct,
        })

    return rows, branch_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--out", type=str, default=RESULTS_DIR, help="Output directory")
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    summary = []
    t_start = time.perf_counter()

    for ds_idx, (name, metric, fn, kwargs) in enumerate(DATASETS):
        ds_seed = make_seed(args.seed, ds_idx)
        rows, branches = run_dataset(name, metric, fn, kwargs, base_seed=ds_seed)
        all_rows.extend(rows)

        bf_t = np.mean([r["bf_time"] for r in rows])
        u_t = np.mean([r["u_time"] for r in rows])
        u_e = np.mean([r["u_err"] for r in rows])
        i_t = np.mean([r["is_time"] for r in rows])
        i_e = np.mean([r["is_err"] for r in rows])
        a_t = np.mean([r["ad_time"] for r in rows])
        a_e = np.mean([r["ad_err"] for r in rows])
        cv_mean = rows[0]["true_cv"]  # fixed data => same CV every trial

        # Check actual CV for text-like; alert if not in low-CV range.
        if name == "Text-like":
            if not (0.2 <= cv_mean <= 0.9):
                print(f"  WARNING: Text-like CV={cv_mean:.3f} outside expected [0.2,0.9] "
                      f"for high_cv=False mode. May need generator tuning.")

        true_M_mean = rows[0]["true_M"]  # fixed data => same M every trial
        bernstein_oracle_val = rows[0]["bernstein_oracle"]
        bernstein_acc = float(np.mean([r["bernstein_correct"] for r in rows]))
        adaptive_cost = float(np.mean([r["ad_T_used"] for r in rows]))
        summary.append({
            "dataset": name,
            "n": rows[0]["n_a"],
            "true_cv_mean": cv_mean,
            "true_cv": cv_mean,
            "true_M": true_M_mean,
            "bernstein_oracle": bernstein_oracle_val,
            "bernstein_accuracy": bernstein_acc,
            "oracle_branch_counts": f"{bernstein_oracle_val}:{len(rows)}",
            "exact_nn_query_cost": rows[0]["exact_nn_query_cost"],
            "uniform_nn_query_cost": rows[0]["uniform_nn_query_cost"],
            "is_nn_query_cost": rows[0]["is_nn_query_cost"],
            "is_T_IS": rows[0]["is_T_IS"],
            "adaptive_nn_query_cost": adaptive_cost,
            "nn_query_equiv_costs": (
                f"exact={rows[0]['exact_nn_query_cost']:.1f};"
                f"uniform={rows[0]['uniform_nn_query_cost']:.1f};"
                f"is={rows[0]['is_nn_query_cost']:.1f};"
                f"adaptive={adaptive_cost:.1f}"
            ),
            "bf_time_ms": bf_t * 1000,
            "unif_time_ms": u_t * 1000, "unif_err": u_e,
            "uniform_error_mean": u_e,
            "is_time_ms": i_t * 1000, "is_err": i_e,
            "is_error_mean": i_e,
            "ad_time_ms": a_t * 1000, "ad_err": a_e,
            "adaptive_error_mean": a_e,
            "ad_branch_uniform": branches.get("uniform", 0),
            "ad_branch_is": branches.get("is", 0),
            "ad_branch_uniform_exact": branches.get("uniform_exact", 0),
            "adaptive_branch_counts": (
                f"uniform={branches.get('uniform', 0)};"
                f"is={branches.get('is', 0)};"
                f"uniform_exact={branches.get('uniform_exact', 0)}"
            ),
        })

    # ---------- Print table ----------
    print("\n\n" + "=" * 120)
    print(f"{'Dataset':<14} | {'CV':>5} | "
          f"{'BF (ms)':>9} | {'U err':>6} {'U t(ms)':>8} | "
          f"{'IS err':>6} {'IS t(ms)':>9} | "
          f"{'AD err':>6} {'AD t(ms)':>9} | {'branch (U/IS/UE)':>16}")
    print("-" * 120)
    for s in summary:
        print(f"{s['dataset']:<14} | {s['true_cv_mean']:>5.2f} | "
              f"{s['bf_time_ms']:>9.2f} | "
              f"{s['unif_err']:>6.3f} {s['unif_time_ms']:>8.2f} | "
              f"{s['is_err']:>6.3f} {s['is_time_ms']:>9.2f} | "
              f"{s['ad_err']:>6.3f} {s['ad_time_ms']:>9.2f} | "
              f"{s['ad_branch_uniform']:>4} / {s['ad_branch_is']:<4} / {s['ad_branch_uniform_exact']:<4}")
    print("=" * 120)

    # Save CSVs.
    raw_path = os.path.join(out_dir, "end_to_end.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {raw_path}")

    sum_path = os.path.join(out_dir, "end_to_end_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Saved {sum_path}")

    print(f"\nTotal time: {time.perf_counter()-t_start:.1f}s")
    return summary


if __name__ == "__main__":
    main()
