"""End-to-end comparison using the ACTUAL datasets from Bakshi et al. 2023.

Downloads happen automatically on first run (cached in data_cache/).

Dataset correspondence with Bakshi Table 1 / Figures 3-4
---------------------------------------------------------
  (a) ShapeNet   → ShapeNet Part HDF5 (Stanford, no registration, ~346 MB)
                   chair (97%) + airplane (3%) → high CV, n=8000, d=3, l1
                   Requires: pip install h5py

  (b) Text       → GloVe 300d (gensim, ~376 MB) + 20 Newsgroups (sklearn)
                   sports (70%) + science (30%) docs embedded as mean GloVe vecs
                   n_a=2500, n_b=1800, d=300, l1
                   Requires: pip install gensim scikit-learn

  (c) Outlier    → Gaussian + planted outlier (synthetic, unchanged)
                   n=1000, d=20, l1, one-sided

Results saved to results/end_to_end_real.csv (separate from synthetic runs).
"""

import sys, os, math, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from chamfer_core import (brute_force_chamfer, uniform_sampling_chamfer,
                          importance_sampling_chamfer)
from crude_nn import crude_nn
from adaptive import adaptive_chamfer
from data_gen import make_outlier_pair
from data_gen_real import (make_shapenet_part_pair, make_glove_20news_pair,
                           download_shapenet_part, _get_glove_20news_embeddings)
from utils import compute_cv, relative_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EPS        = 0.10
T_UNIF     = 50
T_IS       = 20
S          = 50
NUM_LEVELS = 3
N_TRIALS   = 25

# (name, metric, symmetric, make_fn, kwargs)
DATASETS = [
    ("ShapeNet-Part (chair+airplane)",
     "l1", True,
     make_shapenet_part_pair,
     dict(n=8000, high_cv=True)),

    ("GloVe-20News (sports+science)",
     "l1", True,
     make_glove_20news_pair,
     dict(n_a=2500, n_b=1800, d=300, high_cv=True)),

    ("Outlier (synthetic, n=1k d=20)",
     "l1", False,
     make_outlier_pair,
     dict()),
]


def run_dataset(name, metric, symmetric, make_fn, kwargs):
    print(f"\n=== {name} ===", flush=True)
    rows = []
    branch_counts = {"uniform": 0, "is": 0}

    for trial in range(N_TRIALS):
        seed = trial * 31 + 7
        A, B = make_fn(**kwargs, rng=seed)
        if A is None:
            print(f"  [SKIP] data unavailable — check install log above.")
            return None, None

        n_a = len(A)
        tau = math.sqrt(math.log(n_a))

        # Brute force (ground truth)
        t0 = time.perf_counter()
        ch_ab, gamma_ab = brute_force_chamfer(A, B, metric=metric)
        if symmetric:
            ch_ba, _ = brute_force_chamfer(B, A, metric=metric)
            ch_true = ch_ab + ch_ba
        else:
            ch_true = ch_ab
        bf_time = time.perf_counter() - t0
        true_cv = compute_cv(gamma_ab)

        # Uniform (T_UNIF fixed)
        t0 = time.perf_counter()
        u_ab, _, _, _ = uniform_sampling_chamfer(A, B, T=T_UNIF, metric=metric, rng=seed+1)
        u_est = u_ab
        if symmetric:
            u_ba, _, _, _ = uniform_sampling_chamfer(B, A, T=T_UNIF, metric=metric, rng=seed+11)
            u_est += u_ba
        u_time = time.perf_counter() - t0
        u_err = relative_error(u_est, ch_true)

        # IS (T_IS fixed, CrudeNN inside timing)
        t0 = time.perf_counter()
        D_ab = crude_nn(A, B, metric=metric, num_levels=NUM_LEVELS, rng=seed+2)
        is_ab, _, _ = importance_sampling_chamfer(A, B, D_ab, T=T_IS, metric=metric, rng=seed+3)
        is_est = is_ab
        if symmetric:
            D_ba = crude_nn(B, A, metric=metric, num_levels=NUM_LEVELS, rng=seed+12)
            is_ba, _, _ = importance_sampling_chamfer(B, A, D_ba, T=T_IS, metric=metric, rng=seed+13)
            is_est += is_ba
        is_time = time.perf_counter() - t0
        is_err = relative_error(is_est, ch_true)

        # Adaptive (A→B direction only; compare against one-sided truth)
        t0 = time.perf_counter()
        ad_est, diag = adaptive_chamfer(
            A, B, eps=EPS, pilot_size=S, tau=tau,
            return_diagnostics=True, metric=metric,
            num_levels=NUM_LEVELS, rng=seed+4,
        )
        ad_time = time.perf_counter() - t0
        ad_err = relative_error(ad_est, ch_ab)
        branch_counts[diag["branch_taken"]] += 1

        rows.append({
            "dataset":     name,
            "trial":       trial,
            "n_a":         n_a,
            "metric":      metric,
            "symmetric":   symmetric,
            "true_cv":     true_cv,
            "tau":         tau,
            "bf_time":     bf_time,
            "u_time":      u_time,  "u_err":  u_err,
            "is_time":     is_time, "is_err": is_err,
            "ad_time":     ad_time, "ad_err": ad_err,
            "ad_branch":   diag["branch_taken"],
            "ad_pilot_cv": diag["pilot_cv"],
            "ad_T_used":   diag["T_used"],
        })

        if (trial + 1) % 5 == 0 or trial == 0:
            print(f"  trial {trial+1:2d}/{N_TRIALS}  "
                  f"CV={true_cv:.2f}  tau={tau:.2f}  "
                  f"u_err={u_err:.3f}  is_err={is_err:.3f}  "
                  f"ad_err={ad_err:.3f}  branch={diag['branch_taken']}",
                  flush=True)

    return rows, branch_counts


def main():
    print("=" * 70)
    print("END-TO-END — ACTUAL PAPER DATASETS (Bakshi et al. 2023)")
    print("=" * 70)

    # Prefetch / download everything upfront so trial timings are unaffected
    print("\n[Step 1] Prefetching datasets (downloads on first run) ...")
    print("  (a) ShapeNet Part HDF5 ...")
    download_shapenet_part(verbose=True)
    print("  (b) GloVe 300d + 20 Newsgroups ...")
    _get_glove_20news_embeddings(d=300, verbose=True)
    print("  (c) Outlier: synthetic, no download.")

    all_rows = []
    summary  = []
    t_start  = time.perf_counter()

    for name, metric, symmetric, fn, kwargs in DATASETS:
        result = run_dataset(name, metric, symmetric, fn, kwargs)
        if result[0] is None:
            continue
        rows, branches = result
        all_rows.extend(rows)

        summary.append({
            "dataset":           name,
            "metric":            metric,
            "symmetric":         symmetric,
            "true_cv_mean":      float(np.mean([r["true_cv"]  for r in rows])),
            "bf_time_ms":        float(np.mean([r["bf_time"]  for r in rows])) * 1000,
            "unif_time_ms":      float(np.mean([r["u_time"]   for r in rows])) * 1000,
            "unif_err":          float(np.mean([r["u_err"]    for r in rows])),
            "is_time_ms":        float(np.mean([r["is_time"]  for r in rows])) * 1000,
            "is_err":            float(np.mean([r["is_err"]   for r in rows])),
            "ad_time_ms":        float(np.mean([r["ad_time"]  for r in rows])) * 1000,
            "ad_err":            float(np.mean([r["ad_err"]   for r in rows])),
            "ad_branch_uniform": branches["uniform"],
            "ad_branch_is":      branches["is"],
        })

    if not summary:
        print("\nNo datasets completed. Check the install log above.")
        print("Required: pip install h5py gensim scikit-learn requests")
        return []

    # Save CSVs
    raw_path = os.path.join(RESULTS_DIR, "end_to_end_real.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nSaved {raw_path}")

    sum_path = os.path.join(RESULTS_DIR, "end_to_end_real_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"Saved {sum_path}")

    # Print table
    W = 122
    print("\n\n" + "=" * W)
    print("REAL DATA — End-to-End Results  "
          f"(eps=0.1, T_unif={T_UNIF}, T_IS={T_IS}, s={S}, tau=sqrt(log n))")
    print(f"{'Dataset':<36} | {'CV':>5} | "
          f"{'BF ms':>8} | {'U err':>6} {'U ms':>6} | "
          f"{'IS err':>6} {'IS ms':>7} | "
          f"{'AD err':>6} {'AD ms':>7} | {'U/IS':>7}")
    print("-" * W)
    for s in summary:
        sym = "(sym)" if s["symmetric"] else "(1sd)"
        print(f"{s['dataset']:<36} | {s['true_cv_mean']:>5.2f} | "
              f"{s['bf_time_ms']:>8.1f} | "
              f"{s['unif_err']:>6.3f} {s['unif_time_ms']:>6.1f} | "
              f"{s['is_err']:>6.3f} {s['is_time_ms']:>7.1f} | "
              f"{s['ad_err']:>6.3f} {s['ad_time_ms']:>7.1f} | "
              f"{s['ad_branch_uniform']:>3}/{s['ad_branch_is']:<3} {sym}")
    print("=" * W)
    print(f"\nTotal time: {time.perf_counter()-t_start:.1f}s")
    return summary


if __name__ == "__main__":
    main()
