"""Phase 4.3 extension: borderline-CV subset analysis for threshold sweep.

Rebuilds the exact same instance set as exp_threshold_sweep.py (same seeds,
same hyperparameters) and re-runs the tau sweep with per-instance tracking so
that FP/FN rates can be split into "all instances" vs "borderline-CV instances".

Borderline definition (CV-squared vs log-n break-even):
    true_cv_sq = true_cv ** 2
    log_n      = math.log(n)
    borderline = (true_cv_sq >= 0.7 * log_n) and (true_cv_sq <= 1.3 * log_n)

Output: results/threshold_sweep_borderline.csv
Columns: tau, n_total, fp_total, fn_total, n_borderline, fp_borderline, fn_borderline

Original threshold_sweep.csv is NOT modified.
"""
import sys, os, math, csv, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from chamfer_core import brute_force_chamfer
from adaptive import adaptive_chamfer
from utils import compute_cv, compute_true_M, bernstein_oracle

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Must match exp_threshold_sweep.py exactly ----
OUTLIER_FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
OUTLIER_MAGS      = [2.0, 5.0, 10.0]
N_VALUES          = [1000, 2000, 5000]
N_TRIALS          = 3
S                 = 50
D                 = 3
METRIC            = "l2"
EPS               = 0.1
DELTA             = 0.01
C_IS              = 4.0
C_NN              = 1.0
NUM_LEVELS        = 3
TAU_GRID          = np.linspace(0.5, 5.0, 20)


def make_instance(n, d, outlier_fraction, outlier_mag, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d))
    B = rng.standard_normal((n, d))
    n_out = int(round(outlier_fraction * n))
    if n_out > 0:
        dirs = rng.standard_normal((n_out, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        A[:n_out] = dirs * outlier_mag
    return A, B


def instance_id_for(n, outlier_fraction, outlier_mag, trial):
    return f"n={n}|of={outlier_fraction}|om={outlier_mag}|trial={trial}"


def adaptive_side(branch):
    return "uniform" if branch in {"uniform", "uniform_exact"} else "is"


def make_seed(base, *parts):
    """Deterministic seed from base seed and integer/float parts."""
    seed = int(base)
    for p in parts:
        if isinstance(p, float):
            p = int(p * 1000)
        seed = (seed * 1_000_003 + int(p)) % (2**31 - 1)
    return seed


def cost_oracle(true_cv, true_M, n):
    """Bernstein cost-model oracle; no wall-clock measurements."""
    return bernstein_oracle(
        true_cv, true_M, n, eps=EPS, delta=DELTA, C_IS=C_IS, C_NN=C_NN
    )["oracle_branch"]


def is_borderline(true_cv, n):
    true_cv_sq = true_cv ** 2
    log_n      = math.log(n)
    return (true_cv_sq >= 0.7 * log_n) and (true_cv_sq <= 1.3 * log_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--out", type=str, default=RESULTS_DIR, help="Output directory")
    parser.add_argument(
        "--write-per-trial",
        action="store_true",
        help="Write threshold_sweep_borderline_trials.csv with per tau/instance routing details",
    )
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: rebuild instance set (deterministic from seeds)
    # ------------------------------------------------------------------
    instances = []
    print("Building instance set (same seeds as exp_threshold_sweep.py)...")
    for n in N_VALUES:
        for of in OUTLIER_FRACTIONS:
            for om in OUTLIER_MAGS:
                for trial in range(N_TRIALS):
                    seed = make_seed(args.seed, n, of, om, trial)
                    A, B  = make_instance(n, D, of, om, seed)
                    ch_true, gamma = brute_force_chamfer(A, B, metric=METRIC)
                    true_cv = compute_cv(gamma)
                    true_M = compute_true_M(gamma)
                    oracle  = cost_oracle(true_cv, true_M, n)
                    instances.append({
                        "A": A, "B": B,
                        "instance_id": instance_id_for(n, of, om, trial),
                        "dataset_family": "outlier_grid",
                        "ch_true": ch_true,
                        "true_cv": true_cv,
                        "true_M": true_M,
                        "n": n, "of": of, "om": om, "trial": trial,
                        "seed": seed,
                        "oracle": oracle,
                        "borderline": is_borderline(true_cv, n),
                    })
    print(f"  {len(instances)} instances built ({time.perf_counter()-t_start:.1f}s)")

    n_borderline_instances = sum(1 for i in instances if i["borderline"])
    print(f"  Borderline instances: {n_borderline_instances} / {len(instances)} "
          f"({100*n_borderline_instances/len(instances):.1f}%)")

    # ------------------------------------------------------------------
    # Step 2: tau sweep with per-instance tracking
    # ------------------------------------------------------------------
    rows = []
    per_trial_rows = []
    print("\nSweeping tau...")
    for tau_idx, tau in enumerate(TAU_GRID):
        fp_total = fn_total = 0
        fp_bl    = fn_bl    = 0
        n_bl     = sum(1 for i in instances if i["borderline"])

        for inst in instances:
            ad_seed = make_seed(args.seed, inst["n"], inst["of"], inst["om"], inst["trial"], tau_idx)
            _, diag = adaptive_chamfer(
                inst["A"], inst["B"], eps=EPS, pilot_size=S, tau=float(tau),
                return_diagnostics=True, metric=METRIC,
                num_levels=NUM_LEVELS, rng=ad_seed,
            )
            branch = diag["branch_taken"]
            branch_class = adaptive_side(branch)
            oracle = inst["oracle"]
            is_fp = int(oracle == "uniform" and branch_class == "is")
            is_fn = int(oracle == "is" and branch_class == "uniform")

            if args.write_per_trial:
                per_trial_rows.append({
                    "script_name": "exp_threshold_sweep_borderline",
                    "tau": float(tau),
                    "trial_id": inst["trial"],
                    "instance_id": inst["instance_id"],
                    "dataset_family": inst["dataset_family"],
                    "n": inst["n"],
                    "seed_instance": inst["seed"],
                    "seed_pilot": ad_seed,
                    "true_cv": inst["true_cv"],
                    "true_M": inst["true_M"],
                    "cv_hat": diag.get("cv_hat", diag.get("pilot_cv", float("nan"))),
                    "M_hat": diag.get("M_hat", float("nan")),
                    "oracle_branch": oracle,
                    "adaptive_branch": branch,
                    "adaptive_side": branch_class,
                    "is_borderline": int(inst["borderline"]),
                    "is_fp": is_fp,
                    "is_fn": is_fn,
                })

            # All-instances counters
            if is_fp:
                fp_total += 1
            elif is_fn:
                fn_total += 1

            # Borderline counters
            if inst["borderline"]:
                if is_fp:
                    fp_bl += 1
                elif is_fn:
                    fn_bl += 1

        n_oracle_uniform = sum(1 for i in instances if i["oracle"] == "uniform")
        n_oracle_is = sum(1 for i in instances if i["oracle"] == "is")
        n_oracle_uniform_bl = sum(
            1 for i in instances if i["borderline"] and i["oracle"] == "uniform"
        )
        n_oracle_is_bl = sum(
            1 for i in instances if i["borderline"] and i["oracle"] == "is"
        )

        rows.append({
            "tau":          float(tau),
            "n_total":      len(instances),
            "fp_total":     fp_total,
            "fn_total":     fn_total,
            "accuracy_total": 1.0 - (fp_total + fn_total) / len(instances),
            "fp_rate_total": fp_total / max(n_oracle_uniform, 1),
            "fn_rate_total": fn_total / max(n_oracle_is, 1),
            "n_borderline": n_bl,
            "fp_borderline": fp_bl,
            "fn_borderline": fn_bl,
            "accuracy_border": 1.0 - (fp_bl + fn_bl) / n_bl if n_bl else float("nan"),
            "fp_rate_border": fp_bl / max(n_oracle_uniform_bl, 1),
            "fn_rate_border": fn_bl / max(n_oracle_is_bl, 1),
            "n_oracle_uniform_border": n_oracle_uniform_bl,
            "n_oracle_is_border": n_oracle_is_bl,
        })
        print(f"  tau={tau:.3f}  fp_total={fp_total}  fn_total={fn_total}  "
              f"fp_bl={fp_bl}  fn_bl={fn_bl}  n_bl={n_bl}")

    # ------------------------------------------------------------------
    # Step 3: save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(out_dir, "threshold_sweep_borderline.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")

    if args.write_per_trial:
        trial_path = os.path.join(out_dir, "threshold_sweep_borderline_trials.csv")
        with open(trial_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_trial_rows[0].keys()))
            w.writeheader()
            w.writerows(per_trial_rows)
        print(f"Saved {trial_path} ({len(per_trial_rows)} rows)")

    # ------------------------------------------------------------------
    # Step 4: sanity checks
    # ------------------------------------------------------------------
    print("\n--- Sanity checks ---")
    n_bl_values = [r["n_borderline"] for r in rows]
    if all(v > 0 for v in n_bl_values):
        print("  PASS: n_borderline > 0 for all tau values.")
    else:
        zero_taus = [rows[i]["tau"] for i, v in enumerate(n_bl_values) if v == 0]
        print(f"  FAIL: n_borderline == 0 at tau = {zero_taus}")

    ratio = n_bl_values[0] / len(instances)
    ratios = [r["n_borderline"] / r["n_total"] for r in rows]
    ratio_min, ratio_max = min(ratios), max(ratios)
    if ratio_max - ratio_min < 0.01:
        print(f"  PASS: n_borderline / n_total is stable across tau rows "
              f"(range: {ratio_min:.3f}–{ratio_max:.3f}).")
    else:
        print(f"  NOTE: n_borderline / n_total varies: {ratio_min:.3f}–{ratio_max:.3f} "
              f"(expected stable; borderline depends on true_cv and n, not tau).")

    if n_bl_values[0] < 10:
        print(f"  FLAG: n_borderline = {n_bl_values[0]} < 10 — borderline numbers will be noisy.")
    else:
        print(f"  OK: n_borderline = {n_bl_values[0]} (>= 10).")

    # ------------------------------------------------------------------
    # Step 5: key finding for §4.3
    # ------------------------------------------------------------------
    print("\n--- Key findings for §4.3 ---")

    # FP rates (borderline) as fraction
    for r in rows:
        n_oracle_uniform_bl = sum(
            1 for i in instances
            if i["borderline"] and i["oracle"] == "uniform"
        )
        r["n_oracle_uniform_bl"] = n_oracle_uniform_bl
        r["fp_rate_bl"] = r["fp_borderline"] / max(n_oracle_uniform_bl, 1)
        n_oracle_is_bl = sum(
            1 for i in instances
            if i["borderline"] and i["oracle"] == "is"
        )
        r["n_oracle_is_bl"] = n_oracle_is_bl
        r["fn_rate_bl"] = r["fn_borderline"] / max(n_oracle_is_bl, 1)

    eligible = [r for r in rows if r["fp_rate_bl"] < 0.30]
    if eligible:
        first = eligible[0]
        print(f"  Smallest tau where fp_borderline < 0.30:")
        print(f"    tau           = {first['tau']:.4f}")
        print(f"    fp_borderline = {first['fp_borderline']} / {first['n_oracle_uniform_bl']} "
              f"= {first['fp_rate_bl']:.3f}")
        print(f"    fn_borderline = {first['fn_borderline']} / {first['n_oracle_is_bl']} "
              f"= {first['fn_rate_bl']:.3f}")
    else:
        print("  fp_borderline never drops below 0.30 in this sweep range.")

    print(f"\nTotal time: {time.perf_counter()-t_start:.1f}s")
    return rows


if __name__ == "__main__":
    main()
