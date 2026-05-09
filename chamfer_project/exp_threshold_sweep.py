"""Phase 4.3: Threshold sensitivity.

# DEPRECATED OUTPUT: threshold_sweep.csv (kept for backward compatibility).
# Preferred outputs: threshold_sweep_per_instance.csv, threshold_sweep_aggregated.csv.

Fix s = 50; sweep tau across 20 points in [0.5, 5.0] on the same instance set as
exp_cv_calibration.py.

Oracle: cost-model oracle (ALGORITHM_SPEC §2.2) — deterministic, no wall-clock.

Per-instance log: threshold_sweep_per_instance.csv
  columns: n, true_cv, tau, branch_taken, oracle_decision, rel_error, in_borderline

Aggregated: threshold_sweep_aggregated.csv
  columns: tau, n_total, n_borderline, fp_rate_total, fn_rate_total,
           fp_rate_border, fn_rate_border, mean_rel_error_total

Usage:
  python exp_threshold_sweep.py [--seed 0] [--out results/]
"""
import sys, os, math, csv, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chamfer_core import brute_force_chamfer
from adaptive import adaptive_chamfer
from utils import compute_cv, compute_true_M, relative_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

OUTLIER_FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
OUTLIER_MAGS = [2.0, 5.0, 10.0]
N_VALUES = [1000, 2000, 5000]
N_TRIALS = 3
S = 50
D = 3
METRIC = "l2"
EPS = 0.1
DELTA = 0.01
C_IS = 4.0
C_NN = 1.0
NUM_LEVELS = 3
TAU_GRID = np.linspace(0.5, 5.0, 20)


def make_seed(base, *parts):
    """Deterministic seed from base seed and integer/float parts."""
    seed = int(base)
    for p in parts:
        if isinstance(p, float):
            p = int(p * 1000)
        seed = (seed * 1_000_003 + int(p)) % (2**31 - 1)
    return seed


def cost_oracle(true_cv, true_M, n, eps=EPS, delta=DELTA, C_IS=C_IS, C_NN=C_NN):
    """Bernstein-based cost-model oracle. See ALGORITHM_SPEC v3 §2.2."""
    cost_unif = min(
        2 * (true_cv ** 2 + true_M * eps / 3) * math.log(2 / delta) / eps ** 2,
        n,
    )
    cost_is = C_NN * math.log(n) + C_IS * math.log(n) / eps ** 2
    return "uniform" if cost_unif <= cost_is else "is"


def _cost_terms(true_cv, true_M, n, eps=EPS, delta=DELTA, C_IS=C_IS, C_NN=C_NN):
    """Return (cost_unif_chebyshev, cost_unif_bernstein, cost_is) for logging."""
    cost_cheb = min((true_cv ** 2) / (eps ** 2 * delta), n)
    cost_bern = min(
        2 * (true_cv ** 2 + true_M * eps / 3) * math.log(2 / delta) / eps ** 2,
        n,
    )
    cost_is = C_NN * math.log(n) + C_IS * math.log(n) / eps ** 2
    return cost_cheb, cost_bern, cost_is


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--out", type=str, default=RESULTS_DIR, help="Output directory")
    parser.add_argument(
        "--write-per-trial",
        action="store_true",
        help="Write threshold_sweep_trials.csv with per tau/instance routing details",
    )
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: build instance list with cost-model oracle decisions.
    instances = []
    print("Building instance set & computing oracle...")
    t_start = time.perf_counter()
    for n in N_VALUES:
        borderline_lo = 0.7 * math.log(n)
        borderline_hi = 1.3 * math.log(n)
        for of in OUTLIER_FRACTIONS:
            for om in OUTLIER_MAGS:
                for trial in range(N_TRIALS):
                    seed = make_seed(args.seed, n, of, om, trial)
                    A, B = make_instance(n, D, of, om, seed)
                    ch_true, gamma = brute_force_chamfer(A, B, metric=METRIC)
                    true_cv = compute_cv(gamma)
                    true_M = compute_true_M(gamma)
                    oracle = cost_oracle(true_cv, true_M, n)
                    cost_cheb, cost_bern, cost_is_val = _cost_terms(true_cv, true_M, n)
                    in_borderline = int(borderline_lo <= true_cv ** 2 <= borderline_hi)
                    instances.append({
                        "A": A, "B": B,
                        "instance_id": instance_id_for(n, of, om, trial),
                        "dataset_family": "outlier_grid",
                        "ch_true": ch_true,
                        "true_cv": true_cv,
                        "true_M": true_M,
                        "cost_unif_chebyshev": cost_cheb,
                        "cost_unif_bernstein": cost_bern,
                        "cost_is": cost_is_val,
                        "n": n, "of": of, "om": om, "trial": trial,
                        "seed": seed,
                        "oracle_decision": oracle,
                        "in_borderline": in_borderline,
                    })
    print(f"  {len(instances)} instances ready ({time.perf_counter()-t_start:.1f}s)")

    # Step 2: per-tau sweep.
    per_instance_rows = []
    per_trial_rows = []
    agg_rows = []
    legacy_rows = []

    print("\nSweeping tau...")
    for tau_idx, tau in enumerate(TAU_GRID):
        fp_total = fn_total = n_oracle_unif = n_oracle_is = 0
        fp_border = fn_border = n_border_unif = n_border_is = 0
        rel_errs = []

        for inst in instances:
            ad_seed = make_seed(args.seed, inst["n"], inst["of"], inst["om"], inst["trial"], tau_idx)
            est, diag = adaptive_chamfer(
                inst["A"], inst["B"], eps=EPS, pilot_size=S, tau=float(tau),
                delta=DELTA,
                return_diagnostics=True, metric=METRIC,
                num_levels=NUM_LEVELS, rng=ad_seed,
            )
            rerr = relative_error(est, inst["ch_true"])
            branch = diag["branch_taken"]
            # Classify uniform_exact as uniform for FP/FN accounting.
            branch_class = adaptive_side(branch)
            oracle = inst["oracle_decision"]
            ib = inst["in_borderline"]
            is_fp = int(oracle == "uniform" and branch_class == "is")
            is_fn = int(oracle == "is" and branch_class == "uniform")

            rel_errs.append(rerr)

            per_instance_rows.append({
                "n": inst["n"],
                "true_cv": inst["true_cv"],
                "true_M": inst["true_M"],
                "cv_hat": diag.get("cv_hat", diag.get("pilot_cv", float("nan"))),
                "M_hat": diag.get("M_hat", float("nan")),
                "cost_unif_chebyshev": inst["cost_unif_chebyshev"],
                "cost_unif_bernstein": inst["cost_unif_bernstein"],
                "cost_is": inst["cost_is"],
                "tau": float(tau),
                "branch_taken": branch,
                "oracle_decision": oracle,
                "rel_error": rerr,
                "in_borderline": ib,
            })

            if args.write_per_trial:
                per_trial_rows.append({
                    "script_name": "exp_threshold_sweep",
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
                    "is_borderline": ib,
                    "is_fp": is_fp,
                    "is_fn": is_fn,
                })

            # Total FP/FN accounting.
            if oracle == "uniform":
                n_oracle_unif += 1
                if is_fp:
                    fp_total += 1
            else:
                n_oracle_is += 1
                if is_fn:
                    fn_total += 1

            # Borderline FP/FN accounting.
            if ib:
                if oracle == "uniform":
                    n_border_unif += 1
                    if is_fp:
                        fp_border += 1
                else:
                    n_border_is += 1
                    if is_fn:
                        fn_border += 1

        fp_rate_total = fp_total / max(n_oracle_unif, 1)
        fn_rate_total = fn_total / max(n_oracle_is, 1)
        fp_rate_border = fp_border / max(n_border_unif, 1)
        fn_rate_border = fn_border / max(n_border_is, 1)
        accuracy_total = 1.0 - (fp_total + fn_total) / len(instances)
        n_border_total = n_border_unif + n_border_is
        accuracy_border = (
            1.0 - (fp_border + fn_border) / n_border_total
            if n_border_total else float("nan")
        )
        mean_err = float(np.mean(rel_errs))

        agg_rows.append({
            "tau": float(tau),
            "n_total": len(instances),
            "n_borderline": n_border_total,
            "n_oracle_uniform_total": n_oracle_unif,
            "n_oracle_is_total": n_oracle_is,
            "fp_total": fp_total,
            "fn_total": fn_total,
            "n_oracle_uniform_border": n_border_unif,
            "n_oracle_is_border": n_border_is,
            "fp_borderline": fp_border,
            "fn_borderline": fn_border,
            "accuracy_total": accuracy_total,
            "fp_rate_total": fp_rate_total,
            "fn_rate_total": fn_rate_total,
            "accuracy_border": accuracy_border,
            "fp_rate_border": fp_rate_border,
            "fn_rate_border": fn_rate_border,
            "mean_rel_error_total": mean_err,
        })

        # Legacy row (backward-compatible threshold_sweep.csv format).
        legacy_rows.append({
            "tau": float(tau),
            "fp_rate": fp_rate_total,
            "fn_rate": fn_rate_total,
            "mean_rel_error": mean_err,
            "n_oracle_uniform": n_oracle_unif,
            "n_oracle_is": n_oracle_is,
        })

        print(f"  tau={tau:.2f}  FP_tot={fp_rate_total:.3f}  FN_tot={fn_rate_total:.3f}  "
              f"FP_bord={fp_rate_border:.3f}  FN_bord={fn_rate_border:.3f}  "
              f"err={mean_err:.4f}")

    # Save per-instance CSV.
    pi_path = os.path.join(out_dir, "threshold_sweep_per_instance.csv")
    with open(pi_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_instance_rows[0].keys()))
        w.writeheader()
        w.writerows(per_instance_rows)
    print(f"\nSaved {pi_path} ({len(per_instance_rows)} rows)")

    if args.write_per_trial:
        trial_path = os.path.join(out_dir, "threshold_sweep_trials.csv")
        with open(trial_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_trial_rows[0].keys()))
            w.writeheader()
            w.writerows(per_trial_rows)
        print(f"Saved {trial_path} ({len(per_trial_rows)} rows)")

    # Save aggregated CSV (Bernstein oracle).
    agg_path = os.path.join(out_dir, "threshold_sweep_aggregated_bernstein.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Saved {agg_path}")

    # Also write as the canonical aggregated file.
    agg_path2 = os.path.join(out_dir, "threshold_sweep_aggregated.csv")
    with open(agg_path2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Saved {agg_path2}")

    # Save legacy CSV (deprecated).
    csv_path = os.path.join(out_dir, "threshold_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(legacy_rows[0].keys()))
        w.writeheader()
        w.writerows(legacy_rows)
    print(f"Saved {csv_path} (deprecated legacy format)")

    # Report default tau = sqrt(log n) stats (use median n for a representative).
    default_n = N_VALUES[len(N_VALUES) // 2]
    default_tau = math.sqrt(math.log(default_n))
    closest = min(agg_rows, key=lambda r: abs(r["tau"] - default_tau))
    print(f"\nAt default tau~sqrt(log {default_n})={default_tau:.3f} "
          f"(closest grid tau={closest['tau']:.3f}):")
    print(f"  fp_rate_total={closest['fp_rate_total']:.3f}  "
          f"fn_rate_total={closest['fn_rate_total']:.3f}")
    print(f"  fp_rate_border={closest['fp_rate_border']:.3f}  "
          f"fn_rate_border={closest['fn_rate_border']:.3f}")

    # Plot 2-panel figure.
    taus = [r["tau"] for r in agg_rows]
    fp_tot = [r["fp_rate_total"] for r in agg_rows]
    fn_tot = [r["fn_rate_total"] for r in agg_rows]
    fp_brd = [r["fp_rate_border"] for r in agg_rows]
    fn_brd = [r["fn_rate_border"] for r in agg_rows]
    err = [r["mean_rel_error_total"] for r in agg_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(taus, fp_tot, "o-", color="tab:red", label="FP total (IS | oracle=uniform)")
    ax1.plot(taus, fn_tot, "s-", color="tab:blue", label="FN total (uniform | oracle=IS)")
    ax1.plot(taus, fp_brd, "o--", color="tab:orange", label="FP border")
    ax1.plot(taus, fn_brd, "s--", color="tab:cyan", label="FN border")
    ax1.axvline(closest["tau"], ls="--", color="k", alpha=0.5,
                label=f"default tau~{closest['tau']:.2f}")
    ax1.set_xlabel("Threshold tau")
    ax1.set_ylabel("Rate")
    ax1.set_title("FP / FN vs tau (total and borderline)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)

    ax2.plot(taus, err, "o-", color="tab:purple", label="Mean rel. error (total)")
    ax2.set_xlabel("Threshold tau")
    ax2.set_ylabel("Mean relative error")
    ax2.set_title("Mean rel. error vs tau")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig_threshold.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {pdf_path}")

    print(f"\nTotal time: {time.perf_counter()-t_start:.1f}s")
    return agg_rows, closest


if __name__ == "__main__":
    main()
