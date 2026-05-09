"""Phase 4.2: Pilot size sensitivity and decision accuracy.

Same instance grid as exp_cv_calibration.py.

For each instance:
  - Compute cost-model oracle decision (ALGORITHM_SPEC §2.2).
  - For each (s, tau) in {10,30,50,100} x {1.5, sqrt(log n), 3.0}:
      run adaptive(s, tau); record whether branch_taken == oracle_winner.

Output: results/pilot_size_sweep.csv, results/pilot_size_per_instance.csv,
        figures/fig_pilot_size.pdf.

Usage:
  python exp_pilot_size.py [--seed 0] [--out results/]
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
PILOT_SIZES = [10, 30, 50, 100]
D = 3
METRIC = "l2"
EPS = 0.1
DELTA = 0.01
C_IS = 4.0
C_NN = 1.0
NUM_LEVELS = 3


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--out", type=str, default=RESULTS_DIR, help="Output directory")
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    per_instance_rows = []
    t_start = time.perf_counter()

    instance_id = 0
    total = len(N_VALUES) * len(OUTLIER_FRACTIONS) * len(OUTLIER_MAGS) * N_TRIALS
    for n in N_VALUES:
        tau_log = math.sqrt(math.log(n))
        TAUS = [("1.5", 1.5), ("sqrt(log n)", tau_log), ("3.0", 3.0)]
        borderline_lo = 0.7 * math.log(n)
        borderline_hi = 1.3 * math.log(n)

        for outlier_fraction in OUTLIER_FRACTIONS:
            for outlier_mag in OUTLIER_MAGS:
                for trial in range(N_TRIALS):
                    instance_id += 1
                    seed = make_seed(args.seed, n, outlier_fraction, outlier_mag, trial)

                    A, B = make_instance(n, D, outlier_fraction, outlier_mag, seed)
                    ch_true, gamma = brute_force_chamfer(A, B, metric=METRIC)
                    true_cv = compute_cv(gamma)
                    true_M = compute_true_M(gamma)

                    oracle = cost_oracle(true_cv, true_M, n)
                    in_borderline = int(borderline_lo <= true_cv ** 2 <= borderline_hi)

                    for s in PILOT_SIZES:
                        for tau_name, tau_val in TAUS:
                            est, diag = adaptive_chamfer(
                                A, B, eps=EPS, pilot_size=s, tau=tau_val,
                                delta=DELTA,
                                return_diagnostics=True, metric=METRIC,
                                num_levels=NUM_LEVELS,
                                rng=make_seed(args.seed, n, outlier_fraction, outlier_mag, trial, s),
                            )
                            rerr = relative_error(est, ch_true)
                            branch = diag["branch_taken"]
                            # Classify uniform_exact as uniform for match accounting.
                            branch_for_match = "uniform" if branch == "uniform_exact" else branch

                            rows.append({
                                "n": n,
                                "d": D,
                                "outlier_fraction": outlier_fraction,
                                "outlier_mag": outlier_mag,
                                "trial": trial,
                                "true_cv": true_cv,
                                "true_M": true_M,
                                "oracle_decision": oracle,
                                "in_borderline": in_borderline,
                                "s": s,
                                "tau_name": tau_name,
                                "tau": tau_val,
                                "branch_taken": branch,
                                "pilot_cv": diag["pilot_cv"],
                                "M_hat": diag.get("M_hat", float("nan")),
                                "T_used": diag["T_used"],
                                "rel_error": rerr,
                                "match": int(branch_for_match == oracle),
                            })

                            per_instance_rows.append({
                                "n": n,
                                "true_cv": true_cv,
                                "true_M": true_M,
                                "M_hat": diag.get("M_hat", float("nan")),
                                "tau": tau_val,
                                "tau_name": tau_name,
                                "s": s,
                                "branch_taken": branch,
                                "oracle_decision": oracle,
                                "rel_error": rerr,
                                "in_borderline": in_borderline,
                            })

                    if instance_id % 10 == 0 or instance_id == total:
                        el = time.perf_counter() - t_start
                        print(f"  [{instance_id}/{total}] n={n} of={outlier_fraction} "
                              f"om={outlier_mag} CV={true_cv:.2f} oracle={oracle} ({el:.1f}s)",
                              flush=True)

    # Save CSVs
    csv_path = os.path.join(out_dir, "pilot_size_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path} ({len(rows)} rows)")

    bern_path = os.path.join(out_dir, "pilot_size_sweep_bernstein.csv")
    with open(bern_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {bern_path} ({len(rows)} rows)")

    pi_path = os.path.join(out_dir, "pilot_size_per_instance.csv")
    with open(pi_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_instance_rows[0].keys()))
        w.writeheader()
        w.writerows(per_instance_rows)
    print(f"Saved {pi_path} ({len(per_instance_rows)} rows)")

    summary_rows = []
    for s in PILOT_SIZES:
        for tau_name in sorted({r["tau_name"] for r in rows}):
            sel = [r for r in rows if r["s"] == s and r["tau_name"] == tau_name]
            if not sel:
                continue
            n_oracle_uniform = sum(1 for r in sel if r["oracle_decision"] == "uniform")
            n_oracle_is = sum(1 for r in sel if r["oracle_decision"] == "is")
            fp = sum(
                1 for r in sel
                if r["oracle_decision"] == "uniform"
                and ("uniform" if r["branch_taken"] == "uniform_exact" else r["branch_taken"]) == "is"
            )
            fn = sum(
                1 for r in sel
                if r["oracle_decision"] == "is"
                and ("uniform" if r["branch_taken"] == "uniform_exact" else r["branch_taken"]) == "uniform"
            )
            border = [r for r in sel if r["in_borderline"]]
            n_border_uniform = sum(1 for r in border if r["oracle_decision"] == "uniform")
            n_border_is = sum(1 for r in border if r["oracle_decision"] == "is")
            fp_border = sum(
                1 for r in border
                if r["oracle_decision"] == "uniform"
                and ("uniform" if r["branch_taken"] == "uniform_exact" else r["branch_taken"]) == "is"
            )
            fn_border = sum(
                1 for r in border
                if r["oracle_decision"] == "is"
                and ("uniform" if r["branch_taken"] == "uniform_exact" else r["branch_taken"]) == "uniform"
            )
            summary_rows.append({
                "s": s,
                "tau_name": tau_name,
                "tau": sel[0]["tau"],
                "accuracy": sum(r["match"] for r in sel) / len(sel),
                "fp_rate": fp / max(n_oracle_uniform, 1),
                "fn_rate": fn / max(n_oracle_is, 1),
                "n_trials": len(sel),
                "accuracy_border": (
                    sum(
                        1 for r in border
                        if ("uniform" if r["branch_taken"] == "uniform_exact" else r["branch_taken"])
                        == r["oracle_decision"]
                    ) / len(border)
                    if border else float("nan")
                ),
                "fp_rate_border": fp_border / max(n_border_uniform, 1),
                "fn_rate_border": fn_border / max(n_border_is, 1),
                "n_borderline": len(border),
            })

    summary_path = os.path.join(out_dir, "pilot_size_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved {summary_path} ({len(summary_rows)} rows)")

    # Decision accuracy: per (s, tau)
    print("\nDecision accuracy (fraction matching oracle):")
    print(f"{'tau':>14} | " + " | ".join(f"s={s}" for s in PILOT_SIZES))
    print("-" * 60)

    tau_names = []
    seen = set()
    for r in rows:
        if r["tau_name"] not in seen:
            seen.add(r["tau_name"])
            tau_names.append(r["tau_name"])

    accuracy = {}
    for tname in tau_names:
        line = f"{tname:>14} | "
        cells = []
        for s in PILOT_SIZES:
            sel = [r for r in rows if r["tau_name"] == tname and r["s"] == s]
            acc = sum(r["match"] for r in sel) / len(sel) if sel else 0.0
            accuracy[(tname, s)] = acc
            cells.append(f"{acc*100:.1f}%")
        print(line + " | ".join(cells))

    # Plot decision accuracy vs s, one curve per tau
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for tname in tau_names:
        ys = [accuracy[(tname, s)] for s in PILOT_SIZES]
        ax.plot(PILOT_SIZES, ys, "o-", label=f"tau = {tname}")
    ax.set_xlabel("Pilot size s")
    ax.set_ylabel("Decision accuracy (matches oracle)")
    ax.set_title("Pilot Size vs Decision Accuracy (eps = 0.1)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig_pilot_size.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {pdf_path}")

    el = time.perf_counter() - t_start
    print(f"\nTotal time: {el:.1f}s")
    return accuracy


if __name__ == "__main__":
    main()
