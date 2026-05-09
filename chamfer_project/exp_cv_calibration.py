"""Phase 4.1: Pilot CV vs True CV calibration.

Sweep outlier_fraction in {0, 0.01, 0.02, 0.05, 0.1, 0.2},
outlier_mag in {2, 5, 10}, n in {1000, 2000, 5000}, 3 trials each
=> 6*3*3*3 = 162 instances.

For each instance:
  - true_CV = compute_cv on exact gamma (brute force).
  - pilot_CV for s in {10, 30, 50, 100}.

Output: results/cv_calibration.csv, figures/fig_cv_calibration.pdf.
Headline metric: fraction of instances with |pilot - true|/true < 0.30.

Usage:
  python exp_cv_calibration.py [--seed 0] [--out results/]
"""
import sys, os, math, csv, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chamfer_core import brute_force_chamfer
from utils import compute_cv, compute_true_M

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


def make_instance(n, d, outlier_fraction, outlier_mag, seed):
    """Gaussian A,B with a fraction of A points shifted by outlier_mag in random directions."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d))
    B = rng.standard_normal((n, d))
    n_out = int(round(outlier_fraction * n))
    if n_out > 0:
        dirs = rng.standard_normal((n_out, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        A[:n_out] = dirs * outlier_mag
    return A, B


def make_seed(base, *parts):
    """Deterministic seed from base seed and integer/float parts."""
    seed = int(base)
    for p in parts:
        if isinstance(p, float):
            p = int(p * 1000)
        seed = (seed * 1_000_003 + int(p)) % (2**31 - 1)
    return seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--out", type=str, default=RESULTS_DIR, help="Output directory")
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    t_start = time.perf_counter()

    instance_id = 0
    total_instances = len(OUTLIER_FRACTIONS) * len(OUTLIER_MAGS) * len(N_VALUES) * N_TRIALS
    for n in N_VALUES:
        for outlier_fraction in OUTLIER_FRACTIONS:
            for outlier_mag in OUTLIER_MAGS:
                for trial in range(N_TRIALS):
                    instance_id += 1
                    seed_int = make_seed(args.seed, n, outlier_fraction, outlier_mag, trial)

                    A, B = make_instance(n, D, outlier_fraction, outlier_mag, seed_int)
                    _, gamma = brute_force_chamfer(A, B, metric=METRIC)
                    true_cv = compute_cv(gamma)
                    true_M = compute_true_M(gamma)

                    rng = np.random.default_rng(make_seed(args.seed, n, outlier_fraction, outlier_mag, trial, 7))
                    for s in PILOT_SIZES:
                        idx = rng.choice(n, size=s, replace=False)
                        pilot = gamma[idx]
                        pilot_mean = float(pilot.mean())
                        pilot_cv = float(pilot.std(ddof=1) / pilot_mean) if pilot_mean > 0 else 0.0
                        pilot_M = float(pilot.max() / pilot_mean) if pilot_mean > 0 else 1.0
                        cv_rel_error = (
                            abs(pilot_cv - true_cv) / true_cv if true_cv > 0 else abs(pilot_cv)
                        )
                        M_rel_error = (
                            abs(pilot_M - true_M) / true_M if true_M > 0 else abs(pilot_M)
                        )
                        rows.append({
                            "n": n,
                            "d": D,
                            "outlier_fraction": outlier_fraction,
                            "outlier_mag": outlier_mag,
                            "trial": trial,
                            "s": s,
                            "true_cv": true_cv,
                            "true_M": true_M,
                            "cv_hat": pilot_cv,
                            "pilot_cv": pilot_cv,
                            "M_hat": pilot_M,
                            "cv_rel_error": cv_rel_error,
                            "M_rel_error": M_rel_error,
                        })

                    if instance_id % 10 == 0 or instance_id == total_instances:
                        elapsed = time.perf_counter() - t_start
                        print(f"  [{instance_id}/{total_instances}] n={n} of={outlier_fraction} om={outlier_mag} "
                              f"trial={trial} true_CV={true_cv:.3f} ({elapsed:.1f}s)", flush=True)

    # ---------- Save CSV ----------
    csv_path = os.path.join(out_dir, "cv_calibration.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path} ({len(rows)} rows)")

    # ---------- Headline metric ----------
    print("\nHeadline: fraction of instances with |pilot_CV - true_CV| / true_CV < 0.30")
    print(f"{'s':>5} | within-30%")
    print("-" * 22)
    headline = {}
    for s in PILOT_SIZES:
        rs = [r for r in rows if r["s"] == s and r["true_cv"] > 0]
        within = sum(1 for r in rs if abs(r["pilot_cv"] - r["true_cv"]) / r["true_cv"] < 0.30)
        frac = within / len(rs) if rs else 0.0
        headline[s] = frac
        print(f"{s:>5} | {frac*100:.1f}%")

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    cmap = plt.cm.viridis
    of_levels = sorted(set(r["outlier_fraction"] for r in rows))
    color_for = {of: cmap(i / max(1, len(of_levels) - 1)) for i, of in enumerate(of_levels)}

    for ax, s in zip(axes, PILOT_SIZES):
        rs = [r for r in rows if r["s"] == s]
        for of in of_levels:
            sub = [r for r in rs if r["outlier_fraction"] == of]
            xs = [r["true_cv"] for r in sub]
            ys = [r["pilot_cv"] for r in sub]
            ax.scatter(xs, ys, s=18, alpha=0.65, color=color_for[of], label=f"of={of}")
        all_cv = [r["true_cv"] for r in rs] + [r["pilot_cv"] for r in rs]
        if all_cv:
            mx = max(all_cv) * 1.05
            ax.plot([0, mx], [0, mx], "k--", lw=1, alpha=0.6, label="y=x")
        ax.set_title(f"s = {s}  (within-30%: {headline[s]*100:.0f}%)")
        ax.set_xlabel("true CV")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pilot CV_hat")
    axes[-1].legend(loc="upper left", fontsize=7, ncol=1)
    fig.suptitle("Pilot CV vs True CV calibration")
    fig.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig_cv_calibration.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {pdf_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    return headline


if __name__ == "__main__":
    main()
