"""Cost-by-n oracle sanity table for Adaptive-Chamfer.

This is a deterministic cost-model analysis; it does not run estimators.
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import bernstein_oracle

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EPS = 0.1
DELTA = 0.01
C_IS = 4.0
C_NN = 1.0
N_VALUES = [250, 500, 1000, 2000, 5000, 10000, 20000]
CV_VALUES = [0.5, 1.0, 2.0, 3.0]
M_MULTIPLIER = 3.0


def main():
    rows = []
    for n in N_VALUES:
        for cv in CV_VALUES:
            true_M = max(1.0, M_MULTIPLIER * cv)
            oracle = bernstein_oracle(
                cv, true_M, n, eps=EPS, delta=DELTA, C_IS=C_IS, C_NN=C_NN
            )
            rows.append({
                "n": n,
                "true_cv": cv,
                "true_M": true_M,
                "cost_uniform_chebyshev": oracle["cost_uniform_chebyshev"],
                "cost_uniform_bernstein_raw": oracle["cost_uniform_bernstein_raw"],
                "cost_uniform_bernstein": oracle["cost_uniform_bernstein"],
                "cost_is": oracle["cost_is"],
                "oracle_branch": oracle["oracle_branch"],
                "break_even_cv": math.sqrt(
                    max(0.0, oracle["cost_is"] * EPS ** 2 / (2.0 * math.log(2.0 / DELTA)))
                ),
            })

    path = os.path.join(RESULTS_DIR, "cost_by_n.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {path} ({len(rows)} rows)")
    for row in rows:
        if row["true_cv"] == 2.0:
            print(
                f"n={row['n']:>5} cv={row['true_cv']:.1f} "
                f"cheb={row['cost_uniform_chebyshev']:.1f} "
                f"bern={row['cost_uniform_bernstein']:.1f} "
                f"is={row['cost_is']:.1f} oracle={row['oracle_branch']}"
            )
    return rows


if __name__ == "__main__":
    main()
