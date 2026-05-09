"""Reproduce Section 3 experiments from Bakshi et al. 2023.

Figures produced:
  fig3_sample_vs_error.png  — Fig 3 (a/b/c): sample complexity vs relative error
  fig4_runtime.png          — Fig 4: runtime bar chart (ShapeNet-like, Text)
  fig6b_crude_quality.png   — Fig 6(b): D_a/gamma_a ratio vs # LSH levels

Key improvements over Phase 1:
  - Calibrated ShapeNet CV (3% far points at distance 3-8, targets CV 2-4)
  - Real text data (20 Newsgroups TF-IDF) for Fig 3(b)
  - T calibration: sample counts chosen to achieve equal error target

Dataset correspondence (Table 1, Bakshi et al. 2023):
  (a) ShapeNet      |A|=8k, d=3,   l1, symmetric  -> calibrated synthetic 3D clouds
  (b) Text Embeds   |A|=2.5k,d=300,l1, symmetric  -> 20 Newsgroups TF-IDF (real)
  (c) Gaussian Pts  n=1k,   d=20,  l1, one-sided  -> Gaussian + outlier (exact)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chamfer_core import brute_force_chamfer, uniform_sampling_chamfer, importance_sampling_chamfer
from crude_nn import crude_nn
from data_gen import (
    make_shapenet_pair, make_real_text_pair, make_outlier_pair, make_text_pair,
)
from utils import relative_error, compute_cv

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

N_TRIALS = 25
NUM_LEVELS = 3
COLORS = {"uniform": "tab:orange", "is": "tab:blue", "calibrated": "tab:green"}

# Error target for calibration
TARGET_ERROR = 0.10   # 10% relative error


# ---------------------------------------------------------------------------
# T calibration: find min T achieving target error (from pre-computed curves)
# ---------------------------------------------------------------------------

def calibrate_T(make_fn, make_kwargs, metric, symmetric, n_trials,
                T_max_unif=500, T_max_is=100, num_levels=NUM_LEVELS):
    """Find T needed for Uniform and IS to reach TARGET_ERROR.

    Runs a dense sweep, returns (T_unif, T_is) pair.
    For efficiency, samples T on a log-spaced grid.
    """
    T_grid_unif = np.unique(np.geomspace(5, T_max_unif, 15).astype(int))
    T_grid_is = np.unique(np.geomspace(2, T_max_is, 12).astype(int))

    def _sweep(T_grid, estimator_fn):
        for T in T_grid:
            errs = []
            for trial in range(n_trials):
                A, B = make_fn(**make_kwargs, rng=trial)
                D_a = crude_nn(A, B, metric=metric, num_levels=num_levels, rng=trial * 2)
                D_b = None
                if symmetric:
                    D_b = crude_nn(B, A, metric=metric, num_levels=num_levels, rng=trial * 2 + 1)

                ch_true, _ = brute_force_chamfer(A, B, metric=metric)
                if symmetric:
                    ch_ba, _ = brute_force_chamfer(B, A, metric=metric)
                    ch_true += ch_ba

                seed = trial * 10000 + T
                if estimator_fn == "uniform":
                    est, _, _, _ = uniform_sampling_chamfer(A, B, T=T, metric=metric, rng=seed)
                    if symmetric:
                        e2, _, _, _ = uniform_sampling_chamfer(B, A, T=T, metric=metric, rng=seed + 1)
                        est += e2
                else:
                    est, _, _ = importance_sampling_chamfer(A, B, D_a, T=T, metric=metric, rng=seed)
                    if symmetric:
                        e2, _, _ = importance_sampling_chamfer(B, A, D_b, T=T, metric=metric, rng=seed + 1)
                        est += e2
                errs.append(relative_error(est, ch_true))
            mean_err = np.mean(errs)
            if mean_err <= TARGET_ERROR:
                return T
        return T_grid[-1]  # Not reached within range

    T_u = _sweep(T_grid_unif, "uniform")
    T_i = _sweep(T_grid_is, "is")
    return T_u, T_i


# ---------------------------------------------------------------------------
# Core runner: sample complexity experiment
# ---------------------------------------------------------------------------

def run_experiment(make_fn, make_kwargs, T_values, metric, symmetric,
                   n_trials, num_levels=NUM_LEVELS, fixed_pair=False):
    """Run sample-complexity experiment. Returns (unif_mat, is_mat)."""
    unif_mat = np.zeros((len(T_values), n_trials))
    is_mat   = np.zeros((len(T_values), n_trials))

    if fixed_pair:
        A, B = make_fn(**make_kwargs, rng=0)
        D_a = crude_nn(A, B, metric=metric, num_levels=num_levels, rng=42)
        D_b = crude_nn(B, A, metric=metric, num_levels=num_levels, rng=43) if symmetric else None
        ch_true, _ = brute_force_chamfer(A, B, metric=metric)
        if symmetric:
            ch_ba, _ = brute_force_chamfer(B, A, metric=metric)
            ch_true += ch_ba

    for trial in range(n_trials):
        if not fixed_pair:
            A, B = make_fn(**make_kwargs, rng=trial)
            D_a = crude_nn(A, B, metric=metric, num_levels=num_levels, rng=trial * 2)
            D_b = crude_nn(B, A, metric=metric, num_levels=num_levels, rng=trial * 2 + 1) if symmetric else None
            ch_true, _ = brute_force_chamfer(A, B, metric=metric)
            if symmetric:
                ch_ba, _ = brute_force_chamfer(B, A, metric=metric)
                ch_true += ch_ba

        for ti, T in enumerate(T_values):
            seed = trial * 10000 + T

            eu, _, _, _ = uniform_sampling_chamfer(A, B, T=T, metric=metric, rng=seed)
            if symmetric:
                eu_ba, _, _, _ = uniform_sampling_chamfer(B, A, T=T, metric=metric, rng=seed + 1)
                eu += eu_ba
            unif_mat[ti, trial] = relative_error(eu, ch_true)

            ei, _, _ = importance_sampling_chamfer(A, B, D_a, T=T, metric=metric, rng=seed)
            if symmetric:
                ei_ba, _, _ = importance_sampling_chamfer(B, A, D_b, T=T, metric=metric, rng=seed + 1)
                ei += ei_ba
            is_mat[ti, trial] = relative_error(ei, ch_true)

    return unif_mat, is_mat


def _plot_panel(ax, T_values, unif_mat, is_mat, title):
    for mat, label, color in [
        (unif_mat, "Uniform Sampling",    COLORS["uniform"]),
        (is_mat,   "Importance Sampling", COLORS["is"]),
    ]:
        mean = mat.mean(axis=1)
        std  = mat.std(axis=1)
        ax.plot(T_values, mean, linestyle="--", marker="o", markersize=3,
                label=label, color=color)
        ax.fill_between(T_values, np.clip(mean - std, 0, None), mean + std,
                        alpha=0.25, color=color)
    ax.axhline(TARGET_ERROR, color="gray", linestyle=":", alpha=0.5, label=f"{TARGET_ERROR:.0%} error")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("# Samples")
    ax.set_ylabel("Relative Error")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _samples_to_threshold(mat, T_values, threshold=TARGET_ERROR):
    for i, t in enumerate(T_values):
        if mat.mean(axis=1)[i] <= threshold:
            return t
    return ">{}".format(T_values[-1])


# ---------------------------------------------------------------------------
# Fig 3: sample complexity vs relative error
# ---------------------------------------------------------------------------

def fig3_sample_complexity(use_real_text=True):
    print("\n[Fig 3] Sample complexity vs relative error ...")
    print(f"  Target error: {TARGET_ERROR:.0%}, real text data: {use_real_text}")
    T_s = list(range(1, 51, 2))   # ShapeNet: 0-50
    T_t = list(range(1, 51, 2))   # Text: 0-50
    T_o = list(range(1, 101, 4))  # Outlier: 0-100

    # (a) Calibrated ShapeNet
    print("  (a) ShapeNet (calibrated CV, n=8k, d=3, l1, symmetric) ...", flush=True)
    unif_s, is_s = run_experiment(
        make_shapenet_pair, {"high_cv": True}, T_s,
        metric="l1", symmetric=True, n_trials=N_TRIALS, fixed_pair=False)

    # Compute and report true CV
    A_demo, B_demo = make_shapenet_pair(high_cv=True, rng=0)
    _, gamma_demo = brute_force_chamfer(A_demo, B_demo, metric="l1")
    cv_shape = compute_cv(gamma_demo)
    print(f"    True CV(ShapeNet) = {cv_shape:.3f}", flush=True)

    # (b) Real text data
    if use_real_text:
        print("  (b) Real Text (20 Newsgroups TF-IDF, d=300, l1, symmetric) ...", flush=True)
        text_make_fn = make_real_text_pair
    else:
        print("  (b) Synthetic Text (d=300, l1, symmetric) ...", flush=True)
        text_make_fn = make_text_pair

    unif_t, is_t = run_experiment(
        text_make_fn, {"high_cv": True}, T_t,
        metric="l1", symmetric=True, n_trials=N_TRIALS, fixed_pair=False)

    A_demo_t, B_demo_t = text_make_fn(high_cv=True, rng=0)
    _, gamma_demo_t = brute_force_chamfer(A_demo_t, B_demo_t, metric="l1")
    cv_text = compute_cv(gamma_demo_t)
    print(f"    True CV(Text) = {cv_text:.3f}", flush=True)

    # (c) Outlier (same as before)
    print("  (c) Outlier (n=1k, d=20, l1, one-sided, fixed pair) ...", flush=True)
    unif_o, is_o = run_experiment(
        make_outlier_pair, {}, T_o,
        metric="l1", symmetric=False, n_trials=N_TRIALS, fixed_pair=True)

    # Print calibration summary
    print("\n  T calibration (to {}% error):".format(int(TARGET_ERROR * 100)))
    for name, u, s, T_vals in [
        ("ShapeNet (calibrated)", unif_s, is_s, T_s),
        ("Text (real)" if use_real_text else "Text (synthetic)", unif_t, is_t, T_t),
        ("Outlier",               unif_o, is_o, T_o),
    ]:
        u_r = _samples_to_threshold(u, T_vals)
        s_r = _samples_to_threshold(s, T_vals)
        ratio = float(u_r) / float(s_r) if isinstance(u_r, int) and isinstance(s_r, int) else float("nan")
        print(f"  {name:25s}: Uniform T={str(u_r):>5s}, IS T={str(s_r):>5s}, IS advantage = {ratio:.1f}x")

    # Plot
    text_label = "Real Text (20 Newsgroups)" if use_real_text else "Synthetic Text"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _plot_panel(axes[0], T_s, unif_s, is_s,
                f"ShapeNet (calibrated CV={cv_shape:.1f})\n(synthetic 3D, n=8k, l1, symmetric)")
    _plot_panel(axes[1], T_t, unif_t, is_t,
                f"Text Embeddings (CV={cv_text:.1f})\n({text_label}, d=300, l1, symmetric)")
    _plot_panel(axes[2], T_o, unif_o, is_o,
                "Gaussian Points (Outlier)\n(n=1k, d=20, l1, one-sided)")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig3_sample_vs_error.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 4: runtime comparison (T-calibrated)
# ---------------------------------------------------------------------------

def _measure_runtime(make_fn, make_kw, metric, symmetric, T_unif, T_is, n_pairs):
    """Benchmark: measure wall-clock time for BF, Uniform, and IS.

    Also returns the achieved relative error for Uniform and IS so we can
    verify that both methods hit the same accuracy target.

    Returns dict with mean/std times (ms) and achieved errors.
    """
    bf_t, u_t, is_t = [], [], []
    u_errs, is_errs = [], []

    for trial in range(n_pairs):
        A, B = make_fn(**make_kw, rng=trial)

        # Brute force (reference truth)
        t0 = time.perf_counter()
        ch_bf, _ = brute_force_chamfer(A, B, metric=metric)
        if symmetric:
            ch_ba, _ = brute_force_chamfer(B, A, metric=metric)
            ch_bf += ch_ba
        bf_t.append(time.perf_counter() - t0)
        ch_true = ch_bf

        # Uniform sampling
        t0 = time.perf_counter()
        eu, _, _, _ = uniform_sampling_chamfer(A, B, T=T_unif, metric=metric, rng=0)
        if symmetric:
            eu_ba, _, _, _ = uniform_sampling_chamfer(B, A, T=T_unif, metric=metric, rng=1)
            eu += eu_ba
        u_t.append(time.perf_counter() - t0)
        u_errs.append(relative_error(eu, ch_true))

        # Importance sampling (includes CrudeNN build)
        t0 = time.perf_counter()
        D_a = crude_nn(A, B, metric=metric, num_levels=NUM_LEVELS, rng=0)
        ei, _, _ = importance_sampling_chamfer(A, B, D_a, T=T_is, metric=metric, rng=0)
        if symmetric:
            D_b = crude_nn(B, A, metric=metric, num_levels=NUM_LEVELS, rng=1)
            ei_ba, _, _ = importance_sampling_chamfer(B, A, D_b, T=T_is, metric=metric, rng=1)
            ei += ei_ba
        is_t.append(time.perf_counter() - t0)
        is_errs.append(relative_error(ei, ch_true))

    return {
        "bf":    (np.mean(bf_t) * 1000, np.std(bf_t) * 1000),
        "unif":  (np.mean(u_t) * 1000, np.std(u_t) * 1000),
        "is":    (np.mean(is_t) * 1000, np.std(is_t) * 1000),
        "u_err": (np.mean(u_errs), np.std(u_errs)),
        "i_err": (np.mean(is_errs), np.std(is_errs)),
    }


def fig4_runtime(use_real_text=True):
    """Fig 4: Runtime comparison at two error targets.

    Panel layout (2 datasets x 2 error targets):
      Row 1 (ShapeNet):     10% error | Paper T-values
      Row 2 (Text):         10% error | Paper T-values

    The paper uses *fixed* T values chosen to achieve ~3-5% error:
      ShapeNet:  T_unif=500, T_is=100
      Text:      T_unif=450, T_is=20

    We also calibrate to 5% error using the 1/sqrt(T) scaling from our
    10%-error calibration, so both methods are compared at equal accuracy.
    """
    print("\n[Fig 4] Runtime comparison ...")
    N_PAIRS = 20

    if use_real_text:
        text_make_fn = make_real_text_pair
        text_label = "Real Text Embeddings"
    else:
        text_make_fn = make_text_pair
        text_label = "Synthetic Text"

    # ---- Step 1: calibrate T to 10% error (cheap sweep) ----
    print("  Calibrating T to 10% error ...", flush=True)
    T_u_s10, T_i_s10 = calibrate_T(
        make_shapenet_pair, {"high_cv": True},
        metric="l1", symmetric=True, n_trials=8, T_max_unif=500, T_max_is=100)
    T_u_t10, T_i_t10 = calibrate_T(
        text_make_fn, {"high_cv": True},
        metric="l1", symmetric=True, n_trials=8, T_max_unif=500, T_max_is=100)

    # ---- Step 2: extrapolate to 5% error (error ~ 1/sqrt(T)) ----
    # T(5%) = T(10%) * (0.10/0.05)^2 = 4 * T(10%)
    def _extrapolate(T_10):
        return max(int(T_10 * 4.0), 1)

    T_u_s5 = min(_extrapolate(T_u_s10), 2000)
    T_i_s5 = min(_extrapolate(T_i_s10), 400)
    T_u_t5 = min(_extrapolate(T_u_t10), 2000)
    T_i_t5 = min(_extrapolate(T_i_t10), 400)

    print(f"    ShapeNet: T_10% unif={T_u_s10} is={T_i_s10}  |  T_5% unif={T_u_s5} is={T_i_s5}")
    print(f"    Text:     T_10% unif={T_u_t10} is={T_i_t10}  |  T_5% unif={T_u_t5} is={T_i_t5}")

    # ---- Step 3: paper-style fixed T (from Table 1 / Fig 4 of Bakshi et al.) ----
    paper_configs = [
        {"name": "ShapeNet",  "fn": make_shapenet_pair, "kw": {"high_cv": True},
         "T_u": 500, "T_i": 100, "metric": "l1", "sym": True,
         "paper_label": "ShapeNet\n(paper: T_unif=500, T_is=100)"},
        {"name": text_label, "fn": text_make_fn, "kw": {"high_cv": True},
         "T_u": 450, "T_i": 20, "metric": "l1", "sym": True,
         "paper_label": f"{text_label}\n(paper: T_unif=450, T_is=20)"},
    ]

    # ---- Step 4: benchmark all configs ----
    all_results = {}  # key -> result dict

    for key, fn, kw, Tu, Ti, met, sym, label in [
        ("ShapeNet 5%", make_shapenet_pair, {"high_cv": True}, T_u_s5, T_i_s5, "l1", True,
         f"ShapeNet Dataset\n(equal-accuracy ~5% err)"),
        ("Text 5%", text_make_fn, {"high_cv": True}, T_u_t5, T_i_t5, "l1", True,
         f"{text_label}\n(equal-accuracy ~5% err)"),
    ]:
        print(f"  Benchmarking {key} (T_unif={Tu}, T_is={Ti}) ...", flush=True)
        all_results[key] = _measure_runtime(fn, kw, met, sym, Tu, Ti, N_PAIRS)
        all_results[key]["label"] = label
        all_results[key]["T_unif"] = Tu
        all_results[key]["T_is"] = Ti

    paper_keys = []  # track actual keys used for paper configs
    for pc in paper_configs:
        key = pc["name"] + " paper"
        paper_keys.append(key)
        print(f"  Benchmarking {key} (paper T: unif={pc['T_u']}, is={pc['T_i']}) ...", flush=True)
        all_results[key] = _measure_runtime(pc["fn"], pc["kw"], pc["metric"], pc["sym"],
                                            pc["T_u"], pc["T_i"], N_PAIRS)
        all_results[key]["label"] = pc["paper_label"]
        all_results[key]["T_unif"] = pc["T_u"]
        all_results[key]["T_is"] = pc["T_i"]

    # ---- Step 5: plot ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    plot_keys = ["ShapeNet 5%", paper_keys[0], "Text 5%", paper_keys[1]]
    plot_labels = ["Equal ~5% Error", "Paper T Values", "Equal ~5% Error", "Paper T Values"]
    for ax, key, label_short in zip(axes.flatten(), plot_keys, plot_labels):
        r = all_results[key]
        Tu, Ti = r["T_unif"], r["T_is"]
        u_err_pct = r["u_err"][0] * 100
        i_err_pct = r["i_err"][0] * 100

        lbls = [
            "Brute Force\n(cdist)",
            f"Uniform Sampling\nT={Tu}  (err {u_err_pct:.1f}%)",
            f"Importance Sampling\nT={Ti}  (err {i_err_pct:.1f}%)",
        ]
        vals = [r["bf"][0], r["unif"][0], r["is"][0]]
        errs = [r["bf"][1], r["unif"][1], r["is"][1]]
        colors = ["tab:pink", COLORS["uniform"], COLORS["is"]]

        bars = ax.bar(lbls, vals, yerr=errs, capsize=5, color=colors, alpha=0.85)
        ax.set_ylabel("Wall-Clock Time (ms)")
        ax.set_title(f"{r['label']}\n({label_short})", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ymax = max(vals) * 1.2
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + ymax * 0.02, f"{v:.1f}ms",
                    ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, ymax)

        # Print summary
        speedup_bf = r["bf"][0] / r["is"][0] if r["is"][0] > 0 else 0
        speedup_unif = r["unif"][0] / r["is"][0] if r["is"][0] > 0 else 0
        print(f"    {key:20s}: BF={r['bf'][0]:.0f}ms  Unif={r['unif'][0]:.0f}ms(err={u_err_pct:.1f}%)  "
              f"IS={r['is'][0]:.0f}ms(err={i_err_pct:.1f}%)  "
              f"IS/BF={speedup_bf:.1f}x  IS/Unif={speedup_unif:.2f}x")

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_runtime.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 6(b): CrudeNN quality vs number of LSH levels
# ---------------------------------------------------------------------------

def fig6b_crude_quality():
    """Ratio D_a / gamma_a vs number of grid levels."""
    print("\n[Fig 6b] CrudeNN quality vs # LSH levels ...")
    N_PAIRS = 20
    level_range = list(range(1, 11))

    mean_ratios, std_ratios = [], []

    for n_levels in level_range:
        all_ratios = []
        for trial in range(N_PAIRS):
            A, B = make_shapenet_pair(high_cv=False, rng=trial)
            gamma = brute_force_chamfer(A, B, metric="l1")[1]
            D_a = crude_nn(A, B, metric="l1", num_levels=n_levels, rng=trial)
            ratio = D_a / (gamma + 1e-12)
            all_ratios.extend(ratio.tolist())

        mr, sr = np.mean(all_ratios), np.std(all_ratios)
        mean_ratios.append(mr)
        std_ratios.append(sr)
        print(f"  levels={n_levels:2d}  mean D_a/gamma = {mr:.3f} +- {sr:.3f}")

    mean_ratios = np.array(mean_ratios)
    std_ratios = np.array(std_ratios)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(level_range, mean_ratios, "o--", color=COLORS["is"],
            label=r"Mean $D_a / \gamma_a$")
    ax.fill_between(level_range,
                    np.clip(mean_ratios - std_ratios, 1.0, None),
                    mean_ratios + std_ratios,
                    alpha=0.25, color=COLORS["is"])
    ax.axhline(2.0, color="gray", linestyle=":", alpha=0.7, label="Factor-2 threshold")
    ax.set_xlabel("# Levels of LSH Data Structure")
    ax.set_ylabel(r"Ratio $D_a / \gamma_a$")
    ax.set_title("Quality of CrudeNN Approximations $D_a$\nvs Number of LSH Levels (ShapeNet-like, l1)", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(level_range)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6b_crude_quality.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Paper claims <=2 at 3 levels. Ours at 3: {mean_ratios[2]:.2f}")


# ---------------------------------------------------------------------------
# Additional: Pilot CV calibration sensitivity (for Phase 4)
# ---------------------------------------------------------------------------

def supplemental_cv_comparison():
    """Compare synthetic vs real text CV to document the improvement."""
    print("\n[Supplemental] CV comparison: synthetic vs real text data ...")
    rng = np.random.default_rng(42)

    print("  Computing synthetic text CV...", flush=True)
    A_syn, B_syn = make_text_pair(high_cv=True, rng=0)
    _, gamma_syn = brute_force_chamfer(A_syn, B_syn, metric="l1")
    cv_syn = compute_cv(gamma_syn)
    print(f"    Synthetic text CV = {cv_syn:.3f}")

    print("  Computing real text CV...", flush=True)
    A_real, B_real = make_real_text_pair(high_cv=True, rng=0)
    _, gamma_real = brute_force_chamfer(A_real, B_real, metric="l1")
    cv_real = compute_cv(gamma_real)
    print(f"    Real text CV = {cv_real:.3f}")

    print("  Computing calibrated ShapeNet CV...", flush=True)
    A_shape, B_shape = make_shapenet_pair(high_cv=True, rng=0)
    _, gamma_shape = brute_force_chamfer(A_shape, B_shape, metric="l1")
    cv_shape = compute_cv(gamma_shape)
    print(f"    Calibrated ShapeNet CV = {cv_shape:.3f}")

    return cv_syn, cv_real, cv_shape


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 65)
    print("Bakshi et al. 2023 - Section 3 Reproduction (v2: calibrated + real data)")
    print("=" * 65)

    # Supplemental: CV comparison first
    supplemental_cv_comparison()

    # Main figures
    fig3_sample_complexity(use_real_text=True)
    fig4_runtime(use_real_text=True)
    fig6b_crude_quality()

    print("\nAll figures saved to:", FIGURES_DIR)
    print("Done.")
