"""
Plotting and Results Consolidation for S06b.
Stage: S06b

Merges Feature Selection and PCA outputs into master unified tables:
  - outputs/fold_metrics.csv
  - outputs/comparison.csv
  - outputs/paired_fold_comparisons.csv
  - outputs/oof_predictions.csv

Generates 6 publication-ready figures in outputs/plots/:
  1. r2_comparison.png
  2. mae_rmse_comparison.png
  3. feature_counts.png
  4. selection_stability.png
  5. pca_variance.png
  6. subset_performance.png
"""
from __future__ import annotations

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

CURRENT_DIR = pathlib.Path(__file__).resolve().parent

# Nordic / Clean Scientific Theme (matching S06)
NORD0  = "#2e3440"  # Dark background
NORD1  = "#3b4252"  # Axes background
NORD4  = "#d8dee9"  # Text / Foreground
NORD7  = "#8fbcbb"  # Teal
NORD8  = "#88c0d0"  # Frost cyan
NORD9  = "#81a1c1"  # Soft blue
NORD11 = "#bf616a"  # Red / Coral
NORD13 = "#ebcb8b"  # Yellow / Gold
NORD14 = "#a3be8c"  # Green
NORD15 = "#b48ead"  # Purple

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#212529",
    "xtick.color": "#212529",
    "ytick.color": "#212529",
    "text.color": "#212529",
    "grid.color": "#e9ecef",
    "grid.linestyle": "--",
    "grid.alpha": 0.8,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def consolidate_results(output_dir: pathlib.Path) -> None:
    """
    Merges feature selection and PCA results into unified comparison tables safely and idempotently.
    Can be run multiple times without duplicating PCA rows.
    """
    # 1. Load feature selection fold metrics (prefer raw source fs_fold_metrics.csv)
    fs_file = output_dir / "fs_fold_metrics.csv"
    if fs_file.exists():
        fs_fold = pd.read_csv(fs_file)
    else:
        fs_fold = pd.read_csv(output_dir / "fold_metrics.csv")
        fs_fold = fs_fold[~fs_fold["method"].astype(str).str.startswith("PCA_")].copy()

    pca_file = output_dir / "pca_fold_metrics.csv"
    if not pca_file.exists():
        raise FileNotFoundError(f"Missing PCA fold metrics at {pca_file}")
    pca_fold = pd.read_csv(pca_file)

    # Rename n_components to n_features for PCA to align schema
    pca_fold_aligned = pca_fold.rename(columns={
        "n_components": "n_features",
        "pca_time_sec": "selection_time_sec",
    })

    # Combine fold metrics safely
    combined_fold_metrics = pd.concat([fs_fold, pca_fold_aligned], ignore_index=True)
    dups = combined_fold_metrics.duplicated(subset=["method", "fold"])
    if dups.any():
        dup_entries = combined_fold_metrics[dups][["method", "fold"]].to_dict(orient="records")
        raise ValueError(f"Found {dups.sum()} duplicate (method, fold) records in combined fold metrics: {dup_entries}")

    combined_fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    # 2. Combine OOF predictions safely (prefer raw source fs_oof_predictions.csv)
    fs_oof_file = output_dir / "fs_oof_predictions.csv"
    if fs_oof_file.exists():
        fs_oof = pd.read_csv(fs_oof_file)
    else:
        fs_oof = pd.read_csv(output_dir / "oof_predictions.csv")
        fs_oof = fs_oof[~fs_oof["method"].astype(str).str.startswith("PCA_")].copy()

    pca_oof_file = output_dir / "pca_oof_predictions.csv"
    if not pca_oof_file.exists():
        raise FileNotFoundError(f"Missing PCA OOF predictions at {pca_oof_file}")
    pca_oof = pd.read_csv(pca_oof_file)

    combined_oof = pd.concat([fs_oof, pca_oof], ignore_index=True)
    oof_dups = combined_oof.duplicated(subset=["case", "target", "method", "fold"])
    if oof_dups.any():
        raise ValueError(f"Found {oof_dups.sum()} duplicate OOF predictions in combined table!")

    combined_oof.to_csv(output_dir / "oof_predictions.csv", index=False)

    # 3. Recompute unified comparison table
    methods = combined_fold_metrics["method"].unique()
    comp_rows = []
    baseline_df = combined_fold_metrics[combined_fold_metrics["method"] == "Baseline (34 features)"]
    if baseline_df.empty:
        raise ValueError("Missing 'Baseline (34 features)' in fold metrics!")

    for m in methods:
        m_df = combined_fold_metrics[combined_fold_metrics["method"] == m]
        m_oof = combined_oof[combined_oof["method"] == m]

        mean_r2 = float(np.mean(m_df["r2"]))
        std_r2 = float(np.std(m_df["r2"], ddof=0))
        mean_mae = float(np.mean(m_df["mae"]))
        std_mae = float(np.std(m_df["mae"], ddof=0))
        mean_rmse = float(np.mean(m_df["rmse"]))
        std_rmse = float(np.std(m_df["rmse"], ddof=0))
        mean_feats = float(np.mean(m_df["n_features"]))
        total_sel_time = float(np.sum(m_df["selection_time_sec"]))
        mean_train_time = float(np.mean(m_df["train_time_sec"]))

        pooled_r2 = float(r2_score(m_oof["actual"], m_oof["predicted"]))
        pooled_mae = float(mean_absolute_error(m_oof["actual"], m_oof["predicted"]))
        pooled_rmse = float(root_mean_squared_error(m_oof["actual"], m_oof["predicted"]))

        comp_rows.append({
            "method": m,
            "mean_n_features": round(mean_feats, 1),
            "mean_r2": round(mean_r2, 6),
            "std_r2": round(std_r2, 6),
            "mean_mae": round(mean_mae, 6),
            "std_mae": round(std_mae, 6),
            "mean_rmse": round(mean_rmse, 6),
            "std_rmse": round(std_rmse, 6),
            "pooled_oof_r2": round(pooled_r2, 6),
            "pooled_oof_mae": round(pooled_mae, 6),
            "pooled_oof_rmse": round(pooled_rmse, 6),
            "total_selection_time_sec": round(total_sel_time, 2),
            "mean_train_time_sec": round(mean_train_time, 4),
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(output_dir / "comparison.csv", index=False)

    # 4. Recompute unified paired fold comparisons
    paired_rows = []
    b_df = baseline_df.sort_values("fold")
    n_folds = len(b_df)

    for m in methods:
        m_df = combined_fold_metrics[combined_fold_metrics["method"] == m].sort_values("fold")
        for f_idx in range(n_folds):
            b_r2 = b_df[b_df["fold"] == f_idx]["r2"].values[0]
            m_r2 = m_df[m_df["fold"] == f_idx]["r2"].values[0]
            b_mae = b_df[b_df["fold"] == f_idx]["mae"].values[0]
            m_mae = m_df[m_df["fold"] == f_idx]["mae"].values[0]
            b_rmse = b_df[b_df["fold"] == f_idx]["rmse"].values[0]
            m_rmse = m_df[m_df["fold"] == f_idx]["rmse"].values[0]

            paired_rows.append({
                "method": m,
                "fold": f_idx,
                "baseline_r2": round(b_r2, 6),
                "method_r2": round(m_r2, 6),
                "paired_r2_diff": round(m_r2 - b_r2, 6),
                "baseline_mae": round(b_mae, 6),
                "method_mae": round(m_mae, 6),
                "baseline_rmse": round(b_rmse, 6),
                "method_rmse": round(m_rmse, 6),
            })

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(output_dir / "paired_fold_comparisons.csv", index=False)

    # 5. Combine subset metrics safely
    fs_sub_file = output_dir / "fs_subset_metrics.csv"
    if fs_sub_file.exists():
        fs_sub = pd.read_csv(fs_sub_file)
    else:
        fs_sub = pd.read_csv(output_dir / "subset_metrics.csv")
        fs_sub = fs_sub[~fs_sub["method"].astype(str).str.startswith("PCA_")].copy()

    pca_sub_file = output_dir / "pca_subset_metrics.csv"
    if pca_sub_file.exists():
        pca_sub = pd.read_csv(pca_sub_file)
        pca_sub_aligned = pca_sub.rename(columns={"mean_n_components": "mean_n_features"})
        combined_sub = pd.concat([fs_sub, pca_sub_aligned], ignore_index=True)
        sub_dups = combined_sub.duplicated(subset=["subset", "method"])
        if sub_dups.any():
            raise ValueError(f"Found duplicate subset records: {combined_sub[sub_dups]}")
        combined_sub.to_csv(output_dir / "subset_metrics.csv", index=False)

    print("Unified CSVs successfully consolidated idempotently in", output_dir)


def generate_plots(output_dir: pathlib.Path) -> None:
    """Generates all 6 required diagnostic and comparative figures."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    comp_df = pd.read_csv(output_dir / "comparison.csv")
    stability_df = pd.read_csv(output_dir / "feature_selection_stability.csv")
    pca_var_df = pd.read_csv(output_dir / "pca_explained_variance.csv")
    subset_df = pd.read_csv(output_dir / "subset_metrics.csv")

    baseline_r2 = comp_df[comp_df["method"] == "Baseline (34 features)"]["mean_r2"].values[0]

    # -----------------------------------------------------------------------
    # Plot 1: R² across methods with fold variability
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = comp_df["method"].tolist()
    means = comp_df["mean_r2"].tolist()
    stds = comp_df["std_r2"].tolist()

    colors = []
    for m in methods:
        if "Baseline" in m:
            colors.append("#34495e")
        elif "Corr" in m:
            colors.append("#2980b9")
        elif "Mutual" in m:
            colors.append("#16a085")
        elif "F_test" in m:
            colors.append("#27ae60")
        elif "Wrapper" in m:
            colors.append("#8e44ad")
        elif "PCA" in m:
            colors.append("#d35400")
        else:
            colors.append("#7f8c8d")

    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors, alpha=0.88, edgecolor="#222222", width=0.65)
    ax.axhline(baseline_r2, color="#c0392b", linestyle="--", linewidth=1.8, label=f"Baseline 34 Features (R² = {baseline_r2:.4f})")
    s06c = pd.read_csv(pathlib.Path(__file__).resolve().parents[1] / "S06_case_models/outputs/s06c/model_comparison.csv")
    b1_r2 = s06c.query("model == 'B1_rf_basic_untuned' and target == 'U_mean'")["r2"].item()
    ax.axhline(b1_r2, color="#f39c12", linestyle=":", linewidth=1.5, label=f"B1 Baseline Reference ({b1_r2:.3f})")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Grouped 5-Fold CV R²", fontweight="bold")
    ax.set_title("S06b: Cross-Validation Prediction Performance (U_mean) Across Methods", fontweight="bold", pad=12)
    ax.set_ylim(0.60, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95)

    # Value labels
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean_val + 0.012, f"{mean_val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "1_r2_comparison.png", dpi=200)
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 2: MAE and RMSE across methods
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x_pos = np.arange(len(methods))
    width = 0.35

    mae_vals = comp_df["mean_mae"].tolist()
    rmse_vals = comp_df["mean_rmse"].tolist()

    ax.bar(x_pos - width / 2, mae_vals, width, label="Mean Absolute Error (MAE)", color="#2980b9", alpha=0.85, edgecolor="#222")
    ax.bar(x_pos + width / 2, rmse_vals, width, label="Root Mean Squared Error (RMSE)", color="#e67e22", alpha=0.85, edgecolor="#222")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Error Metric Value (U/u*)", fontweight="bold")
    ax.set_title("S06b: Error Comparison (MAE & RMSE) Across Methods", fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    fig.savefig(plots_dir / "2_mae_rmse_comparison.png", dpi=200)
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 3: Number of features or components
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    feat_counts = comp_df["mean_n_features"].tolist()
    bars = ax.barh(methods, feat_counts, color=colors, alpha=0.85, edgecolor="#222", height=0.6)
    ax.set_xlabel("Number of Features / Retained Components", fontweight="bold")
    ax.set_title("S06b: Dimensionality Reduction Comparison Across Methods", fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.axvline(34, color="#c0392b", linestyle="--", linewidth=1.5, label="Original 34 Features")
    ax.legend(loc="lower right")

    for bar, count in zip(bars, feat_counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{count:.1f}", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "3_feature_counts.png", dpi=200)
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 4: Feature-selection stability
    # -----------------------------------------------------------------------
    if not stability_df.empty:
        # Group by feature and see selection count across key methods
        fig, ax = plt.subplots(figsize=(11, 8))
        pivot_stab = stability_df.pivot(index="feature", columns="method", values="selection_count").fillna(0)
        # Sort by total selections
        pivot_stab["total"] = pivot_stab.sum(axis=1)
        pivot_stab = pivot_stab.sort_values("total", ascending=True).drop(columns=["total"])

        pivot_stab.plot(kind="barh", stacked=False, ax=ax, colormap="viridis", width=0.8, edgecolor="#333", alpha=0.88)
        ax.set_xlabel("Selection Frequency (Folds out of 5)", fontweight="bold")
        ax.set_ylabel("Feature Name", fontweight="bold")
        ax.set_title("S06b: Feature Selection Stability Across Methods (5 Outer Folds)", fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.legend(title="Method", loc="lower right", framealpha=0.95)

        plt.tight_layout()
        fig.savefig(plots_dir / "4_selection_stability.png", dpi=200)
        plt.close()

    # -----------------------------------------------------------------------
    # Plot 5: PCA explained and cumulative variance
    # -----------------------------------------------------------------------
    if not pca_var_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        # Look at fold 0 for detailed scree plot
        fold0_pca = pca_var_df[pca_var_df["fold"] == 0]
        # Choose 99% variance run as it contains the most components
        p99 = fold0_pca[fold0_pca["method"] == "PCA_99%_var"].sort_values("component")

        comps = p99["component"].values
        evr = p99["explained_variance_ratio"].values * 100
        cvr = p99["cumulative_variance_ratio"].values * 100

        ax1.bar(comps, evr, color="#2980b9", alpha=0.85, edgecolor="#222")
        ax1.set_xlabel("Principal Component Index", fontweight="bold")
        ax1.set_ylabel("Individual Explained Variance (%)", fontweight="bold")
        ax1.set_title("PCA Scree Plot (Individual Variance)", fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)

        ax2.plot(comps, cvr, marker="o", color="#e74c3c", linewidth=2.2, markersize=5)
        ax2.axhline(90, color="#f39c12", linestyle="--", label="90% Variance Target")
        ax2.axhline(95, color="#27ae60", linestyle="--", label="95% Variance Target")
        ax2.axhline(99, color="#8e44ad", linestyle="--", label="99% Variance Target")
        ax2.set_xlabel("Number of Components", fontweight="bold")
        ax2.set_ylabel("Cumulative Explained Variance (%)", fontweight="bold")
        ax2.set_title("Cumulative Explained Variance Trajectory", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="lower right", framealpha=0.95)

        plt.suptitle("S06b: PCA Variance Retention Analysis (Fold 0)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(plots_dir / "5_pca_variance.png", dpi=200)
        plt.close()

    # -----------------------------------------------------------------------
    # Plot 6: Idealized vs Realistic performance
    # -----------------------------------------------------------------------
    if not subset_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ideal_df = subset_df[subset_df["subset"] == "idealized"].set_index("method")
        real_df = subset_df[subset_df["subset"] == "realistic"].set_index("method")

        common_methods = [m for m in ideal_df.index if m in real_df.index]
        x_pos = np.arange(len(common_methods))
        width = 0.35

        ideal_r2s = [ideal_df.loc[m, "mean_r2"] for m in common_methods]
        real_r2s = [real_df.loc[m, "mean_r2"] for m in common_methods]

        ax.bar(x_pos - width / 2, ideal_r2s, width, label="Idealized Subset (N=190)", color="#27ae60", alpha=0.85, edgecolor="#222")
        ax.bar(x_pos + width / 2, real_r2s, width, label="Realistic Subset (N=292)", color="#e67e22", alpha=0.85, edgecolor="#222")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(common_methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Grouped CV R²", fontweight="bold")
        ax.set_title("S06b: Performance Decomposition: Idealized vs. Realistic Subsets", fontweight="bold", pad=12)
        ax.set_ylim(0.40, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="lower left", framealpha=0.95)

        for i, (i_r2, r_r2) in enumerate(zip(ideal_r2s, real_r2s)):
            ax.text(x_pos[i] - width / 2, i_r2 + 0.015, f"{i_r2:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.text(x_pos[i] + width / 2, r_r2 + 0.015, f"{r_r2:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        fig.savefig(plots_dir / "6_subset_performance.png", dpi=200)
        plt.close()

    print("All 6 figures generated successfully in", plots_dir)


if __name__ == "__main__":
    out_dir = CURRENT_DIR / "outputs"
    consolidate_results(out_dir)
    generate_plots(out_dir)
