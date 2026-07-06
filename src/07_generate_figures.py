#!/usr/bin/env python3
"""
07_generate_figures.py
======================
Produce all publication-ready figures and tables for the report.

Usage
-----
    python src/07_generate_figures.py \
        --uas-dir        results/uas_tables/ \
        --regression-dir results/regression/ \
        --output-dir     results/figures/ \
        --config         configs/experiment_config.yaml

Generates
---------
    fig1_uas_heatmaps.pdf     – 2×2 UAS heatmaps (Figure 1)
    fig2_uas_barchart.pdf     – Best-head vs baselines  (Figure 2)
    fig3_accuracy_length.pdf  – Accuracy vs dep length  (Figure 3)
"""

import argparse
import csv
import json
import yaml
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Style setup ─────────────────────────────────────────────

LANG_LABELS = {
    "en": "English", "hi": "Hindi", "fr": "French",
    "ja": "Japanese", "ko": "Korean", "es": "Spanish",
}
# SVO cluster first, SOV cluster second — makes bar charts typologically readable
LANG_ORDER = ["en", "fr", "es", "hi", "ja", "ko"]
PALETTE = {
    "en": "#4C72B0", "fr": "#55A868", "hi": "#C44E52",
    "ja": "#8172B3", "ko": "#CCB974", "es": "#E87D3E",
}


def setup_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ── Figure 1: UAS heatmaps ─────────────────────────────────

def load_uas_grid(uas_dir, lang, num_layers=12, num_heads=12):
    """Load the 12×12 UAS grid from CSV."""
    path = Path(uas_dir) / f"{lang}_head_uas.csv"
    grid = np.zeros((num_layers, num_heads))
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            l = int(row["layer"])
            for h in range(num_heads):
                grid[l, h] = float(row[f"head_{h}"])
    return grid


def plot_uas_heatmaps(uas_dir, output_dir, languages):
    """Figure 1: Grid of UAS heatmaps (adapts to number of languages)."""
    n_langs = len(languages)
    ncols = min(3, n_langs)
    nrows = (n_langs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if n_langs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Find global vmax for consistent color scale
    grids = {}
    for lang in languages:
        grids[lang] = load_uas_grid(uas_dir, lang)
    vmax = max(g.max() for g in grids.values())

    for idx, lang in enumerate(languages):
        ax = axes[idx]
        sns.heatmap(
            grids[lang],
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            annot=False,
            cbar=idx == (ncols - 1),  # colorbar on rightmost of top row
            cbar_kws={"label": "UAS"} if idx == (ncols - 1) else {},
            xticklabels=range(12),
            yticklabels=range(12),
        )
        ax.set_title(LANG_LABELS.get(lang, lang), fontweight="bold")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

    # Hide unused subplots
    for idx in range(n_langs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Per-Head Unlabeled Attachment Score by Layer and Head",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save_fig(fig, output_dir, "fig1_uas_heatmaps")


# ── Figure 2: Best-head UAS vs baselines ───────────────────

def plot_uas_barchart(uas_dir, output_dir, languages):
    """Figure 2: Grouped bar chart comparing best head to baselines."""
    data = []
    for lang in languages:
        with open(Path(uas_dir) / f"{lang}_best_head.json") as f:
            best = json.load(f)
        with open(Path(uas_dir) / f"{lang}_baselines.json") as f:
            bl = json.load(f)

        data.append({"Language": LANG_LABELS[lang],
                      "Method": "Best Attention Head",
                      "UAS": best["best_uas"]})
        data.append({"Language": LANG_LABELS[lang],
                      "Method": "Right-Branching",
                      "UAS": bl["right_branching_uas"]})
        data.append({"Language": LANG_LABELS[lang],
                      "Method": "Left-Branching",
                      "UAS": bl["left_branching_uas"]})
        data.append({"Language": LANG_LABELS[lang],
                      "Method": "Random",
                      "UAS": bl["random_uas"]})

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(7, 4))
    method_order = [
        "Best Attention Head", "Right-Branching",
        "Left-Branching", "Random",
    ]
    method_colors = ["#2C3E50", "#3498DB", "#E67E22", "#95A5A6"]

    sns.barplot(
        data=df, x="Language", y="UAS", hue="Method",
        hue_order=method_order, palette=method_colors,
        ax=ax, edgecolor="white",
    )
    ax.set_title(
        "Best Attention Head UAS vs. Structural Baselines",
        fontweight="bold",
    )
    ax.set_ylabel("Unlabeled Attachment Score")
    ax.set_xlabel("")
    ax.legend(title="", loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_fig(fig, output_dir, "fig2_uas_barchart")


# ── Figure 3: Accuracy vs dependency length ─────────────────

def plot_accuracy_vs_length(regression_dir, output_dir):
    """Figure 3: Mean arc accuracy by dependency length, per language."""
    df = pd.read_csv(Path(regression_dir) / "regression_data.csv")

    # Bin dependency lengths
    def bin_dep_length(d):
        if d <= 5:
            return str(d)
        elif d <= 10:
            return "6–10"
        else:
            return "11+"

    df["dep_length_bin"] = df["dep_length"].apply(bin_dep_length)
    bin_order = ["1", "2", "3", "4", "5", "6–10", "11+"]

    # Compute mean accuracy per bin per language
    grouped = (
        df.groupby(["language", "dep_length_bin"])["arc_correct"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    # 95% CI
    grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for lang in LANG_ORDER:
        sub = grouped[grouped["language"] == lang].copy()
        # Ensure correct bin order
        sub["dep_length_bin"] = pd.Categorical(
            sub["dep_length_bin"], categories=bin_order, ordered=True
        )
        sub = sub.sort_values("dep_length_bin")

        ax.errorbar(
            sub["dep_length_bin"],
            sub["mean"],
            yerr=sub["ci95"],
            marker="o",
            label=LANG_LABELS[lang],
            color=PALETTE[lang],
            capsize=3,
            linewidth=1.5,
            markersize=5,
        )

    ax.set_xlabel("Dependency Length (tokens)")
    ax.set_ylabel("Mean Arc Accuracy")
    ax.set_title(
        "Per-Arc Accuracy by Dependency Length",
        fontweight="bold",
    )
    ax.legend(title="Language", framealpha=0.9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, output_dir, "fig3_accuracy_length")


# ── Figure 4: Accuracy by Relation Class ────────────────────

def plot_accuracy_by_relation_class(regression_dir, output_dir):
    """Figure 4: Mean arc accuracy grouped by Dependency Relation Class (Core, Functional, Other)."""
    df = pd.read_csv(Path(regression_dir) / "regression_data.csv")

    grouped = (
        df.groupby(["language", "deprel_class"])["arc_correct"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Standardize classes
    class_order = ["core", "functional", "other"]
    
    sns.barplot(
        data=df, x="deprel_class", y="arc_correct", hue="language",
        hue_order=LANG_ORDER, palette=PALETTE, order=class_order,
        ax=ax, edgecolor="white", errorbar=('ci', 95), capsize=0.1
    )
    
    ax.set_xlabel("Dependency Relation Class")
    ax.set_ylabel("Mean Arc Accuracy")
    ax.set_title(
        "Accuracy by Syntactic Relation Class (Best Head)",
        fontweight="bold",
    )
    ax.set_xticklabels(["Core", "Functional", "Other"])
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [LANG_LABELS.get(l, l) for l in labels]
    ax.legend(handles, new_labels, title="Language", framealpha=0.9, loc="upper right")
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    save_fig(fig, output_dir, "fig4_accuracy_relation")
# ── Figure 5: Fragmentation vs UAS scatter ─────────────────

def plot_frag_vs_uas(uas_dir, regression_dir, output_dir, languages):
    """Scatter plot: mean subword fragmentation ratio vs best-head UAS."""
    df = pd.read_csv(Path(regression_dir) / "regression_data.csv")

    xs, ys, langs_plotted = [], [], []
    for lang in languages:
        best_path = Path(uas_dir) / f"{lang}_best_head.json"
        if not best_path.exists():
            continue
        with open(best_path) as f:
            best_info = json.load(f)
        if "frag_ratio" not in df.columns:
            continue
        lang_df = df[df["language"] == lang]
        if lang_df.empty:
            continue
        x = lang_df["frag_ratio"].mean()
        y = best_info["best_uas"]
        xs.append(x)
        ys.append(y)
        langs_plotted.append(lang)

    if len(xs) < 2:
        print("  [!] Not enough data for frag_vs_uas plot, skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, lang in enumerate(langs_plotted):
        ax.scatter(xs[i], ys[i], color=PALETTE.get(lang, "#333"),
                   s=100, zorder=3, edgecolors="white", linewidth=0.5)
        ax.annotate(LANG_LABELS.get(lang, lang), (xs[i], ys[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    # Trendline
    m, b = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs) - 0.1, max(xs) + 0.1, 100)
    ax.plot(xr, m * xr + b, color="#888", linewidth=1, linestyle="--", zorder=1)

    ax.set_xlabel("Mean Subword Fragmentation Ratio")
    ax.set_ylabel("Best-Head UAS")
    ax.set_title("Subword Fragmentation vs. Syntactic Recovery",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, output_dir, "fig6_frag_vs_uas")


# ── Figure 6: Head transfer ─────────────────────────────────

def plot_transfer_heatmap(output_dir):
    """Heatmap of Δ UAS when transferring English's best head to other languages."""
    transfer_path = Path("results/transfer/transfer_results.csv")
    if not transfer_path.exists():
        print("  [!] transfer_results.csv not found, skipping transfer plot.")
        return

    df = pd.read_csv(transfer_path)
    # Overall transfer (no per-relation breakdown)
    overall = df[df["relation"] == "_overall_"].copy()
    if overall.empty:
        print("  [!] No _overall_ rows in transfer data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    target_order = [l for l in LANG_ORDER if l != "en"]
    plot_data = []
    for lang in target_order:
        row = overall[overall["target_lang"] == lang]
        if not row.empty:
            plot_data.append({
                "Language": LANG_LABELS.get(lang, lang),
                "Transferred (L5-H8)": row["transferred_uas"].values[0],
                "Native Best": row["native_uas"].values[0],
            })

    if not plot_data:
        return

    pdf = pd.DataFrame(plot_data)
    x = np.arange(len(pdf))
    w = 0.35
    ax.bar(x - w/2, pdf["Native Best"], w, label="Native Best Head",
           color="#2C3E50", edgecolor="white")
    ax.bar(x + w/2, pdf["Transferred (L5-H8)"], w,
           label="English L5-H8 (transferred)",
           color="#3498DB", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(pdf["Language"])
    ax.set_ylabel("UAS")
    ax.set_title("Cross-Lingual Head Transfer: English L5-H8 -> Other Languages",
                 fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, output_dir, "fig7_head_transfer")


# ── Helpers ─────────────────────────────────────────────────

def save_fig(fig, output_dir, name):
    """Save figure as both PDF and PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    print(f"  Saved {name}.pdf and {name}.png")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures."
    )
    parser.add_argument("--uas-dir", required=True)
    parser.add_argument("--regression-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    languages = list(cfg["languages"].keys())
    setup_style()

    print("Generating figures …")
    plot_uas_heatmaps(args.uas_dir, args.output_dir, languages)
    plot_uas_barchart(args.uas_dir, args.output_dir, languages)
    plot_accuracy_vs_length(args.regression_dir, args.output_dir)
    plot_accuracy_by_relation_class(args.regression_dir, args.output_dir)
    plot_frag_vs_uas(args.uas_dir, args.regression_dir, args.output_dir, languages)
    plot_transfer_heatmap(args.output_dir)
    print("\n[V] All figures generated.")


if __name__ == "__main__":
    main()
