#!/usr/bin/env python3
"""
11_visualize_head_clusters.py
=============================
Visualizes the "distributed syntax" of mBERT by treating each of the 144
attention heads as a vector of grammatical specializations (UAS per relation).
Uses PCA and t-SNE to project these 144 heads into a 2D space, demonstrating
how syntax-specific experts (like subject vs object experts) cluster together.

Usage
-----
    python src/11_visualize_head_clusters.py
"""

import argparse
import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.conllu_utils import read_jsonl

def main():
    parser = argparse.ArgumentParser(description="Visualize attention head clusters in 2D.")
    parser.add_argument("--predictions", default="results/predicted_trees/en_trees.jsonl")
    parser.add_argument("--gold", default="data/processed/en.jsonl")
    parser.add_argument("--output-dir", default="results/figures/")
    parser.add_argument("--lang-name", default="English")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    gold_path = Path(args.gold)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists() or not gold_path.exists():
        print(f"Error: Could not find data for {args.lang_name} clustering.")
        sys.exit(1)

    print(f"Loading data to compute head matrix for {args.lang_name}...")
    gold_records = read_jsonl(str(gold_path))
    pred_records = read_jsonl(str(pred_path))
    gold_by_id = {r["sent_id"]: r for r in gold_records}

    # Relations to analyze (features for our vectors)
    target_rels = ["nsubj", "obj", "obl", "amod", "advmod", "det", "case", "mark", "cc", "conj", "nmod"]
    
    # Store correct counts and total counts per head per relation
    # dict[head_key][rel] = correct_count
    head_correct = defaultdict(lambda: defaultdict(int))
    rel_total = defaultdict(int)

    # 1. Count totals and correct predictions
    for prec in pred_records:
        sid = prec["sent_id"]
        if sid not in gold_by_id:
            continue
            
        gold = gold_by_id[sid]
        g_heads = gold["gold_heads"]
        g_rels = gold["gold_deprels"]

        # Count total possible connections for each relation type in this sentence
        for i, rel in enumerate(g_rels):
            base_rel = rel.split(":")[0] 
            if base_rel in target_rels:
                rel_total[base_rel] += 1

        # Check every head's prediction
        for head_key, p_heads in prec["predictions"].items():
            for i, rel in enumerate(g_rels):
                base_rel = rel.split(":")[0]
                if base_rel in target_rels:
                    if i < len(p_heads) and p_heads[i] == g_heads[i]:
                        head_correct[head_key][base_rel] += 1

    # 2. Build the Matrix X (144 heads x N relations)
    head_keys = [f"{l}-{h}" for l in range(12) for h in range(12)]
    matrix = np.zeros((144, len(target_rels)))
    
    for i, hk in enumerate(head_keys):
        for j, rel in enumerate(target_rels):
            if rel_total[rel] > 0:
                matrix[i, j] = head_correct[hk][rel] / rel_total[rel]
            else:
                matrix[i, j] = 0.0

    # Determine the "primary specialization" of each head for coloring
    specializations = []
    top_rel_per_head = []
    
    for i in range(144):
        best_rel_idx = np.argmax(matrix[i])
        best_rel = target_rels[best_rel_idx]
        best_score = matrix[i, best_rel_idx]
        
        # If the head is just generally bad at everything, mark as 'Weak'
        if best_score < 0.15:
            specializations.append("Weak/Diffuse")
        elif best_rel in ["nsubj", "obj", "obl"]:
            specializations.append("Core Arguments")
        elif best_rel in ["amod", "advmod", "nmod"]:
            specializations.append("Modifiers")
        elif best_rel in ["det", "case", "mark", "cc"]:
            specializations.append("Functional/Lexical")
        else:
            specializations.append("Other")
            
        top_rel_per_head.append(best_rel if best_score >= 0.15 else "none")

    # 3. Dimensionality Reduction
    print("Running PCA...")
    # Standardize features before PCA
    from sklearn.preprocessing import StandardScaler
    matrix_scaled = StandardScaler().fit_transform(matrix)
    
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(matrix_scaled)
    
    # 4. Plotting
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Define a clean palette
    palette = {
        "Core Arguments": "#E63946",       # Red
        "Modifiers": "#2A9D8F",            # Teal
        "Functional/Lexical": "#457B9D",   # Blue
        "Other": "#F4A261",                # Orange
        "Weak/Diffuse": "#D3D3D3"          # Light Grey
    }
    
    df_plot = pd.DataFrame({
        "PCA1": pca_result[:, 0],
        "PCA2": pca_result[:, 1],
        "Specialization": specializations,
        "Head": head_keys,
        "BestRel": top_rel_per_head
    })
    
    # Sort so 'Weak' is plotted first (underneath)
    df_plot['sort_key'] = df_plot['Specialization'].map(lambda x: 0 if x == 'Weak/Diffuse' else 1)
    df_plot = df_plot.sort_values('sort_key')
    
    ax = sns.scatterplot(
        data=df_plot,
        x="PCA1",
        y="PCA2",
        hue="Specialization",
        palette=palette,
        s=150,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5
    )
    
    # Annotate specific notable heads
    # L5-H8
    l5h8_row = df_plot[df_plot["Head"] == "5-8"].iloc[0]
    plt.annotate(
        "L5-H8\n(Generalist)", 
        (l5h8_row["PCA1"], l5h8_row["PCA2"]),
        xytext=(10, 10), textcoords='offset points',
        fontsize=10, fontweight='bold',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="black")
    )
    
    # Find the best 'nsubj' head
    nsubj_idx = target_rels.index("nsubj")
    best_nsubj_head_idx = np.argmax(matrix[:, nsubj_idx])
    best_nsubj_head = head_keys[best_nsubj_head_idx]
    if best_nsubj_head != "5-8":
        row = df_plot[df_plot["Head"] == best_nsubj_head].iloc[0]
        plt.annotate(
            f"{best_nsubj_head}\n(Best nsubj)", 
            (row["PCA1"], row["PCA2"]),
            xytext=(-40, 15), textcoords='offset points',
            fontsize=9,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#E63946")
        )
        
    # Find the best 'obj' head
    obj_idx = target_rels.index("obj")
    best_obj_head_idx = np.argmax(matrix[:, obj_idx])
    best_obj_head = head_keys[best_obj_head_idx]
    if best_obj_head != "5-8" and best_obj_head != best_nsubj_head:
        row = df_plot[df_plot["Head"] == best_obj_head].iloc[0]
        plt.annotate(
            f"{best_obj_head}\n(Best obj)", 
            (row["PCA1"], row["PCA2"]),
            xytext=(10, -20), textcoords='offset points',
            fontsize=9,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#E63946")
        )

    plt.title(f"mBERT 'Brain Map': Attention Heads Clustered by Syntactic Specialization\n(PCA projection of {len(target_rels)}-dimensional accuracy vectors)", 
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    
    # Clean up legend
    plt.legend(title="Head Specialization", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_file = out_dir / "fig8_head_clusters.pdf"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.savefig(out_dir / "fig8_head_clusters.png", bbox_inches="tight", dpi=300)
    print(f"Saved PCA brain map -> {out_file}")

if __name__ == "__main__":
    main()
