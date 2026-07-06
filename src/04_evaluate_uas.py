#!/usr/bin/env python3
"""
04_evaluate_uas.py
==================
Compute per-head UAS, identify the best head, and compute structural
baselines for one language.

Usage
-----
    python src/04_evaluate_uas.py \
        --predictions results/predicted_trees/en_trees.jsonl \
        --gold        data/processed/en.jsonl \
        --output-dir  results/uas_tables/ \
        --lang        en \
        --config      configs/experiment_config.yaml

Outputs
-------
    results/uas_tables/en_head_uas.csv    – 12×12 UAS grid
    results/uas_tables/en_best_head.json  – best (layer, head, UAS)
    results/uas_tables/en_baselines.json  – baseline UAS values
"""

import argparse
import csv
import json
import yaml
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.conllu_utils import read_jsonl
from src.utils.eval_utils import (
    compute_uas,
    find_best_head,
    right_branching_heads,
    left_branching_heads,
    random_baseline_heads,
    compute_baseline_uas,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate UAS.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    num_layers = cfg["model"]["num_layers"]
    num_heads = cfg["model"]["num_heads"]
    random_runs = cfg.get("evaluation", {}).get("random_baseline_runs", 100)

    # Load gold
    gold_records = read_jsonl(args.gold)
    gold_by_id = {r["sent_id"]: r for r in gold_records}

    # Load predictions
    pred_records = read_jsonl(args.predictions)

    # ── Per-head UAS ────────────────────────────────────────
    print(f"Computing per-head UAS for {args.lang} …")

    # Accumulators: correct[l][h], total[l][h]
    correct = np.zeros((num_layers, num_heads), dtype=np.int64)
    total = np.zeros((num_layers, num_heads), dtype=np.int64)

    for prec in pred_records:
        sid = prec["sent_id"]
        gold = gold_by_id[sid]["gold_heads"]
        for l in range(num_layers):
            for h in range(num_heads):
                pred = prec["predictions"][f"{l}-{h}"]
                for p, g in zip(pred, gold):
                    if g == 0:
                        continue
                    total[l, h] += 1
                    if p == g:
                        correct[l, h] += 1

    uas_grid = np.where(total > 0, correct / total, 0.0)

    # Save UAS grid as CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{args.lang}_head_uas.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + [f"head_{h}" for h in range(num_heads)])
        for l in range(num_layers):
            writer.writerow(
                [l] + [f"{uas_grid[l, h]:.4f}" for h in range(num_heads)]
            )
    print(f"  Saved UAS grid -> {csv_path}")

    # Best head
    best_l, best_h, best_uas = find_best_head(uas_grid)
    best_info = {
        "lang": args.lang,
        "best_layer": best_l,
        "best_head": best_h,
        "best_uas": round(best_uas, 4),
    }
    best_path = out_dir / f"{args.lang}_best_head.json"
    with open(best_path, "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"  Best head: layer {best_l}, head {best_h}, UAS = {best_uas:.4f}")

    # ── Baselines ───────────────────────────────────────────
    print("Computing baselines …")
    gold_heads_list = [r["gold_heads"] for r in gold_records]

    rb_uas = compute_baseline_uas(gold_heads_list, right_branching_heads)
    lb_uas = compute_baseline_uas(gold_heads_list, left_branching_heads)
    rand_uas = compute_baseline_uas(
        gold_heads_list, random_baseline_heads, n_runs=random_runs
    )

    baselines = {
        "lang": args.lang,
        "right_branching_uas": round(rb_uas, 4),
        "left_branching_uas": round(lb_uas, 4),
        "random_uas": round(rand_uas, 4),
    }
    bl_path = out_dir / f"{args.lang}_baselines.json"
    with open(bl_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  Baselines: RB={rb_uas:.4f}, LB={lb_uas:.4f}, Rand={rand_uas:.4f}")


if __name__ == "__main__":
    main()
