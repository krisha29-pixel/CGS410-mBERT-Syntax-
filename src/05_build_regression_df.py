#!/usr/bin/env python3
"""
05_build_regression_df.py
=========================
Assemble the arc-level DataFrame used by the mixed-effects model.
Uses the **best head per language** identified in step 04.

Usage
-----
    python src/05_build_regression_df.py \
        --predictions-dir results/predicted_trees/ \
        --gold-dir        data/processed/ \
        --uas-dir         results/uas_tables/ \
        --output          results/regression/regression_data.csv \
        --config          configs/experiment_config.yaml

Output columns
--------------
    arc_correct, dep_length, log_dep_length, sent_length, log_sent_length,
    language, deprel, deprel_class, sent_id, treebank
"""

import argparse
import json
import yaml
import sys
import os
import math
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.conllu_utils import read_jsonl
from src.utils.eval_utils import per_arc_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Build regression DataFrame from arc-level results."
    )
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--uas-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    languages = list(cfg["languages"].keys())
    sample_n = cfg.get("statistics", {}).get("sample_per_language", None)

    rows = []

    for lang in languages:
        # Load best head info
        best_path = Path(args.uas_dir) / f"{lang}_best_head.json"
        with open(best_path) as f:
            best_info = json.load(f)
        best_key = f"{best_info['best_layer']}-{best_info['best_head']}"
        print(f"[{lang}] Using best head: layer {best_info['best_layer']}, "
              f"head {best_info['best_head']} (UAS={best_info['best_uas']})")

        # Load gold
        gold_records = read_jsonl(
            str(Path(args.gold_dir) / f"{lang}.jsonl")
        )
        gold_by_id = {r["sent_id"]: r for r in gold_records}

        # Load predictions
        pred_records = read_jsonl(
            str(Path(args.predictions_dir) / f"{lang}_trees.jsonl")
        )

        # Optional subsampling
        if sample_n and len(pred_records) > sample_n:
            import random
            random.seed(42)
            pred_records = random.sample(pred_records, sample_n)

        for prec in pred_records:
            sid = prec["sent_id"]
            gold = gold_by_id[sid]
            pred_heads = prec["predictions"][best_key]
            arc_acc = per_arc_accuracy(pred_heads, gold["gold_heads"])

            for i in range(len(arc_acc)):
                if arc_acc[i] == -1:
                    continue  # skip root token

                dep_len = gold["dep_lengths"][i]
                if dep_len == 0:
                    continue  # safety check (root)

                rows.append({
                    "arc_correct": arc_acc[i],
                    "dep_length": dep_len,
                    "log_dep_length": math.log(dep_len),
                    "sent_length": gold["sent_len"],
                    "log_sent_length": math.log(gold["sent_len"]),
                    "language": lang,
                    "deprel": gold["gold_deprels"][i],
                    "deprel_class": gold["deprel_classes"][i],
                    "sent_id": sid,
                    "treebank": gold["treebank"],
                    "frag_ratio": gold.get("frag_ratio", 1.0),
                })

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nRegression DataFrame: {len(df)} arcs -> {args.output}")
    print(df.groupby("language")["arc_correct"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
