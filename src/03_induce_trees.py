#!/usr/bin/env python3
"""
03_induce_trees.py
==================
Apply Chu-Liu/Edmonds maximum spanning arborescence to every attention
head for every sentence, producing predicted dependency trees.

Usage
-----
    python src/03_induce_trees.py \
        --attention data/attention/en_attention.npz \
        --gold      data/processed/en.jsonl \
        --output    results/predicted_trees/en_trees.jsonl \
        --config    configs/experiment_config.yaml

Output format (one JSON object per line):
    {
        "sent_id": "en-ewt-train-00001",
        "predictions": {
            "0-0": [2, 0, 2, ...],   // layer 0, head 0 → list of heads
            "0-1": [...],
            ...
            "11-11": [...]
        }
    }
"""

import argparse
import json
import yaml
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.conllu_utils import read_jsonl
from src.utils.mst import attention_to_tree


def main():
    parser = argparse.ArgumentParser(
        description="Induce dependency trees from attention via CLE MST."
    )
    parser.add_argument("--attention", required=True, help=".npz attention file.")
    parser.add_argument("--gold", required=True, help="Gold .jsonl file.")
    parser.add_argument("--output", required=True, help="Output .jsonl file.")
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
        help="Experiment config YAML.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root_strategy = cfg.get("tree_induction", {}).get(
        "root_weight_strategy", "mean"
    )
    num_layers = cfg["model"]["num_layers"]
    num_heads = cfg["model"]["num_heads"]

    # Load data
    print(f"Loading attention from {args.attention} …")
    attn_data = np.load(args.attention, allow_pickle=True)
    records = read_jsonl(args.gold)

    # Build a lookup for records by sent_id
    rec_by_id = {r["sent_id"]: r for r in records}

    # Process
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    outf = open(args.output, "w", encoding="utf-8")
    processed = 0

    sent_ids = [r["sent_id"] for r in records if r["sent_id"] in attn_data]

    for sid in tqdm(sent_ids, desc="Inducing trees"):
        attn_tensor = attn_data[sid]  # (12, 12, n_words, n_words)
        # Handle float16 → float64 for numerical stability in CLE
        attn_tensor = attn_tensor.astype(np.float64)

        preds = {}
        for l in range(num_layers):
            for h in range(num_heads):
                A = attn_tensor[l, h]  # (n_words, n_words)
                tree = attention_to_tree(A, root_weight_strategy=root_strategy)
                preds[f"{l}-{h}"] = tree

        outf.write(json.dumps({
            "sent_id": sid,
            "predictions": preds,
        }, ensure_ascii=False) + "\n")
        processed += 1

    outf.close()
    print(f"Wrote {processed} sentence predictions to {args.output}")


if __name__ == "__main__":
    main()
