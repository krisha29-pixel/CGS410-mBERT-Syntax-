#!/usr/bin/env python3
"""
10_head_transfer.py
===================
Cross-lingual head transfer experiment: evaluate English's best overall
head (and optionally per-relation best heads) on all other languages.

Key insight: the predicted trees files already contain predictions for ALL
144 heads per sentence, so we just read the appropriate head key and score
against gold — no new model runs needed.

Usage
-----
    python src/10_head_transfer.py \
        --predictions-dir results/predicted_trees/ \
        --gold-dir        data/processed/ \
        --uas-dir         results/uas_tables/ \
        --output          results/transfer/transfer_results.csv \
        --config          configs/experiment_config.yaml
"""

import argparse
import json
import yaml
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.conllu_utils import read_jsonl
from src.utils.eval_utils import compute_uas


def evaluate_head_on_language(pred_records, gold_by_id, layer, head,
                              filter_deprel=None):
    """Evaluate a specific (layer, head) on a language's data.

    Parameters
    ----------
    pred_records : list of dicts with 'predictions' key
    gold_by_id : dict mapping sent_id -> gold record
    layer, head : ints identifying the attention head
    filter_deprel : optional str — if set, only count arcs with this
                    base deprel (e.g. 'nsubj')

    Returns
    -------
    uas : float
    """
    head_key = f"{layer}-{head}"
    total_correct = 0
    total_arcs = 0

    for prec in pred_records:
        sid = prec["sent_id"]
        if sid not in gold_by_id:
            continue
        gold = gold_by_id[sid]
        pred_heads = prec["predictions"].get(head_key)
        if pred_heads is None:
            continue

        gold_heads = gold["gold_heads"]
        gold_deprels = gold.get("gold_deprels", [])

        for i in range(len(gold_heads)):
            if gold_heads[i] == 0:
                continue  # skip root

            if filter_deprel is not None:
                if i >= len(gold_deprels):
                    continue
                base_rel = gold_deprels[i].split(":")[0]
                if base_rel != filter_deprel:
                    continue

            total_arcs += 1
            if i < len(pred_heads) and pred_heads[i] == gold_heads[i]:
                total_correct += 1

    return total_correct / total_arcs if total_arcs > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Cross-lingual head transfer experiment."
    )
    parser.add_argument("--predictions-dir", default="results/predicted_trees/")
    parser.add_argument("--gold-dir", default="data/processed/")
    parser.add_argument("--uas-dir", default="results/uas_tables/")
    parser.add_argument("--output", default="results/transfer/transfer_results.csv")
    parser.add_argument("--source", default="en",
                        help="Source language whose best head is transferred.")
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_langs = list(cfg["languages"].keys())
    target_langs = [l for l in all_langs if l != args.source]

    # Load source language's best head
    source_best_path = Path(args.uas_dir) / f"{args.source}_best_head.json"
    with open(source_best_path) as f:
        source_best = json.load(f)
    src_layer = source_best["best_layer"]
    src_head = source_best["best_head"]
    print(f"Source: {args.source} best head = L{src_layer}-H{src_head} "
          f"(UAS={source_best['best_uas']})")

    # Relations to test per-relation transfer on
    target_rels = ["nsubj", "obj", "det", "case", "amod", "advmod", "obl"]

    records = []

    for target in target_langs:
        pred_path = Path(args.predictions_dir) / f"{target}_trees.jsonl"
        gold_path = Path(args.gold_dir) / f"{target}.jsonl"

        if not pred_path.exists() or not gold_path.exists():
            print(f"  [{target}] skipping -- data not found")
            continue

        pred_records = read_jsonl(str(pred_path))
        gold_records = read_jsonl(str(gold_path))
        gold_by_id = {r["sent_id"]: r for r in gold_records}

        # Get target's native best head
        target_best_path = Path(args.uas_dir) / f"{target}_best_head.json"
        with open(target_best_path) as f:
            target_best = json.load(f)
        native_layer = target_best["best_layer"]
        native_head = target_best["best_head"]
        native_uas = target_best["best_uas"]

        # Overall transfer
        transferred_uas = evaluate_head_on_language(
            pred_records, gold_by_id, src_layer, src_head
        )
        delta = transferred_uas - native_uas

        records.append({
            "source_lang": args.source,
            "target_lang": target,
            "relation": "_overall_",
            "source_head": f"L{src_layer}-H{src_head}",
            "native_head": f"L{native_layer}-H{native_head}",
            "transferred_uas": round(transferred_uas, 4),
            "native_uas": round(native_uas, 4),
            "delta_uas": round(delta, 4),
        })
        print(f"  [{target}] overall: transferred={transferred_uas:.4f}, "
              f"native={native_uas:.4f}, delta={delta:+.4f}")

        # Per-relation transfer
        for rel in target_rels:
            t_uas = evaluate_head_on_language(
                pred_records, gold_by_id, src_layer, src_head,
                filter_deprel=rel
            )
            n_uas = evaluate_head_on_language(
                pred_records, gold_by_id, native_layer, native_head,
                filter_deprel=rel
            )
            d = t_uas - n_uas if n_uas > 0 else None

            records.append({
                "source_lang": args.source,
                "target_lang": target,
                "relation": rel,
                "source_head": f"L{src_layer}-H{src_head}",
                "native_head": f"L{native_layer}-H{native_head}",
                "transferred_uas": round(t_uas, 4),
                "native_uas": round(n_uas, 4),
                "delta_uas": round(d, 4) if d is not None else None,
            })

    # Save results
    df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows -> {args.output}")

    # Print summary
    overall = df[df["relation"] == "_overall_"]
    if not overall.empty:
        print("\n=== Transfer Summary (overall) ===")
        print(overall[["target_lang", "transferred_uas", "native_uas",
                        "delta_uas"]].to_string(index=False))


if __name__ == "__main__":
    main()
