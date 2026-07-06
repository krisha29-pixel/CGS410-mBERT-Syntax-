#!/usr/bin/env python3
"""
08_best_head_per_relation.py
============================
Finds the single best attention head for specific dependency relations
(e.g., nsubj, obj, amod) across all 144 heads. This demonstrates that
different heads specialize in different grammatical functions.

Usage
-----
    python src/08_best_head_per_relation.py
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.conllu_utils import read_jsonl

def main():
    parser = argparse.ArgumentParser(description="Find best head for specific relations.")
    parser.add_argument("--predictions-dir", default="results/predicted_trees/")
    parser.add_argument("--gold-dir", default="data/processed/")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    languages = list(cfg["languages"].keys())
    target_rels = ["nsubj", "obj", "amod", "case", "mark"]
    
    print("Finding specialized heads for specific relations...\n")

    for lang in languages:
        gold_path = Path(args.gold_dir) / f"{lang}.jsonl"
        pred_path = Path(args.predictions_dir) / f"{lang}_trees.jsonl"
        
        if not gold_path.exists() or not pred_path.exists():
            continue

        gold_records = read_jsonl(str(gold_path))
        pred_records = read_jsonl(str(pred_path))
        
        gold_by_id = {r["sent_id"]: r for r in gold_records}

        # Track correctly identified arcs per relation, per head.
        # {rel: {head_key: correct_count}}
        correct_counts = defaultdict(lambda: defaultdict(int))
        total_counts = defaultdict(int)

        for prec in pred_records:
            sid = prec["sent_id"]
            if sid not in gold_by_id:
                continue
                
            gold = gold_by_id[sid]
            g_heads = gold["gold_heads"]
            g_rels = gold["gold_deprels"]

            # Count total possible connections for each relation type in this sentence
            for i, rel in enumerate(g_rels):
                # Using split(":") to handle subtypes like nsubj:pass
                base_rel = rel.split(":")[0] 
                if base_rel in target_rels:
                    total_counts[base_rel] += 1

            # Check every head's prediction
            for head_key, p_heads in prec["predictions"].items():
                for i, rel in enumerate(g_rels):
                    base_rel = rel.split(":")[0]
                    if base_rel in target_rels:
                        # 0-indexed alignment between lengths, ignore root
                        if i < len(p_heads) and p_heads[i] == g_heads[i]:
                            correct_counts[base_rel][head_key] += 1
        
        print(f"=== {lang.upper()} ===")
        for rel in target_rels:
            if total_counts[rel] == 0:
                continue
            
            best_head = None
            best_acc = -1.0
            
            for head_key, correct in correct_counts[rel].items():
                acc = correct / total_counts[rel]
                if acc > best_acc:
                    best_acc = acc
                    best_head = head_key
                    
            print(f"  {rel.ljust(6)} | Best Head: L{best_head.split('-')[0].zfill(2)}-H{best_head.split('-')[1].zfill(2)} | Acc: {best_acc:.3f} (N={total_counts[rel]})")
        print()

if __name__ == "__main__":
    main()
