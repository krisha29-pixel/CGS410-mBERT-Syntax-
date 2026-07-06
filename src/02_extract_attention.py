#!/usr/bin/env python3
"""
02_extract_attention.py
=======================
Run bert-base-multilingual-cased on preprocessed sentences and save
word-level attention tensors to a compressed .npz archive.

Usage
-----
    python src/02_extract_attention.py \
        --input  data/processed/en.jsonl \
        --output data/attention/en_attention.npz \
        --config configs/experiment_config.yaml

Each entry in the .npz file is keyed by sent_id and contains a NumPy
array of shape (12, 12, n_words, n_words) — one 2-D attention matrix
per (layer, head) pair, aggregated to word level.
"""

import argparse
import yaml
import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.conllu_utils import read_jsonl
from src.utils.attention_utils import (
    load_model_and_tokenizer,
    extract_word_attention,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract word-level attention from mBERT."
    )
    parser.add_argument("--input", required=True, help="Input .jsonl file.")
    parser.add_argument("--output", required=True, help="Output .npz file.")
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
        help="Experiment config YAML.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    attn_cfg = cfg.get("attention", {})
    device = attn_cfg.get("device", "cpu")
    precision = attn_cfg.get("precision", "float32")

    print(f"Loading model '{model_name}' on {device} …")
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    records = read_jsonl(args.input)
    print(f"Processing {len(records)} sentences …")

    results = {}
    skipped = 0

    for rec in tqdm(records, desc="Extracting attention"):
        sid = rec["sent_id"]
        toks = rec["tokens"]
        try:
            attn = extract_word_attention(toks, model, tokenizer, device)
            if precision == "float16":
                attn = attn.astype(np.float16)
            results[sid] = attn
        except ValueError as exc:
            print(f"  [!] skipping {sid}: {exc}")
            skipped += 1

    # Save as .npz
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **results)
    print(f"Saved {len(results)} attention tensors to {args.output}")
    if skipped:
        print(f"  [!] {skipped} sentences skipped due to alignment errors.")


if __name__ == "__main__":
    main()
