#!/usr/bin/env python3
"""
01_preprocess_treebank.py
=========================
Parse a SUD CoNLL-U treebank and produce a filtered, preprocessed
.jsonl file ready for the attention extraction pipeline.

Usage
-----
    python src/01_preprocess_treebank.py \
        --input  data/raw/SUD_English-EWT \
        --output data/processed/en.jsonl \
        --lang   en \
        --config configs/experiment_config.yaml

The script reads **all** .conllu files under --input (train, dev, test)
and writes a single .jsonl.  Filtering parameters (min/max sentence
length, MWT policy) are read from the YAML config.
"""

import argparse
import yaml
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer

from src.utils.conllu_utils import (
    load_treebank,
    sentence_to_record,
    write_jsonl,
)

# ── fragmentation ratio ────────────────────────────────────

_FRAG_TOKENIZER = None

def _get_frag_tokenizer():
    """Lazy-load the mBERT tokenizer for fragmentation computation."""
    global _FRAG_TOKENIZER
    if _FRAG_TOKENIZER is None:
        _FRAG_TOKENIZER = AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
    return _FRAG_TOKENIZER

def compute_fragmentation(tokens: list) -> float:
    """Compute subword fragmentation ratio for a sentence.

    Returns n_subwords / n_words.  A ratio of 1.0 means no
    fragmentation; higher values indicate heavier WordPiece splitting.
    """
    tokenizer = _get_frag_tokenizer()
    n_words = len(tokens)
    if n_words == 0:
        return 1.0
    n_subwords = sum(len(tokenizer.tokenize(t)) for t in tokens)
    return n_subwords / n_words


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a SUD treebank into .jsonl."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the SUD treebank directory (contains .conllu files).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .jsonl path.",
    )
    parser.add_argument(
        "--lang", required=True,
        choices=["en", "hi", "fr", "ja", "ko", "es"],
        help="ISO 639-1 language code.",
    )
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
        help="Path to the experiment config YAML.",
    )
    args = parser.parse_args()

    # ── load config ─────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preproc = cfg.get("preprocessing", {})
    min_len = preproc.get("min_sent_length", 5)
    max_len = preproc.get("max_sent_length", 40)
    skip_mwt = preproc.get("skip_mwt", True)

    lang_cfg = cfg["languages"][args.lang]
    treebank_name = lang_cfg["treebank_name"]

    # ── load and filter ─────────────────────────────────────
    print(f"Loading treebank from {args.input} …")
    sentences = load_treebank(args.input)
    print(f"  Found {len(sentences)} raw sentences.")

    records = []
    skipped = 0
    for sent in sentences:
        rec = sentence_to_record(
            sent,
            lang=args.lang,
            treebank_name=treebank_name,
            min_len=min_len,
            max_len=max_len,
            skip_mwt=skip_mwt,
        )
        if rec is not None:
            rec["frag_ratio"] = compute_fragmentation(rec["tokens"])
            records.append(rec)
        else:
            skipped += 1

    n = write_jsonl(records, args.output)
    print(f"  Wrote {n} records to {args.output}")
    print(f"  Skipped {skipped} sentences (length/MWT filters).")


if __name__ == "__main__":
    main()
