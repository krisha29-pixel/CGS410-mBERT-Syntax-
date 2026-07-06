"""
conllu_utils.py
---------------
Helpers for parsing SUD/UD CoNLL-U files and producing the
preprocessed .jsonl records expected by the rest of the pipeline.
"""

import json
import glob
import os
from pathlib import Path
from typing import List, Dict, Optional

from conllu import parse as conllu_parse


# ── deprel → class mapping ─────────────────────────────────

CORE_DEPRELS = {
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp",
}
FUNCTIONAL_DEPRELS = {
    "det", "case", "mark", "aux", "cop", "clf",
}


def classify_deprel(deprel: str) -> str:
    """Map a UD/SUD deprel label to one of {core, functional, other}."""
    # Strip any subtypes (e.g.  "nsubj:pass" → "nsubj")
    base = deprel.split(":")[0] if deprel else "other"
    if base in CORE_DEPRELS:
        return "core"
    if base in FUNCTIONAL_DEPRELS:
        return "functional"
    return "other"


# ── sentence-level checks ──────────────────────────────────

def _has_multiword_tokens(token_list) -> bool:
    """Return True if the sentence contains multi-word tokens (MWTs)."""
    for tok in token_list:
        # MWT entries have a range id like "1-2"
        if isinstance(tok["id"], tuple):
            return True
    return False


def _token_is_word(tok) -> bool:
    """Return True if the token is a regular word (not MWT range, not
    empty node)."""
    return isinstance(tok["id"], int)


# ── main parsing routine ───────────────────────────────────

def parse_conllu_file(filepath: str) -> list:
    """Parse a single .conllu file and return a list of conllu TokenList
    objects."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = fh.read()
    return conllu_parse(data)


def load_treebank(treebank_dir: str) -> list:
    """Load standard UD train/dev/test .conllu files from *treebank_dir*.

    Only loads top-level .conllu files to avoid picking up raw source
    files in subdirectories like ``not-to-release/``.
    """
    # Only top-level .conllu files (no recursive descent)
    pattern = os.path.join(treebank_dir, "*.conllu")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No .conllu files found in {treebank_dir}"
        )
    sentences = []
    for fp in files:
        sentences.extend(parse_conllu_file(fp))
    return sentences


def sentence_to_record(
    token_list,
    lang: str,
    treebank_name: str,
    min_len: int = 5,
    max_len: int = 40,
    skip_mwt: bool = True,
) -> Optional[Dict]:
    """Convert a conllu TokenList into a pipeline-ready dict.

    Returns None if the sentence should be skipped (too short/long,
    contains MWTs when *skip_mwt* is True, etc.).
    """
    if skip_mwt and _has_multiword_tokens(token_list):
        return None

    # Keep only real word tokens
    words = [tok for tok in token_list if _token_is_word(tok)]
    n = len(words)

    if n < min_len or n > max_len:
        return None

    tokens = []
    gold_heads = []
    gold_deprels = []
    deprel_classes = []
    dep_lengths = []

    for tok in words:
        form = tok["form"]
        head = tok["head"]           # 1-indexed; 0 = root
        deprel = tok["deprel"] or "dep"

        tokens.append(form)
        # Convert to 0-indexed (root → -1 internally, we'll store as 0
        # to match standard MST convention where root id = 0).
        gold_heads.append(head)       # keep 1-indexed for now
        gold_deprels.append(deprel)
        deprel_classes.append(classify_deprel(deprel))

        # dep_length = |position_of_dependent − position_of_head|
        # For root tokens (head == 0) we store length as 0 (excluded
        # from regression later).
        pos = tok["id"]               # 1-indexed position
        if head == 0:
            dep_lengths.append(0)
        else:
            dep_lengths.append(abs(pos - head))

    sent_id_meta = token_list.metadata.get("sent_id", "")
    sent_id = f"{lang}-{sent_id_meta}" if sent_id_meta else f"{lang}-unk"

    return {
        "sent_id": sent_id,
        "lang": lang,
        "treebank": treebank_name,
        "tokens": tokens,
        "gold_heads": gold_heads,       # 1-indexed; 0 = root
        "gold_deprels": gold_deprels,
        "deprel_classes": deprel_classes,
        "dep_lengths": dep_lengths,
        "sent_len": n,
    }


def write_jsonl(records: List[Dict], output_path: str) -> int:
    """Write a list of dicts to a JSON-lines file.  Returns the count
    of records written."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def read_jsonl(filepath: str) -> List[Dict]:
    """Read a JSON-lines file back into a list of dicts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
