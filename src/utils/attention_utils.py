"""
attention_utils.py
------------------
Extract word-level attention matrices from bert-base-multilingual-cased.

Key design decisions
~~~~~~~~~~~~~~~~~~~~
* Columns (attended-to): use the **first** subword of each word.
* Rows   (attending):    **sum** over all subwords of each word.
* [CLS] and [SEP] are stripped before the word-level matrix is returned.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple


# ── model loading ───────────────────────────────────────────

_CACHE = {}


def load_model_and_tokenizer(
    model_name: str = "bert-base-multilingual-cased",
    device: str = "cpu",
):
    """Load and cache the model + tokenizer.  Returns (model, tokenizer)."""
    key = (model_name, device)
    if key not in _CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, output_attentions=True
        )
        model.eval()
        model.to(device)
        _CACHE[key] = (model, tokenizer)
    return _CACHE[key]


# ── subword → word mapping ──────────────────────────────────

def _build_word_map(encoding) -> List[List[int]]:
    """Build a list-of-lists mapping word index → list of subword
    positions (excluding [CLS]=None and [SEP]=None in word_ids).

    encoding: output of tokenizer(text, return_tensors="pt")

    Returns
    -------
    word_groups : list[list[int]]
        word_groups[w] = list of subword indices belonging to word w.
    """
    word_ids = encoding.word_ids(batch_index=0)  # None for special tokens
    groups = {}
    for sw_idx, wid in enumerate(word_ids):
        if wid is None:
            continue  # skip [CLS], [SEP], [PAD]
        groups.setdefault(wid, []).append(sw_idx)

    # Return in word order (0, 1, 2, …)
    n_words = max(groups.keys()) + 1 if groups else 0
    return [groups.get(w, []) for w in range(n_words)]


# ── core extraction function ───────────────────────────────

@torch.no_grad()
def extract_word_attention(
    tokens: List[str],
    model,
    tokenizer,
    device: str = "cpu",
) -> np.ndarray:
    """Run a forward pass and return word-level attention.

    Parameters
    ----------
    tokens : list[str]
        The gold-standard word-level tokens for one sentence.
    model : transformers model (with output_attentions=True).
    tokenizer : matching tokenizer.
    device : "cpu" or "cuda".

    Returns
    -------
    attn : np.ndarray, shape (num_layers, num_heads, n_words, n_words)
        Word-level attention matrices (float32).
    """
    # Tokenize – join words with space; the tokenizer will re-split
    text = " ".join(tokens)
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
    raw_attentions = outputs.attentions  # tuple of length num_layers

    word_groups = _build_word_map(encoding)
    n_words = len(word_groups)

    if n_words != len(tokens):
        # Alignment failure – this can happen if the tokenizer merges
        # characters in unexpected ways (especially for CJK scripts).
        # Callers should handle this gracefully.
        raise ValueError(
            f"Word alignment mismatch: expected {len(tokens)} words, "
            f"got {n_words} from tokenizer."
        )

    num_layers = len(raw_attentions)
    num_heads = raw_attentions[0].shape[1]

    word_attn = np.zeros(
        (num_layers, num_heads, n_words, n_words), dtype=np.float32
    )

    for layer_idx in range(num_layers):
        # (1, heads, seq, seq) → (heads, seq, seq)
        attn_l = raw_attentions[layer_idx][0].cpu().numpy()

        for h in range(num_heads):
            A = attn_l[h]  # (seq, seq)

            for w_i, sw_rows in enumerate(word_groups):
                for w_j, sw_cols in enumerate(word_groups):
                    # Row aggregation: SUM over subwords of w_i
                    # Col selection:  FIRST subword of w_j only
                    first_col = sw_cols[0]
                    val = sum(A[sr, first_col] for sr in sw_rows)
                    word_attn[layer_idx, h, w_i, w_j] = val

    return word_attn


# ── batch helper ────────────────────────────────────────────

def extract_attention_batch(
    records: list,
    model,
    tokenizer,
    device: str = "cpu",
    precision: str = "float32",
) -> dict:
    """Extract word-level attention for a list of sentence records.

    Parameters
    ----------
    records : list[dict]
        Each dict must have a "tokens" key (list of word strings) and
        a "sent_id" key.

    Returns
    -------
    result : dict[str, np.ndarray]
        Mapping sent_id → attention array.  Sentences that fail
        alignment are silently skipped (with a printed warning).
    """
    result = {}
    for rec in records:
        sid = rec["sent_id"]
        toks = rec["tokens"]
        try:
            attn = extract_word_attention(toks, model, tokenizer, device)
            if precision == "float16":
                attn = attn.astype(np.float16)
            result[sid] = attn
        except ValueError as exc:
            print(f"  [!] skipping {sid}: {exc}")
    return result
