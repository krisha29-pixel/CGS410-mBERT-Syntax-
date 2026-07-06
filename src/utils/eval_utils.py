"""
eval_utils.py
-------------
Evaluation helpers: UAS computation, baselines, and per-arc
accuracy extraction for the regression DataFrame.
"""

import numpy as np
from typing import List, Dict, Tuple


# ── UAS ─────────────────────────────────────────────────────

def compute_uas(
    predicted_heads: List[int],
    gold_heads: List[int],
    exclude_root: bool = True,
) -> float:
    """Compute Unlabeled Attachment Score.

    Both lists are 1-indexed: head = 0 means root.
    If *exclude_root* is True, tokens whose gold head is 0 (i.e., the
    root token) are excluded from the count.

    Returns UAS as a float in [0, 1].
    """
    assert len(predicted_heads) == len(gold_heads)
    correct = 0
    total = 0
    for pred, gold in zip(predicted_heads, gold_heads):
        if exclude_root and gold == 0:
            continue
        total += 1
        if pred == gold:
            correct += 1
    return correct / total if total > 0 else 0.0


def per_arc_accuracy(
    predicted_heads: List[int],
    gold_heads: List[int],
) -> List[int]:
    """Return a list of 0/1 for each token (1 = correct head).
    Root tokens (gold head == 0) get value -1 (to be filtered later)."""
    result = []
    for pred, gold in zip(predicted_heads, gold_heads):
        if gold == 0:
            result.append(-1)   # root token – exclude from regression
        elif pred == gold:
            result.append(1)
        else:
            result.append(0)
    return result


# ── structural baselines ────────────────────────────────────

def right_branching_heads(n: int) -> List[int]:
    """Right-branching baseline: head[i] = i-1 (previous word).
    First word is root (head = 0)."""
    heads = [0]  # word 1's head is root
    for i in range(2, n + 1):
        heads.append(i - 1)
    return heads


def left_branching_heads(n: int) -> List[int]:
    """Left-branching baseline: head[i] = i+1 (next word).
    Last word is root (head = 0)."""
    heads = []
    for i in range(1, n):
        heads.append(i + 1)
    heads.append(0)  # last word is root
    return heads


def random_baseline_heads(n: int, rng=None) -> List[int]:
    """Random baseline: each word's head is a uniformly random other word
    (or root).  This does NOT guarantee a valid tree—it's only used as
    a loose lower bound."""
    if rng is None:
        rng = np.random.default_rng()
    heads = []
    for i in range(1, n + 1):
        candidates = [j for j in range(0, n + 1) if j != i]
        heads.append(int(rng.choice(candidates)))
    return heads


def compute_baseline_uas(
    gold_heads_list: List[List[int]],
    baseline_fn,
    n_runs: int = 1,
    rng=None,
) -> float:
    """Compute the mean UAS of a baseline across all sentences.
    For random baselines, average over *n_runs* runs."""
    if rng is None:
        rng = np.random.default_rng(42)

    all_uas = []
    for _ in range(n_runs):
        run_correct = 0
        run_total = 0
        for gold in gold_heads_list:
            n = len(gold)
            if baseline_fn == random_baseline_heads:
                pred = baseline_fn(n, rng=rng)
            else:
                pred = baseline_fn(n)
            for p, g in zip(pred, gold):
                if g == 0:
                    continue
                run_total += 1
                if p == g:
                    run_correct += 1
        all_uas.append(run_correct / run_total if run_total > 0 else 0.0)
    return float(np.mean(all_uas))


# ── summary helpers ─────────────────────────────────────────

def find_best_head(
    uas_grid: np.ndarray,
) -> Tuple[int, int, float]:
    """Given a 12×12 UAS grid, return (best_layer, best_head, best_uas)."""
    idx = np.unravel_index(np.argmax(uas_grid), uas_grid.shape)
    return int(idx[0]), int(idx[1]), float(uas_grid[idx])
