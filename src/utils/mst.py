"""
mst.py
------
Chu-Liu / Edmonds (CLE) maximum spanning arborescence for inducing
dependency trees from attention weight matrices.

We implement a self-contained CLE so the pipeline does not depend on
stanza or networkx for this critical step.  The algorithm runs in
O(V·E) = O(n³) worst case for a dense n×n graph, which is acceptable
for sentences up to length 40.
"""

import numpy as np
from typing import List


def chu_liu_edmonds(weights: np.ndarray, root: int = 0) -> List[int]:
    """Find the maximum spanning arborescence rooted at *root*.

    Parameters
    ----------
    weights : np.ndarray, shape (n, n)
        weights[i][j] = weight of arc  j → i  (j is head, i is dep).
        Diagonal is ignored (self-loops) .
    root : int
        Index of the root node.  Typically 0.

    Returns
    -------
    heads : list[int]
        heads[i] = index of the predicted head of node i.
        heads[root] = -1 (root has no head).

    Notes
    -----
    This is a standard implementation following Edmonds (1967) /
    Chu & Liu (1965).  We solve the *maximum* arborescence by keeping
    the original weights (many implementations negate weights and find
    the minimum; we directly track maxima to avoid sign confusion).
    """
    n = weights.shape[0]
    assert weights.shape == (n, n), "Weight matrix must be square."

    # Make a copy; zero the diagonal
    W = weights.copy()
    np.fill_diagonal(W, 0.0)
    # No arcs into root
    W[root, :] = 0.0

    return _cle_recursive(W, root, list(range(n)))


def _cle_recursive(
    W: np.ndarray,
    root: int,
    nodes: List[int],
) -> List[int]:
    """Recursive CLE core."""
    n_total = W.shape[0]
    # Step 1: For each non-root node, pick the best incoming arc
    best_heads = {}
    for v in nodes:
        if v == root:
            continue
        candidates = [(W[v, u], u) for u in nodes if u != v and W[v, u] > 0]
        if not candidates:
            # No incoming edges – attach to root as fallback
            best_heads[v] = root
        else:
            best_heads[v] = max(candidates, key=lambda x: x[0])[1]

    # Step 2: Check for cycles among the chosen arcs
    visited = {}
    cycles = []
    in_cycle = [False] * n_total

    for v in nodes:
        if v == root:
            continue
        path = []
        cur = v
        while cur not in visited and cur != root:
            visited[cur] = v  # mark that we visited cur on v's walk
            path.append(cur)
            cur = best_heads.get(cur, root)

        # If cur is in the current path, we have a cycle
        if cur != root and visited.get(cur) == v and cur in path:
            cycle_start = path.index(cur)
            cycle = path[cycle_start:]
            if len(cycle) > 1:
                cycles.append(cycle)
                for c in cycle:
                    in_cycle[c] = True

    # If no cycles, we're done
    if not cycles:
        heads = [-1] * n_total
        for v, h in best_heads.items():
            heads[v] = h
        heads[root] = -1
        return heads

    # Step 3: Contract cycles
    # We only contract one cycle at a time (simplest correct approach)
    cycle = cycles[0]
    cycle_set = set(cycle)
    rep = cycle[0]  # the representative node for the contracted cycle

    # Build new node list (replace cycle with rep)
    new_nodes = [v for v in nodes if v not in cycle_set or v == rep]

    # Build new weight matrix
    new_W = W.copy()
    for v in nodes:
        if v in cycle_set and v != rep:
            continue
        if v == rep:
            continue
        # Incoming to cycle: pick best way to enter the cycle
        best_in = -np.inf
        for c in cycle:
            w_incoming = W[c, v]
            # Subtract the arc that c already uses inside the cycle
            w_existing = W[c, best_heads[c]]
            adjusted = w_incoming - w_existing
            if adjusted > best_in:
                best_in = adjusted
                new_W[rep, v] = w_incoming  # keep original for tracing
        new_W[rep, v] = best_in + max(W[c, best_heads[c]] for c in cycle)

        # Outgoing from cycle
        best_out = -np.inf
        for c in cycle:
            if W[v, c] > best_out:
                best_out = W[v, c]
        new_W[v, rep] = best_out

    # Zero out old cycle nodes
    for c in cycle:
        if c == rep:
            continue
        for v in nodes:
            new_W[c, v] = 0.0
            new_W[v, c] = 0.0

    # Recurse
    contracted_heads = _cle_recursive(new_W, root, new_nodes)

    # Step 4: Expand the cycle
    # Who enters the cycle?  contracted_heads[rep] = some external node
    entering_head = contracted_heads[rep]
    # Find which cycle node that external head actually connects to
    best_entry_node = rep
    best_entry_weight = W[rep, entering_head]
    for c in cycle:
        if W[c, entering_head] > best_entry_weight:
            best_entry_weight = W[c, entering_head]
            best_entry_node = c

    # Assign heads within the cycle
    heads = contracted_heads[:]
    heads[best_entry_node] = entering_head
    for c in cycle:
        if c == best_entry_node:
            continue
        heads[c] = best_heads[c]

    heads[root] = -1
    return heads


# ── convenience wrapper ─────────────────────────────────────

def attention_to_tree(
    attn_matrix: np.ndarray,
    root_weight_strategy: str = "mean",
) -> List[int]:
    """Convert an n_words × n_words attention matrix into a dependency
    tree using CLE.

    Parameters
    ----------
    attn_matrix : np.ndarray, shape (n_words, n_words)
        attn_matrix[i, j] = how much word i attends to word j.
        Interpreted as the weight of arc  j → i  (j = head, i = dep).
    root_weight_strategy : "mean" | "max"
        How to set root → word edge weights.

    Returns
    -------
    predicted_heads : list[int]
        Length n_words.  predicted_heads[i] = 1-indexed head of word i.
        Root word gets head = 0.
    """
    n = attn_matrix.shape[0]
    # Build (n+1) × (n+1) matrix with virtual root at index 0
    W = np.zeros((n + 1, n + 1), dtype=np.float64)

    # Fill word-word block (indices 1..n)
    W[1:, 1:] = attn_matrix.astype(np.float64)

    # Root → word edges (W[word, root=0])
    for i in range(n):
        if root_weight_strategy == "max":
            W[i + 1, 0] = float(np.max(attn_matrix[:, i]))
        else:
            W[i + 1, 0] = float(np.mean(attn_matrix[:, i]))

    # No arcs into root
    W[0, :] = 0.0
    # No self-loops
    np.fill_diagonal(W, 0.0)

    heads_0indexed = chu_liu_edmonds(W, root=0)

    # Convert back: internal uses 0-indexed with root=0;
    # output uses 1-indexed gold-standard convention (0 = root)
    predicted_heads = []
    for i in range(1, n + 1):
        h = heads_0indexed[i]
        if h == 0 or h == -1:
            predicted_heads.append(0)  # attached to root
        else:
            predicted_heads.append(h)  # already 1-indexed since words
                                        # start at index 1 in W

    return predicted_heads
